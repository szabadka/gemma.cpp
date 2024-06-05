// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "backprop/optimizer.h"

#include <random>

#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

namespace {
class WeightInitializer {
 public:
  WeightInitializer(std::mt19937& gen) : dist_(0.0f, 1.0f), gen_(gen) {}

  template <size_t N>
  void operator()(const char* name, std::array<float, N>& tensor) {
    for (size_t i = 0; i < N; ++i) {
      tensor[i] = dist_(gen_);
    }
  }
 private:
  std::normal_distribution<float> dist_;
  std::mt19937& gen_;
};

template <typename TConfig>
void RandInitWeights(ByteStorageT& weights_u8, hwy::ThreadPool& pool,
                     std::mt19937& gen) {
  auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  // TODO(szabadka) Use the same weight initialization method as in the python
  // version.
  WeightInitializer init(gen);
  ForEachTensor1<float, TConfig>(init, weights);
}

class WeightUpdater {
 public:
  explicit WeightUpdater(float lr) : lr_(lr) {}

  template <size_t kCapacity>
  void operator()(const char* name, const std::array<float, kCapacity>& grad,
                  std::array<float, kCapacity>& weights) {
    for (size_t i = 0; i < kCapacity; ++i) {
      weights[i] += lr_ * grad[i];
    }
  }

 private:
  float lr_;
};

template <typename TConfig>
void UpdateWeights(const ByteStorageT& grad_u8, float scale,
                   ByteStorageT& weights_u8, hwy::ThreadPool& pool) {
  const auto& grad =
      *reinterpret_cast<const WeightsF<TConfig>*>(grad_u8.get());
  auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  WeightUpdater updater(scale);
  ForEachTensor2<float, TConfig>(updater, grad, weights);
}

class AdamUpdater {
 public:
  explicit AdamUpdater(float alpha, float beta1, float beta2, float epsilon,
                       size_t t)
      : alpha_(alpha), beta1_(beta1), beta2_(beta2), cbeta1_(1.0f - beta1),
        cbeta2_(1.0f - beta2), norm1_(1.0 / (1.0 - std::pow(beta1, t))),
        norm2_(1.0 / (1.0 - std::pow(beta2, t))), epsilon_(epsilon) {}

  template <size_t kCapacity>
  void operator()(const char* name, const std::array<float, kCapacity>& grad,
                  std::array<float, kCapacity>& weights,
                  std::array<float, kCapacity>& grad_m,
                  std::array<float, kCapacity>& grad_v) {
    for (size_t i = 0; i < kCapacity; ++i) {
      grad_m[i] *= beta1_;
      grad_m[i] += cbeta1_ * grad[i];
      grad_v[i] *= beta2_;
      grad_v[i] += cbeta2_ * grad[i] * grad[i];
      const float mhat = grad_m[i] * norm1_;
      const float vhat = grad_v[i] * norm2_;
      weights[i] -= alpha_ * mhat / (std::sqrt(vhat) + epsilon_);
    }
  }

 private:
  float alpha_;
  float beta1_;
  float beta2_;
  float cbeta1_;
  float cbeta2_;
  float norm1_;
  float norm2_;
  float epsilon_;
};

template <typename TConfig>
void AdamUpdate(const ByteStorageT& grad_u8, float alpha, float beta1,
                float beta2, float epsilon, size_t t, ByteStorageT& weights_u8,
                ByteStorageT& grad_m_u8, ByteStorageT& grad_v_u8,
                hwy::ThreadPool& pool) {
  const auto& grad =
      *reinterpret_cast<const WeightsF<TConfig>*>(grad_u8.get());
  auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  auto& grad_m = *reinterpret_cast<WeightsF<TConfig>*>(grad_m_u8.get());
  auto& grad_v = *reinterpret_cast<WeightsF<TConfig>*>(grad_v_u8.get());
  AdamUpdater updater(alpha, beta1, beta2, epsilon, t);
  ForEachTensor4<float, TConfig>(updater, grad, weights, grad_m, grad_v);
}

}  // namespace

void RandInitWeights(Model model, ByteStorageT& weights_u8,
                     hwy::ThreadPool& pool, std::mt19937& gen) {
  switch (model) {
    case Model::GEMMA_2B:
      RandInitWeights<ConfigGemma2B>(weights_u8, pool, gen);
      break;
    case Model::GEMMA_7B:
      RandInitWeights<ConfigGemma7B>(weights_u8, pool, gen);
      break;
    case Model::GRIFFIN_2B:
      RandInitWeights<ConfigGriffin2B>(weights_u8, pool, gen);
      break;
    case Model::GEMMA_TINY:
      RandInitWeights<ConfigGemmaTiny>(weights_u8, pool, gen);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

void UpdateWeights(Model model, const ByteStorageT& grad, float scale,
                   ByteStorageT& weights, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      UpdateWeights<ConfigGemma2B>(grad, scale, weights, pool);
      break;
    case Model::GEMMA_7B:
      UpdateWeights<ConfigGemma7B>(grad, scale, weights, pool);
      break;
    case Model::GRIFFIN_2B:
      UpdateWeights<ConfigGriffin2B>(grad, scale, weights, pool);
      break;
    case Model::GEMMA_TINY:
      UpdateWeights<ConfigGemmaTiny>(grad, scale, weights, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

void AdamUpdate(Model model, const ByteStorageT& grad, float alpha, float beta1,
                float beta2, float epsilon, size_t t, ByteStorageT& weights,
                ByteStorageT& grad_m, ByteStorageT& grad_v,
                hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      AdamUpdate<ConfigGemma2B>(
          grad, alpha, beta1, beta2, epsilon, t, weights, grad_m, grad_v, pool);
      break;
    case Model::GEMMA_7B:
      AdamUpdate<ConfigGemma7B>(
          grad, alpha, beta1, beta2, epsilon, t, weights, grad_m, grad_v, pool);
      break;
    case Model::GRIFFIN_2B:
      AdamUpdate<ConfigGriffin2B>(
          grad, alpha, beta1, beta2, epsilon, t, weights, grad_m, grad_v, pool);
      break;
    case Model::GEMMA_TINY:
      AdamUpdate<ConfigGemmaTiny>(
          grad, alpha, beta1, beta2, epsilon, t, weights, grad_m, grad_v, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace gcpp
