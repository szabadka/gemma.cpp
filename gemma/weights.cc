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

#include "gemma/weights.h"

#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/stats.h"

namespace gcpp {

ByteStorageT AllocateWeights(Model model, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return AllocateWeights<float, ConfigGemma2B>(pool);
    case Model::GEMMA_7B:
      return AllocateWeights<float, ConfigGemma7B>(pool);
    case Model::GRIFFIN_2B:
      return AllocateWeights<float, ConfigGriffin2B>(pool);
    case Model::GEMMA_TINY:
      return AllocateWeights<float, ConfigGemmaTiny>(pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

namespace {
template <typename TConfig>
void ZeroInitWeightsT(ByteStorageT& weights, hwy::ThreadPool& pool) {
  ZeroInit<float, TConfig>(
      *reinterpret_cast<Weights<float, TConfig>*>(weights.get()));
}
}  // namespace

void ZeroInitWeights(Model model, ByteStorageT& weights,
                     hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      ZeroInitWeightsT<ConfigGemma2B>(weights, pool);
      break;
    case Model::GEMMA_7B:
      ZeroInitWeightsT<ConfigGemma7B>(weights, pool);
      break;
    case Model::GRIFFIN_2B:
      ZeroInitWeightsT<ConfigGriffin2B>(weights, pool);
      break;
    case Model::GEMMA_TINY:
      ZeroInitWeightsT<ConfigGemmaTiny>(weights, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

namespace {
void LogVec(const char* name, const float* data, float scale, size_t len) {
  hwy::Stats stats;
  for (size_t i = 0; i < len; ++i) {
    stats.Notify(data[i] * scale);
  }
  printf("%-20s  %12zu   %13.10f   %8.5f   %13.10f\n",
         name, len, stats.Min(), stats.Mean(), stats.Max());
}

class WeightLogger {
 public:
  explicit WeightLogger(float scale) : scale_(scale) {}
  template <size_t N>
  void operator()(const char* name, const std::array<float, N>& tensor) {
    LogVec(name, tensor.data(), scale_, N);
    total_weights += N;
  }
  size_t total_weights = 0;

 private:
  float scale_;
};

template <typename TConfig>
    void LogWeightStats(const ByteStorageT& weights_u8, float scale) {
  const auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  WeightLogger logger(scale);
  ForEachTensor1<float, TConfig>(logger, weights);
  printf("%-20s  %12zu\n", "Total", logger.total_weights);
}
}  // namespace

void LogWeightStats(gcpp::Model model, const ByteStorageT& weights,
                    float scale) {
  switch (model) {
    case Model::GEMMA_2B:
      return LogWeightStats<ConfigGemma2B>(weights, scale);
    case Model::GEMMA_7B:
      return LogWeightStats<ConfigGemma7B>(weights, scale);
    case Model::GRIFFIN_2B:
      return LogWeightStats<ConfigGriffin2B>(weights, scale);
    case Model::GEMMA_TINY:
      return LogWeightStats<ConfigGemmaTiny>(weights, scale);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace gcpp
