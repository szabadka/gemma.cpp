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

#include "gemma/optimizer.h"

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
  // TODO(szabadka) Implement multi-threaded initialization.
  WeightInitializer init(gen);
  ForEachTensor1<float, TConfig>(init, weights);
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

}  // namespace gcpp
