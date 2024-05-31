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

}  // namespace gcpp
