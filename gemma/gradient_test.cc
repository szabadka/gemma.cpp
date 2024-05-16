// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include <stddef.h>

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "compression/compress.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gradient_test.cc"  //NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
// After highway.h
#include "gemma/ops.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void TestSoftmaxCrossEntropyLossGrad() {
  static const size_t kMaxVocabSize = 64;
  HWY_ALIGN float logits[kMaxVocabSize];
  HWY_ALIGN float grad[kMaxVocabSize];
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 5.0f);
  for (size_t vocab_size = 2; vocab_size <= kMaxVocabSize; vocab_size *= 2) {
    for (size_t iter = 0; iter < 10; ++iter) {
      memset(logits, 0, vocab_size * sizeof(logits[0]));
      if (iter == 0) {
        logits[0] = 5.0f;
      } else if (iter == 1) {
        logits[1] = 5.0f;
      } else {
        for (size_t i = 0; i < vocab_size; ++i) {
          logits[i] = dist(gen);
        }
      }
      for (size_t i = vocab_size; i < kMaxVocabSize; ++i) {
        logits[i] = dist(gen);
      }
      SoftmaxCrossEntropyLossGrad(logits, vocab_size, 0, grad);
      static constexpr double kStep = 0.5e-3;
      static constexpr double kStepScale = 0.5f / kStep;
      for (size_t i = 0; i < vocab_size; ++i) {
        const float x0 = logits[i];
        logits[i] = x0 + kStep;
        double loss1 = SoftmaxCrossEntropyLoss(logits, vocab_size, 0);
        logits[i] = x0 - kStep;
        double loss2 = SoftmaxCrossEntropyLoss(logits, vocab_size, 0);
        const double diff = loss1 - loss2;
        const double exp_grad = diff * kStepScale;
        ASSERT_GE(grad[i] * exp_grad, 0.0)
            << "vocab_size=" << vocab_size << " iter=" << iter << " idx=" << i
            << " loss1=" << loss1 << " loss2=" << loss2 << " diff=" << diff
            << " exp_grad=" << exp_grad << " grad=" << grad[i];
        ASSERT_NEAR(grad[i], exp_grad, 5e-3)
            << "vocab_size=" << vocab_size << " iter=" << iter << " idx=" << i
            << " loss1=" << loss1 << " loss2=" << loss2 << " diff=" << diff;
      }
    }
  }
}

void TestLossGradient() {
  static const size_t kVocabSize = 8;
  static const size_t kSeqLen = 8;
  hwy::ThreadPool pool(8);
  std::vector<int> prompt = { 0, 1, 2, 3, 4 };
  const size_t context_size = 1;
  const size_t num_tokens = prompt.size() - 1;
  HWY_ALIGN float logits[kSeqLen * kVocabSize];
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < num_tokens * kVocabSize; ++i) {
    logits[i] = dist(gen);
  }
  HWY_ALIGN float grad[kSeqLen * kVocabSize];
  LossGradient<kVocabSize>(logits, prompt, context_size, grad, pool);

  static constexpr float kStep = 0.5e-3;
  static constexpr float kStepScale = 0.5f / kStep;
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t i = 0; i < kVocabSize; ++i) {
      size_t idx = pos * kVocabSize + i;
      const float x0 = logits[idx];
      logits[idx] = x0 + kStep;
      float loss1 = CrossEntropyLoss<kVocabSize>(
          logits, prompt, context_size, pool);
      logits[idx] = x0 - kStep;
      float loss2 = CrossEntropyLoss<kVocabSize>(
          logits, prompt, context_size, pool);
      const float diff = loss1 - loss2;
      const float exp_grad = diff * kStepScale;
      ASSERT_NEAR(grad[idx], exp_grad, 5e-3)
          << "idx=" << idx << " loss1=" << loss1 << " loss2=" << loss2
          << " diff=" << diff;
    }
  }
}

#if 0
void TestEndToEnd() {
  hwy::ThreadPool pool(8);
  std::mt19937 gen(42);
  const Model model = Model::GEMMA_TINY;
  WeightStorageT weights_u8 = AllocateWeights(model, pool);
  WeightStorageT grad_u8 = AllocateWeights(model, pool);
  WeightStorageT forward_u8 = AllocateForwardPass(model);
  WeightStorageT backward_u8 = AllocateForwardPass(model);

  using TConfig = ConfigGemmaTiny;
  using TWeights = Weights<TConfig>;
  auto& weights = *reinterpret_cast<TWeights*>(weights_u8.get());
  auto& grad = *reinterpret_cast<TWeights*>(grad_u8.get());

  InitWeights(model, weights_u8, InitMode::RAND_INIT, pool, &gen);
  std::vector<int> prompt = { 0, 1, 2, 3, 4 };
  const size_t context_size = 1;
  float loss = CrossEntropyLossForwardStep(
      prompt, context_size, model, weights_u8, forward_u8, pool);
  printf("loss = %f\n", loss);

  InitWeights(model, grad_u8, InitMode::ZERO_INIT, pool);
  CrossEntropyLossBackwardStep(
      prompt, context_size, model, weights_u8, forward_u8, grad_u8,
      backward_u8, pool);

  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kModelDim = TConfig::kModelDim;

  static constexpr float kStep = 1e-2;
  static constexpr float kStepScale = 0.5f / kStep;
#if 0
  for (size_t j = 0; j < kVocabSize; ++j) {
    for (size_t i = 0; i < kModelDim; ++i) {
      const float* g = grad.embedder_input_embedding.data();
      float* w = weights.embedder_input_embedding.data();
      const size_t idx = j * kVocabSize + i;
      const float w0 = w[idx];
      w[idx] = w0 + kStep;
      float loss1 = CrossEntropyLossForwardStep(
          prompt, context_size, model, weights_u8, forward_u8, pool);
      w[idx] = w0 - kStep;
      float loss2 = CrossEntropyLossForwardStep(
          prompt, context_size, model, weights_u8, forward_u8, pool);
      const float exp_grad = (loss1 - loss2) * kStepScale;
      const float rel_diff = std::abs((g[idx] - exp_grad) / exp_grad);
      ASSERT_LT(rel_diff, 1e-5)
          << "Embedding gradient token " << j << " index " << i
          << " expected " << exp_grad << " [loss1 = " << loss1
          << " loss2 = " << loss2 << "] actual " << g[idx];
    }
  }
#endif

  for (size_t i = 0; i < kModelDim; ++i) {
    const float* g = grad.final_norm_scale.data();
    float* w = weights.final_norm_scale.data();
    const float w0 = w[i];
    w[i] = w0 + kStep;
    float loss1 = CrossEntropyLossForwardStep(
        prompt, context_size, model, weights_u8, forward_u8, pool);
    w[i] = w0 - kStep;
    float loss2 = CrossEntropyLossForwardStep(
        prompt, context_size, model, weights_u8, forward_u8, pool);
    const float exp_grad = (loss1 - loss2) * kStepScale;
    //const float rel_diff = std::abs((g[i] - exp_grad) / exp_grad);
    ASSERT_NEAR(g[i], exp_grad, 1e-2)
        << "Final norm scale gradient index " << i
        << " expected " << exp_grad << " [loss1 = " << loss1
        << " loss2 = " << loss2 << "] actual " << g[i];
  }
}
#endif

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(GradientTest);
HWY_EXPORT_AND_TEST_P(GradientTest, TestSoftmaxCrossEntropyLossGrad);
HWY_EXPORT_AND_TEST_P(GradientTest, TestLossGradient);
#ifdef HWY_AFTER_TEST
HWY_AFTER_TEST();
#endif

}  // namespace gcpp

#endif
