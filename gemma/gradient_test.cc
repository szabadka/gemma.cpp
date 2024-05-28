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
#include <complex>
#include <random>
#include <vector>

#include "compression/compress.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "gemma/backprop_scalar.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"

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

template<size_t kLen>
void ZeroInit(std::array<float, kLen>& x) {
  for (size_t i = 0; i < kLen; ++i) {
    x[i] = 0.0f;
  }
}

template<typename T, size_t kLen>
void RandInit(std::array<T, kLen>& x, T stddev, std::mt19937& gen) {
  std::normal_distribution<T> dist(0.0, stddev);
  for (size_t i = 0; i < kLen; ++i) {
    x[i] = dist(gen);
  }
}

template<typename T, typename U, size_t kLen>
void Complexify(const std::array<T, kLen>& x,
                std::array<std::complex<U>, kLen>& c_x) {
  for (size_t i = 0; i < kLen; ++i) {
    c_x[i] = std::complex<U>(x[i], 0.0);
  }
}

template<typename T, typename U, size_t kLen>
void StaticCast(const std::array<T, kLen>& a, std::array<U, kLen>& b) {
  for (size_t i = 0; i < kLen; ++i) {
    b[i] = static_cast<U>(a[i]);
  }
}

template<typename T, typename U, size_t N>
void TestNear(const std::array<T, N>& actual, const std::array<U, N>& expected,
              double max_abs_err, double max_rel_err, int line) {
  for (size_t i = 0; i < N; ++i) {
    ASSERT_NEAR(actual[i], expected[i],
                std::max(max_abs_err, std::abs(expected[i]) * max_rel_err))
        << "line: " << line << " dim=" << N << " i=" << i;
  }
}

template<typename T, typename U, size_t N, typename FUNC>
void TestGradient(const std::array<T, N>& grad,
                  std::array<std::complex<U>, N>& x, FUNC func,
                  U step, T max_abs_err, T max_rel_err, int line) {
  const U inv_step = 1.0 / step;
  for (size_t i = 0; i < N; ++i) {
    const U x0 = std::real(x[i]);
    const std::complex<U> x1 = std::complex<U>(x0, step);
    x[i] = x1;
    const std::complex<U> f1 = func();
    const T exp_grad = std::imag(f1) * inv_step;
    x[i] = x0;
    ASSERT_NEAR(grad[i], exp_grad,
                std::max(max_abs_err, std::abs(exp_grad) * max_rel_err))
        << "line: " << line << " dim=" << N << " i=" << i << " f1=" << f1;
  }
}

template<size_t N, typename FUNC>
void TestGradient(const std::array<float, N>& grad,
                  std::array<std::complex<float>, N>& x, FUNC func,
                  float max_abs_err, float max_rel_error, int line) {
  TestGradient(grad, x, func, 1e-30f, max_abs_err, max_rel_error, line);
}

template<size_t N, typename FUNC>
void TestGradient(const std::array<float, N>& grad,
                  std::array<std::complex<double>, N>& x, FUNC func,
                  float max_abs_err, float max_rel_error, int line) {
  TestGradient(grad, x, func, 1e-50, max_abs_err, max_rel_error, line);
}

template<size_t N, typename FUNC>
void TestGradient(const std::array<double, N>& grad,
                  std::array<std::complex<double>, N>& x, FUNC func,
                  double max_abs_err, double max_rel_error, int line) {
  TestGradient(grad, x, func, 1e-50, max_abs_err, max_rel_error, line);
}

static constexpr float kDefaultEpsilon = 1.0 / (1 << 20);

template<typename T>
void TestGradient(const float* grad, size_t dim, float* x, T func, int line,
                  const float f_epsilon = kDefaultEpsilon) {
  const float h = std::pow(f_epsilon, 1.0 / 3.0);
  const float max_err = std::max<float>(
      1e-5, 2.0 * f_epsilon * std::abs(func()) / h);
  for (size_t i = 0; i < dim; ++i) {
    const float x0 = x[i];
    volatile float x1 = x[i] + h;
    volatile float x2 = x[i] - h;
    x[i] = x1;
    const double f1 = func();
    x[i] = x2;
    const double f2 = func();
    x[i] = x0;
    const double diff = f1 - f2;
    const double exp_grad = diff / (x1 - x2);
    ASSERT_NEAR(exp_grad, grad[i], max_err)
          << "line: " << line << " dim=" << dim << " i=" << i
          << " f1=" << f1 << " f2=" << f2;
  }
}

template<size_t N, typename T>
void TestGradient(const std::array<float, N>& grad, std::array<float, N>& x,
                  T func, int line, float max_err = kDefaultEpsilon) {
  TestGradient(grad.data(), N, x.data(), func, line, max_err);
}


void TestSoftmaxCrossEntropyLossGrad() {
  static const size_t kMaxVocabSize = 64;
  HWY_ALIGN float logits[kMaxVocabSize];
  HWY_ALIGN float grad[kMaxVocabSize];
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 50.0f);
  for (size_t vocab_size = 2; vocab_size <= kMaxVocabSize; vocab_size *= 2) {
    auto func = [&]() HWY_ATTR {
      return SoftmaxCrossEntropyLoss(logits, vocab_size, 0);
    };
    for (size_t iter = 0; iter < 10; ++iter) {
      memset(logits, 0, vocab_size * sizeof(logits[0]));
      if (iter == 0) {
        logits[0] = 30.0f;
      } else if (iter == 1) {
        logits[1] = 30.0f;
      } else {
        for (size_t i = 0; i < vocab_size; ++i) {
          logits[i] = dist(gen);
        }
      }
      for (size_t i = vocab_size; i < kMaxVocabSize; ++i) {
        logits[i] = dist(gen);
      }
      SoftmaxCrossEntropyLossGrad(logits, vocab_size, 0, grad);
      TestGradient(grad, vocab_size, logits, func, __LINE__);
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
  auto func = [&]() HWY_ATTR {
    return CrossEntropyLoss<kVocabSize>(logits, prompt, context_size, pool);
  };
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 50.0f);
  for (size_t i = 0; i < num_tokens * kVocabSize; ++i) {
    logits[i] = dist(gen);
  }
  HWY_ALIGN float grad[kSeqLen * kVocabSize];
  LossGradient<kVocabSize>(logits, prompt, context_size, grad, pool);
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    TestGradient(&grad[pos * kVocabSize], kVocabSize, &logits[pos * kVocabSize],
                 func, __LINE__);
  }
}

void TestMatMulVJP() {
  static const size_t kRows = 8;
  static const size_t kCols = 64;
  static const size_t kTokens = 5;
  hwy::ThreadPool pool(8);
  std::mt19937 gen(42);
  HWY_ALIGN std::array<float, kRows * kCols> weights;
  HWY_ALIGN std::array<float, kTokens * kCols> x;
  HWY_ALIGN std::array<float, kTokens * kRows> dy;
  HWY_ALIGN std::array<float, kRows * kCols> grad;
  HWY_ALIGN std::array<float, kTokens * kCols> dx;
  HWY_ALIGN std::array<float, kRows * kCols> f_grad;
  HWY_ALIGN std::array<float, kTokens * kCols> f_dx;
  HWY_ALIGN std::array<double, kRows * kCols> d_weights;
  HWY_ALIGN std::array<double, kTokens * kCols> d_x;
  HWY_ALIGN std::array<double, kRows * kCols> d_grad;
  HWY_ALIGN std::array<double, kTokens * kCols> d_dx;
  HWY_ALIGN std::array<double, kTokens * kRows> d_dy;
  using TC = std::complex<double>;
  std::array<TC, kRows * kCols> c_weights;
  std::array<TC, kTokens * kCols> c_x;
  std::array<TC, kTokens * kRows> c_y;
  std::array<TC, kTokens * kRows> c_dy;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    Complexify(dy, c_dy);
    StaticCast(weights, d_weights);
    StaticCast(x, d_x);
    StaticCast(dy, d_dy);
    auto func = [&]() {
      MatMulT(c_weights.data(), c_x.data(), c_y.data(), kRows, kCols, kTokens);
      return DotT(c_dy.data(), c_y.data(), kTokens * kRows);
    };
    memset(&d_grad, 0, sizeof(d_grad));
    MatMulVJPT(d_weights.data(), d_x.data(), d_dy.data(), d_grad.data(),
               d_dx.data(), kRows, kCols, kTokens);
    TestGradient(d_dx, c_x, func, 1e-12, 1e-12, __LINE__);
    TestGradient(d_grad, c_weights, func, 1e-12, 1e-12, __LINE__);

    memset(&f_grad, 0, sizeof(f_grad));
    MatMulVJPT(weights.data(), x.data(), dy.data(), f_grad.data(), f_dx.data(),
               kRows, kCols, kTokens);

    memset(&grad, 0, sizeof(grad));
    MatMulVJP<kCols, kRows>(weights, x.data(), dy.data(), kTokens, nullptr,
                            grad, dx.data(), pool);
    TestNear(dx, f_dx, 0, 0, __LINE__);
    TestNear(grad, f_grad, 0, 0, __LINE__);
    TestGradient(dx, c_x, func, 5e-5, 5e-5, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5, 5e-5, __LINE__);
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
  auto& forward = *reinterpret_cast<ForwardPass<TConfig>*>(forward_u8.get());
  auto& backward = *reinterpret_cast<ForwardPass<TConfig>*>(backward_u8.get());
  static constexpr size_t kModelDim = TConfig::kModelDim;

  std::vector<int> prompt = { 0, 1, 2, 3, 4 };
  const size_t context_size = 1;
  const size_t num_tokens = prompt.size() - 1;
  const size_t xdim = num_tokens * kModelDim;

  auto func = [&]() HWY_ATTR {
    return CrossEntropyLossForwardStep(
        prompt, context_size, model, weights_u8, forward_u8, pool);
  };

  //RandInit(forward.layers[0].input, 10.0f, gen);
  RandInit(forward.layers[0].att_out, 10.0f, gen);
  //RandInit(forward.layers[0].attention_out, 10.0f, gen);
  //RandInit(forward.final_layer_output, 1.0f, gen);
  //RandInit(forward.final_norm_output, 1.0f, gen);
  //RandInit(forward.raw_logits, 50.0f, gen);
  InitWeights(model, weights_u8, InitMode::RAND_INIT, pool, &gen);
  CrossEntropyLossForwardStep(
      prompt, context_size, model, weights_u8, forward_u8, pool);

  //ZeroInit(backward.layers[0].input);
  ZeroInit(backward.layers[0].att_out);
  //ZeroInit(backward.layers[0].attention_out);
  //ZeroInit(backward.final_layer_output);
  //ZeroInit(backward.final_norm_output);
  //ZeroInit(backward.raw_logits);
  InitWeights(model, grad_u8, InitMode::ZERO_INIT, pool);
  CrossEntropyLossBackwardStep(
      prompt, context_size, model, weights_u8, forward_u8, grad_u8,
      backward_u8, pool);

  //TestGradient(grad.final_norm_scale.data(), kModelDim,
  //             weights.final_norm_scale.data(), func, __LINE__);
  //TestGradient(backward.raw_logits.data(), backward.raw_logits.size(),
  //             forward.raw_logits.data(), func, __LINE__);
  //TestGradient(backward.final_norm_output.data(), xdim,
  //             forward.final_norm_output.data(), func, __LINE__);
  //TestGradient(backward.final_layer_output.data(), xdim,
  //             forward.final_layer_output.data(), func, __LINE__);
  //TestGradient(grad.embedder_input_embedding,
  //             weights.embedder_input_embedding, func, __LINE__);
  TestGradient(grad.final_norm_scale,
               weights.final_norm_scale, func, __LINE__);
  TestGradient(grad.GetLayer(0)->linear_w,
               weights.GetLayer(0)->linear_w, func, __LINE__);
  TestGradient(grad.GetLayer(0)->gating_einsum_w,
               weights.GetLayer(0)->gating_einsum_w, func, __LINE__);
  TestGradient(grad.GetLayer(0)->pre_ffw_norm_scale,
               weights.GetLayer(0)->pre_ffw_norm_scale, func, __LINE__);
  //TestGradient(backward.layers[0].attention_out,
  //             forward.layers[0].attention_out, func, __LINE__);
  TestGradient(grad.GetLayer(0)->attn_vec_einsum_w,
               weights.GetLayer(0)->attn_vec_einsum_w, func, __LINE__);
  TestGradient(backward.layers[0].att_out,
               forward.layers[0].att_out, func, __LINE__);
  //TestGradient(backward.layers[0].input,
  //             forward.layers[0].input, func, __LINE__);
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
HWY_EXPORT_AND_TEST_P(GradientTest, TestMatMulVJP);
//HWY_EXPORT_AND_TEST_P(GradientTest, TestEndToEnd);
#ifdef HWY_AFTER_TEST
HWY_AFTER_TEST();
#endif

}  // namespace gcpp

#endif
