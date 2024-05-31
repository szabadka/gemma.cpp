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
#include "gemma/sampler.h"
#include "gemma/weights.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gradient_test.cc"  //NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
// After highway.h
#include "gemma/backward-inl.h"
#include "gemma/forward-inl.h"
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
  double sum0 = 0;
  double sum1 = 0;
  double sum01 = 0;
  for (size_t i = 0; i < N; ++i) {
    sum0 += actual[i] * actual[i];
    sum1 += expected[i] * expected[i];
    sum01 += actual[i] * expected[i];
    ASSERT_NEAR(actual[i], expected[i],
                std::max(max_abs_err, std::abs(expected[i]) * max_rel_err))
        << "line: " << line << " dim=" << N << " i=" << i;
  }
  double norm_dot = sum01 / std::sqrt(sum0) / std::sqrt(sum1);
  ASSERT_NEAR(norm_dot, 1.0, 1e-7) << "line: " << line;
}

template<typename T, typename U, size_t N, typename FUNC>
void TestGradient(const std::array<T, N>& grad,
                  std::array<std::complex<U>, N>& x, FUNC func,
                  U step, T max_abs_err, T max_rel_err, int line) {
  std::array<T, N> exp_grad;
  const U inv_step = 1.0 / step;
  for (size_t i = 0; i < N; ++i) {
    const U x0 = std::real(x[i]);
    const std::complex<U> x1 = std::complex<U>(x0, step);
    x[i] = x1;
    const std::complex<U> f1 = func();
    exp_grad [i] = std::imag(f1) * inv_step;
    x[i] = x0;
  }
  TestNear(grad, exp_grad, max_abs_err, max_rel_err, line);
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
    MatMulVJP<kCols, kRows>(weights, x.data(), dy.data(), kTokens,
                            grad, dx.data(), pool);
    TestNear(dx, f_dx, 0, 0, __LINE__);
    TestNear(grad, f_grad, 0, 0, __LINE__);
    TestGradient(dx, c_x, func, 5e-5, 5e-5, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5, 5e-5, __LINE__);
  }
}

template<typename T, typename TConfig>
void RandInit(Layer<T, TConfig>& w, T stddev, std::mt19937& gen) {
  RandInit(w.pre_attention_norm_scale, stddev, gen);
  RandInit(w.attn_vec_einsum_w, stddev, gen);
  RandInit(w.qkv_einsum_w, stddev, gen);
  RandInit(w.pre_ffw_norm_scale, stddev, gen);
  RandInit(w.gating_einsum_w, stddev, gen);
  RandInit(w.linear_w, stddev, gen);
}

template<typename T, typename U, typename TConfig>
void Complexify(const Layer<T, TConfig>& w,
                Layer<std::complex<U>, TConfig>& c_w) {
  Complexify(w.pre_attention_norm_scale, c_w.pre_attention_norm_scale);
  Complexify(w.attn_vec_einsum_w, c_w.attn_vec_einsum_w);
  Complexify(w.qkv_einsum_w, c_w.qkv_einsum_w);
  Complexify(w.pre_ffw_norm_scale, c_w.pre_ffw_norm_scale);
  Complexify(w.gating_einsum_w, c_w.gating_einsum_w);
  Complexify(w.linear_w, c_w.linear_w);
}

template<typename T, typename U, typename TConfig, typename FUNC>
void TestGradient(const Layer<T, TConfig>& grad,
                  Layer<std::complex<U>, TConfig>& c_weights,
                  FUNC func, T max_err) {
  TestGradient(grad.pre_attention_norm_scale,
               c_weights.pre_attention_norm_scale,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.attn_vec_einsum_w, c_weights.attn_vec_einsum_w,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.qkv_einsum_w, c_weights.qkv_einsum_w,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.pre_ffw_norm_scale, c_weights.pre_ffw_norm_scale,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.gating_einsum_w, c_weights.gating_einsum_w,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.linear_w, c_weights.linear_w,
               func, max_err, max_err, __LINE__);
}

template<typename T, typename TConfig>
void RandInit(Weights<T, TConfig>& w, T stddev, std::mt19937& gen) {
  static constexpr size_t kLayers = TConfig::kLayers;
  RandInit(w.embedder_input_embedding, stddev, gen);
  RandInit(w.final_norm_scale, stddev, gen);
  for (size_t i = 0; i < kLayers; ++i) {
    RandInit(*w.GetLayer(i), stddev, gen);
  }
}

template<typename T, typename U, typename TConfig>
void Complexify(const Weights<T, TConfig>& w,
                Weights<std::complex<U>, TConfig>& c_w) {
  static constexpr size_t kLayers = TConfig::kLayers;
  Complexify(w.embedder_input_embedding, c_w.embedder_input_embedding);
  Complexify(w.final_norm_scale, c_w.final_norm_scale);
  for (size_t i = 0; i < kLayers; ++i) {
    Complexify(*w.GetLayer(i), *c_w.GetLayer(i));
  }
}

template<typename T, typename U, typename TConfig, typename FUNC>
void TestGradient(const Weights<T, TConfig>& grad,
                  Weights<std::complex<U>, TConfig>& c_weights,
                  FUNC func, T max_err) {
  TestGradient(grad.embedder_input_embedding,
                 c_weights.embedder_input_embedding,
                 func,  2 * max_err, max_err, __LINE__);
  TestGradient(grad.final_norm_scale, c_weights.final_norm_scale,
               func, max_err, max_err, __LINE__);
  for (int i = 0; i < TConfig::kLayers; ++i) {
    TestGradient(*grad.GetLayer(i), *c_weights.GetLayer(i), func, max_err);
  }
}

struct TestConfig {
  static constexpr int kSeqLen = 24;
  static constexpr int kVocabSize = 16;
  static constexpr int kModelDim = 32;
  static constexpr int kHeads = 3;
  static constexpr int kQKVDim = 16;
  static constexpr int kFFHiddenDim = 64;
  static constexpr int kLayers = 2;

  static constexpr int kKVHeads = 1;
  static constexpr int kConv1dWidth = 0;
  static constexpr bool kFFBiases = false;
  static constexpr bool kSoftmaxAttnOutputBiases = false;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kGriffinLayers = 0;
  static constexpr int kNumTensorScales = 0;
};

template<typename TConfig>
using LayerF = Layer<float, TConfig>;
template<typename TConfig>
using WeightsF = Weights<float, TConfig>;

void TestEndToEnd() {
  std::mt19937 gen(42);
  hwy::ThreadPool pool(8);
  WeightsWrapper<float, TestConfig> weights;
  WeightsWrapper<float, TestConfig> grad;
  ActivationsWrapper<float, TestConfig> forward0;
  ActivationsWrapper<float, TestConfig> forward1;
  ActivationsWrapper<float, TestConfig> backward;
  using TC = std::complex<double>;
  WeightsWrapper<TC, TestConfig> c_weights;
  ForwardPass<TC, TestConfig> c_forward;

  ReverseSequenceSampler training_task({0, 0, 1, 1});
  std::vector<Prompt> batch = training_task.SampleBatch(10, gen);

  for (const Prompt& prompt : batch) {
    ReverseSequenceSampler::LogPrompt(prompt);
    RandInit(weights.get(), 1.0f, gen);

    float loss0 = CrossEntropyLossForwardPass(
        prompt, weights.get(), forward0.get());

    float loss1 = CrossEntropyLossForwardStep<TestConfig, WeightsF, LayerF>(
        prompt.tokens, prompt.context_size, weights.get(), forward1.get(),
        pool);

    EXPECT_NEAR(loss1, loss0, std::abs(loss0) * 1e-5);

    grad.clear();
    CrossEntropyLossBackwardStep(
        prompt, weights.get(), forward1.get(), grad.get(), backward.get(),
        pool);

    Complexify(weights.get(), c_weights.get());
    auto func = [&]() {
      return CrossEntropyLossForwardPass(prompt, c_weights.get(), c_forward);
    };

    TestGradient(grad.get(), c_weights.get(), func, 2e-3f);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(GradientTest);
HWY_EXPORT_AND_TEST_P(GradientTest, TestMatMulVJP);
HWY_EXPORT_AND_TEST_P(GradientTest, TestEndToEnd);
#ifdef HWY_AFTER_TEST
HWY_AFTER_TEST();
#endif

}  // namespace gcpp

#endif
