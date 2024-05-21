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

#include "gemma/backprop_scalar.h"

#include <array>
#include <complex>
#include <random>

#include "gtest/gtest.h"

namespace gcpp {

template<typename T, size_t kLen>
void RandInit(std::array<T, kLen>& x, T stddev, std::mt19937& gen) {
  std::normal_distribution<T> dist(0.0, stddev);
  for (size_t i = 0; i < kLen; ++i) {
    x[i] = dist(gen);
  }
}

template<typename T, size_t kLen>
void Complexify(const std::array<T, kLen>& x,
                std::array<std::complex<T>, kLen>& c_x) {
  for (size_t i = 0; i < kLen; ++i) {
    c_x[i] = std::complex<T>(x[i], 0.0);
  }
}

template<typename T, typename TConfig>
void RandInit(AllWeights<T, TConfig>& w, std::mt19937& gen) {
  RandInit(w.embedder_input_embedding, 1.0, gen);
  RandInit(w.final_norm_scale, 1.0, gen);
}

template<typename T, typename TConfig>
void Complexify(const AllWeights<T, TConfig>& w,
                AllWeights<std::complex<T>, TConfig>& c_w) {
  Complexify(w.embedder_input_embedding, c_w.embedder_input_embedding);
  Complexify(w.final_norm_scale, c_w.final_norm_scale);
}

template<typename T, size_t N, typename FUNC>
void TestGradient(const std::array<T, N>& grad,
                  std::array<std::complex<T>, N>& x, FUNC func, int line) {
  const T kStep = 1e-50;
  const T kInvStep = 1.0 / kStep;
  for (size_t i = 0; i < N; ++i) {
    const T x0 = std::real(x[i]);
    const std::complex<T> x1 = std::complex<T>(x0, kStep);
    x[i] = x1;
    const std::complex<T> f1 = func();
    const T exp_grad = std::imag(f1) * kInvStep;
    x[i] = x0;
    ASSERT_NEAR(grad[i], exp_grad, std::max(1e-15, std::abs(exp_grad) * 1e-14))
        << "line: " << line << " dim=" << N << " i=" << i << " f1=" << f1;
  }
}

TEST(BackPropTest, MatMulVJP) {
  static const size_t kRows = 2;
  static const size_t kCols = 64;
  static const size_t kTokens = 3;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, kRows * kCols> weights;
  std::array<T, kTokens * kCols> x;
  std::array<T, kRows * kCols> grad;
  std::array<T, kTokens * kCols> dx;
  std::array<TC, kRows * kCols> c_weights;
  std::array<TC, kTokens * kCols> c_x;
  std::array<TC, kTokens * kRows> c_y;
  std::array<T, kTokens * kRows> dy;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0 * (1 << iter), gen);
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    for (size_t t = 0; t < kTokens; ++t) {
      for (size_t r = 0; r < kRows; ++r) {
        auto func = [&]() {
          MatMul(c_weights.data(), c_x.data(), c_y.data(), kRows, kCols,
                 kTokens);
          return c_y[t * kRows + r];
        };
        memset(&dy, 0, sizeof(dy));
        dy[t * kRows + r] = 1.0;
        memset(&grad, 0, sizeof(grad));
        MatMulVJP(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
                  kRows, kCols, kTokens);
        TestGradient(dx, c_x, func, __LINE__);
        TestGradient(grad, c_weights, func, __LINE__);
      }
    }
  }
}

TEST(BackPropTest, RMSNormVJP) {
  static const size_t K = 2;
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, N> weights;
  std::array<T, N> grad;
  std::array<T, K * N> x;
  std::array<T, K * N> dx;
  std::array<T, K * N> dy;
  std::array<TC, N> c_weights;
  std::array<TC, K * N> c_x;
  std::array<TC, K * N> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0 * (1 << iter), gen);
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < N; ++j) {
        auto func = [&]() {
          RMSNorm(c_weights.data(), c_x.data(), c_y.data(), N, K);
          return c_y[i * N + j];
        };
        memset(&dy, 0, sizeof(dy));
        dy[i * N + j] = 1.0;
        memset(&grad, 0, sizeof(grad));
        RMSNormVJP(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
                   N, K);
        TestGradient(dx, c_x, func, __LINE__);
        TestGradient(grad, c_weights, func, __LINE__);
      }
    }
  }
}

TEST(BackPropTest, SoftmaxVJP) {
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, N> x;
  std::array<T, N> dx;
  std::array<T, N> dy;
  std::array<TC, N> c_x;
  std::array<TC, N> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    for (size_t j = 0; j < N; ++j) {
      auto func = [&]() {
        Softmax(c_x.data(), c_y.data(), N);
        return c_y[j];
      };
      memset(&dy, 0, sizeof(dy));
      dy[j] = 1.0;
      SoftmaxVJP(x.data(), dy.data(), dx.data(), N);
      TestGradient(dx, c_x, func, __LINE__);
    }
  }
}

TEST(BackPropTest, SoftcapVJP) {
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, N> x;
  std::array<T, N> dx;
  std::array<T, N> dy;
  std::array<TC, N> c_x;
  std::array<TC, N> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    for (size_t j = 0; j < N; ++j) {
      auto func = [&]() {
        Softcap(c_x.data(), c_y.data(), N);
        return c_y[j];
      };
      memset(&dy, 0, sizeof(dy));
      dy[j] = 1.0;
      SoftcapVJP(x.data(), dy.data(), dx.data(), N);
      TestGradient(dx, c_x, func, __LINE__);
    }
  }
}

TEST(BackPropTest, CrossEntropyLossGrad) {
  static const size_t K = 4;
  static const size_t V = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, K * V> x;
  std::array<T, K * V> dx;
  std::array<TC, K * V> c_x;
  std::vector<int> prompt = { 0, 1, 2, 3, 0 };
  size_t context_size = 1;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Softcap(x.data(), x.data(), V * K);
    Softmax(x.data(), x.data(), V, K);
    CrossEntropyLossGrad(x.data(), dx.data(), prompt, context_size, V);
    Complexify(x, c_x);
    auto func = [&]() {
      return CrossEntropyLoss(c_x.data(), prompt, context_size, V);
    };
    TestGradient(dx, c_x, func, __LINE__);
  }
}

struct TestConfig {
  static constexpr int kSeqLen = 4;
  static constexpr int kVocabSize = 4;
  static constexpr int kModelDim = 8;
  static constexpr int kHeads = 2;
  static constexpr int kQKVDim = 5;
  static constexpr int kFFHiddenDim = 16;
  static constexpr int kLayers = 1;
};

TEST(BackPropTest, EndToEnd) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  AllWeights<T, TestConfig> weights;
  AllWeights<T, TestConfig> grad;
  AllActivations<T, TestConfig> forward;
  AllActivations<T, TestConfig> backward;
  AllWeights<TC, TestConfig> c_weights;
  AllActivations<TC, TestConfig> c_forward;
  std::vector<int> prompt = { 0, 1, 2, 3, 0 };
  size_t context_size = 1;

  RandInit(weights, gen);
  ForwardPass(prompt, context_size, weights, forward);
  BackwardPass(prompt, context_size, weights, forward, grad, backward);

  Complexify(weights, c_weights);
  auto func = [&]() {
    return ForwardPass(prompt, context_size, c_weights, c_forward);
  };

  TestGradient(grad.embedder_input_embedding,
               c_weights.embedder_input_embedding,
               func,  __LINE__);
  TestGradient(grad.final_norm_scale, c_weights.final_norm_scale,
               func, __LINE__);
}

}  // namespace gcpp
