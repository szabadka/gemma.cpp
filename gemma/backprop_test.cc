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

template<typename T, typename U, size_t kLen>
void Complexify(const std::array<T, kLen>& x,
                std::array<std::complex<U>, kLen>& c_x) {
  for (size_t i = 0; i < kLen; ++i) {
    c_x[i] = std::complex<U>(x[i], 0.0);
  }
}

template<typename T, typename U, size_t N, typename FUNC>
void TestGradient(const std::array<T, N>& grad,
                  std::array<std::complex<U>, N>& x, FUNC func,
                  U step, T max_abs_err, T max_rel_error, int line) {
  const U inv_step = 1.0 / step;
  for (size_t i = 0; i < N; ++i) {
    const U x0 = std::real(x[i]);
    const std::complex<U> x1 = std::complex<U>(x0, step);
    x[i] = x1;
    const std::complex<U> f1 = func();
    const T exp_grad = std::imag(f1) * inv_step;
    x[i] = x0;
    ASSERT_NEAR(grad[i], exp_grad,
                std::max(max_abs_err, std::abs(exp_grad) * max_rel_error))
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

TEST(BackPropTest, MatMulVJP) {
  static const size_t kRows = 8;
  static const size_t kCols = 64;
  static const size_t kTokens = 5;
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
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MatMulT(c_weights.data(), c_x.data(), c_y.data(), kRows, kCols, kTokens);
      return DotT(dy.data(), c_y.data(), kTokens * kRows);
    };
    memset(&grad, 0, sizeof(grad));
    MatMulVJPT(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
               kRows, kCols, kTokens);
    TestGradient(dx, c_x, func, 1e-11, 1e-12,__LINE__);
    TestGradient(grad, c_weights, func, 1e-14, 1e-12,__LINE__);
  }
}

TEST(BackPropTest, MultiHeadMatMulVJP) {
  static const size_t kRows = 2;
  static const size_t kCols = 16;
  static const size_t kHeads = 4;
  static const size_t kTokens = 3;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, kRows * kCols * kHeads> weights;
  std::array<T, kTokens * kCols * kHeads> x;
  std::array<T, kRows * kCols * kHeads> grad;
  std::array<T, kTokens * kCols * kHeads> dx;
  std::array<TC, kRows * kCols * kHeads> c_weights;
  std::array<TC, kTokens * kCols * kHeads> c_x;
  std::array<TC, kTokens * kRows> c_y;
  std::array<T, kTokens * kRows> dy;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0 * (1 << iter), gen);
    RandInit(x, 1.0 * (1 << iter), gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MultiHeadMatMul(c_weights.data(), c_x.data(), c_y.data(), kHeads, kRows,
                      kCols, kTokens);
      return DotT(dy.data(), c_y.data(), kTokens * kRows);
    };
    memset(&grad, 0, sizeof(grad));
    MultiHeadMatMulVJP(weights.data(), x.data(), dy.data(), grad.data(),
                       dx.data(), kHeads, kRows, kCols, kTokens);
    TestGradient(dx, c_x, func, 1e-15, 1e-13,__LINE__);
    TestGradient(grad, c_weights, func, 1e-15, 1e-13,__LINE__);
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
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      RMSNorm(c_weights.data(), c_x.data(), c_y.data(), N, K);
      return DotT(dy.data(), c_y.data(), K * N);
    };
    memset(&grad, 0, sizeof(grad));
    RMSNormVJP(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
               N, K);
    TestGradient(dx, c_x, func, 1e-15, 1e-14, __LINE__);
    TestGradient(grad, c_weights, func, 1e-15, 1e-14, __LINE__);
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
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      memcpy(c_y.data(), c_x.data(), sizeof(c_x));
      Softmax(c_y.data(), N);
      return DotT(dy.data(), c_y.data(), N);
    };
    Softmax(x.data(), N);
    memcpy(dx.data(), dy.data(), N * sizeof(dx[0]));
    SoftmaxVJP(x.data(), dx.data(), N);
    TestGradient(dx, c_x, func, 1e-15, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, MaskedSoftmaxVJP) {
  static const size_t kSeqLen = 16;
  static const size_t kHeads = 2;
  static const size_t kTokens = 14;
  static const size_t N = kHeads * kSeqLen * kSeqLen;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, N> x;
  std::array<T, N> dy;
  std::array<T, N> dx = {};
  std::array<TC, N> c_x;
  std::array<TC, N> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      memcpy(c_y.data(), c_x.data(),
             kTokens * kHeads * kSeqLen * sizeof(c_x[0]));
      MaskedSoftmax(c_y.data(), kTokens, kHeads, kSeqLen);
      return DotT(dy.data(), c_y.data(), N);
    };
    MaskedSoftmax(x.data(), kTokens, kHeads, kSeqLen);
    memcpy(dx.data(), dy.data(), kTokens * kHeads * kSeqLen * sizeof(dx[0]));
    MaskedSoftmaxVJP(x.data(), dx.data(), kTokens, kHeads, kSeqLen);
    TestGradient(dx, c_x, func, 1e-14, 1e-15, __LINE__);
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
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      memcpy(c_y.data(), c_x.data(), N * sizeof(c_x[0]));
      Softcap(c_y.data(), N);
      return DotT(dy.data(), c_y.data(), N);
    };
    Softcap(x.data(), N);
    memcpy(dx.data(), dy.data(), N * sizeof(dx[0]));
    SoftcapVJP(x.data(), dx.data(), N);
    TestGradient(dx, c_x, func, 1e-15, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, CrossEntropyLossGrad) {
  static const size_t K = 8;
  static const size_t V = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, K * V> x;
  std::array<T, K * V> dx;
  std::array<TC, K * V> c_x;
  Prompt prompt;
  prompt.tokens = { 0, 1, 2, 3, 0, 3, 2, 1, 0 };

  for (int iter = 0; iter < 10; ++iter) {
    prompt.context_size = 1 + (iter % 6);
    RandInit(x, 1.0 * (1 << iter), gen);
    Softcap(x.data(), V * K);
    Softmax(x.data(), V, K);
    CrossEntropyLossGrad(x.data(), dx.data(), prompt, V);
    Complexify(x, c_x);
    auto func = [&]() {
      return CrossEntropyLoss(c_x.data(), prompt, V);
    };
    TestGradient(dx, c_x, func, 1e-100, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, GatedGeluVJP) {
  static const size_t K = 2;
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, K * 2 * N> x;
  std::array<T, K * 2 * N> dx;
  std::array<T, K * N> dy;
  std::array<TC, K * 2 * N> c_x;
  std::array<TC, K * N> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0, gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      GatedGelu(c_x.data(), c_y.data(), N, K);
      return DotT(dy.data(), c_y.data(), N * K);
    };
    GatedGeluVJP(x.data(), dy.data(), dx.data(), N, K);
    TestGradient(dx, c_x, func, 1e-15, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, MaskedAttentionVJP) {
  static const size_t kSeqLen = 16;
  static const size_t kHeads = 2;
  static const size_t kQKVDim = 8;
  static const size_t kTokens = 14;
  static const size_t kQKVSize = kSeqLen * (kHeads + 2) * kQKVDim;
  static const size_t kOutSize = kSeqLen * kHeads * kSeqLen;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, kQKVSize> x;
  std::array<T, kQKVSize> dx = {};
  std::array<T, kOutSize> dy;
  std::array<TC, kQKVSize> c_x;
  std::array<TC, kOutSize> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0, gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      MaskedAttention(c_x.data(), c_y.data(), kTokens, kHeads, kQKVDim,
                      kSeqLen);
      return DotT(dy.data(), c_y.data(), kOutSize);
    };
    MaskedAttentionVJP(x.data(), dy.data(), dx.data(),
                       kTokens, kHeads, kQKVDim, kSeqLen);
    TestGradient(dx, c_x, func, 1e-14, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, MixByAttentionVJP) {
  static const size_t kSeqLen = 16;
  static const size_t kHeads = 2;
  static const size_t kQKVDim = 8;
  static const size_t kTokens = 14;
  static const size_t kQKVSize = kSeqLen * (kHeads + 2) * kQKVDim;
  static const size_t kAttnSize = kSeqLen * kHeads * kSeqLen;
  static const size_t kOutSize = kSeqLen * kHeads * kQKVDim;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, kQKVSize> qkv;
  std::array<T, kQKVSize> dqkv = {};
  std::array<T, kAttnSize> attn;
  std::array<T, kAttnSize> dattn = {};
  std::array<T, kOutSize> dy;
  std::array<TC, kQKVSize> c_qkv;
  std::array<TC, kAttnSize> c_attn;
  std::array<TC, kOutSize> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(qkv, 1.0, gen);
    RandInit(attn, 1.0, gen);
    Complexify(qkv, c_qkv);
    Complexify(attn, c_attn);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      MixByAttention(c_qkv.data(), c_attn.data(), c_y.data(),
                     kTokens, kHeads, kQKVDim, kSeqLen);
      return DotT(dy.data(), c_y.data(), kOutSize);
    };
    MixByAttentionVJP(qkv.data(), attn.data(), dy.data(), dqkv.data(),
                      dattn.data(), kTokens, kHeads, kQKVDim, kSeqLen);
    TestGradient(dqkv, c_qkv, func, 1e-14, 1e-15, __LINE__);
    TestGradient(dattn, c_attn, func, 1e-14, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, InputEmbeddingVJP) {
  static const size_t kSeqLen = 8;
  static const size_t kVocabSize = 4;
  static const size_t kModelDim = 16;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  std::array<T, kVocabSize * kModelDim> weights;
  std::array<T, kVocabSize * kModelDim> grad;
  std::array<T, kSeqLen * kModelDim> dy;
  std::array<TC, kVocabSize * kModelDim> c_weights;
  std::array<TC, kSeqLen * kModelDim> c_y;
  std::vector<int> tokens = { 0, 1, 2, 3, 0, 1, 2 };
  size_t num_tokens = tokens.size() - 1;

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    auto func = [&]() {
      InputEmbedding(c_weights.data(), tokens, TC(3.0), c_y.data(), kModelDim);
      return DotT(dy.data(), c_y.data(), num_tokens * kModelDim);
    };
    memset(&grad, 0, sizeof(grad));
    InputEmbeddingVJP(weights.data(), tokens, 3.0, dy.data(), grad.data(),
                      kModelDim);
    TestGradient(grad, c_weights, func, 1e-16, 1e-14, __LINE__);
  }
}

static constexpr int kReverseToken = 10;
static constexpr int kEndToken = 11;

struct TestConfig {
  static constexpr int kSeqLen = 18;
  static constexpr int kVocabSize = 12;
  static constexpr int kModelDim = 32;
  static constexpr int kHeads = 3;
  static constexpr int kQKVDim = 12;
  static constexpr int kFFHiddenDim = 48;
  static constexpr int kLayers = 2;

  static constexpr int kKVHeads = 1;
  static constexpr int kConv1dWidth = 0;
  static constexpr bool kFFBiases = false;
  static constexpr bool kSoftmaxAttnOutputBiases = false;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kGriffinLayers = 0;
  static constexpr int kNumTensorScales = 0;
};

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

TEST(BackPropTest, LayerVJP) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  const size_t kOutputSize = TestConfig::kSeqLen * TestConfig::kModelDim;
  Layer<T, TestConfig> weights;
  Layer<T, TestConfig> grad;
  ForwardLayer<T, TestConfig> forward;
  ForwardLayer<T, TestConfig> backward = {};
  Layer<TC, TestConfig> c_weights;
  ForwardLayer<TC, TestConfig> c_forward;
  std::array<T, kOutputSize> y;
  std::array<T, kOutputSize> dy;
  std::array<TC, kOutputSize> c_y;
  const size_t num_tokens = 3;

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0, gen);
    RandInit(forward.input, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(forward.input, c_forward.input);
    auto func = [&]() {
      ApplyLayer(c_weights, c_forward, num_tokens, c_y.data());
      return DotT(dy.data(), c_y.data(), num_tokens * TestConfig::kModelDim);
    };
    memset(&grad, 0, sizeof(grad));
    ApplyLayer(weights, forward, num_tokens, y.data());
    LayerVJP(weights, forward, dy.data(), grad, backward, num_tokens);
    TestGradient(backward.input, c_forward.input, func, 1e-11, 1e-11,
                 __LINE__);
    TestGradient(grad, c_weights, func, 1e-11);
  }
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

class PromptSampler {
 public:
  virtual Prompt Sample(std::mt19937& gen) = 0;

  std::vector<Prompt> SampleBatch(size_t batch_size, std::mt19937& gen) {
    std::vector<Prompt> batch;
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch.emplace_back(Sample(gen));
    }
    return batch;
  }
};

class ReverseSequenceSampler : public PromptSampler {
 public:
  explicit ReverseSequenceSampler(const std::vector<int>& length_histo)
      : token_dist_(0, 9) {
    for (int i = 0; i < length_histo.size(); ++i) {
      const int count = length_histo[i];
      for (int j = 0; j < count; ++j) {
        length_lut_.push_back(i + 1);
      }
    }
    length_dist_ = std::uniform_int_distribution<>(0, length_lut_.size() - 1);
  }

  Prompt Sample(std::mt19937& gen) override {
    Prompt prompt;
    int len = length_lut_[length_dist_(gen)];
    prompt.tokens.resize(2 * len + 2);
    prompt.tokens[len] = kReverseToken;
    prompt.tokens[2 * len + 1] = kEndToken;
    for (size_t i = 0; i < len; ++i) {
      prompt.tokens[i] = prompt.tokens[2 * len - i] = token_dist_(gen);
    }
    prompt.context_size = len + 1;
    return prompt;
  }

 private:
  std::uniform_int_distribution<> token_dist_;
  std::uniform_int_distribution<> length_dist_;
  std::vector<int> length_lut_;
};

void LogPrompt(const Prompt& prompt) {
  static const char* kVocab[] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-->", "|",
  };
  for (int token : prompt.tokens) printf("%s", kVocab[token]);
  printf("  [context_size: %zu]\n", prompt.context_size);
}

template<typename T, typename TConfig>
class WeightsWrapper {
 public:
  WeightsWrapper()
      : pool_(0), data_(AllocateWeights<T, TConfig>(pool_)),
        weights_(reinterpret_cast<Weights<T, TConfig>*>(data_.get())) {}

  const Weights<T, TConfig>& get() const { return *weights_; }
  Weights<T, TConfig>& get() { return *weights_; }
  void clear() { ZeroInit(get()); }
  void copy(const WeightsWrapper<T, TConfig>& other) {
    Copy(get(), other.get());
  }

 private:
  hwy::ThreadPool pool_;
  hwy::AlignedFreeUniquePtr<uint8_t[]> data_;
  Weights<T, TConfig>* weights_;
};

template<typename T, typename U, typename TConfig, typename FUNC>
void TestGradient(const Weights<T, TConfig>& grad,
                  Weights<std::complex<U>, TConfig>& c_weights,
                  FUNC func, T max_err) {
  TestGradient(grad.embedder_input_embedding,
                 c_weights.embedder_input_embedding,
                 func,  max_err, max_err, __LINE__);
  TestGradient(grad.final_norm_scale, c_weights.final_norm_scale,
               func, max_err, max_err, __LINE__);
  for (int i = 0; i < TestConfig::kLayers; ++i) {
    TestGradient(*grad.GetLayer(i), *c_weights.GetLayer(i), func, max_err);
  }

}

TEST(BackPropTest, EndToEnd) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  WeightsWrapper<T, TestConfig> weights;
  WeightsWrapper<T, TestConfig> grad;
  ForwardPass<T, TestConfig> forward;
  ForwardPass<T, TestConfig> backward;
  WeightsWrapper<TC, TestConfig> c_weights;
  ForwardPass<TC, TestConfig> c_forward;

  printf("Num weights: %zu\n", sizeof(weights.get()) / sizeof(T));

  ReverseSequenceSampler training_task({0, 0, 1, 1});
  std::vector<Prompt> batch = training_task.SampleBatch(10, gen);

  for (const Prompt& prompt : batch) {
    LogPrompt(prompt);
    RandInit(weights.get(), 1.0, gen);
    CrossEntropyLossForwardPass(prompt, weights.get(), forward);
    grad.clear();
    CrossEntropyLossBackwardPass(
        prompt, weights.get(), forward, grad.get(), backward);

    Complexify(weights.get(), c_weights.get());
    auto func = [&]() {
      return CrossEntropyLossForwardPass(prompt, c_weights.get(), c_forward);
    };

    TestGradient(grad.get(), c_weights.get(), func, 1e-11);
  }
}

template<typename T, typename TConfig>
T CrossEntropyLossForwardPass(const std::vector<Prompt>& batch,
                              const WeightsWrapper<T, TConfig>& weights,
                              ForwardPass<T, TConfig>& forward) {
  T loss = 0.0;
  for (const Prompt& prompt : batch) {
    loss += CrossEntropyLossForwardPass(prompt, weights.get(), forward);
  }
  T scale = 1.0 / batch.size();
  return loss * scale;
}

template<typename T, typename TConfig>
T CrossEntropyLossForwardPass(T learning_rate,
                              const std::vector<Prompt>& batch,
                              const WeightsWrapper<T, TConfig>& weights,
                              const WeightsWrapper<T, TConfig>& grad,
                              WeightsWrapper<T, TConfig>& tmp,
                              ForwardPass<T, TConfig>& forward) {
  tmp.copy(weights);
  const T scale = -learning_rate / batch.size();
  MulByConstAndAddT(scale, grad.get(), tmp.get());
  return CrossEntropyLossForwardPass(batch, tmp, forward);
}

template<typename T, typename TConfig>
T FindOptimalUpdate(const WeightsWrapper<T, TConfig>& grad,
                    WeightsWrapper<T, TConfig>& weights,
                    WeightsWrapper<T, TConfig>& tmp,
                    ForwardPass<T, TConfig>& forward,
                    const std::vector<Prompt>& batch,
                    T loss, T initial_learning_rate) {
  T lr0 = initial_learning_rate;
  T loss0 = CrossEntropyLossForwardPass(
      lr0, batch, weights, grad, tmp, forward);
  for (size_t iter = 0; iter < 30; ++iter) {
    T lr1 = lr0 * 0.5;
    T loss1 = CrossEntropyLossForwardPass(
        lr1, batch, weights, grad, tmp, forward);
    if (loss0 < loss && loss1 >= loss0) {
      break;
    }
    loss0 = loss1;
    lr0 = lr1;
  }
  for (size_t iter = 0; iter < 30; ++iter) {
    T lr1 = lr0 * 2.0;
    T loss1 = CrossEntropyLossForwardPass(
        lr1, batch, weights, grad, tmp, forward);
    if (loss1 >= loss0) {
      break;
    }
    loss0 = loss1;
    lr0 = lr1;
  }
  const T scale = -lr0 / batch.size();
  MulByConstAndAddT(scale, grad.get(), weights.get());
  return lr0;
}

TEST(BackProptest, Convergence) {
  std::mt19937 gen(42);
  using T = float;
  using TC = std::complex<double>;
  WeightsWrapper<T, TestConfig> weights;
  WeightsWrapper<T, TestConfig> grad;
  WeightsWrapper<T, TestConfig> tmp;
  ForwardPass<T, TestConfig> forward;
  ForwardPass<T, TestConfig> backward;
  WeightsWrapper<TC, TestConfig> c_weights;
  ForwardPass<TC, TestConfig> c_forward;
  constexpr size_t kBatchSize = 10;
  ReverseSequenceSampler training_task({0, 0, 0, 1, 1});
  T learning_rate = 0.01;

  printf("Num weights: %zu\n", sizeof(weights) / sizeof(T));
  RandInit(weights.get(), T(1.0), gen);

  printf("Sample batch:\n");
  for (size_t i = 0; i < 10; ++i) {
    LogPrompt(training_task.Sample(gen));
  }

  T prev_loss = std::numeric_limits<T>::max();
  bool stop = false;
  size_t step = 0;
  while (!stop) {
    T loss = 0.0;
    grad.clear();
    std::mt19937 sgen(42);
    std::vector<Prompt> batch = training_task.SampleBatch(kBatchSize, sgen);
    for (const Prompt& prompt : batch) {
      loss += CrossEntropyLossForwardPass(prompt, weights.get(), forward);
      CrossEntropyLossBackwardPass(
          prompt, weights.get(), forward, grad.get(), backward);
    }

    if (step % 200 == 0) {
      printf("Checking gradient...\n");
      Complexify(weights.get(), c_weights.get());
      auto func = [&]() {
        TC scale = batch.size();
        return CrossEntropyLossForwardPass(batch, c_weights, c_forward) * scale;
      };

      TestGradient(grad.get(), c_weights.get(), func, 4e-3f);
    }

    loss /= batch.size();
    EXPECT_LT(loss, prev_loss);
    stop = step >= 10000 || loss < 1e-2;
    if (step % 10 == 0 || stop) {
      printf("step: %5zu  loss: %.15f  learning_rate: %.15f\n",
             step, loss, learning_rate);
    }
    if (!stop) {
      learning_rate = FindOptimalUpdate(
          grad, weights, tmp, forward, batch, loss, learning_rate);
      ++step;
    }
    prev_loss = loss;
  }

  EXPECT_LT(step, 1000);
}

}  // namespace gcpp
