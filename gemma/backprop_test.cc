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

template<typename T, size_t N, typename FUNC>
void TestGradient(const std::array<T, N>& grad,
                  std::array<std::complex<T>, N>& x, FUNC func,
                  T max_abs_err, T max_rel_error, int line) {
  const T kStep = 1e-50;
  const T kInvStep = 1.0 / kStep;
  for (size_t i = 0; i < N; ++i) {
    const T x0 = std::real(x[i]);
    const std::complex<T> x1 = std::complex<T>(x0, kStep);
    x[i] = x1;
    const std::complex<T> f1 = func();
    const T exp_grad = std::imag(f1) * kInvStep;
    x[i] = x0;
    ASSERT_NEAR(grad[i], exp_grad,
                std::max(max_abs_err, std::abs(exp_grad) * max_rel_error))
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
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MatMul(c_weights.data(), c_x.data(), c_y.data(), kRows, kCols, kTokens);
      return Dot(dy.data(), c_y.data(), kTokens * kRows);
    };
    memset(&grad, 0, sizeof(grad));
    MatMulVJP(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
              kRows, kCols, kTokens);
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
      return Dot(dy.data(), c_y.data(), K * N);
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
      Softmax(c_x.data(), c_y.data(), N);
      return Dot(dy.data(), c_y.data(), N);
    };
    SoftmaxVJP(x.data(), dy.data(), dx.data(), N);
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
  std::array<T, N> dx = {};
  std::array<T, N> dy;
  std::array<TC, N> c_x;
  std::array<TC, N> c_y;

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      MaskedSoftmax(c_x.data(), c_y.data(), kTokens, kHeads, kSeqLen);
      return Dot(dy.data(), c_y.data(), N);
    };
    MaskedSoftmaxVJP(x.data(), dy.data(), dx.data(), kTokens, kHeads, kSeqLen);
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
      Softcap(c_x.data(), c_y.data(), N);
      return Dot(dy.data(), c_y.data(), N);
    };
    SoftcapVJP(x.data(), dy.data(), dx.data(), N);
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
  std::vector<int> prompt = { 0, 1, 2, 3, 0, 3, 2, 1, 0 };
  size_t context_size;

  for (int iter = 0; iter < 10; ++iter) {
    context_size = 1 + (iter % 6);
    RandInit(x, 1.0 * (1 << iter), gen);
    Softcap(x.data(), x.data(), V * K);
    Softmax(x.data(), x.data(), V, K);
    CrossEntropyLossGrad(x.data(), dx.data(), prompt, context_size, V);
    Complexify(x, c_x);
    auto func = [&]() {
      return CrossEntropyLoss(c_x.data(), prompt, context_size, V);
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
      return Dot(dy.data(), c_y.data(), N * K);
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
      return Dot(dy.data(), c_y.data(), kOutSize);
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
      return Dot(dy.data(), c_y.data(), kOutSize);
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
  std::vector<int> prompt = { 0, 1, 2, 3, 0, 1, 2 };
  size_t num_tokens = prompt.size() - 1;

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    auto func = [&]() {
      InputEmbedding(c_weights.data(), prompt, TC(3.0), c_y.data(), kModelDim);
      return Dot(dy.data(), c_y.data(), num_tokens * kModelDim);
    };
    memset(&grad, 0, sizeof(grad));
    InputEmbeddingVJP(weights.data(), prompt, 3.0, dy.data(), grad.data(),
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
  static constexpr int kHeads = 4;
  static constexpr int kQKVDim = 10;
  static constexpr int kFFHiddenDim = 64;
  static constexpr int kLayers = 3;
};

template<typename T, typename TConfig>
void RandInit(AttnWeights<T, TConfig>& w, std::mt19937& gen) {
  RandInit(w.pre_attention_norm_scale, 1.0, gen);
  RandInit(w.attn_vec_einsum_w, 1.0, gen);
  RandInit(w.qkv_einsum_w, 1.0, gen);
}

template<typename T, typename TConfig>
void Complexify(const AttnWeights<T, TConfig>& w,
                AttnWeights<std::complex<T>, TConfig>& c_w) {
  Complexify(w.pre_attention_norm_scale, c_w.pre_attention_norm_scale);
  Complexify(w.attn_vec_einsum_w, c_w.attn_vec_einsum_w);
  Complexify(w.qkv_einsum_w, c_w.qkv_einsum_w);
}

template<typename T, typename TConfig, typename FUNC>
void TestGradient(const AttnWeights<T, TConfig>& grad,
                  AttnWeights<std::complex<T>, TConfig>& c_weights,
                  FUNC func) {
  TestGradient(grad.pre_attention_norm_scale,
               c_weights.pre_attention_norm_scale,
               func, 1e-11, 1e-11, __LINE__);
  TestGradient(grad.attn_vec_einsum_w, c_weights.attn_vec_einsum_w,
               func, 1e-11, 1e-11, __LINE__);
  TestGradient(grad.qkv_einsum_w, c_weights.qkv_einsum_w,
               func, 1e-11, 1e-11, __LINE__);
}

TEST(BackPropTest, AttnBlockVJP) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  const size_t kOutputSize = TestConfig::kSeqLen * TestConfig::kModelDim;
  AttnWeights<T, TestConfig> weights;
  AttnWeights<T, TestConfig> grad;
  AttnActivations<T, TestConfig> forward;
  AttnActivations<T, TestConfig> backward = {};
  AttnWeights<TC, TestConfig> c_weights;
  AttnActivations<TC, TestConfig> c_forward;
  std::array<T, kOutputSize> y;
  std::array<T, kOutputSize> dy;
  std::array<TC, kOutputSize> c_y;
  const size_t num_tokens = 3;

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, gen);
    RandInit(forward.input, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(forward.input, c_forward.input);
    auto func = [&]() {
      ApplyAttentionBlock(c_weights, c_forward, num_tokens, c_y.data());
      return Dot(dy.data(), c_y.data(), num_tokens * TestConfig::kModelDim);
    };
    memset(&grad, 0, sizeof(grad));
    ApplyAttentionBlock(weights, forward, num_tokens, y.data());
    AttentionBlockVJP(weights, forward, dy.data(), grad, backward, num_tokens);
    TestGradient(backward.input, c_forward.input, func, 1e-11, 1e-11,
                 __LINE__);
    TestGradient(grad, c_weights, func);
  }
}

template<typename T, typename TConfig>
void RandInit(FFWWeights<T, TConfig>& w, std::mt19937& gen) {
  RandInit(w.pre_ffw_norm_scale, 1.0, gen);
  RandInit(w.gating_einsum_w, 1.0, gen);
  RandInit(w.linear_w, 1.0, gen);
}

template<typename T, typename TConfig>
void Complexify(const FFWWeights<T, TConfig>& w,
                FFWWeights<std::complex<T>, TConfig>& c_w) {
  Complexify(w.pre_ffw_norm_scale, c_w.pre_ffw_norm_scale);
  Complexify(w.gating_einsum_w, c_w.gating_einsum_w);
  Complexify(w.linear_w, c_w.linear_w);
}

template<typename T, typename TConfig, typename FUNC>
void TestGradient(const FFWWeights<T, TConfig>& grad,
                  FFWWeights<std::complex<T>, TConfig>& c_weights,
                  FUNC func) {
  TestGradient(grad.pre_ffw_norm_scale, c_weights.pre_ffw_norm_scale,
               func, 1e-11, 1e-11, __LINE__);
  TestGradient(grad.gating_einsum_w, c_weights.gating_einsum_w,
               func, 1e-11, 1e-11, __LINE__);
  TestGradient(grad.linear_w, c_weights.linear_w,
               func, 1e-11, 1e-11, __LINE__);
}

TEST(BackPropTest, FFWBlockVJP) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  const size_t kOutputSize = TestConfig::kSeqLen * TestConfig::kModelDim;
  FFWWeights<T, TestConfig> weights;
  FFWWeights<T, TestConfig> grad;
  FFWActivations<T, TestConfig> forward;
  FFWActivations<T, TestConfig> backward = {};
  FFWWeights<TC, TestConfig> c_weights;
  FFWActivations<TC, TestConfig> c_forward;
  std::array<T, kOutputSize> y;
  std::array<T, kOutputSize> dy;
  std::array<TC, kOutputSize> c_y;
  const size_t num_tokens = 3;

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, gen);
    RandInit(forward.input, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(forward.input, c_forward.input);
    auto func = [&]() {
      ApplyFFWBlock(c_weights, c_forward, num_tokens, c_y.data());
      return Dot(dy.data(), c_y.data(), num_tokens * TestConfig::kModelDim);
    };
    memset(&grad, 0, sizeof(grad));
    ApplyFFWBlock(weights, forward, num_tokens, y.data());
    FFWBlockVJP(weights, forward, dy.data(), grad, backward, num_tokens);
    TestGradient(backward.input, c_forward.input, func, 1e-11, 1e-11,
                 __LINE__);
    TestGradient(grad, c_weights, func);
  }
}

template<typename T, typename TConfig>
void RandInit(AllWeights<T, TConfig>& w, std::mt19937& gen) {
  static constexpr size_t kLayers = TConfig::kLayers;
  RandInit(w.embedder_input_embedding, 1.0, gen);
  RandInit(w.final_norm_scale, 1.0, gen);
  for (size_t i = 0; i < kLayers; ++i) {
    RandInit(w.layers[i].attn, gen);
    RandInit(w.layers[i].ffw, gen);
  }
}

template<typename T, typename TConfig>
void Complexify(const AllWeights<T, TConfig>& w,
                AllWeights<std::complex<T>, TConfig>& c_w) {
  static constexpr size_t kLayers = TConfig::kLayers;
  Complexify(w.embedder_input_embedding, c_w.embedder_input_embedding);
  Complexify(w.final_norm_scale, c_w.final_norm_scale);
  for (size_t i = 0; i < kLayers; ++i) {
    Complexify(w.layers[i].attn, c_w.layers[i].attn);
    Complexify(w.layers[i].ffw, c_w.layers[i].ffw);
  }
}

class PromptSampler {
 public:
  virtual size_t Sample(std::mt19937& gen, std::vector<int>& sample) = 0;
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

  size_t Sample(std::mt19937& gen, std::vector<int>& sample) override {
    int len = length_lut_[length_dist_(gen)];
    sample.resize(2 * len + 2);
    sample[len] = kReverseToken;
    sample[2 * len + 1] = kEndToken;
    for (size_t i = 0; i < len; ++i) {
      sample[i] = sample[2 * len - i] = token_dist_(gen);
    }
    return len + 1;
  }

 private:
  std::uniform_int_distribution<> token_dist_;
  std::uniform_int_distribution<> length_dist_;
  std::vector<int> length_lut_;
};

void LogPrompt(const std::vector<int>& prompt, size_t context_size) {
  static const char* kVocab[] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-->", "|",
  };
  for (int token : prompt) printf("%s", kVocab[token]);
  printf("  [context_size: %zu]\n", context_size);
}

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

  printf("Num weights: %zu\n", sizeof(weights) / sizeof(T));

  ReverseSequenceSampler training_task({0, 0, 1, 1});
  for (size_t iter = 0; iter < 10; ++iter) {
    std::vector<int> prompt;
    size_t context_size = training_task.Sample(gen, prompt);
    LogPrompt(prompt, context_size);
    RandInit(weights, gen);
    ForwardPass(prompt, context_size, weights, forward);
    memset(&grad, 0, sizeof(grad));
    BackwardPass(prompt, context_size, weights, forward, grad, backward);

    Complexify(weights, c_weights);
    auto func = [&]() {
      return ForwardPass(prompt, context_size, c_weights, c_forward);
    };

    TestGradient(grad.embedder_input_embedding,
                 c_weights.embedder_input_embedding,
                 func,  1e-11, 1e-11, __LINE__);
    TestGradient(grad.final_norm_scale, c_weights.final_norm_scale,
                 func, 1e-11, 1e-11, __LINE__);
    for (int i = 0; i < TestConfig::kLayers; ++i) {
      TestGradient(grad.layers[i].attn, c_weights.layers[i].attn, func);
      TestGradient(grad.layers[i].ffw, c_weights.layers[i].ffw, func);
    }
  }
}

TEST(BackProptest, Convergence) {
  std::mt19937 gen(42);
  using T = double;
  AllWeights<T, TestConfig> weights;
  AllWeights<T, TestConfig> grad;
  AllActivations<T, TestConfig> forward;
  AllActivations<T, TestConfig> backward;
  constexpr size_t kBatchSize = 20;

  RandInit(weights, gen);
  const T learning_rate = 0.01;

  ReverseSequenceSampler training_task({0, 0, 1, 1});

  bool stop = false;
  size_t step = 0;
  while (!stop) {
    T loss = 0.0;
    memset(&grad, 0, sizeof(grad));
    std::mt19937 sampler_gen(42);
    for (size_t i = 0; i < kBatchSize; ++i) {
      std::vector<int> prompt;
      size_t context_size = training_task.Sample(sampler_gen, prompt);
      if (step == 0) {
        LogPrompt(prompt, context_size);
      }
      ASSERT_LE(prompt.size() - 1, TestConfig::kSeqLen);
      loss += ForwardPass(prompt, context_size, weights, forward);
      BackwardPass(prompt, context_size, weights, forward, grad, backward);
    }
    loss /= kBatchSize;
    stop = step >= 10000 || loss < 1e-2;
    if (step % 10 == 0 || stop) {
      printf("step: %5zu  loss: %.15f\n", step, loss);
    }
    if (!stop) {
      const T scale = -learning_rate / kBatchSize;
      MulByConstAndAdd(scale, grad, weights);
      ++step;
    }
  }

  EXPECT_LT(step, 3000);
}

}  // namespace gcpp
