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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_BACKPROP_SCALAR_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BACKPROP_SCALAR_H_

#include <stddef.h>
#include <string.h>

#include <cmath>
#include <complex>
#include <vector>

namespace gcpp {

template<typename T>
T Dot(const T* a, const T* b, size_t N) {
  T sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template<typename T>
void MulByConst(T c, const T* x, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = c * x[i];
  }
}

// out += c * x
template<typename T>
void MulByConstAndAdd(T c, const T* x, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] += c * x[i];
  }
}

// w is N x M matrix in row-major order, x is M x K matrix in column-major order
// y = w * x is N x K matrix in column-major order.
template<typename T>
void MatMul(const T* w, const T* x, T* y, size_t N, size_t M, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      y[i * N + j] = Dot(&w[j * M], &x[i * M], M);
    }
  }
}

template<typename T>
void MatMulVJP(const T* w, const T* x, const T* dy, T* dw, T* dx,
               size_t N, size_t M, size_t K) {
  memset(dx, 0, M * K * sizeof(dx[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      MulByConstAndAdd(dy[i * N + j], &x[i * M], &dw[j * M], M);
      MulByConstAndAdd(dy[i * N + j], &w[j * M], &dx[i * M], M);
    }
  }
}

template<typename T>
T SquaredL2(const T* x, size_t N) {
  T sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += x[i] * x[i];
  }
  return sum;
}

template<typename T>
void RMSNorm(const T* w, const T* x, T* out, size_t N, size_t K) {
  constexpr T eps(1e-6);
  for (size_t i = 0; i < K; ++i) {
    T ss = SquaredL2(x + i * N, N);
    ss = T(1.0) / std::sqrt(ss / T(N) + eps);
    for (size_t j = 0; j < N; j++) {
      out[i * N + j] = (T(1.0) + w[j]) * (ss * x[i * N + j]);
    }
  }
}

template<typename T>
void RMSNormVJP(const T* w, const T* x, const T* dy, T* dw, T* dx,
                size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    const size_t offset = i * N;
    constexpr T eps(1e-6);
    T ss = SquaredL2(x + i * N, N);
    ss = T(1.0) / std::sqrt(ss / T(N) + eps);
    for (size_t j = 0; j < N; ++j) {
      dw[j] += dy[i * N + j] * x[i * N + j] * ss;
    }
    const T ss3 = ss * ss * ss / T(N);
    T tmp = 0.0;
    for (size_t j = 0; j < N; ++j) {
      tmp += (1.0f + w[j]) * dy[i* N + j] * x[i * N + j];
    }
    tmp *= ss3;
    for (size_t j = 0; j < N; ++j) {
      dx[i * N + j] = ss * (1.0 + w[j]) * dy[i* N + j] - tmp * x[i * N + j];
    }
  }
}

template<typename T>
void Softmax(const T* x, T* out, size_t N) {
  T sum = {};
  auto maxreal = std::real(x[0]);
  for (size_t i = 1; i < N; ++i) {
    if (std::real(x[i]) > maxreal) {
      maxreal = std::real(x[i]);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    out[i] = std::exp(x[i] - maxreal);
    sum += out[i];
  }
  T scale = T(1.0) / sum;
  for (size_t i = 0; i < N; ++i) {
    out[i] *= scale;
  }
}

template<typename T>
void SoftmaxVJP(const T* x, const T* dy, T* dx, size_t N) {
  std::vector<T> y(N);
  Softmax(x, &y[0], N);
  T sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += y[i] * dy[i];
  }
  for (size_t i = 0; i < N; ++i) {
    dx[i] = y[i] * (dy[i] - sum);
  }
}

template<typename T>
void Softmax(const T* x, T* out, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    Softmax(x + i * N, out + i * N, N);
  }
}

template<typename T>
void SoftmaxVJP(const T* x, const T* dy, T* dx, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    SoftmaxVJP(x + i * N, dy + i * N, dx + i * N, N);
  }
}

template<typename T>
void Softcap(const T* x, T* out, size_t N) {
  T cap = 30.0;
  T inv_cap = 1.0 / cap;
  for (size_t i = 0; i < N; ++i) {
    out[i] = cap * std::tanh(x[i] * inv_cap);
  }
}

template<typename T>
void SoftcapVJP(const T* x, const T* dy, T* dx, size_t N) {
  std::vector<T> y(N);
  Softcap(x, &y[0], N);
  T cap = 30.0;
  T inv_cap = 1.0 / cap;
  for (size_t i = 0; i < N; ++i) {
    T scaled = y[i] * inv_cap;
    dx[i] = dy[i] * (1.0 - scaled * scaled);
  }
}

template <typename T, typename TConfig>
struct AttnActivations {
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  std::array<T, kSeqLen * kModelDim> input;
  std::array<T, kSeqLen * kModelDim> pre_att_rms_out;
  std::array<T, kSeqLen * (kHeads + 2) * kQKVDim> qkv;
  std::array<T, kSeqLen * kHeads * kSeqLen> att;
  std::array<T, kSeqLen * kHeads * kQKVDim> att_out;
  std::array<T, kSeqLen * kModelDim> att_post2;
};

template <typename T, typename TConfig>
struct FFWActivations {
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  std::array<T, kSeqLen * kModelDim> input;
  std::array<T, kSeqLen * kModelDim> bf_pre_ffw_rms_out;
  std::array<T, kSeqLen * kFFHiddenDim * 2> ffw_hidden;
  std::array<T, kSeqLen * kFFHiddenDim> ffw_hidden_gated;
  std::array<T, kSeqLen * kModelDim> ffw_out;
};

template <typename T, typename TConfig>
struct LayerActivations {
  AttnActivations<T, TConfig> attn;
  FFWActivations<T, TConfig> ffw;
};

template <typename T, typename TConfig>
struct AllActivations {
  AllActivations() {}  // prevents placement-new calling memset

  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;

  std::array<LayerActivations<T, TConfig>, kLayers> layers;
  std::array<T, kSeqLen * kModelDim> final_layer_output;
  std::array<T, kSeqLen * kModelDim> final_norm_output;
  std::array<T, kSeqLen * kVocabSize> raw_logits;
  std::array<T, kSeqLen * kVocabSize> logits;
  std::array<T, kSeqLen * kVocabSize> probs;
};

template <typename T, typename TConfig>
struct AttnWeights {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;

  std::array<T, kModelDim> pre_attention_norm_scale;
  std::array<T, kHeads * kQKVDim * kModelDim> attn_vec_einsum_w;
  std::array<T, (kHeads + 2) * kQKVDim * kModelDim> qkv_einsum_w;
};

template <typename T, typename TConfig>
struct FFWWeights {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  std::array<T, kModelDim> pre_ffw_norm_scale;
  std::array<T, 2 * kFFHiddenDim * kModelDim> gating_einsum_w;
  std::array<T, kModelDim * kFFHiddenDim> linear_w;
};

template <typename T, typename TConfig>
struct LayerWeights {
  AttnWeights<T, TConfig> attn;
  FFWWeights<T, TConfig> ffw;
};

template <typename T, typename TConfig>
struct AllWeights {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;

  std::array<T, kVocabSize * kModelDim> embedder_input_embedding;
  std::array<T, kModelDim> final_norm_scale;
  std::array<LayerWeights<T, TConfig>, kLayers> layers;
};

template<typename T>
void InputEmbedding(const T* w, const std::vector<int>& prompt, T scaling,
                    T* y, size_t N) {
  for (size_t i = 0; i + 1 < prompt.size(); ++i) {
    int token = prompt[i];
    memcpy(y + i * N, w + token * N, N * sizeof(y[0]));
    MulByConst(scaling, y + i * N, N);
  }
}

template<typename T>
void InputEmbeddingVJP(const T* w, const std::vector<int>& prompt, T scaling,
                       const T* dy, T* dw, size_t N) {
  for (size_t i = 0; i + 1 < prompt.size(); ++i) {
    int token = prompt[i];
    MulByConstAndAdd(scaling, dy + i * N, dw + token * N, N);
  }
}


template<typename T, typename TConfig>
void ApplyFowrardLayer(const LayerWeights<T, TConfig>& weights,
                       LayerActivations<T, TConfig>& forward, size_t num_tokens,
                       T* output) {
}

template<typename T, typename TConfig>
void LayerVJP(const LayerWeights<T, TConfig>& weights,
              const LayerActivations<T, TConfig>& forward,  const T* dy,
              LayerWeights<T, TConfig>& grad,
              LayerActivations<T, TConfig>& backward, size_t num_tokens) {
}

template<typename T>
T CrossEntropyLoss(const T* x, const std::vector<int>& prompt,
                   size_t context_size, size_t V) {
  T loss = {};
  for (size_t i = 0; i + 1 < prompt.size(); ++i) {
    if (i + 1 < context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = prompt[i + 1];
    loss += std::log(x[next_token]);
  }
  T scaling = 1.0 / std::log(2.0);
  return loss * scaling;
}

template<typename T>
void CrossEntropyLossGrad(const T* x, T* dx, const std::vector<int>& prompt,
                          size_t context_size, size_t V) {
  T scaling = 1.0 / std::log(2.0);
  size_t num_tokens = prompt.size() - 1;
  memset(dx, 0, V * num_tokens * sizeof(x[0]));
  for (size_t i = 0; i + 1 < prompt.size(); ++i) {
    if (i + 1 < context_size) {
      continue;
    }
    const int next_token = prompt[i + 1];
    dx[next_token] = scaling / x[next_token];
  }
}

template<typename T, typename TConfig>
float ForwardPass(const std::vector<int>& prompt,
                  size_t context_size,
                  const AllWeights<T, TConfig>& weights,
                  AllActivations<T, TConfig>& forward) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;
  const size_t num_tokens = prompt.size() - 1;

  const T kEmbScaling = std::sqrt(kModelDim);
  InputEmbedding(weights.embedder_input_embedding.data(), prompt, kEmbScaling,
                 forward.layers[0].attn.input.data(), kModelDim);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    T* output = layer + 1 < kLayers ?
                forward.layers[layer + 1].attn.input.data() :
                forward.final_layer_output.data();
    ApplyForwardLayer(weights.layers[layer], forward.layers[layer],
                      num_tokens, output);
  }

  RMSNorm(weights.final_norm_scale.data(),
          forward.final_layer_output.data(),
          forward.final_norm_output.data(), kModelDim, num_tokens);

  MatMul(weights.embedder_input_embedding.data(),
         forward.final_norm_output.data(),
         forward.raw_logits.data(), kVocabSize, kModelDim, num_tokens);

  Softcap(forward.raw_logits.data(), forward.logits.data(),
          num_tokens * kVocabSize);

  Softmax(forward.logits.data(), forward.probs.data(), kVocabSize, num_tokens);

  return CrossEntropyLoss(forward.probs.data(), prompt, context_size);
}

template<typename T, typename TConfig>
float BackwardPass(const std::vector<int>& prompt,
                   size_t context_size,
                   const AllWeights<T, TConfig>& weights,
                   const AllActivations<T, TConfig>& forward,
                   AllWeights<T, TConfig>& grad,
                   AllActivations<T, TConfig>& backward) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;
  const size_t num_tokens = prompt.size() - 1;

  CrossEntropyLossGrad(forward.probs.data(), backward.probs.data(), prompt,
                       context_size);

  SoftmaxVJP(forward.logits.data(), backward.probs.data(),
             backward.logits.data(), kVocabSize, num_tokens);

  SoftcapVJP(forward.raw_logits.data(), backward.logits.data(),
             backward.raw_logits.data(), num_tokens * kVocabSize);

  MatMulVJP(weights.embedder_input_embedding.data()(),
            forward.final_norm_output.data(),
            backward.raw_logits.data(),
            grad.embedder_input_emdebbing.data(),
            backward.final_norm_output.data(),
            kVocabSize, kModelDim, num_tokens);

  RMSNormVJP(weights.final_norm_scale.data(),
             forward.final_layer_output.data(),
             backward.final_norm_output.data(),
             grad.final_norm_scale.data(),
             backward.final_norm_scale.data(), kModelDim, num_tokens);

  for (int layer = static_cast<int>(kLayers) - 1; layer >= 0; --layer) {
    T* next_layer_grad =
        layer + 1 < kLayers ? backward.layers[layer + 1].attn.input.data()
                            : backward.final_layer_output.data();
    LayerVJP(weights.layers[layer], forward.layers[layer], next_layer_grad,
             grad.layers[layer], backward.layers[layer], num_tokens);
  }

  const T kEmbScaling = std::sqrt(kModelDim);
  InputEmbeddingVJP(weights.embedder_input_embedding.data(),
                    prompt, kEmbScaling,
                    backward.layers[0].attn.input.data(),
                    grad.embedder_input_embedding.data(), kModelDim);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BACKPROP_SCALAR_H_
