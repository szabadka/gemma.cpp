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

#include <iostream>
#include <cstdio>

#include <cmath>
#include <complex>
#include <vector>

namespace gcpp {

template<typename T, size_t N>
void LogVec(const char* name, const std::array<T, N>& x) {
  std::cout << name;
  for (const auto& v : x) std::cout << "  " << v;
  std::cout << std::endl;
}

template<typename T, typename U>
U Dot(const T* a, const U* b, size_t N) {
  U sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template<typename T>
void MulByConst(T c, T* x, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    x[i] *= c;
  }
}

// out += c * x
template<typename T>
void MulByConstAndAdd(T c, const T* x, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] += c * x[i];
  }
}

template<typename T>
void Add(const T* a, const T* b, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = a[i] + b[i];
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

template<typename T>
T Gelu(T x) {
  static const T kMul = 0.044715;
  static const T kSqrt2OverPi = 0.797884560804236;

  const T x3 = x * x * x;
  const T arg = kSqrt2OverPi * (kMul * x3 + x);
  const T cdf = 0.5 * (T(1.0) + std::tanh(arg));
  return x * cdf;
}

template<typename T>
T GeluDerivative(T x) {
  static const T kMul = 0.044715;
  static const T kSqrt2OverPi = 0.797884560804236;
  static const T kMul2 = kSqrt2OverPi * T(3.0) * kMul;

  const T x2 = x * x;
  const T x3 = x2 * x;
  const T arg = kSqrt2OverPi * (kMul * x3 + x);
  const T tanh = std::tanh(arg);
  const T cdf = 0.5 * (T(1.0) + tanh);
  const T dtanh = T(1.0) - tanh * tanh;
  const T darg = kMul2 * x2 + kSqrt2OverPi;
  return 0.5 * x * dtanh * darg + cdf;
}

template<typename T>
void GatedGelu(const T* in, T* out, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    const T* x1 = in + i * 2 * N;
    const T* x2 = x1 + N;
    T* y = out + i * N;
    for (size_t j = 0; j < N; ++j) {
      y[j] = x2[j] * Gelu(x1[j]);
    }
  }
}

template<typename T>
void GatedGeluVJP(const T* in, const T* d_out, T* d_in, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    const T* x1 = in + i * 2 * N;
    const T* x2 = x1 + N;
    const T* v = d_out + i * N;
    T* dx1 = d_in + i * 2 * N;
    T* dx2 = dx1 + N;
    for (size_t j = 0; j < N; ++j) {
      dx1[j] = v[j] * x2[j] * GeluDerivative(x1[j]);
      dx2[j] = v[j] * Gelu(x1[j]);
    }
  }
}

template<typename T>
void Rope(T* x, size_t N, int i) {
  const size_t N2 = N / 2;
  for (size_t dim = 0; dim < N2; ++dim) {
    const T freq_exponents = T(2 * dim) / T(N);
    const T timescale = std::pow(10000.0, freq_exponents);
    const T theta = T(i) / timescale;
    const T cos_val = std::cos(theta);
    const T sin_val = std::sin(theta);
    const T x0 = x[dim];
    const T x1 = x[dim + N2];
    x[dim] = x0 * cos_val - x1 * sin_val;
    x[dim + N2] = x0 * sin_val + x1 * cos_val;
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
  std::array<T, kSeqLen * kHeads * kSeqLen> att_sm;
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

template<typename T>
void MaskedAttention(const T* qkv, T* output, size_t num_tokens,
                     size_t kHeads, size_t kQKVDim, size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      const size_t qoffset = pos * (kHeads + 2) * kQKVDim;
      const size_t aoffset = pos * kHeads * kSeqLen + head * kSeqLen;
      const T* q = qkv + qoffset + head * kQKVDim;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const T* k = qkv + (pos2 * (kHeads + 2) + kHeads) * kQKVDim;
        output[aoffset + pos2] = Dot(q, k, kQKVDim);
      }
    }
  }
}

template<typename T>
void MaskedAttentionVJP(const T* qkv, const T* doutput, T* dqkv,
                        size_t num_tokens, size_t kHeads, size_t kQKVDim,
                        size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * (kHeads + 2) * kQKVDim;
    memset(dqkv + offset, 0, (kHeads + 1) * kQKVDim * sizeof(qkv[0]));
  }
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t qoffs = (pos * (kHeads + 2) + head) * kQKVDim;
      const size_t aoffs = head * kSeqLen + pos * kHeads * kSeqLen;
      const T* q = qkv + qoffs;
      const T* dout = doutput + aoffs;
      T* dq = dqkv + qoffs;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const size_t koffs = (pos2 * (kHeads + 2) + kHeads) * kQKVDim;
        const T* k = qkv + koffs;
        T* dk = dqkv + koffs;
        MulByConstAndAdd(dout[pos2], k, dq, kQKVDim);
        MulByConstAndAdd(dout[pos2], q, dk, kQKVDim);
      }
    }
  }
}

template<typename T>
void MaskedSoftmax(const T* x, T* y, size_t num_tokens,
                   size_t kHeads, size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      size_t offset = pos * kHeads * kSeqLen + head * kSeqLen;
      Softmax(x + offset, y + offset, pos + 1);
    }
  }
}

template<typename T>
void MaskedSoftmaxVJP(const T* x, const T* dy, T* dx, size_t num_tokens,
                      size_t kHeads, size_t kSeqLen) {
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      size_t offset = pos * kHeads * kSeqLen + head * kSeqLen;
      SoftmaxVJP(x + offset, dy + offset, dx + offset, pos + 1);
    }
  }
}

template<typename T>
void MixByAttention(const T* qkv, const T* attention, T* output,
                    size_t num_tokens, size_t kHeads, size_t kQKVDim,
                    size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      const T* att = &attention[pos * kHeads * kSeqLen + head * kSeqLen];
      T* out = &output[head * kQKVDim + pos * kHeads * kQKVDim];
      memset(out, 0, kQKVDim * sizeof(out[0]));
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        size_t v_offset = (pos2 * (kHeads + 2) + kHeads + 1) * kQKVDim;
        const T* v = &qkv[v_offset];
        MulByConstAndAdd(att[pos2], v, out, kQKVDim);
      }
    }
  }
}

template<typename T>
void MixByAttentionVJP(const T* qkv, const T* attention, const T* doutput,
                       T* dqkv, T* dattention, size_t num_tokens,
                       size_t kHeads, size_t kQKVDim, size_t kSeqLen) {
  auto v_offset = [&](size_t pos) {
    return (pos * (kHeads + 2) + kHeads + 1) * kQKVDim;
  };
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    memset(&dqkv[v_offset(pos)], 0, kQKVDim * sizeof(qkv[0]));
  }
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t offset = head * kQKVDim + pos * kHeads * kQKVDim;
      const size_t aoffset = head * kSeqLen + pos * kHeads * kSeqLen;
      const T* att = &attention[aoffset];
      const T* dout = &doutput[offset];
      T* datt = &dattention[aoffset];
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        datt[pos2] = Dot(dout, &qkv[v_offset(pos2)], kQKVDim);
        MulByConstAndAdd(att[pos2], dout, &dqkv[v_offset(pos2)], kQKVDim);
      }
    }
  }
}

template<typename T, typename TConfig>
void ApplyAttentionBlock(const AttnWeights<T, TConfig>& weights,
                         AttnActivations<T, TConfig>& activations,
                         size_t num_tokens, T* output) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static const T kQueryScale = 1.0 / std::sqrt(T(kQKVDim));

  RMSNorm(weights.pre_attention_norm_scale.data(), activations.input.data(),
          activations.pre_att_rms_out.data(), kModelDim, num_tokens);

  MatMul(weights.qkv_einsum_w.data(), activations.pre_att_rms_out.data(),
         activations.qkv.data(), (kHeads + 2) * kQKVDim, kModelDim, num_tokens);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * kQKVDim, kQKVDim, pos);
    }
  }

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    MulByConst(kQueryScale, qkv, kHeads * kQKVDim);
  }

  MaskedAttention(activations.qkv.data(), activations.att.data(),
                  num_tokens, kHeads, kQKVDim, kSeqLen);

  MaskedSoftmax(activations.att.data(), activations.att_sm.data(),
                num_tokens, kHeads, kSeqLen);

  MixByAttention(activations.qkv.data(), activations.att_sm.data(),
                 activations.att_out.data(), num_tokens, kHeads, kQKVDim,
                 kSeqLen);

  MatMul(weights.attn_vec_einsum_w.data(), activations.att_out.data(),
         output, kModelDim, kHeads * kQKVDim, num_tokens);

  Add(activations.input.data(), output, output, num_tokens * kModelDim);
}

template<typename T, typename TConfig>
void AttentionBlockVJP(const AttnWeights<T, TConfig>& weights,
                       const AttnActivations<T, TConfig>& forward,
                       const T* dy,
                       AttnWeights<T, TConfig>& grad,
                       AttnActivations<T, TConfig>& backward,
                       size_t num_tokens) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static const T kQueryScale = 1.0 / std::sqrt(T(kQKVDim));

  MatMulVJP(weights.attn_vec_einsum_w.data(), forward.att_out.data(),
            dy, grad.attn_vec_einsum_w.data(), backward.att_out.data(),
            kModelDim, kHeads * kQKVDim, num_tokens);

  MixByAttentionVJP(forward.qkv.data(), forward.att_sm.data(),
                    backward.att_out.data(), backward.qkv.data(),
                    backward.att_sm.data(), num_tokens, kHeads, kQKVDim,
                    kSeqLen);

  MaskedSoftmaxVJP(forward.att.data(), backward.att_sm.data(),
                   backward.att.data(), num_tokens, kHeads, kSeqLen);

  MaskedAttentionVJP(forward.qkv.data(), backward.att.data(),
                     backward.qkv.data(), num_tokens, kHeads, kQKVDim, kSeqLen);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    MulByConst(kQueryScale, qkv, kHeads * kQKVDim);
  }

  for (int pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * kQKVDim, kQKVDim, -pos);
    }
  }

  MatMulVJP(weights.qkv_einsum_w.data(), forward.pre_att_rms_out.data(),
            backward.qkv.data(), grad.qkv_einsum_w.data(),
            backward.pre_att_rms_out.data(),
            (kHeads + 2) * kQKVDim, kModelDim, num_tokens);
  RMSNormVJP(weights.pre_attention_norm_scale.data(), forward.input.data(),
             backward.pre_att_rms_out.data(),
             grad.pre_attention_norm_scale.data(),
             backward.input.data(), kModelDim, num_tokens);

  Add(dy, backward.input.data(), backward.input.data(), num_tokens * kModelDim);
}

template<typename T, typename TConfig>
void ApplyFFWBlock(const FFWWeights<T, TConfig>& weights,
                   FFWActivations<T, TConfig>& activations,
                   size_t num_tokens, T* output) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  RMSNorm(weights.pre_ffw_norm_scale.data(), activations.input.data(),
          activations.bf_pre_ffw_rms_out.data(), kModelDim, num_tokens);

  MatMul(weights.gating_einsum_w.data(), activations.bf_pre_ffw_rms_out.data(),
         activations.ffw_hidden.data(), kFFHiddenDim * 2, kModelDim,
         num_tokens);

  GatedGelu(activations.ffw_hidden.data(), activations.ffw_hidden_gated.data(),
            kFFHiddenDim, num_tokens);

  MatMul(weights.linear_w.data(), activations.ffw_hidden_gated.data(),
         activations.ffw_out.data(), kModelDim, kFFHiddenDim, num_tokens);

  Add(activations.input.data(), activations.ffw_out.data(), output,
      num_tokens * kModelDim);
}

template<typename T, typename TConfig>
void FFWBlockVJP(const FFWWeights<T, TConfig>& weights,
                 const FFWActivations<T, TConfig>& forward,
                 const T* dy,
                 FFWWeights<T, TConfig>& grad,
                 FFWActivations<T, TConfig>& backward,
                 size_t num_tokens) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  MatMulVJP(weights.linear_w.data(), forward.ffw_hidden_gated.data(),
            dy, grad.linear_w.data(), backward.ffw_hidden_gated.data(),
            kModelDim, kFFHiddenDim, num_tokens);

  GatedGeluVJP(forward.ffw_hidden.data(), backward.ffw_hidden_gated.data(),
               backward.ffw_hidden.data(), kFFHiddenDim, num_tokens);

  MatMulVJP(weights.gating_einsum_w.data(), forward.bf_pre_ffw_rms_out.data(),
            backward.ffw_hidden.data(), grad.gating_einsum_w.data(),
            backward.bf_pre_ffw_rms_out.data(), kFFHiddenDim * 2, kModelDim,
            num_tokens);

  RMSNormVJP(weights.pre_ffw_norm_scale.data(), forward.input.data(),
             backward.bf_pre_ffw_rms_out.data(),
             grad.pre_ffw_norm_scale.data(), backward.input.data(),
             kModelDim, num_tokens);

  Add(dy, backward.input.data(), backward.input.data(), num_tokens * kModelDim);
}

template<typename T, typename TConfig>
void ApplyLayer(const LayerWeights<T, TConfig>& weights,
                LayerActivations<T, TConfig>& forward, size_t num_tokens,
                T* output) {
  ApplyAttentionBlock(weights.attn, forward.attn, num_tokens,
                      forward.ffw.input.data());
  ApplyFFWBlock(weights.ffw, forward.ffw, num_tokens, output);
}

template<typename T, typename TConfig>
void LayerVJP(const LayerWeights<T, TConfig>& weights,
              const LayerActivations<T, TConfig>& forward,  const T* dy,
              LayerWeights<T, TConfig>& grad,
              LayerActivations<T, TConfig>& backward, size_t num_tokens) {
  FFWBlockVJP(weights.ffw, forward.ffw, dy, grad.ffw, backward.ffw, num_tokens);
  AttentionBlockVJP(weights.attn, forward.attn, backward.ffw.input.data(),
                    grad.attn, backward.attn, num_tokens);
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
    loss += std::log(x[i * V + next_token]);
  }
  T scaling = -1.0 / std::log(2.0);
  return loss * scaling;
}

template<typename T>
void CrossEntropyLossGrad(const T* x, T* dx, const std::vector<int>& prompt,
                          size_t context_size, size_t V) {
  T scaling = -1.0 / std::log(2.0);
  size_t num_tokens = prompt.size() - 1;
  memset(dx, 0, V * num_tokens * sizeof(x[0]));
  for (size_t i = 0; i + 1 < prompt.size(); ++i) {
    if (i + 1 < context_size) {
      continue;
    }
    const int next_token = prompt[i + 1];
    dx[i * V + next_token] = scaling / x[i * V + next_token];
  }
}

template<typename T, typename TConfig>
T ForwardPass(const std::vector<int>& prompt,
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
    ApplyLayer(weights.layers[layer], forward.layers[layer], num_tokens,
               output);
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

  return CrossEntropyLoss(forward.probs.data(), prompt, context_size,
                          kVocabSize);
}

template<typename T, typename TConfig>
void BackwardPass(const std::vector<int>& prompt,
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
                       context_size, kVocabSize);

  SoftmaxVJP(forward.logits.data(), backward.probs.data(),
             backward.logits.data(), kVocabSize, num_tokens);

  SoftcapVJP(forward.raw_logits.data(), backward.logits.data(),
             backward.raw_logits.data(), num_tokens * kVocabSize);

  MatMulVJP(weights.embedder_input_embedding.data(),
            forward.final_norm_output.data(),
            backward.raw_logits.data(),
            grad.embedder_input_embedding.data(),
            backward.final_norm_output.data(),
            kVocabSize, kModelDim, num_tokens);

  RMSNormVJP(weights.final_norm_scale.data(),
             forward.final_layer_output.data(),
             backward.final_norm_output.data(),
             grad.final_norm_scale.data(),
             backward.final_layer_output.data(), kModelDim, num_tokens);

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
