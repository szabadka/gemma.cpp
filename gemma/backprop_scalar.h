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

#include "gemma/activations.h"
#include "gemma/weights.h"

namespace gcpp {

template<typename T, size_t N>
void LogVec(const char* name, const std::array<T, N>& x) {
  std::cout << name;
  for (const auto& v : x) std::cout << "  " << v;
  std::cout << std::endl;
}

template<typename T, typename U>
U DotT(const T* a, const U* b, size_t N) {
  U sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template<typename T>
void MulByConstT(T c, T* x, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    x[i] *= c;
  }
}

// out += c * x
template<typename T>
void MulByConstAndAddT(T c, const T* x, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] += c * x[i];
  }
}

template<typename T, size_t N>
void MulByConstAndAddT(T c, const std::array<T, N>& x, std::array<T, N>& out) {
  MulByConstAndAddT(c, x.data(), out.data(), N);
}

template<typename T>
void AddFromT(const T* a, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] += a[i];
  }
}

// w is N x M matrix in row-major order, x is M x K matrix in column-major order
// y = w * x is N x K matrix in column-major order.
template<typename T>
void MatMulT(const T* w, const T* x, T* y, size_t N, size_t M, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      y[i * N + j] = DotT(&w[j * M], &x[i * M], M);
    }
  }
}

template<typename T>
void MatMulVJPT(const T* w, const T* x, const T* dy, T* dw, T* dx,
                size_t N, size_t M, size_t K) {
  memset(dx, 0, M * K * sizeof(dx[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      MulByConstAndAddT(dy[i * N + j], &x[i * M], &dw[j * M], M);
      MulByConstAndAddT(dy[i * N + j], &w[j * M], &dx[i * M], M);
    }
  }
}

// w is H concatenated N x M matrix in row-major order, x is HM x K matrix in
// column-major order and y = w' * x is N x K matrix in column-major order,
// where w' is the rearrangement of w into an N x HM matrix.
template<typename T>
void MultiHeadMatMul(const T* w, const T* x, T* y, size_t H, size_t N,
                     size_t M, size_t K) {
  memset(y, 0, N * K * sizeof(y[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t j = 0; j < N; ++j) {
        y[i * N + j] += DotT(&w[h * N * M + j * M], &x[i * H * M + h * M], M);
      }
    }
  }
}

template<typename T>
void MultiHeadMatMulVJP(const T* w, const T* x, const T* dy, T* dw, T* dx,
                        size_t H, size_t N, size_t M, size_t K) {
  memset(dx, 0, H * M * K * sizeof(dx[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t h = 0; h < H; ++h) {
        MulByConstAndAddT(dy[i * N + j], &x[i * H * M + h * M],
                          &dw[h * N * M + j * M], M);
        MulByConstAndAddT(dy[i * N + j], &w[h * N * M + j * M],
                          &dx[i * H * M + h * M], M);
      }
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
      tmp += (T(1.0) + w[j]) * dy[i* N + j] * x[i * N + j];
    }
    tmp *= ss3;
    for (size_t j = 0; j < N; ++j) {
      dx[i * N + j] = ss * (T(1.0) + w[j]) * dy[i* N + j] - tmp * x[i * N + j];
    }
  }
}

template<typename T>
void Softmax(T* x, size_t N) {
  T sum = {};
  auto maxreal = std::real(x[0]);
  for (size_t i = 1; i < N; ++i) {
    if (std::real(x[i]) > maxreal) {
      maxreal = std::real(x[i]);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    x[i] = std::exp(x[i] - maxreal);
    sum += x[i];
  }
  T scale = T(1.0) / sum;
  for (size_t i = 0; i < N; ++i) {
    x[i] *= scale;
  }
}

template<typename T>
void SoftmaxVJP(const T* y, T* dy, size_t N) {
  T sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += y[i] * dy[i];
  }
  for (size_t i = 0; i < N; ++i) {
    dy[i] = y[i] * (dy[i] - sum);
  }
}

template<typename T>
void Softmax(T* x, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    Softmax(x + i * N, N);
  }
}

template<typename T>
void SoftmaxVJP(const T* y, T* dy, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    SoftmaxVJP(y + i * N, dy + i * N, N);
  }
}

template<typename T>
void Softcap(T* x, size_t N) {
  T cap = 30.0;
  T inv_cap = T(1.0) / cap;
  for (size_t i = 0; i < N; ++i) {
    x[i] = cap * std::tanh(x[i] * inv_cap);
  }
}

template<typename T>
void SoftcapVJP(const T* y, T* dy, size_t N) {
  T cap = 30.0;
  T inv_cap = T(1.0) / cap;
  for (size_t i = 0; i < N; ++i) {
    T scaled = y[i] * inv_cap;
    dy[i] *= (T(1.0) - scaled * scaled);
  }
}

template<typename T>
T Gelu(T x) {
  static const T kMul = 0.044715;
  static const T kSqrt2OverPi = 0.797884560804236;

  const T x3 = x * x * x;
  const T arg = kSqrt2OverPi * (kMul * x3 + x);
  const T cdf = T(0.5) * (T(1.0) + std::tanh(arg));
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
  const T cdf = T(0.5) * (T(1.0) + tanh);
  const T dtanh = T(1.0) - tanh * tanh;
  const T darg = kMul2 * x2 + kSqrt2OverPi;
  return T(0.5) * x * dtanh * darg + cdf;
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

template<typename T, typename U>
void Rope(T* x, U base, size_t N, int i) {
  const size_t N2 = N / 2;
  for (size_t dim = 0; dim < N2; ++dim) {
    const T freq_exponents = T(2 * dim) / T(N);
    const T timescale = std::pow(base, freq_exponents);
    const T theta = T(i) / timescale;
    const T cos_val = std::cos(theta);
    const T sin_val = std::sin(theta);
    const T x0 = x[dim];
    const T x1 = x[dim + N2];
    x[dim] = x0 * cos_val - x1 * sin_val;
    x[dim + N2] = x0 * sin_val + x1 * cos_val;
  }
}

template<typename T>
void Rope(T* x, size_t N, int i) {
  Rope(x, T(10000.0), N, i);
}

template<typename T>
void Rope(std::complex<T>* x, size_t N, int i) {
  Rope(x, T(10000.0), N, i);
}

template <typename T, typename TConfig>
struct AllActivations {
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;

  std::array<ForwardLayer<T, TConfig>, kLayers> layers;
  std::array<T, kSeqLen * kModelDim> final_layer_output;
  std::array<T, kSeqLen * kModelDim> final_norm_output;
  std::array<T, kSeqLen * kVocabSize> logits;
  std::array<T, kSeqLen * kVocabSize> probs;
};

template<typename T, typename TConfig>
class ActivationsWrapper {
 public:
  ActivationsWrapper()
      : data_(hwy::AllocateAligned<uint8_t>(
            sizeof(AllActivations<T, TConfig>))),
        activations_(
            reinterpret_cast<AllActivations<T, TConfig>*>(data_.get())) {}

  const AllActivations<T, TConfig>& get() const { return *activations_; }
  AllActivations<T, TConfig>& get() { return *activations_; }

 private:
  hwy::AlignedFreeUniquePtr<uint8_t[]> data_;
  AllActivations<T, TConfig>* activations_;
};

template<typename T, typename TConfig>
void MulByConstAndAddT(T c, const Layer<T, TConfig>& x,
                      Layer<T, TConfig>& out) {
  MulByConstAndAddT(c, x.pre_attention_norm_scale,
                    out.pre_attention_norm_scale);
  MulByConstAndAddT(c, x.attn_vec_einsum_w, out.attn_vec_einsum_w);
  MulByConstAndAddT(c, x.qkv_einsum_w, out.qkv_einsum_w);
  MulByConstAndAddT(c, x.pre_ffw_norm_scale, out.pre_ffw_norm_scale);
  MulByConstAndAddT(c, x.gating_einsum_w, out.gating_einsum_w);
  MulByConstAndAddT(c, x.linear_w, out.linear_w);
}

template<typename T, typename TConfig>
void MulByConstAndAddT(T c, const Weights<T, TConfig>& x,
                       Weights<T, TConfig>& out) {
  static constexpr size_t kLayers = TConfig::kLayers;
  MulByConstAndAddT(c, x.embedder_input_embedding,
                    out.embedder_input_embedding);
  MulByConstAndAddT(c, x.final_norm_scale, out.final_norm_scale);
  for (size_t i = 0; i < kLayers; ++i) {
    MulByConstAndAddT(c, *x.GetLayer(i), *out.GetLayer(i));
  }
}

template<typename T>
void InputEmbedding(const T* w, const std::vector<int>& tokens, T scaling,
                    T* y, size_t N) {
  for (size_t i = 0; i + 1 < tokens.size(); ++i) {
    int token = tokens[i];
    memcpy(y + i * N, w + token * N, N * sizeof(y[0]));
    MulByConstT(scaling, y + i * N, N);
  }
}

template<typename T>
void InputEmbeddingVJP(const T* w, const std::vector<int>& tokens, T scaling,
                       const T* dy, T* dw, size_t N) {
  for (size_t i = 0; i + 1 < tokens.size(); ++i) {
    int token = tokens[i];
    MulByConstAndAddT(scaling, dy + i * N, dw + token * N, N);
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
        output[aoffset + pos2] = DotT(q, k, kQKVDim);
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
        MulByConstAndAddT(dout[pos2], k, dq, kQKVDim);
        MulByConstAndAddT(dout[pos2], q, dk, kQKVDim);
      }
    }
  }
}

template<typename T>
void MaskedSoftmax(T* x, size_t num_tokens, size_t kHeads, size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      size_t offset = pos * kHeads * kSeqLen + head * kSeqLen;
      Softmax(x + offset, pos + 1);
      memset(x + offset + pos + 1, 0, (kSeqLen - pos - 1) * sizeof(T));
    }
  }
}

template<typename T>
void MaskedSoftmaxVJP(const T* y, T* dy, size_t num_tokens,
                      size_t kHeads, size_t kSeqLen) {
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      size_t offset = pos * kHeads * kSeqLen + head * kSeqLen;
      SoftmaxVJP(y + offset, dy + offset, pos + 1);
      memset(dy + offset + pos + 1, 0, (kSeqLen - pos - 1) * sizeof(T));
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
        MulByConstAndAddT(att[pos2], v, out, kQKVDim);
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
        datt[pos2] = DotT(dout, &qkv[v_offset(pos2)], kQKVDim);
        MulByConstAndAddT(att[pos2], dout, &dqkv[v_offset(pos2)], kQKVDim);
      }
    }
  }
}

template<typename T, typename TConfig>
void ApplyLayer(const Layer<T, TConfig>& weights,
                ForwardLayer<T, TConfig>& activations,
                size_t num_tokens, T* output) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static const T kQueryScale = T(1.0) / std::sqrt(T(kQKVDim));

  RMSNorm(weights.pre_attention_norm_scale.data(), activations.input.data(),
          activations.pre_att_rms_out.data(), kModelDim, num_tokens);

  MatMulT(weights.qkv_einsum_w.data(), activations.pre_att_rms_out.data(),
          activations.qkv.data(), (kHeads + 2) * kQKVDim, kModelDim,
          num_tokens);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * kQKVDim, kQKVDim, pos);
    }
  }

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    MulByConstT(kQueryScale, qkv, kHeads * kQKVDim);
  }

  MaskedAttention(activations.qkv.data(), activations.att.data(),
                  num_tokens, kHeads, kQKVDim, kSeqLen);

  MaskedSoftmax(activations.att.data(), num_tokens, kHeads, kSeqLen);

  MixByAttention(activations.qkv.data(), activations.att.data(),
                 activations.att_out.data(), num_tokens, kHeads, kQKVDim,
                 kSeqLen);

  MultiHeadMatMul(weights.attn_vec_einsum_w.data(), activations.att_out.data(),
                  activations.attention_out.data(), kHeads, kModelDim, kQKVDim,
                  num_tokens);

  AddFromT(activations.input.data(), activations.attention_out.data(),
           num_tokens * kModelDim);

  RMSNorm(weights.pre_ffw_norm_scale.data(), activations.attention_out.data(),
          activations.bf_pre_ffw_rms_out.data(), kModelDim, num_tokens);

  MatMulT(weights.gating_einsum_w.data(), activations.bf_pre_ffw_rms_out.data(),
          activations.ffw_hidden.data(), kFFHiddenDim * 2, kModelDim,
          num_tokens);

  GatedGelu(activations.ffw_hidden.data(), activations.ffw_hidden_gated.data(),
            kFFHiddenDim, num_tokens);

  MatMulT(weights.linear_w.data(), activations.ffw_hidden_gated.data(),
          output, kModelDim, kFFHiddenDim, num_tokens);

  AddFromT(activations.attention_out.data(), output, num_tokens * kModelDim);
}

template<typename T, typename TConfig>
void LayerVJP(const Layer<T, TConfig>& weights,
              const ForwardLayer<T, TConfig>& forward,
              const T* dy,
              Layer<T, TConfig>& grad,
              ForwardLayer<T, TConfig>& backward,
              size_t num_tokens) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static const T kQueryScale = 1.0 / std::sqrt(T(kQKVDim));

  MatMulVJPT(weights.linear_w.data(), forward.ffw_hidden_gated.data(),
             dy, grad.linear_w.data(), backward.ffw_hidden_gated.data(),
             kModelDim, kFFHiddenDim, num_tokens);

  GatedGeluVJP(forward.ffw_hidden.data(), backward.ffw_hidden_gated.data(),
               backward.ffw_hidden.data(), kFFHiddenDim, num_tokens);

  MatMulVJPT(weights.gating_einsum_w.data(), forward.bf_pre_ffw_rms_out.data(),
             backward.ffw_hidden.data(), grad.gating_einsum_w.data(),
             backward.bf_pre_ffw_rms_out.data(), kFFHiddenDim * 2, kModelDim,
             num_tokens);

  RMSNormVJP(weights.pre_ffw_norm_scale.data(), forward.attention_out.data(),
             backward.bf_pre_ffw_rms_out.data(),
             grad.pre_ffw_norm_scale.data(), backward.attention_out.data(),
             kModelDim, num_tokens);

  AddFromT(dy, backward.attention_out.data(), num_tokens * kModelDim);

  MultiHeadMatMulVJP(weights.attn_vec_einsum_w.data(), forward.att_out.data(),
                     backward.attention_out.data(),
                     grad.attn_vec_einsum_w.data(),
                     backward.att_out.data(),
                     kHeads, kModelDim, kQKVDim, num_tokens);

  MixByAttentionVJP(forward.qkv.data(), forward.att.data(),
                    backward.att_out.data(), backward.qkv.data(),
                    backward.att.data(), num_tokens, kHeads, kQKVDim,
                    kSeqLen);

  MaskedSoftmaxVJP(forward.att.data(), backward.att.data(),
                   num_tokens, kHeads, kSeqLen);

  MaskedAttentionVJP(forward.qkv.data(), backward.att.data(),
                     backward.qkv.data(), num_tokens, kHeads, kQKVDim, kSeqLen);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    MulByConstT(kQueryScale, qkv, kHeads * kQKVDim);
  }

  for (int pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * kQKVDim, kQKVDim, -pos);
    }
  }

  MatMulVJPT(weights.qkv_einsum_w.data(), forward.pre_att_rms_out.data(),
             backward.qkv.data(), grad.qkv_einsum_w.data(),
            backward.pre_att_rms_out.data(),
            (kHeads + 2) * kQKVDim, kModelDim, num_tokens);
  RMSNormVJP(weights.pre_attention_norm_scale.data(), forward.input.data(),
             backward.pre_att_rms_out.data(),
             grad.pre_attention_norm_scale.data(),
             backward.input.data(), kModelDim, num_tokens);

  AddFromT(backward.attention_out.data(), backward.input.data(),
           num_tokens * kModelDim);
}

struct Prompt {
  std::vector<int> tokens;
  size_t context_size;
};

template<typename T>
T CrossEntropyLoss(const T* x, const Prompt& prompt, size_t V) {
  T loss = {};
  for (size_t i = 0; i + 1 < prompt.tokens.size(); ++i) {
    if (i + 1 < prompt.context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = prompt.tokens[i + 1];
    loss += std::log(x[i * V + next_token]);
  }
  T scaling = -1.0 / std::log(2.0);
  return loss * scaling;
}

template<typename T>
void CrossEntropyLossGrad(const T* x, T* dx, const Prompt& prompt, size_t V) {
  T scaling = -1.0 / std::log(2.0);
  size_t num_tokens = prompt.tokens.size() - 1;
  memset(dx, 0, V * num_tokens * sizeof(x[0]));
  for (size_t i = 0; i + 1 < prompt.tokens.size(); ++i) {
    if (i + 1 < prompt.context_size) {
      continue;
    }
    const int next_token = prompt.tokens[i + 1];
    dx[i * V + next_token] = scaling / x[i * V + next_token];
  }
}

template<typename T, typename TConfig>
T CrossEntropyLossForwardPass(const Prompt& prompt,
                              const Weights<T, TConfig>& weights,
                              AllActivations<T, TConfig>& forward) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;
  const size_t num_tokens = prompt.tokens.size() - 1;

  const T kEmbScaling = std::sqrt(kModelDim);
  InputEmbedding(weights.embedder_input_embedding.data(), prompt.tokens,
                 kEmbScaling, forward.layers[0].input.data(), kModelDim);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    T* output = layer + 1 < kLayers ?
                forward.layers[layer + 1].input.data() :
                forward.final_layer_output.data();
    ApplyLayer(*weights.GetLayer(layer), forward.layers[layer], num_tokens,
               output);
  }

  RMSNorm(weights.final_norm_scale.data(),
          forward.final_layer_output.data(),
          forward.final_norm_output.data(), kModelDim, num_tokens);

  MatMulT(weights.embedder_input_embedding.data(),
          forward.final_norm_output.data(),
          forward.logits.data(), kVocabSize, kModelDim, num_tokens);

  Softcap(forward.logits.data(), num_tokens * kVocabSize);

  memcpy(forward.probs.data(), forward.logits.data(),
         num_tokens * kVocabSize * sizeof(forward.logits[0]));
  Softmax(forward.probs.data(), kVocabSize, num_tokens);

  return CrossEntropyLoss(forward.probs.data(), prompt, kVocabSize);
}

template<typename T, typename TConfig>
void CrossEntropyLossBackwardPass(const Prompt& prompt,
                                  const Weights<T, TConfig>& weights,
                                  const AllActivations<T, TConfig>& forward,
                                  Weights<T, TConfig>& grad,
                                  AllActivations<T, TConfig>& backward) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;
  const size_t num_tokens = prompt.tokens.size() - 1;

  CrossEntropyLossGrad(forward.probs.data(), backward.logits.data(), prompt,
                       kVocabSize);

  SoftmaxVJP(forward.probs.data(), backward.logits.data(),
             kVocabSize, num_tokens);

  SoftcapVJP(forward.logits.data(), backward.logits.data(),
             num_tokens * kVocabSize);

  MatMulVJPT(weights.embedder_input_embedding.data(),
             forward.final_norm_output.data(),
             backward.logits.data(),
             grad.embedder_input_embedding.data(),
             backward.final_norm_output.data(),
             kVocabSize, kModelDim, num_tokens);

  RMSNormVJP(weights.final_norm_scale.data(),
             forward.final_layer_output.data(),
             backward.final_norm_output.data(),
             grad.final_norm_scale.data(),
             backward.final_layer_output.data(), kModelDim, num_tokens);

  for (int layer = static_cast<int>(kLayers) - 1; layer >= 0; --layer) {
    T* next_layer_grad = layer + 1 < kLayers
                         ? backward.layers[layer + 1].input.data()
                         : backward.final_layer_output.data();
    LayerVJP(*weights.GetLayer(layer), forward.layers[layer], next_layer_grad,
             *grad.GetLayer(layer), backward.layers[layer], num_tokens);
  }

  const T kEmbScaling = std::sqrt(kModelDim);
  InputEmbeddingVJP(weights.embedder_input_embedding.data(),
                    prompt.tokens, kEmbScaling,
                    backward.layers[0].input.data(),
                    grad.embedder_input_embedding.data(), kModelDim);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BACKPROP_SCALAR_H_
