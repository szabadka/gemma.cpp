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

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BACKPROP_SCALAR_H_
