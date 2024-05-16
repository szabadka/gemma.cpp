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
#define HWY_TARGET_INCLUDE "gemma/ops_test.cc"  //NOLINT
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

template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    hwy::RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};

    for (size_t count = 0; count < 2 * N; ++count) {
      for (size_t ma : misalignments) {
        for (size_t mb : misalignments) {
          Test()(d, count, ma, mb, rng);
        }
      }
    }
  }
};

template <typename T>
T Random(hwy::RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  const double val = (bits - 512) / 64.0;
  // Clamp negative to zero for unsigned types.
  return hwy::ConvertScalarTo<T>(
      HWY_MAX(hwy::ConvertScalarTo<double>(hwy::LowestValue<T>()), val));
}

HWY_NOINLINE void SourceAddFrom(const float* HWY_RESTRICT other,
                                float* HWY_RESTRICT x, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    x[i] += other[i];
  }
}

HWY_NOINLINE void SourceMulBy(const float* HWY_RESTRICT other,
                              float* HWY_RESTRICT x, size_t size,
                              size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= other[i];
  }
}

HWY_NOINLINE void SourceMulByConst(float c, float* HWY_RESTRICT x, size_t size,
                                   size_t max_pos) {
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= c;
  }
}

HWY_NOINLINE void SourceMulByConstAndAdd(float c, const float* HWY_RESTRICT x,
                                         float* HWY_RESTRICT out, size_t size,
                                         size_t max_pos) {
  for (size_t i = 0; i < max_pos; ++i) {
    out[i] += x[i] * c;
  }
}

HWY_NOINLINE void SourceSoftmax(float* HWY_RESTRICT x, size_t size,
                                size_t mask_pos) {
  HWY_DASSERT(size != 0);
  HWY_DASSERT(mask_pos <= size);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  const size_t N = hn::Lanes(d);

  const hn::Vec<D> vmin = hn::Set(d, hwy::LowestValue<float>());
  hn::Vec<D> vmax = vmin;
  size_t idx = 0;
  if (mask_pos >= N) {
    for (; idx <= mask_pos - N; idx += N) {
      vmax = hn::Max(vmax, LoadU(d, x + idx));
    }
  }
  vmax = hn::Max(vmax, LoadNOr(vmin, d, x + idx, mask_pos - idx));
  vmax = hn::MaxOfLanes(d, vmax);  // broadcast

  hn::Vec<D> sum = hn::Zero(d);
  idx = 0;
  if (mask_pos >= N) {
    for (; idx <= mask_pos - N; idx += N) {
      const hn::Vec<D> out = hn::Exp(d, hn::Sub(hn::LoadU(d, x + idx), vmax));
      sum = hn::Add(sum, out);
      hn::StoreU(out, d, x + idx);
    }
  }
  if (mask_pos > idx) {
    const size_t remaining = mask_pos - idx;
    const hn::Vec<D> out =
        hn::Exp(d, hn::Sub(hn::LoadN(d, x + idx, remaining), vmax));
    sum = hn::Add(sum, out);
    hn::StoreN(out, d, x + idx, remaining);
  }

  const float mul = 1.0f / hn::ReduceSum(d, sum);
  SourceMulByConst(mul, x, size, mask_pos);
}

template <size_t k>
HWY_NOINLINE std::discrete_distribution<int> SourceCreateDistribution(
    std::array<float, k>& top_k, float temperature) {
  // re-normalize distribution
  for (size_t i = 0; i < k; ++i) {
    top_k[i] = exp(log(top_k[i]) / temperature);
  }
  float denominator = 0.0f;
  for (size_t i = 0; i < k; ++i) {
    denominator += top_k[i];
  }
  denominator = 1.0f / denominator;
  MulByConst(denominator, top_k.data(), k);
  return std::discrete_distribution<int>(std::begin(top_k), std::end(top_k));
}

struct TestAddFrom {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }

    SourceAddFrom(o, e, count);
    AddFrom(o, x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulBy {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }

    SourceMulBy(o, e, count, count);
    MulBy(o, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulByConstAndAdd {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }
    T constant = Random<T>(rng);

    SourceMulByConstAndAdd(constant, o, e, count, count);
    MulByConstAndAdd(constant, o, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulByConst {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
    }
    T constant = Random<T>(rng);

    SourceMulByConst(constant, e, count, count);
    MulByConst(constant, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestSoftmax {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    if (count == 0) return;  // *Softmax would assert
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
    }

    SourceSoftmax(e, count, count);
    Softmax(x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

template <size_t k>
struct TestCreateDistribution {
  void operator()(hwy::RandomState& rng) {
    std::array<float, k> x;
    std::array<float, k> e;

    for (size_t i = 0; i < k; ++i) {
      x[i] = Random<float>(rng);
      e[i] = x[i];
    }
    const float constant = Random<float>(rng);
    auto expected = SourceCreateDistribution(e, constant);
    auto output = create_distribution(x, constant);

    AssertEqual(expected, output, hwy::TargetName(HWY_TARGET), __FILE__,
                __LINE__);
  }
};

void TestAllAddFrom() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestAddFrom>>()(float());
}

void TestAllMulBy() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulBy>>()(float());
}

void TestAllMulByConst() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConst>>()(float());
}

void TestAllMulByConstAndAdd() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConstAndAdd>>()(
      float());
}

void TestAllSoftmax() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestSoftmax>>()(float());
}

void TestAllCreateDistribution() {
  TestCreateDistribution<2048>();
  TestCreateDistribution<5000>();
}

template <size_t kOuter, size_t kInner>
CompressedArray<float, kOuter * kInner> GenerateMat(size_t offset) {
  hwy::ThreadPool pool(0);
  gcpp::CompressWorkingSet ws;
  CompressedArray<float, kOuter * kInner> mat;
  std::array<float, kOuter * kInner> content;
  const float scale = 1.0f / kInner;
  for (size_t i = 0; i < kOuter; i++) {
    for (size_t j = 0; j < kInner; j++) {
      content[i * kInner + j] = static_cast<float>((i + j + offset) * scale);
    }
  }
  Compress(content, ws, mat, pool);
  mat.set_scale(1.0f);
  return mat;
}

template <size_t kOuter, size_t kInner>
CompressedArray<float, kOuter * kInner> GenerateZeroMat(size_t offset) {
  hwy::ThreadPool pool(static_cast<size_t>(std::clamp(
      static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 4)));
  gcpp::CompressWorkingSet ws;
  CompressedArray<float, kOuter * kInner> mat;
  std::array<float, kOuter * kInner> content;

  pool.Run(0, kOuter, [&](const size_t i, size_t thread) {
    for (size_t j = 0; j < kInner; j++) {
      content[i * kInner + j] = 0.0f;
    }
  });

  Compress(content, ws, mat, pool);
  mat.set_scale(1.0f);
  return mat;
}

template <size_t length>
hwy::AlignedFreeUniquePtr<float[]> GenerateVec(size_t offset) {
  hwy::AlignedFreeUniquePtr<float[]> vec = hwy::AllocateAligned<float>(length);
  HWY_ASSERT(vec);
  for (size_t idx = 0; idx < length; idx++) {
    vec[idx] = static_cast<float>(idx + offset);
  }
  return vec;
}

// A simple matrix multiplication. No optimization / tiling.
template <size_t kM, size_t kN, size_t kK>
hwy::AlignedFreeUniquePtr<float[]> SimpleMatMul(
    const hwy::AlignedFreeUniquePtr<float[]>& a,
    const hwy::AlignedFreeUniquePtr<float[]>& b) {
  hwy::AlignedFreeUniquePtr<float[]> out = hwy::AllocateAligned<float>(kM * kK);
  hwy::ZeroBytes(out.get(), kM * kK * sizeof(float));

  int i, j, k;
  for (i = 0; i < kM; ++i) {
    for (j = 0; j < kK; ++j) {
      for (k = 0; k < kN; ++k) {
        out[i * kK + j] += a[i * kN + k] * b[k * kK + j];
      }
    }
  }
  return out;
}

template <size_t kOuter, size_t kInner>
hwy::AlignedFreeUniquePtr<float[]> SimpleMatVecAdd(
    const CompressedArray<float, kOuter * kInner>& mat,
    const hwy::AlignedFreeUniquePtr<float[]>& vec,
    const hwy::AlignedFreeUniquePtr<float[]>& add) {
  hwy::AlignedFreeUniquePtr<float[]> uncompressed_mat =
      hwy::AllocateAligned<float>(kOuter * kInner);
  hwy::AlignedFreeUniquePtr<float[]> out = hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(uncompressed_mat && out);
  Decompress(mat, 0, uncompressed_mat.get(), kOuter * kInner);
  for (size_t idx_row = 0; idx_row < kOuter; idx_row++) {
    out[idx_row] = add[idx_row];
    for (size_t idx_col = 0; idx_col < kInner; idx_col++) {
      out[idx_row] +=
          uncompressed_mat[kInner * idx_row + idx_col] * vec[idx_col];
    }
  }
  return out;
}

template <size_t length>
void AssertClose(const hwy::AlignedFreeUniquePtr<float[]>& a,
                 const hwy::AlignedFreeUniquePtr<float[]>& b) {
  for (size_t idx = 0; idx < length; idx++) {
    const float rel_abs_delta = std::abs(a[idx] - b[idx]) /
                                std::max(std::abs(a[idx]), std::abs(b[idx]));
    EXPECT_LT(rel_abs_delta, 2e-6)
        << "a[" << idx << "]=" << a[idx] << ", b[" << idx << "]=" << b[idx];
  }
}

template <typename MatT>
void AssertClose(const hwy::AlignedFreeUniquePtr<MatT[]>& expected,
                 const hwy::AlignedFreeUniquePtr<MatT[]>& actual, size_t num) {
  for (size_t idx = 0; idx < num; idx++) {
    double expected_value = hwy::ConvertScalarTo<double>(expected[idx]);
    double actual_value = hwy::ConvertScalarTo<double>(actual[idx]);

    const double tolerance =
        expected_value * 20 * 1.0 / (1ULL << hwy::MantissaBits<MatT>());

    if (!(expected_value - tolerance <= actual_value &&
          actual_value <= expected_value + tolerance)) {
      fprintf(stderr, "expected[%lu]: %f, actual[%lu]: %f\n", idx,
              expected_value, idx, actual_value);
      HWY_ASSERT(0);
    }
  }
}

void TestMatMul() {
  hwy::ThreadPool pool(0);
  constexpr size_t kM = 128 * 3;  // 384
  constexpr size_t kK = 128 * 5;  // 640
  constexpr size_t kN = 128 * 6;  // 768

  CompressedArray<float, kM * kN> a1 = GenerateMat<kM, kN>(0);
  CompressedArray<float, kN * kK> b1 = GenerateMat<kN, kK>(0);

  hwy::AlignedFreeUniquePtr<float[]> a = hwy::AllocateAligned<float>(kM * kN);
  Decompress(a1, 0, a.get(), kM * kN);

  hwy::AlignedFreeUniquePtr<float[]> b = hwy::AllocateAligned<float>(kN * kK);
  Decompress(b1, 0, b.get(), kN * kK);

  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatMul<kM, kN, kK>(a, b);

  CompressedArray<float, kM * kK> compressed_c = GenerateZeroMat<kM, kK>(0);
  hwy::AlignedFreeUniquePtr<float[]> c = hwy::AllocateAligned<float>(kM * kK);
  Decompress(compressed_c, 0, c.get(), kM * kK);

  MatMul<kM, kN, kK>(a.get(), b.get(), c.get());

  AssertClose(expected_out1, c, kM * kK);
}

void TestMatVecAdd() {
  hwy::ThreadPool pool(0);
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat = GenerateMat<kOuter, kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> even_odd =
      hwy::AllocateAligned<float>(kInner * pool.NumWorkers());
  hwy::AlignedFreeUniquePtr<float[]> expected_out =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add);
  hwy::AlignedFreeUniquePtr<float[]> actual_out =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add && even_odd && expected_out && actual_out);
  MatVecAdd</*kAdd=*/true, kOuter, kInner>(
      mat, 0, vec.get(), add.get(), even_odd.get(), actual_out.get(), pool);
  AssertClose<kOuter>(actual_out, expected_out);
}

void TestTwoMatVecAdd() {
  hwy::ThreadPool pool(0);
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat0 = GenerateMat<kOuter, kInner>(0);
  CompressedArray<float, kOuter * kInner> mat1 = GenerateMat<kOuter, kInner>(1);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add0 = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> add1 = GenerateVec<kOuter>(1);
  hwy::AlignedFreeUniquePtr<float[]> expected_out0 =
      SimpleMatVecAdd<kOuter, kInner>(mat0, vec, add0);
  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatVecAdd<kOuter, kInner>(mat1, vec, add1);
  hwy::AlignedFreeUniquePtr<float[]> actual_out0 =
      hwy::AllocateAligned<float>(kOuter);
  hwy::AlignedFreeUniquePtr<float[]> actual_out1 =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoMatVecAdd<true, kOuter, kInner>(mat0, mat1, 0, vec.get(), add0.get(),
                                     add1.get(), actual_out0.get(),
                                     actual_out1.get(), pool);
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

void TestTwoOfsMatVecAddLoop() {
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat = GenerateMat<kOuter, kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add0 = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> add1 = GenerateVec<kOuter>(1);
  hwy::AlignedFreeUniquePtr<float[]> expected_out0 =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add0);
  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add1);
  hwy::AlignedFreeUniquePtr<float[]> actual_out0 =
      hwy::AllocateAligned<float>(kOuter);
  hwy::AlignedFreeUniquePtr<float[]> actual_out1 =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoOfsMatVecAddLoop<true, kOuter, kInner>(mat, 0, 0, vec.get(), add0.get(),
                                            add1.get(), actual_out0.get(),
                                            actual_out1.get());
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

void TestSigmoid() {
  std::vector<float> values;
  for (int i = -150; i <= 150; ++i) {
    values.push_back(.1f * i);
  }
  std::vector<float> result = values;
  Sigmoid(result.data(), result.size());

  for (size_t i = 0; i < values.size(); i++) {
    const float max_error = 0.00007;
    float value = values[i];
    float approx = result[i];
    float expected = (1 / (1 + std::exp(-values[i])));
    EXPECT_NEAR(approx, expected, max_error) << "Input: " << value;
  }
}

void TestLossGradient() {
  static const size_t kVocabSize = 8;
  static const size_t kSeqLen = 8;
  hwy::ThreadPool pool(8);
  std::vector<int> prompt = { 0, 1, 2, 3, 4 };
  const size_t context_size = 1;
  const size_t num_tokens = prompt.size() - 1;
  std::array<float, kSeqLen * kVocabSize> logits;
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < num_tokens * kVocabSize; ++i) {
    logits[i] = dist(gen);
  }
  std::array<float, kSeqLen * kVocabSize> grad;
  LossGradient<kVocabSize>(logits.data(), prompt, context_size, grad.data(),
                           pool);

  static constexpr float kStep = 1e-2;
  static constexpr float kStepScale = 0.5f / kStep;
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t i = 0; i < kVocabSize; ++i) {
      size_t idx = pos * kVocabSize + i;
      const float x0 = logits[idx];
      logits[i] = x0 + kStep;
      float loss1 = CrossEntropyLoss<kVocabSize>(
          logits.data(), prompt, context_size, pool);
      logits[i] = x0 - kStep;
      float loss2 = CrossEntropyLoss<kVocabSize>(
          logits.data(), prompt, context_size, pool);
      const float exp_grad = (loss1 - loss2) * kStepScale;
      ASSERT_NEAR(grad[idx], exp_grad, 1e-2)
        << "Cross entropy loss gradient index " << idx
        << " expected " << exp_grad << " [loss1 = " << loss1
        << " loss2 = " << loss2 << "] actual " << grad[idx];
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(OpsTest);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllAddFrom);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulBy);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConst);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConstAndAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllSoftmax);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllCreateDistribution);
HWY_EXPORT_AND_TEST_P(OpsTest, TestMatMul);
HWY_EXPORT_AND_TEST_P(OpsTest, TestMatVecAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestTwoMatVecAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestTwoOfsMatVecAddLoop);
HWY_EXPORT_AND_TEST_P(OpsTest, TestSigmoid);
HWY_EXPORT_AND_TEST_P(OpsTest, TestLossGradient);
#ifdef HWY_AFTER_TEST
HWY_AFTER_TEST();
#endif

}  // namespace gcpp

#endif
