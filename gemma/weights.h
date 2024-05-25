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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

template <typename T, class TConfig>
struct Layer {
  Layer() = default;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = kHeads * kQKVDim * kModelDim;
  static constexpr size_t kQKVEinsumWSize =
      (kHeads + 2 * kKVHeads) * kQKVDim * kModelDim;
  // 2x for (gelu gating vector, gated vector)
  static constexpr size_t kGatingEinsumWSize = 2 * kFFHiddenDim * kModelDim;
  static constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
  static constexpr bool kFFBiases = TConfig::kFFBiases;
  static constexpr size_t kAOBiasDim =
      TConfig::kSoftmaxAttnOutputBiases ? kModelDim : 0;
  static constexpr size_t kGriffinDim =
      TConfig::kGriffinLayers > 0 ? kModelDim : 0;

  union {
    struct {
      std::array<T, kAttVecEinsumWSize> attn_vec_einsum_w;
      std::array<T, kQKVEinsumWSize> qkv_einsum_w;
      std::array<T, kAOBiasDim> attention_output_biases;
    };

    struct {
      std::array<T, kGriffinDim * kGriffinDim> linear_x_w;
      std::array<T, kGriffinDim> linear_x_biases;
      std::array<T, kGriffinDim * kGriffinDim> linear_y_w;
      std::array<T, kGriffinDim> linear_y_biases;
      std::array<T, kGriffinDim * kGriffinDim> linear_out_w;
      std::array<T, kGriffinDim> linear_out_biases;
      std::array<T, kConv1dWidth * kGriffinDim> conv_w;
      std::array<T, kGriffinDim> conv_biases;
      std::array<T, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      std::array<T, kGriffinDim * 2> gate_biases;
      std::array<T, kGriffinDim> a;
    } griffin;
  };

  std::array<T, kGatingEinsumWSize> gating_einsum_w;
  std::array<T, kModelDim * kFFHiddenDim> linear_w;
  std::array<T, kModelDim> pre_attention_norm_scale;
  std::array<T, kModelDim> pre_ffw_norm_scale;

  std::array<T, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  std::array<T, kFFBiases ? kModelDim : 0> ffw_output_biases;
};

// Array instead of single large allocation for parallel mem init. Split out of
// Weights so that only these pointers are initialized.
template <typename T, class TConfig>
struct LayerPointers {
  explicit LayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->layers[task] = hwy::AllocateAligned<Layer<T, TConfig>>(1);
    });
  }

  using TLayer = Layer<T, TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<TLayer[]>, TConfig::kLayers> layers;
};

template <typename T, class TConfig>
struct Weights {
  // No ctor/dtor, allocated via AllocateAligned.

  std::array<T, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  std::array<T, TConfig::kModelDim> final_norm_scale;

  LayerPointers<T, TConfig> layer_ptrs;

  std::array<T, TConfig::kNumTensorScales> scales;

  const Layer<T, TConfig>* GetLayer(size_t layer) const {
    return layer_ptrs.layers[layer].get();
  }
  Layer<T, TConfig>* GetLayer(size_t layer) {
    return layer_ptrs.layers[layer].get();
  }
};

template <typename T, typename TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> AllocateWeights(hwy::ThreadPool& pool) {
  using TWeights = Weights<T, TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(TWeights));
  TWeights* weights = reinterpret_cast<TWeights*>(weights_u8.get());
  new (&weights->layer_ptrs) LayerPointers<T, TConfig>(pool);
  return weights_u8;
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
