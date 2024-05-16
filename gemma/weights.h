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

namespace gcpp {

template <class TConfig>
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

  template <class T, size_t N>
  using ArrayT = std::array<T, N>;

  union {
    struct {
      ArrayT<float, kAttVecEinsumWSize> attn_vec_einsum_w;
      ArrayT<float, kQKVEinsumWSize> qkv_einsum_w;
      ArrayT<float, kAOBiasDim> attention_output_biases;
    };

    struct {
      ArrayT<float, kGriffinDim * kGriffinDim> linear_x_w;
      ArrayT<float, kGriffinDim> linear_x_biases;
      ArrayT<float, kGriffinDim * kGriffinDim> linear_y_w;
      ArrayT<float, kGriffinDim> linear_y_biases;
      ArrayT<float, kGriffinDim * kGriffinDim> linear_out_w;
      ArrayT<float, kGriffinDim> linear_out_biases;
      ArrayT<float, kConv1dWidth * kGriffinDim> conv_w;
      ArrayT<float, kGriffinDim> conv_biases;
      ArrayT<float, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      ArrayT<float, kGriffinDim * 2> gate_biases;
      ArrayT<float, kGriffinDim> a;
    } griffin;
  };

  ArrayT<float, kGatingEinsumWSize> gating_einsum_w;
  ArrayT<float, kModelDim * kFFHiddenDim> linear_w;
  ArrayT<float, kModelDim> pre_attention_norm_scale;
  ArrayT<float, kModelDim> pre_ffw_norm_scale;

  ArrayT<float, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  ArrayT<float, kFFBiases ? kModelDim : 0> ffw_output_biases;
};

// Array instead of single large allocation for parallel mem init. Split out of
// Weights so that only these pointers are initialized.
template <class TConfig>
struct LayerPointers {
  explicit LayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->layers[task] = hwy::AllocateAligned<Layer<TConfig>>(1);
    });
  }

  using TLayer = Layer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<TLayer[]>, TConfig::kLayers> layers;
};

template <class TConfig>
struct Weights {
  // No ctor/dtor, allocated via AllocateAligned.

  std::array<float, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  std::array<float, TConfig::kModelDim> final_norm_scale;

  LayerPointers<TConfig> layer_ptrs;

  std::array<float, TConfig::kNumTensorScales> scales;

  const Layer<TConfig>* GetLayer(size_t layer) const {
    return layer_ptrs.layers[layer].get();
  }
  Layer<TConfig>* GetLayer(size_t layer) {
    return layer_ptrs.layers[layer].get();
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
