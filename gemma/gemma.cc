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

// Lightweight C++ implementation of the gemma model.

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gemma.cc"  // NOLINT
#include "hwy/foreach_target.h"        // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "compression/compress-inl.h"
#include "gemma/ops.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "compression/compress.h"
#include "compression/io.h"  // Path
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

// Setting this to true disables fread() calls that read the model file.
constexpr bool kDryRunFread = false;

// Setting this to false will load and use uncompressed weights.
constexpr bool kWeightsAreCompressed = true;

// Set this to true to debug tokenizer tokens.
constexpr bool kShowTokenization = false;

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

float ScaleWeights(float* data, size_t len) {
  float maxabs = 0.0;
  for (size_t i = 0; i < len; ++i) {
    maxabs = std::max(maxabs, std::abs(data[i]));
  }
  const float kMaxRange = 1.875f;
  if (maxabs <= kMaxRange) {
    return 1.0f;
  }
  const float scale = maxabs / kMaxRange;
  const float inv_scale = 1.0f / scale;
  for (size_t i = 0; i < len; ++i) {
    data[i] *= inv_scale;
  }
  return scale;
}

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

template <typename TConfig>
WeightStorageT AllocateWeights(hwy::ThreadPool& pool) {
  using TWeights = Weights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(TWeights));
  TWeights* weights = reinterpret_cast<TWeights*>(weights_u8.get());
  new (&weights->layer_ptrs) LayerPointers<TConfig>(pool);
  return weights_u8;
}

template <typename TConfig>
WeightStorageT AllocateForwardPass() {
  using TForward = ForwardPass<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> forward_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(TForward));
  TForward* forward = reinterpret_cast<TForward*>(forward_u8.get());
  return forward_u8;
}

template <typename TConfig>
WeightStorageT AllocateBackwardPass() {
  using TBackward = BackwardPass<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> backward_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(TBackward));
  TBackward* backward = reinterpret_cast<TBackward*>(backward_u8.get());
  return backward_u8;
}

template <typename TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadWeights(
    const Path& checkpoint, hwy::ThreadPool& pool,
    bool scale_for_compression = false) {
  PROFILER_ZONE("Startup.LoadWeights");
  if (!checkpoint.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              checkpoint.path.c_str());
  }

  WeightStorageT weights_u8 = AllocateWeights<TConfig>(pool);
  auto* weights = reinterpret_cast<Weights<TConfig>*>(weights_u8.get());

  size_t scale_pos = 0;
  FILE* fptr;
  if constexpr (kDryRunFread) {
    fprintf(stderr, "Dry-Run, not reading model-file.\n");
  } else {
    fptr = fopen(checkpoint.path.c_str(), "rb");
    if (fptr == nullptr) {
      HWY_ABORT("Failed to open model file %s - does it exist?",
                checkpoint.path.c_str());
    }
  }
  bool ok = true;
  uint64_t total_size = 0;
  auto do_fread = [&](void* var, int layer, const char* name, size_t size) {
    if (layer == -1) {
      fprintf(stderr, "Loading Parameters (size %zu): %s\n", size, name);
    } else {
      fprintf(stderr, "Loading Parameters (layer=%d, size %zu): %s\n", layer,
              size, name);
    }
    if constexpr (!kDryRunFread) {
      ok &= 1 == fread(var, size, 1, fptr);
      total_size += size;
    }
  };
  do_fread(&(weights->embedder_input_embedding), -1, "embedder_input_embedding",
           sizeof(weights->embedder_input_embedding));
  do_fread(&(weights->final_norm_scale), -1, "final_norm_scale",
           sizeof(weights->final_norm_scale));
  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    Layer<TConfig>* layer_view = weights->GetLayer(layer);

#define READ_WEIGHTS(name)                                                 \
  do {                                                                     \
    do_fread(&(layer_view->name), layer, #name, sizeof(layer_view->name)); \
  } while (0)

#define SCALE_WEIGHTS(name)                                               \
  do {                                                                    \
    if (ok && !kDryRunFread && scale_for_compression) {                   \
      weights->scales[scale_pos++] =                                      \
          ScaleWeights(layer_view->name.data(), layer_view->name.size()); \
    }                                                                     \
  } while (0)
    // Make sure we don't have uninitialized memory.
    hwy::ZeroBytes(layer_view, sizeof(*layer_view));
    if (type == LayerAttentionType::kGemma) {
      READ_WEIGHTS(attn_vec_einsum_w);
      READ_WEIGHTS(qkv_einsum_w);
      SCALE_WEIGHTS(attn_vec_einsum_w);
      SCALE_WEIGHTS(qkv_einsum_w);
    } else {
      READ_WEIGHTS(griffin.linear_x_w);
      READ_WEIGHTS(griffin.linear_x_biases);
      READ_WEIGHTS(griffin.linear_y_w);
      READ_WEIGHTS(griffin.linear_y_biases);
      READ_WEIGHTS(griffin.linear_out_w);
      READ_WEIGHTS(griffin.linear_out_biases);
      READ_WEIGHTS(griffin.conv_w);
      READ_WEIGHTS(griffin.conv_biases);
      READ_WEIGHTS(griffin.gate_w);
      READ_WEIGHTS(griffin.gate_biases);
      READ_WEIGHTS(griffin.a);
      SCALE_WEIGHTS(griffin.linear_x_w);
      SCALE_WEIGHTS(griffin.linear_y_w);
      SCALE_WEIGHTS(griffin.linear_out_w);
      SCALE_WEIGHTS(griffin.gate_w);
    }
    READ_WEIGHTS(gating_einsum_w);
    READ_WEIGHTS(linear_w);
    SCALE_WEIGHTS(gating_einsum_w);
    SCALE_WEIGHTS(linear_w);
    READ_WEIGHTS(pre_attention_norm_scale);
    READ_WEIGHTS(pre_ffw_norm_scale);
    if (TConfig::kFFBiases) {
      READ_WEIGHTS(ffw_gating_biases);
      READ_WEIGHTS(ffw_output_biases);
    }
    if (TConfig::kSoftmaxAttnOutputBiases &&
        type == LayerAttentionType::kGemma) {
      READ_WEIGHTS(attention_output_biases);
    }
#undef READ_WEIGHTS
  }
  if (!ok) {
    HWY_ABORT(
        "Failed to read from %s - might be a directory, or too small? "
        "expected size: %d kB",
        checkpoint.path.c_str(), static_cast<uint32_t>(total_size >> 10));
  }
  if (!kDryRunFread) {
    HWY_ASSERT(0 == fclose(fptr));
    if (scale_for_compression) {
      HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
    }
  }
  return weights_u8;
}

template <class TConfig>
struct CompressedLayer {
  // No ctor/dtor, allocated via AllocateAligned.

  using TLayer = gcpp::Layer<TConfig>;
  using WeightT = typename TConfig::WeightT;

  static constexpr size_t kHeads = TLayer::kHeads;
  static constexpr size_t kKVHeads = TLayer::kKVHeads;
  static constexpr size_t kModelDim = TLayer::kModelDim;
  static constexpr size_t kQKVDim = TLayer::kQKVDim;
  static constexpr size_t kFFHiddenDim = TLayer::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = TLayer::kAttVecEinsumWSize;
  static constexpr size_t kQKVEinsumWSize = TLayer::kQKVEinsumWSize;
  static constexpr size_t kGatingEinsumWSize = TLayer::kGatingEinsumWSize;
  static constexpr size_t kConv1dWidth = TLayer::kConv1dWidth;
  static constexpr bool kFFBiases = TLayer::kFFBiases;
  static constexpr size_t kAOBiasDim = TLayer::kAOBiasDim;
  static constexpr size_t kGriffinDim = TLayer::kGriffinDim;

  // Compressed Parameters

  template <class T, size_t N>
  using ArrayT = CompressedArray<T, N>;

  union {
    struct {
      ArrayT<WeightT, kAttVecEinsumWSize> attn_vec_einsum_w;
      ArrayT<WeightT, kQKVEinsumWSize> qkv_einsum_w;
      ArrayT<float, kAOBiasDim> attention_output_biases;
    };

    struct {
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_x_w;
      ArrayT<float, kGriffinDim> linear_x_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_y_w;
      ArrayT<float, kGriffinDim> linear_y_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_out_w;
      ArrayT<float, kGriffinDim> linear_out_biases;
      ArrayT<float, TConfig::kConv1dWidth * kGriffinDim> conv_w;
      ArrayT<float, kGriffinDim> conv_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      ArrayT<float, kGriffinDim * 2> gate_biases;
      ArrayT<float, kGriffinDim> a;
    } griffin;
  };

  ArrayT<WeightT, TLayer::kGatingEinsumWSize> gating_einsum_w;
  ArrayT<WeightT, kModelDim * kFFHiddenDim> linear_w;
  // We don't yet have an RMSNorm that accepts all WeightT.
  ArrayT<hwy::bfloat16_t, kModelDim> pre_attention_norm_scale;
  ArrayT<hwy::bfloat16_t, kModelDim> pre_ffw_norm_scale;

  ArrayT<float, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  ArrayT<float, kFFBiases ? kModelDim : 0> ffw_output_biases;
};

// Array instead of single large allocation for parallel mem init. Split out
// of CompressedWeights so that only these pointers are initialized, not the
// CompressedArray.
template <class TConfig>
struct CompressedLayerPointers {
  explicit CompressedLayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->c_layers[task] = hwy::AllocateAligned<CompressedLayer<TConfig>>(1);
    });
  }

  using CLayer = CompressedLayer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<CLayer[]>, TConfig::kLayers> c_layers;
};

template <class TConfig>
struct CompressedWeights {
  // No ctor/dtor, allocated via AllocateAligned.

  CompressedArray<EmbedderInputT, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  CompressedArray<hwy::bfloat16_t, TConfig::kModelDim> final_norm_scale;

  // Must be last so that the other arrays remain aligned.
  CompressedLayerPointers<TConfig> c_layer_ptrs;

  const CompressedLayer<TConfig>* GetLayer(size_t layer) const {
    return c_layer_ptrs.c_layers[layer].get();
  }
  CompressedLayer<TConfig>* GetLayer(size_t layer) {
    return c_layer_ptrs.c_layers[layer].get();
  }
};

template <class TConfig>
using WeightsT = hwy::If<kWeightsAreCompressed, CompressedWeights<TConfig>,
                         Weights<TConfig>>;

// Aligned.
template <class TConfig, size_t TBatchSize>
struct Activations {
  static constexpr size_t kBatchSize = TBatchSize;
  using LayerConfig = Layer<TConfig>;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kCacheLayerSize = kKVHeads * kQKVDim * 2;
  static constexpr size_t kCachePosSize =
      TConfig::kGemmaLayers * kCacheLayerSize;
  static constexpr size_t kQDim = kHeads == kKVHeads ? kQKVDim * 3 : kQKVDim;

  std::array<float, kBatchSize * kModelDim> x;  // input
  std::array<float, kBatchSize * kModelDim> pre_att_rms_out;
  std::array<float, kBatchSize * kHeads * kQDim> q;  // query vector
  std::array<float, kBatchSize * kHeads * TConfig::kSeqLen>
      att;                                                   // attention vector
  std::array<float, kBatchSize * kHeads * kQKVDim> att_out;  // attention output
  std::array<float, kHeads * kBatchSize * kModelDim>
      att_post1;  // attention output after linear transformation, per head
  std::array<float, kBatchSize * kModelDim>
      att_post2;  // accumulation of attention outputs over heads
  std::array<hwy::bfloat16_t, kBatchSize * kModelDim> bf_pre_ffw_rms_out;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim * 2> ffw_hidden;
  // bf_ version can't be used until GeluMulToBF16 issue in FFW() is resolved.
  // std::array<hwy::bfloat16_t, kBatchSize * 2 * TConfig::kFFHiddenDim>
  //     bf_ffw_hidden;
  std::array<float, kBatchSize * kModelDim> ffw_out;
  std::array<float, kBatchSize * TConfig::kVocabSize> logits;

  // For bf16/f32 vectors * bf16 matrix: faster to unpack once beforehand, into
  // per-thread storage.
  std::array<float, kModelDim * kMaxThreads> even_odd;

  // Griffin layer internal activations
  static constexpr size_t kGriffinDim =
      TConfig::kGriffinLayers > 0 ? kModelDim : 0;
  std::array<float, kBatchSize * kGriffinDim> griffin_x;
  std::array<float, kBatchSize * kGriffinDim> griffin_y;
  std::array<float, kBatchSize * kGriffinDim> griffin_gate_x;
  std::array<float, kBatchSize * kGriffinDim> griffin_multiplier;
};

// GemmaImpl is a template and thus cannot be exposed in gemma.h, hence we
// define an abstract base class.
struct GemmaInterface {
  virtual ~GemmaInterface() = default;

  virtual const GemmaTokenizer* Tokenizer() const = 0;
  virtual const WeightStorageT& Weights() const = 0;

  virtual void Generate(size_t max_tokens, size_t max_generated_tokens,
                        float temperature, const std::vector<int>& prompt,
                        size_t start_pos, KVCache& kv_cache,
                        hwy::ThreadPool& pool, const StreamFunc& stream_token,
                        const AcceptFunc& accept_token, std::mt19937& gen,
                        int verbosity, LayersOutputT* layers_output) = 0;

  virtual float ComputeCrossEntropy(size_t max_tokens,
                                    const std::vector<int>& prompt,
                                    KVCache& kv_cache, hwy::ThreadPool& pool,
                                    int verbosity) = 0;
};

template <class Config>
KVCache CreateKVCacheT() {
  constexpr size_t kConv1dWidth = Config::kConv1dWidth;
  return CreateKVCache(
      Config::kGemmaLayers * Config::kKVHeads * Config::kQKVDim,
      Config::kSeqLen + kPrefillBatchSize,
      Config::kGriffinLayers * (kConv1dWidth == 0 ? 0 : kConv1dWidth - 1) *
          Config::kModelDim,
      Config::kGriffinLayers * Config::kModelDim);
}

KVCache CreateKVCache(Model type) {
  switch (type) {
    case Model::GEMMA_2B:
      return CreateKVCacheT<ConfigGemma2B>();
    case Model::GEMMA_7B:
      return CreateKVCacheT<ConfigGemma7B>();
    case Model::GRIFFIN_2B:
      return CreateKVCacheT<ConfigGriffin2B>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(type));
  }
}

class GemmaTokenizerImpl : public GemmaTokenizer {
 public:
  GemmaTokenizerImpl(
      std::unique_ptr<sentencepiece::SentencePieceProcessor>&& impl)
      : impl_(std::move(impl)) {}
  bool Encode(const std::string& input,
              std::vector<std::string>* pieces) const override {
    return impl_->Encode(input, pieces).ok();
  }
  bool Encode(const std::string& input,
              std::vector<int>* pieces) const override {
    if constexpr (kShowTokenization) {
      bool is_ok = impl_->Encode(input, pieces).ok();
      for (int i = 0; i < static_cast<int>(pieces->size()); i++) {
        fprintf(stderr, "%3d: %d\n", i, (*pieces)[i]);
      }
      return is_ok;
    } else {
      return impl_->Encode(input, pieces).ok();
    }
  }
  // Given a sequence of ids, decodes it into a detokenized output.
  bool Decode(const std::vector<int>& ids,
              std::string* detokenized) const override {
    return impl_->Decode(ids, detokenized).ok();
  }

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> impl_;
};

namespace {
template <class Config>
void DeleteLayersPtrs(CompressedWeights<Config>* c_weights) {
  c_weights->c_layer_ptrs.~CompressedLayerPointers<Config>();
}
template <class Config>
void DeleteLayersPtrs(Weights<Config>* weights) {
  weights->layer_ptrs.~LayerPointers<Config>();
}
}  // namespace

template <class Config>
struct GemmaImpl : public GemmaInterface {
  GemmaImpl(std::unique_ptr<sentencepiece::SentencePieceProcessor>& tokenizer,
            hwy::AlignedFreeUniquePtr<uint8_t[]>& weights_u8,
            hwy::ThreadPool& pool);

  ~GemmaImpl() {
    WeightsT<Config>* weights =
        reinterpret_cast<WeightsT<Config>*>(weights_u8.get());
    DeleteLayersPtrs(weights);
  }

  const GemmaTokenizer* Tokenizer() const override { return &tokenizer; }
  const WeightStorageT& Weights() const override { return weights_u8; }

  void Generate(size_t max_tokens, size_t max_generated_tokens,
                float temperature, const std::vector<int>& prompt,
                size_t start_pos, KVCache& kv_cache, hwy::ThreadPool& pool,
                const StreamFunc& stream_token, const AcceptFunc& accept_token,
                std::mt19937&, int verbosity,
                LayersOutputT* layers_output) override;

  float ComputeCrossEntropy(size_t max_tokens, const std::vector<int>& prompt,
                            KVCache& kv_cache, hwy::ThreadPool& pool,
                            int verbosity) override;

  GemmaTokenizerImpl tokenizer;
  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8;
  hwy::AlignedUniquePtr<Activations<Config, kPrefillBatchSize>> prefill;
  hwy::AlignedUniquePtr<Activations<Config, 1>> state;
};

std::string TokenString(const GemmaTokenizer* tokenizer, int token) {
  std::string token_str;
  tokenizer->Decode({token}, &token_str);
  return "'" + std::regex_replace(token_str, std::regex("\n"), "\\n") + "'";
}

}  // namespace gcpp
#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <size_t kBatchSize, typename LayerT, class TConfig>
HWY_NOINLINE void GriffinRecurrent(
    size_t batch_start, size_t num_tokens, size_t layer,
    Activations<TConfig, kBatchSize>& activations, const LayerT* layer_weights,
    KVCache& kv_cache, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Griffin");
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize>::kModelDim;
  static constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr bool kAdd = true;

  // X / Y linear layers.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    float* HWY_RESTRICT y = activations.griffin_y.data() + batch_offset;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    TwoMatVecAdd<kAdd, kModelDim, kModelDim>(
        layer_weights->griffin.linear_x_w, layer_weights->griffin.linear_y_w, 0,
        activations.pre_att_rms_out.data() + batch_offset,
        /*add0=*/layer_weights->griffin.linear_x_biases.data(),
        /*add1=*/layer_weights->griffin.linear_y_biases.data(), /*out0=*/x,
        /*out1=*/y, pool);
    Gelu(y, kModelDim);
  }

  // Conv1D.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    HWY_FULL(float) df;
    HWY_DASSERT(kModelDim % Lanes(df) == 0);
    const size_t layer_offset = layer * kModelDim * (kConv1dWidth - 1);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[HWY_MAX(kConv1dWidth, 1)];
    cache[0] = x;
    for (size_t i = 1; i < kConv1dWidth; i++) {
      cache[i] =
          kv_cache.conv1d_cache.get() + layer_offset +
          ((pos + kConv1dWidth - 1 - i) % (kConv1dWidth - 1)) * kModelDim;
    }
    for (size_t i = 0; i < kModelDim; i += Lanes(df)) {
      auto xv = hn::Load(df, x + i);
      auto accum0 =
          hn::Load(df, layer_weights->griffin.conv_biases.data() + i);
      auto accum1 = hn::Zero(df);
      static_assert(kConv1dWidth % 2 == 0, "Conv width must be even");
      for (size_t l = 0; 2 * l < kConv1dWidth; l++) {
        auto wv0 = hn::Load(df, layer_weights->griffin.conv_w.data() +
                                (kConv1dWidth - 1 - 2 * l) * kModelDim + i);
        auto wv1 = hn::Load(df, layer_weights->griffin.conv_w.data() +
                                (kConv1dWidth - 2 - 2 * l) * kModelDim + i);
        accum0 = hn::MulAdd(wv0, hn::Load(df, cache[l * 2] + i), accum0);
        accum1 = hn::MulAdd(wv1, hn::Load(df, cache[l * 2 + 1] + i), accum1);
      }
      hn::Store(hn::Add(accum0, accum1), df, x + i);
      hn::Store(xv, df, cache[HWY_MAX(kConv1dWidth, 1) - 1] + i);
    }
  }

  // RGLRU
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT y = activations.griffin_y.data() + batch_offset;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    float* HWY_RESTRICT gate_x =
        activations.griffin_gate_x.data() + batch_offset;
    float* HWY_RESTRICT a =
        activations.griffin_multiplier.data() + batch_offset;
    float* HWY_RESTRICT rnn_state =
        kv_cache.rglru_cache.get() + layer * kModelDim;

    pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      constexpr size_t kHeadDim = kModelDim / kHeads;
      constexpr size_t kMatrixSize = kHeadDim * kHeadDim;
      size_t head_offset = head * kHeadDim;
      TwoOfsMatVecAddLoop<kAdd, kHeadDim, kHeadDim>(
          layer_weights->griffin.gate_w, kMatrixSize * head,
          kMatrixSize * (kHeads + head), x + head_offset,
          /*add0=*/layer_weights->griffin.gate_biases.data() + head_offset,
          /*add1=*/layer_weights->griffin.gate_biases.data() + kModelDim +
          head_offset,
          /*out0=*/gate_x + head_offset, /*out1=*/a + head_offset);
      Sigmoid(gate_x + head_offset, kHeadDim);
      Sigmoid(a + head_offset, kHeadDim);
      const auto fn_mul = [](D d, hn::Vec<D> x, hn::Vec<D> gate_x)
                          HWY_ATTR { return hn::Mul(x, gate_x); };
      hn::Transform1(D(), a + head_offset, kHeadDim,
                     layer_weights->griffin.a.data() + head_offset, fn_mul);
      hn::Transform1(D(), x + head_offset, kHeadDim, gate_x + head_offset,
                     fn_mul);
      // RNN scan
      HWY_FULL(float) df;
      HWY_DASSERT(kHeadDim % Lanes(df) == 0);
      for (size_t i = 0; i < kHeadDim; i += Lanes(df)) {
        auto log_a = hn::Load(df, a + head_offset + i);
        auto gated_x = hn::Load(df, x + head_offset + i);
        auto rnn = hn::Load(df, rnn_state + head_offset + i);
        auto a = hn::Exp(df, log_a);
        auto x_multiplier = hn::Sqrt(hn::NegMulAdd(a, a, hn::Set(df, 1.0)));
        if (pos == 0) {
          x_multiplier = hn::Set(df, 1.0);
        }
        auto new_x = hn::MulAdd(x_multiplier, gated_x, hn::Mul(a, rnn));
        hn::Store(new_x, df, rnn_state + head_offset + i);

        // Join branches.
        auto yv = hn::Load(df, y + head_offset + i);
        auto pre_out = hn::Mul(yv, new_x);
        hn::Store(pre_out, df, x + head_offset + i);
      }
    });
  }

  // Final linear layer.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    float* out_ptr = activations.att_post2.data() + batch_idx * kModelDim;
    MatVecAdd<kAdd, kModelDim, kModelDim>(
        layer_weights->griffin.linear_out_w, 0, x,
        layer_weights->griffin.linear_out_biases.data(),
        activations.even_odd.data(), out_ptr, pool);
  }
}

template <size_t kBatchSize, typename LayerT, class TConfig>
HWY_NOINLINE void Attention(size_t batch_start, size_t num_tokens, size_t layer,
                            Activations<TConfig, kBatchSize>& activations,
                            const LayerT* layer_weights, KVCache& kv_cache,
                            hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kQKVDim = gcpp::Activations<TConfig, 1>::kQKVDim;
  static constexpr size_t kCachePosSize =
      gcpp::Activations<TConfig, kBatchSize>::kCachePosSize;
  static constexpr size_t kCacheLayerSize =
      gcpp::Activations<TConfig, kBatchSize>::kCacheLayerSize;
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize>::kModelDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static const float kQueryScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));

  auto Attn = [&](float* q, uint64_t head, size_t head_offset, size_t batch_idx,
                  size_t thread) HWY_ATTR {
    const size_t pos = batch_start + batch_idx;
    // Calculate scores
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * kSeqLen +
                                   batch_idx * kHeads * kSeqLen;

    Rope(q, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);

    // Compute Q dot K scores
    const size_t start_pos = pos - std::min(kSeqLen - 1, pos);
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset = cache_pos * kCachePosSize +
                               layer * kCacheLayerSize + head_offset;
      const float* HWY_RESTRICT k2 = kv_cache.kv_cache.get() + kv_offset;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2 % kSeqLen] = score;
    }
    Softmax(head_att, std::min(pos + 1, kSeqLen));

    // Weighted summation
    float* HWY_RESTRICT att_out = activations.att_out.data() + head * kQKVDim +
                                  batch_idx * kHeads * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset = cache_pos * kCachePosSize +
                               layer * kCacheLayerSize + head_offset;
      float* HWY_RESTRICT v2 = kv_cache.kv_cache.get() + kv_offset + kQKVDim;
      MulByConstAndAdd(head_att[pos2 % kSeqLen], v2, att_out, kQKVDim);
    }
  };

  if constexpr (kHeads == kKVHeads) {
    // Multi-Head Attention
    static_assert(TConfig::kInterleaveQKV);

    for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
      float* x = activations.pre_att_rms_out.data() + batch_idx * kModelDim;
      float* HWY_RESTRICT qkv =
          activations.q.data() + batch_idx * kHeads * kQKVDim * 3;
      MatVec<kHeads * kQKVDim * 3, kModelDim>(
          layer_weights->qkv_einsum_w, 0, x, activations.even_odd.data(), qkv,
          pool);
    }
    const size_t num_tasks = kHeads * num_tokens;
    pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
      const size_t head = task % kHeads;
      const size_t batch_idx = task / kHeads;
      const size_t pos = batch_start + batch_idx;
      float* HWY_RESTRICT q =
          activations.q.data() + (batch_idx * kHeads + head) * kQKVDim * 3;
      const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset = cache_pos * kCachePosSize +
                               layer * kCacheLayerSize + head * kQKVDim * 2;
      float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
      memcpy(kv, q + kQKVDim, 2 * kQKVDim * sizeof(float));
      Rope(kv, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
    });
    pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
      const size_t head = task % kHeads;
      const size_t batch_idx = task / kHeads;
      float* HWY_RESTRICT q =
          activations.q.data() + (batch_idx * kHeads + head) * kQKVDim * 3;
      Attn(q, head, head * kQKVDim * 2, batch_idx, thread);
    });
  } else {
    // Multi-Query Attention
    for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
      const size_t pos = batch_start + batch_idx;
      float* x = activations.pre_att_rms_out.data() + batch_idx * kModelDim;

      float* HWY_RESTRICT q =
          activations.q.data() + batch_idx * kHeads * kQKVDim;
      MatVec<kHeads * kQKVDim, kModelDim>(layer_weights->qkv_einsum_w, 0, x,
                                          activations.even_odd.data(), q, pool);

      const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset = cache_pos * kCachePosSize +
                               layer * kCacheLayerSize;
      float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
      MatVec<kQKVDim * 2, kModelDim>(layer_weights->qkv_einsum_w,
                                     kHeads * kQKVDim * kModelDim, x,
                                     activations.even_odd.data(), kv, pool);
      Rope(kv, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
    }
    const size_t num_tasks = kHeads * num_tokens;
    pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
      const size_t head = task % kHeads;
      const size_t batch_idx = task / kHeads;
      float* HWY_RESTRICT q =
          activations.q.data() + batch_idx * kHeads * kQKVDim;
      Attn(q + head * kQKVDim, head, 0, batch_idx, thread);
    });
  }

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    // TODO(szabadka) Use a single MatVecAdd like in GriffinRecurrent() after
    // rearranging the weights.
    float* HWY_RESTRICT att_out =
        activations.att_out.data() + batch_idx * kHeads * kQKVDim;
    float* HWY_RESTRICT layer_out =
        activations.att_post2.data() + batch_idx * kModelDim;
    MatVecAdd<TConfig::kSoftmaxAttnOutputBiases, kModelDim, kQKVDim>(
        layer_weights->attn_vec_einsum_w, 0, att_out,
        layer_weights->attention_output_biases.data(),
        activations.even_odd.data(), layer_out, pool);
    for (size_t head = 1; head < kHeads; ++head) {
      float* HWY_RESTRICT head_out =
          activations.att_post1.data() + head * kBatchSize * kModelDim;
      MatVec<kModelDim, kQKVDim>(
          layer_weights->attn_vec_einsum_w, head * kModelDim * kQKVDim,
          att_out + head * kQKVDim,
          activations.even_odd.data(), head_out, pool);
      AddFrom(head_out, layer_out, kModelDim);
    }
  }
}

template <size_t kBatchSize, typename LayerT, typename TConfig>
HWY_NOINLINE void FFW(Activations<TConfig, kBatchSize>& activations,
                      size_t num_tokens, const LayerT* layer_weights,
                      hwy::ThreadPool& pool) {
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  float* HWY_RESTRICT even_odd = activations.even_odd.data();

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;
    PROFILER_ZONE("Gen.FFW.GatedGELU");
    const hwy::bfloat16_t* HWY_RESTRICT vec =
        activations.bf_pre_ffw_rms_out.data() + batch_idx * kModelDim;
    float* HWY_RESTRICT out = activations.ffw_hidden.data() + hidden_offset;
    float* HWY_RESTRICT out_mul = out + kFFHiddenDim;

    // Same matrix, first and second half of rows. Could fuse into one MatVec.
    MatVecAdd<TConfig::kFFBiases, kFFHiddenDim, kModelDim>(
        layer_weights->gating_einsum_w, kFFHiddenDim * kModelDim, vec,
        layer_weights->ffw_gating_biases.data() + kFFHiddenDim, even_odd,
        out_mul, pool);
    // Gate, will go through the nonlinearity.
    MatVecAdd<TConfig::kFFBiases, kFFHiddenDim, kModelDim>(
        layer_weights->gating_einsum_w, 0, vec,
        layer_weights->ffw_gating_biases.data(), even_odd, out, pool);

    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    hn::Transform1(DF(), out, kFFHiddenDim, out_mul,
                   [](DF df, VF v, VF mul)
                       HWY_ATTR { return hn::Mul(mul, Gelu(df, v)); });
  }

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    PROFILER_ZONE("Gen.FFW\\GatedGELU");
    const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;
    MatVecAdd<TConfig::kFFBiases, kModelDim, kFFHiddenDim>(
        layer_weights->linear_w, 0,
        activations.ffw_hidden.data() + hidden_offset,
        layer_weights->ffw_output_biases.data(), even_odd,
        activations.ffw_out.data() + batch_idx * kModelDim, pool);
  }
}

// `EmbeddingScaling` can be constexpr only if `Sqrt` and `hwy::ConvertScalarTo`
// are both constexpr
#if HWY_COMPILER_GCC_ACTUAL
#define GEMMA_CONSTEXPR_EMBSCALING HWY_BF16_CONSTEXPR
#else
#define GEMMA_CONSTEXPR_EMBSCALING
#endif

template <typename TConfig>
GEMMA_CONSTEXPR_EMBSCALING float EmbeddingScaling() {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::bfloat16_t>(
      Sqrt(static_cast<float>(TConfig::kModelDim))));
}

template <size_t kBatchSize, typename WeightArrayT, typename TConfig>
HWY_NOINLINE void Prefill(const int* tokens, size_t num_tokens, size_t pos,
                          const WeightArrayT& weights,
                          Activations<TConfig, kBatchSize>& activations,
                          KVCache& kv_cache, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Prefill\\Att\\FFW");
  static constexpr size_t kModelDim = TConfig::kModelDim;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        const int token = tokens[token_idx];
        HWY_ASSERT(token >= 0);
        HWY_ASSERT(token < TConfig::kVocabSize);
        Decompress(weights.embedder_input_embedding, token * kModelDim,
                   activations.x.data() + token_idx * kModelDim, kModelDim);
        MulByConst(kEmbScaling, activations.x.data() + token_idx * kModelDim,
                   kModelDim);
        if constexpr (TConfig::kAbsolutePE) {
          AddAbsolutePositionalEmbeddings(
              activations.x.data() + token_idx * kModelDim, TConfig::kModelDim,
              pos);
        };
      });

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    const auto* layer_weights = weights.GetLayer(layer);
    size_t layer_of_type =
        NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);

    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      RMSNorm(activations.x.data() + token_idx * kModelDim,
              layer_weights->pre_attention_norm_scale.data(),
              activations.pre_att_rms_out.data() + token_idx * kModelDim,
              kModelDim);
    }
    if (type == LayerAttentionType::kGemma) {
      Attention<kBatchSize>(pos, num_tokens, layer_of_type, activations,
                            layer_weights, kv_cache, pool);
    } else {
      GriffinRecurrent<kBatchSize>(pos, num_tokens, layer_of_type, activations,
                                   layer_weights, kv_cache, pool);
    }

    pool.Run(0, num_tokens, [&](const uint64_t token_idx,
                                size_t /*thread*/) HWY_ATTR {
      AddFrom(activations.att_post2.data() + token_idx * kModelDim,
              activations.x.data() + token_idx * kModelDim, kModelDim);
      RMSNorm(activations.x.data() + token_idx * kModelDim,
              layer_weights->pre_ffw_norm_scale.data(),
              activations.bf_pre_ffw_rms_out.data() + token_idx * kModelDim,
              kModelDim);
    });
    FFW<kBatchSize>(activations, num_tokens, layer_weights, pool);
    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      AddFrom(activations.ffw_out.data() + token_idx * kModelDim,
              activations.x.data() + token_idx * kModelDim, kModelDim);
    }
  }  // foreach layer

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        RMSNormInplace(weights.final_norm_scale.data(),
                       activations.x.data() + token_idx * kModelDim, kModelDim);
      });
}

// n = 1 specialization
template <typename WeightArrayT, class TConfig>
void Transformer(int token, size_t pos, const WeightArrayT& weights,
                 Activations<TConfig, 1>& activations, KVCache& kv_cache,
                 hwy::ThreadPool& pool, LayersOutputT* layers_output) {
  if (layers_output != nullptr) {
    float token_f = token;
    (*layers_output)(pos, "Tokens", &token_f, 1);
  }
  static constexpr size_t kModelDim = TConfig::kModelDim;
  Decompress(weights.embedder_input_embedding, token * kModelDim,
             activations.x.data(), kModelDim);

  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();
  MulByConst(kEmbScaling, activations.x.data(), kModelDim);
  if constexpr (TConfig::kAbsolutePE) {
    AddAbsolutePositionalEmbeddings(activations.x.data(), TConfig::kModelDim,
                                    pos);
  };
  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    const auto* layer_weights = weights.GetLayer(layer);
    size_t layer_of_type =
        NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);
    RMSNorm(activations.x.data(),
            layer_weights->pre_attention_norm_scale.data(),
            activations.pre_att_rms_out.data(), kModelDim);
    if (type == LayerAttentionType::kGemma) {
      Attention<1>(pos, 1, layer_of_type, activations, layer_weights, kv_cache,
                   pool);
    } else {
      GriffinRecurrent<1>(pos, 1, layer_of_type, activations, layer_weights,
                          kv_cache, pool);
    }
    AddFrom(activations.att_post2.data(), activations.x.data(), kModelDim);
    RMSNorm(activations.x.data(), layer_weights->pre_ffw_norm_scale.data(),
            activations.bf_pre_ffw_rms_out.data(), kModelDim);
    FFW<1>(activations, /* num_tokens = */ 1, layer_weights, pool);
    AddFrom(activations.ffw_out.data(), activations.x.data(), kModelDim);
    if (layers_output != nullptr) {
      std::string block_name = "blocks." + std::to_string(layer);
      (*layers_output)(pos, block_name, activations.x.data(), kModelDim);
    }
  }
  RMSNormInplace(weights.final_norm_scale.data(), activations.x.data(),
                 kModelDim);
  if (layers_output != nullptr) {
    (*layers_output)(pos, "final_norm", activations.x.data(), kModelDim);
  }
}

template <class TConfig>
void RangeChecks(size_t& max_tokens, size_t& max_generated_tokens,
                 size_t& prompt_size) {
  if (!TConfig::kUseLocalAttention) {
    if (max_tokens > TConfig::kSeqLen) {
      fprintf(stderr, "WARNING: max_tokens %zu > kSeqLen %d, truncating.\n",
              max_tokens, TConfig::kSeqLen);
      max_tokens = static_cast<size_t>(TConfig::kSeqLen);
    }
  }

  if (max_generated_tokens > max_tokens) {
    fprintf(stderr,
            "WARNING: max_generated_tokens %zu > max_tokens %zu, truncating.\n",
            max_generated_tokens, max_tokens);
    max_generated_tokens = max_tokens - 1;
  }

  if (!TConfig::kUseLocalAttention) {
    if (prompt_size + max_generated_tokens > max_tokens) {
      fprintf(stderr,
              "WARNING: prompt_size %zu + max_generated_tokens %zu > "
              "max_tokens %zu, truncating to ",
              prompt_size, max_generated_tokens, max_tokens);
      prompt_size = std::min(prompt_size, max_tokens - max_generated_tokens);
      fprintf(stderr, "%zu\n", prompt_size);
    }
  }
}

template <class TConfig>
void GenerateImpl(GemmaImpl<TConfig>& gemma, size_t max_tokens,
                  size_t max_generated_tokens, float temperature,
                  const std::vector<int>& prompt, size_t pos, KVCache& kv_cache,
                  hwy::ThreadPool& pool, const StreamFunc& stream_token,
                  const AcceptFunc& accept_token, std::mt19937& gen,
                  int verbosity, LayersOutputT* layers_output) {
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  Activations<TConfig, 1>& activations = *gemma.state.get();
  Activations<TConfig, kPrefillBatchSize>& prefill_activations =
      *gemma.prefill.get();

  const WeightsT<TConfig>& weights =
      *reinterpret_cast<WeightsT<TConfig>*>(gemma.weights_u8.get());

  size_t prompt_size = prompt.size();
  RangeChecks<TConfig>(max_tokens, max_generated_tokens, prompt_size);
  if (pos >= max_tokens) {
    fprintf(stderr, "Warning: pos %zu >= max_tokens %zu, aborting.\n", pos,
            max_tokens);
    return;
  }
  HWY_ASSERT(prompt_size > 0);

  // pos indexes the KV cache. In the first turn of a chat, pos = 0.
  //
  // After the first turn, pos gets passed in with > 0 corresponding to the
  // current token position in the KV cache.
  //
  // pos_offset keeps track of the relative position within the turn, starting
  // at 0 each turn. During prefill, pos_offset corresponds to the index into
  // the prompt vector.
  //
  // In single-turn (non-chat) usage, pos and pos_offset start at 0 and are
  // always equal.
  size_t pos_offset = 0;  // offset relative to pos
  const double prefill_start = hwy::platform::Now();

  // Prefill stops before prompt_size - 1 since the last prompt token is the
  // first input token for generation.
  while (pos_offset < prompt_size - 1) {
    const size_t batch_size =
        std::min(kPrefillBatchSize, prompt_size - 1 - pos_offset);
    HWY_DASSERT(batch_size <= kPrefillBatchSize);
    HWY_DASSERT(pos_offset + batch_size <= prompt_size - 1);
    const int* batch_tokens = prompt.data() + pos_offset;
    Prefill<kPrefillBatchSize>(batch_tokens, batch_size, pos, weights,
                               prefill_activations, kv_cache, pool);
    for (size_t idx = 0; idx < batch_size; ++idx) {
      if (!stream_token(batch_tokens[idx], 0.0f)) return;
    }
    pos += batch_size;
    pos_offset += batch_size;
  }

  if (verbosity >= 2) {
    // in the future this output should not occur in GenerateImpl but instead
    // should be available as observable state for frontend code to handle I/O.
    const double prefill_end = hwy::platform::Now();
    const double prefill_tok_sec =
        static_cast<double>(pos_offset) / (prefill_end - prefill_start);
    std::cout << "\n[ Prefill tokens / sec = " << prefill_tok_sec << " ]";
  }

  const double gen_start = hwy::platform::Now();

  HWY_DASSERT(pos_offset == prompt_size - 1);

  size_t pos_gen_start = pos_offset;
  int token = prompt.at(pos_offset);
  stream_token(token, 0);
  for (size_t generate_pos = 0;
       pos < max_tokens && generate_pos < max_generated_tokens;
       ++pos, ++pos_offset, ++generate_pos) {
    const bool is_generating_phase = pos_offset >= prompt_size - 1;
    Transformer(token, pos, weights, activations, kv_cache, pool,
                layers_output);
    float* final_activation = activations.x.data();
    // The condition below is always true if we are doing Prefill above.
    // We keep it here for clarity so that the code is correct even if Prefill
    // is disabled.
    if (is_generating_phase) {
      PROFILER_ZONE("Gen.Embedding");
      // Generation phase
      MatVec<kVocabSize, TConfig::kModelDim>(
          weights.embedder_input_embedding, 0, final_activation,
          activations.even_odd.data(), activations.logits.data(), pool);
      // Barrier: must have all logits so we can subtract max.
      Softmax(activations.logits.data(), kVocabSize);
      token = SampleTopK<TConfig::kTopK>(activations.logits.data(), kVocabSize,
                                         gen, temperature, accept_token);
      if (!stream_token(token, activations.logits[token])) {
        token = EOS_ID;
      }
    } else {
      // We would take this branch if we were not doing Prefill but would
      // process the tokens of the prompt one at a time.
      token = prompt.at(pos_offset + 1);
      if (!stream_token(token, 0)) {
        token = EOS_ID;
      }
    }
    if (token == EOS_ID) {
      if (verbosity >= 2) {
        const double gen_end = hwy::platform::Now();
        const double gen_tok_sec =
            static_cast<double>(pos_offset - pos_gen_start) /
            (gen_end - gen_start);
        std::cout << "\n[ Generation tokens / sec = " << gen_tok_sec << " ]\n";
      }
      break;
    }
  }
}

#define TOKEN(token_id) TokenString(tokenizer, token_id).c_str()

void LogTopK(const GemmaTokenizer* tokenizer, float* logits, float* dist,
             size_t len, size_t k) {
  std::vector<std::pair<float, int>> sorted(len);
  for (size_t i = 0; i < len; ++i) {
    sorted[i] = std::make_pair(dist[i], static_cast<int>(i));
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
  for (size_t i = 0; i < k; ++i) {
    printf("  [#%-2d token %6d = %-12s  %.2e  %f]\n", static_cast<int>(i + 1),
           sorted[i].second, TOKEN(sorted[i].second), sorted[i].first,
           logits[sorted[i].second]);
  }
}

template <class TConfig>
float ComputeCrossEntropyImpl(const WeightStorageT& weights_u8,
                              Activations<TConfig, 1>& activations,
                              const GemmaTokenizer* tokenizer,
                              size_t max_tokens,
                              const std::vector<int>& prompt, KVCache& kv_cache,
                              hwy::ThreadPool& pool, int verbosity) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  const WeightsT<TConfig>& weights =
      *reinterpret_cast<const WeightsT<TConfig>*>(weights_u8.get());
  std::vector<float> logits(kVocabSize);
  Softmax(activations.logits.data(), kVocabSize);
  float total_entropy = 0.0f;
  for (size_t pos = 0; pos < max_tokens && pos < prompt.size(); ++pos) {
    if (verbosity >= 4 && tokenizer) {
      LogTopK(tokenizer, logits.data(), activations.logits.data(),
              kVocabSize, 10);
    }
    const int token = prompt[pos];
    const float prob = activations.logits[token];
    if (verbosity >= 3) {
      printf("pos %4zu token %6d = %-12s  %.10e  %14.10f bits\n", pos, token,
             TOKEN(token), prob, -std::log(prob) / std::log(2.0));
    }
    if (pos > 0) {
      total_entropy -= std::max(std::log(prob), -64.0f);
    }
    if (verbosity >= 2 && pos % 100 == 99) {
      printf("Processed %zu tokens, cross-entropy per token: %f\n", pos + 1,
             total_entropy / std::log(2.0) / (pos + 1));
    }
    Transformer(token, pos, weights, activations, kv_cache, pool,
                /*layers_output=*/nullptr);
    MatVec<kVocabSize, kModelDim>(
        weights.embedder_input_embedding, 0, activations.x.data(),
        activations.even_odd.data(), activations.logits.data(), pool);
    LogitsSoftCap(30.0f, activations.logits.data(), kVocabSize);
    memcpy(logits.data(), activations.logits.data(),
           kVocabSize * sizeof(logits[0]));
    Softmax(activations.logits.data(), kVocabSize);
  }
  return total_entropy / std::log(2.0);
}

template <class TConfig>
float ComputeCrossEntropyImpl(GemmaImpl<TConfig>& gemma, size_t max_tokens,
                              const std::vector<int>& prompt, KVCache& kv_cache,
                              hwy::ThreadPool& pool, int verbosity) {
  return ComputeCrossEntropyImpl<TConfig>(gemma.weights_u8,
                                          *gemma.state.get(),
                                          gemma.Tokenizer(),
                                          max_tokens, prompt, kv_cache, pool,
                                          verbosity);
}

#undef TOKEN

void Generate2B(GemmaImpl<ConfigGemma2B>& gemma, size_t max_tokens,
                size_t max_generated_tokens, float temperature,
                const std::vector<int>& prompt, size_t start_pos,
                KVCache& kv_cache, hwy::ThreadPool& pool,
                const StreamFunc& stream_token, const AcceptFunc& accept_token,
                std::mt19937& gen, int verbosity,
                LayersOutputT* layers_output) {
  GenerateImpl(gemma, max_tokens, max_generated_tokens, temperature, prompt,
               start_pos, kv_cache, pool, stream_token, accept_token, gen,
               verbosity, layers_output);
}

void Generate7B(GemmaImpl<ConfigGemma7B>& gemma, size_t max_tokens,
                size_t max_generated_tokens, float temperature,
                const std::vector<int>& prompt, size_t start_pos,
                KVCache& kv_cache, hwy::ThreadPool& pool,
                const StreamFunc& stream_token, const AcceptFunc& accept_token,
                std::mt19937& gen, int verbosity,
                LayersOutputT* layers_output) {
  GenerateImpl(gemma, max_tokens, max_generated_tokens, temperature, prompt,
               start_pos, kv_cache, pool, stream_token, accept_token, gen,
               verbosity, layers_output);
}

void GenerateGriffin2B(GemmaImpl<ConfigGriffin2B>& gemma, size_t max_tokens,
                       size_t max_generated_tokens, float temperature,
                       const std::vector<int>& prompt, size_t start_pos,
                       KVCache& kv_cache, hwy::ThreadPool& pool,
                       const StreamFunc& stream_token,
                       const AcceptFunc& accept_token, std::mt19937& gen,
                       int verbosity, LayersOutputT* layers_output) {
  GenerateImpl(gemma, max_tokens, max_generated_tokens, temperature, prompt,
               start_pos, kv_cache, pool, stream_token, accept_token, gen,
               verbosity, layers_output);
}

float ComputeCrossEntropy2B(GemmaImpl<ConfigGemma2B>& gemma, size_t max_tokens,
                            const std::vector<int>& prompt, KVCache& kv_cache,
                            hwy::ThreadPool& pool, int verbosity) {
  return ComputeCrossEntropyImpl(gemma, max_tokens, prompt, kv_cache, pool,
                                 verbosity);
}

float ComputeCrossEntropy7B(GemmaImpl<ConfigGemma7B>& gemma, size_t max_tokens,
                            const std::vector<int>& prompt, KVCache& kv_cache,
                            hwy::ThreadPool& pool, int verbosity) {
  return ComputeCrossEntropyImpl(gemma, max_tokens, prompt, kv_cache, pool,
                                 verbosity);
}

float ComputeCrossEntropyGriffin2B(GemmaImpl<ConfigGriffin2B>& gemma,
                                   size_t max_tokens,
                                   const std::vector<int>& prompt,
                                   KVCache& kv_cache, hwy::ThreadPool& pool,
                                   int verbosity) {
  return ComputeCrossEntropyImpl(gemma, max_tokens, prompt, kv_cache, pool,
                                 verbosity);
}

// Calls func(name, float*, CompressedArray*) for each tensor. float* is null
// if weights = null, and CompressedArray* is null if c_weights is null.
//
// This avoids repeating the list of tensors between loading and compressing.
template <class TConfig, class Func>
void ForEachTensor(Weights<TConfig>* weights,
                   CompressedWeights<TConfig>* c_weights, Func& func) {
  func("c_embedding",
       weights ? weights->embedder_input_embedding.data() : nullptr,
       c_weights ? &c_weights->embedder_input_embedding : nullptr );
  func("c_final_norm", weights ? weights->final_norm_scale.data() : nullptr,
       c_weights ? &c_weights->final_norm_scale : nullptr);

  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    Layer<TConfig>* layer = weights ? weights->GetLayer(idx) : nullptr;
    CompressedLayer<TConfig>* layer_weights =
        c_weights ? c_weights->GetLayer(idx) : nullptr;

#define CALL_FUNC(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx); \
  func(name_buf, layer ? layer->member.data() : nullptr,       \
       layer_weights ? &layer_weights->member : nullptr)

    CALL_FUNC("pre_ff_ns", pre_ffw_norm_scale);
    CALL_FUNC("gating_ein", gating_einsum_w);
    CALL_FUNC("linear_w", linear_w);
    if (type == LayerAttentionType::kGemma) {
      CALL_FUNC("qkv_ein", qkv_einsum_w);
      CALL_FUNC("att_ein", attn_vec_einsum_w);
    } else {
      CALL_FUNC("gr_lin_x_w", griffin.linear_x_w);
      CALL_FUNC("gr_lin_x_b", griffin.linear_x_biases);
      CALL_FUNC("gr_lin_y_w", griffin.linear_y_w);
      CALL_FUNC("gr_lin_y_b", griffin.linear_y_biases);
      CALL_FUNC("gr_lin_out_w", griffin.linear_out_w);
      CALL_FUNC("gr_lin_out_b", griffin.linear_out_biases);
      CALL_FUNC("gr_conv_w", griffin.conv_w);
      CALL_FUNC("gr_conv_b", griffin.conv_biases);
      CALL_FUNC("gr_gate_w", griffin.gate_w);
      CALL_FUNC("gr_gate_b", griffin.gate_biases);
      CALL_FUNC("gr_a", griffin.a);
    }
    CALL_FUNC("pre_att_ns", pre_attention_norm_scale);

    if (TConfig::kFFBiases) {
      CALL_FUNC("ffw_gat_b", ffw_gating_biases);
      CALL_FUNC("ffw_out_b", ffw_output_biases);
    }

    if (TConfig::kSoftmaxAttnOutputBiases &&
        type == LayerAttentionType::kGemma) {
      CALL_FUNC("attn_ob", attention_output_biases);
    }
#undef CALL_FUNC
  }
}

template <class TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadCompressedWeights(
    const Path& weights, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.LoadCompressedWeights");
  if (!weights.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights.path.c_str());
  }

  // Allocate compressed weights.
  using CWeights = CompressedWeights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> c_weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(CWeights));
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

  std::array<float, TConfig::kNumTensorScales> scales;
  CacheLoader loader(weights);
  ForEachTensor<TConfig>(nullptr, c_weights, loader);
  loader.LoadScales(scales.data(), scales.size());
  if (!loader.ReadAll(pool)) {
    HWY_ABORT("Failed to load model weights.");
  }
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    CompressedLayer<TConfig>* layer_weights = c_weights->GetLayer(idx);
    if (type == LayerAttentionType::kGemma) {
      static constexpr size_t kHeads = TConfig::kHeads;
      static constexpr size_t kModelDim = TConfig::kModelDim;
      static constexpr size_t kQKVDim = TConfig::kQKVDim;
      std::array<SfpStream, kHeads * kQKVDim * kModelDim> tmp;
      SfpStream* attn_vec_einsum_w = layer_weights->attn_vec_einsum_w.data();
      for (size_t i = 0; i < kModelDim; ++i) {
        for (size_t h = 0; h < kHeads; ++h) {
          memcpy(&tmp[i * kHeads * kQKVDim + h * kQKVDim],
                 &attn_vec_einsum_w[h * kQKVDim * kModelDim + i * kQKVDim],
                 kQKVDim * sizeof(tmp[0]));
        }
      }
      memcpy(attn_vec_einsum_w, tmp.data(), tmp.size() * sizeof(tmp[0]));
    }
  }
  if (TConfig::kNumTensorScales > 0) {
    size_t scale_pos = 0;
    for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
      auto type = TConfig::kLayerConfig[layer_idx];
      const size_t idx = static_cast<size_t>(layer_idx);
      CompressedLayer<TConfig>* layer_weights = c_weights->GetLayer(idx);
      if (type == LayerAttentionType::kGemma) {
        layer_weights->attn_vec_einsum_w.set_scale(scales[scale_pos++]);
        layer_weights->qkv_einsum_w.set_scale(scales[scale_pos++]);
      } else {
        layer_weights->griffin.linear_x_w.set_scale(scales[scale_pos++]);
        layer_weights->griffin.linear_y_w.set_scale(scales[scale_pos++]);
        layer_weights->griffin.linear_out_w.set_scale(scales[scale_pos++]);
        layer_weights->griffin.gate_w.set_scale(scales[scale_pos++]);
      }
      layer_weights->gating_einsum_w.set_scale(scales[scale_pos++]);
      layer_weights->linear_w.set_scale(scales[scale_pos++]);
    }
    HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
  }
  return c_weights_u8;
}

// Type-erased because this function is called via a function pointer.
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadCompressedWeightsT(
    gcpp::Model model, const Path& weights, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return LoadCompressedWeights<ConfigGemma2B>(weights, pool);
    case Model::GEMMA_7B:
      return LoadCompressedWeights<ConfigGemma7B>(weights, pool);
    case Model::GRIFFIN_2B:
      return LoadCompressedWeights<ConfigGriffin2B>(weights, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

hwy::AlignedFreeUniquePtr<uint8_t[]> LoadWeightsT(gcpp::Model model,
                                                  const Path& weights,
                                                  hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return LoadWeights<ConfigGemma2B>(weights, pool);
    case Model::GEMMA_7B:
      return LoadWeights<ConfigGemma7B>(weights, pool);
    case Model::GRIFFIN_2B:
      return LoadWeights<ConfigGriffin2B>(weights, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

WeightStorageT AllocateWeightsT(gcpp::Model model, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return AllocateWeights<ConfigGemma2B>(pool);
    case Model::GEMMA_7B:
      return AllocateWeights<ConfigGemma7B>(pool);
    case Model::GRIFFIN_2B:
      return AllocateWeights<ConfigGriffin2B>(pool);
    case Model::GEMMA_TINY:
      return AllocateWeights<ConfigGemmaTiny>(pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

WeightStorageT AllocateForwardPassT(gcpp::Model model) {
  switch (model) {
    case Model::GEMMA_2B:
      return AllocateForwardPass<ConfigGemma2B>();
    case Model::GEMMA_TINY:
      return AllocateForwardPass<ConfigGemmaTiny>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

WeightStorageT AllocateBackwardPassT(gcpp::Model model) {
  switch (model) {
    case Model::GEMMA_2B:
      return AllocateBackwardPass<ConfigGemma2B>();
    case Model::GEMMA_TINY:
      return AllocateBackwardPass<ConfigGemmaTiny>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

template <class TConfig>
void CompressWeights(const Path& weights_path,
                     const Path& compressed_weights_path,
                     hwy::ThreadPool& pool) {
  if (!weights_path.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights_path.path.c_str());
  }

  // Allocate compressed weights.
  using CWeights = CompressedWeights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> c_weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(CWeights));
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

  // Get weights, compress, and store.
  const bool scale_for_compression = TConfig::kNumTensorScales > 0;
  const hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8 =
      LoadWeights<TConfig>(weights_path, pool, scale_for_compression);
  Weights<TConfig>* weights =
      reinterpret_cast<Weights<TConfig>*>(weights_u8.get());
  Compressor compressor(pool);
  ForEachTensor<TConfig>(weights, c_weights, compressor);
  compressor.AddScales(weights->scales.data(), weights->scales.size());
  compressor.WriteAll(pool, compressed_weights_path);

  weights->layer_ptrs.~LayerPointers<TConfig>();
  c_weights->c_layer_ptrs.~CompressedLayerPointers<TConfig>();
}

void CompressWeightsT(gcpp::Model model, const Path& weights,
                      const Path& compressed_weights, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      CompressWeights<ConfigGemma2B>(weights, compressed_weights, pool);
      break;
    case Model::GEMMA_7B:
      CompressWeights<ConfigGemma7B>(weights, compressed_weights, pool);
      break;
    case Model::GRIFFIN_2B:
      CompressWeights<ConfigGriffin2B>(weights, compressed_weights, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

void LogVec(const char* name, const float* data, size_t len) {
  float minval = std::numeric_limits<float>::max();
  float maxval = std::numeric_limits<float>::min();
  double sum = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    minval = std::min(minval, data[i]);
    maxval = std::max(maxval, data[i]);
    sum += data[i];
  }
  float avg = sum / len;
  printf("%-20s  %12zu   %13.10f   %8.5f   %13.10f\n",
         name, len, minval, avg, maxval);
}

class WeightLogger {
 public:
  template <typename MatT, size_t kCapacity>
  void operator()(const char* name, const float* data,
                  CompressedArray<MatT, kCapacity>* compressed) {
    LogVec(name, data, kCapacity);
    total_weights += kCapacity;
  }
  size_t total_weights = 0;
};

template <typename TConfig>
void LogWeightStats(const WeightStorageT& weights_u8) {
  auto* weights = reinterpret_cast<Weights<TConfig>*>(weights_u8.get());
  WeightLogger logger;
  ForEachTensor<TConfig>(weights, nullptr, logger);
  printf("%-20s  %12zu\n", "Total", logger.total_weights);
}

void LogWeightStatsT(gcpp::Model model, const WeightStorageT& weights) {
  switch (model) {
    case Model::GEMMA_2B:
      return LogWeightStats<ConfigGemma2B>(weights);
    case Model::GEMMA_7B:
      return LogWeightStats<ConfigGemma7B>(weights);
    case Model::GRIFFIN_2B:
      return LogWeightStats<ConfigGriffin2B>(weights);
    case Model::GEMMA_TINY:
      return LogWeightStats<ConfigGemmaTiny>(weights);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

class WeightInitializer {
 public:
  WeightInitializer(InitMode mode, std::mt19937* gen)
      : mode_(mode), dist_(0.0f, 1.0f), gen_(gen) {}

  template <typename MatT, size_t kCapacity>
  void operator()(const char* name, float* data,
                  CompressedArray<MatT, kCapacity>* compressed) {
    if (mode_ == InitMode::RAND_INIT) {
      for (size_t i = 0; i < kCapacity; ++i) {
        data[i] = dist_(*gen_);
      }
    } else if (mode_ == InitMode::ZERO_INIT) {
      memset(data, 0, kCapacity * sizeof(data[0]));
    }
  }
 private:
  InitMode mode_;
  std::normal_distribution<float> dist_;
  std::mt19937* gen_;
};

template <typename TConfig>
void InitWeights(InitMode mode, WeightStorageT& weights_u8,
                 std::mt19937* gen) {
  auto* weights = reinterpret_cast<Weights<TConfig>*>(weights_u8.get());
  // TODO(szabadka) Use the same weight initialization method as in the python
  // version.
  // TODO(szabadka) Implement multi-threaded initialization.
  WeightInitializer init(mode, gen);
  ForEachTensor<TConfig>(weights, nullptr, init);
}

void InitWeightsT(gcpp::Model model, WeightStorageT& weights,
                  InitMode mode, std::mt19937* gen) {
  switch (model) {
    case Model::GEMMA_2B:
      InitWeights<ConfigGemma2B>(mode, weights, gen);
      break;
    case Model::GEMMA_7B:
      InitWeights<ConfigGemma7B>(mode, weights, gen);
      break;
    case Model::GRIFFIN_2B:
      InitWeights<ConfigGriffin2B>(mode, weights, gen);
      break;
    case Model::GEMMA_TINY:
      InitWeights<ConfigGemmaTiny>(mode, weights, gen);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

template<size_t kCapacity>
void UpdateTensor(const std::array<float, kCapacity>& grad, float scale,
                  std::array<float, kCapacity>& weights) {
  // TODO(szabadka) SIMDify this.
  for (size_t i = 0; i < kCapacity; ++i) {
    weights[i] += scale * grad[i];
  }
}

template <typename TConfig>
void UpdateWeights(const WeightStorageT& grad_u8, float scale,
                   WeightStorageT& weights_u8, hwy::ThreadPool& pool) {
  const auto& grad = *reinterpret_cast<const Weights<TConfig>*>(grad_u8.get());
  auto* weights = reinterpret_cast<Weights<TConfig>*>(weights_u8.get());

  UpdateTensor(grad.embedder_input_embedding, scale,
               weights->embedder_input_embedding);
  UpdateTensor(grad.final_norm_scale, scale, weights->final_norm_scale);

  pool.Run(0, TConfig::kLayers, [&](uint64_t idx, size_t /*thread*/) {
    const Layer<TConfig>* gl = grad.GetLayer(idx);
    Layer<TConfig>* wl = weights->GetLayer(idx);

#define UPDATE_FUNC(member) UpdateTensor(gl->member, scale, wl->member);
    // TODO(szabadka) Implement it for Griffin as well.
    UPDATE_FUNC(pre_ffw_norm_scale);
    UPDATE_FUNC(gating_einsum_w);
    UPDATE_FUNC(linear_w);
    UPDATE_FUNC(qkv_einsum_w);
    UPDATE_FUNC(attn_vec_einsum_w);
    UPDATE_FUNC(pre_attention_norm_scale);
#undef UPDATE_FUNC
  });
}

void UpdateWeightsT(Model model, const WeightStorageT& grad, float scale,
                   WeightStorageT& weights, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      UpdateWeights<ConfigGemma2B>(grad, scale, weights, pool);
      break;
    case Model::GEMMA_7B:
      UpdateWeights<ConfigGemma7B>(grad, scale, weights, pool);
      break;
    case Model::GRIFFIN_2B:
      UpdateWeights<ConfigGriffin2B>(grad, scale, weights, pool);
      break;
    case Model::GEMMA_TINY:
      UpdateWeights<ConfigGemmaTiny>(grad, scale, weights, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

template <typename ArrayT>
void InputEmbedding(const ArrayT& weights, const std::vector<int>& prompt,
                    const float scaling, float* HWY_RESTRICT output,
                    size_t model_dim) {
  for (size_t pos = 0; pos + 1 < prompt.size(); ++pos) {
    int token = prompt[pos];
    Decompress(weights, token * model_dim, output + pos * model_dim, model_dim);
    MulByConst(scaling, output + pos * model_dim, model_dim);
  }
}

template<typename WT, typename XT, typename OutT>
void ApplyRMSNorm(const WT* HWY_RESTRICT weights, const XT* HWY_RESTRICT x,
                  size_t model_dim, size_t num_tokens,
                  OutT* HWY_RESTRICT output,
                  hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * model_dim;
    RMSNorm(x + offset, weights, output + offset, model_dim);
  }
}

void RMSNormVJP(const float* HWY_RESTRICT weights, const float* HWY_RESTRICT x,
                const float* HWY_RESTRICT v, size_t model_dim,
                size_t num_tokens, float* HWY_RESTRICT grad_w,
                float* HWY_RESTRICT grad_x, hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * model_dim;
    constexpr float eps = 1e-6f;
    float ss = SquaredL2(x + offset, model_dim);
    ss = 1.0f / sqrtf(ss / StaticCast<float>(model_dim) + eps);
    for (size_t i = 0; i < model_dim; ++i) {
      grad_w[i] += v[offset + i] * x[offset + i] * ss;
    }
    const float ss3 = ss * ss * ss / StaticCast<float>(model_dim);
    float tmp = 0.0f;
    for (size_t i = 0; i < model_dim; ++i) {
      tmp += (1.0f + weights[i]) * v[offset + i] * x[offset + i];
    }
    tmp *= ss3;
    for (size_t i = 0; i < model_dim; ++i) {
      grad_x[offset + i] = ss * (1.0f + weights[i]) * v[offset + i] -
                           tmp * x[offset + i];
    }
  }
}

template <size_t kCols, size_t kRows>
void MatMulVJP(const std::array<float, kRows * kCols>& weights,
               const float* HWY_RESTRICT x,  // num_tokens * kCols
               const float* HWY_RESTRICT v,  // num_tokens * kRows
               size_t num_tokens, float* HWY_RESTRICT even_odd,
               std::array<float, kRows * kCols>& grad_w,
               float* HWY_RESTRICT grad_x,  // num_tokens * kCols
               hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t voffs = pos * kRows;
    const size_t xoffs = pos * kCols;
    for (size_t j = 0; j < kRows; ++j) {
      MulByConstAndAdd(v[voffs + j], &x[xoffs], &grad_w[j * kCols], kCols);
    }
    // &grad_x[xoffs] = &v[voffs] * weights (row vec * matrix)
    memset(&grad_x[xoffs], 0, kCols * sizeof(grad_x[0]));
    for (size_t j = 0; j < kRows; ++j) {
      MulByConstAndAdd(v[voffs + j], &weights[j * kCols], &grad_x[xoffs],
                       kCols);
    }
  }
}

template <typename TConfig>
void ApplyForwardLayer(const Layer<TConfig>& weights,
                       ForwardLayer<TConfig>& activations,
                       size_t num_tokens,
                       float* HWY_RESTRICT even_odd,
                       float* HWY_RESTRICT output,
                       hwy::ThreadPool& pool) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static const float kQueryScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));

#if 0
  ApplyRMSNorm(weights.pre_attention_norm_scale.data(),
               activations.input.data(), kModelDim, num_tokens,
               activations.pre_att_rms_out.data(), pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    float* x = activations.pre_att_rms_out.data() + pos * kModelDim;
    float* HWY_RESTRICT q = activations.q.data() + pos * kHeads * kQKVDim;
    MatVec<kHeads * kQKVDim, kModelDim>(
        weights.qkv_einsum_w, 0, x, even_odd, q, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    float* x = activations.pre_att_rms_out.data() + pos * kModelDim;
    float* HWY_RESTRICT kv = activations.kv.data() + pos * kHeads * kQKVDim * 2;
    MatVec<kQKVDim * 2, kModelDim>(weights.qkv_einsum_w,
                                   kHeads * kQKVDim * kModelDim, x,
                                   even_odd, kv, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    float* HWY_RESTRICT kv = activations.kv.data() + pos * kHeads * kQKVDim * 2;
    Rope(kv, kQKVDim, pos);
  }

  const size_t num_tasks = kHeads * num_tokens;
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    float* HWY_RESTRICT q =
        activations.q.data() + (pos * kHeads + head) * kQKVDim;
    Rope(q, kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);
  });
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    float* HWY_RESTRICT q =
        activations.q.data() + (pos * kHeads + head) * kQKVDim;
    // Calculate scores
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * kSeqLen +
                                   pos * kHeads * kSeqLen;
    // Compute Q dot K scores
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      float* HWY_RESTRICT k2 =
          activations.kv.data() + pos2 * kHeads * kQKVDim * 2;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2 % kSeqLen] = score;
    }
  });
#endif
  const size_t num_tasks = kHeads * num_tokens;
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    // Calculate scores
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * kSeqLen +
                                   pos * kHeads * kSeqLen;
    Softmax(head_att, std::min(pos + 1, kSeqLen));
  });
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    // Weighted summation
    const float* HWY_RESTRICT head_att = activations.att.data() +
                                         head * kSeqLen +
                                         pos * kHeads * kSeqLen;
    float* HWY_RESTRICT att_out = activations.att_out.data() + head * kQKVDim +
                                  pos * kHeads * kQKVDim;

    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      float* HWY_RESTRICT v2 =
          activations.kv.data() + pos2 * kHeads * kQKVDim * 2 + kQKVDim;
      MulByConstAndAdd(head_att[pos2 % kSeqLen], v2, att_out, kQKVDim);
    }
  });
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kModelDim, kHeads * kQKVDim>(
        weights.attn_vec_einsum_w, 0,
        activations.att_out.data() + pos * kHeads * kQKVDim, even_odd,
        //activations.att_post2.data() + pos * kModelDim, pool);
        activations.attention_out.data() + pos * kModelDim, pool);
  }
#if 0
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    Add(activations.input.data() + pos * kModelDim,
        activations.att_post2.data() + pos * kModelDim,
        activations.attention_out.data() + pos * kModelDim, kModelDim);
  }
#endif
  ApplyRMSNorm(weights.pre_ffw_norm_scale.data(),
               activations.attention_out.data(), kModelDim, num_tokens,
               activations.bf_pre_ffw_rms_out.data(), pool);
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kFFHiddenDim * 2, kModelDim>(
        weights.gating_einsum_w, 0,
        activations.bf_pre_ffw_rms_out.data() + pos * kModelDim, even_odd,
        activations.ffw_hidden.data() + pos * kFFHiddenDim * 2, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t hidden_offset = pos * kFFHiddenDim * 2;
    const float* HWY_RESTRICT out =
        activations.ffw_hidden.data() + hidden_offset;
    const float* HWY_RESTRICT out_mul = out + kFFHiddenDim;
    float* HWY_RESTRICT out_gated =
        activations.ffw_hidden_gated.data() + pos * kFFHiddenDim;
    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    DF df;
    for (size_t i = 0; i < kFFHiddenDim; i += Lanes(df)) {
      const auto y = Load(df, out + i);
      const auto x = Load(df, out_mul + i);
      hn::Store(hn::Mul(x, Gelu(df, y)), df, out_gated + i);
    }
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kModelDim, kFFHiddenDim>(
        weights.linear_w, 0,
        activations.ffw_hidden_gated.data() + pos * kFFHiddenDim,
        even_odd, activations.ffw_out.data() + pos * kModelDim, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    Add(activations.attention_out.data() + pos * kModelDim,
        activations.ffw_out.data() + pos * kModelDim,
        output + pos * kModelDim, kModelDim);
  }
}

template <typename TConfig>
void LayerVJP(const Layer<TConfig>& weights,
              const ForwardLayer<TConfig>& forward,
              const float* HWY_RESTRICT next_layer_grad,
              size_t num_tokens,
              float* HWY_RESTRICT even_odd,
              Layer<TConfig>& grad,
              ForwardLayer<TConfig>& backward,
              hwy::ThreadPool& pool) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  MatMulVJP<kFFHiddenDim, kModelDim>(
      weights.linear_w, forward.ffw_hidden_gated.data(), next_layer_grad,
      num_tokens, even_odd, grad.linear_w, backward.ffw_hidden_gated.data(),
      pool);
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t hidden_offset = pos * kFFHiddenDim * 2;
    const float* HWY_RESTRICT f_out = forward.ffw_hidden.data() + hidden_offset;
    const float* HWY_RESTRICT f_out_mul = f_out + kFFHiddenDim;
    const float* HWY_RESTRICT b_out_gated =
        backward.ffw_hidden_gated.data() + pos * kFFHiddenDim;
    float* HWY_RESTRICT b_out = backward.ffw_hidden.data() + hidden_offset;
    float* HWY_RESTRICT b_out_mul = b_out + kFFHiddenDim;
    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    DF df;
    for (size_t i = 0; i < kFFHiddenDim; i += Lanes(df)) {
      const auto y = Load(df, f_out + i);
      const auto x = Load(df, f_out_mul + i);
      const auto v = Load(df, b_out_gated + i);
      hn::Store(hn::Mul(v, Gelu(df, y)), df, b_out_mul + i);
      hn::Store(hn::Mul(v, hn::Mul(x, GeluGrad(df, y))), df, b_out + i);
    }
  }
  MatMulVJP<kModelDim, kFFHiddenDim * 2>(
      weights.gating_einsum_w,
      forward.bf_pre_ffw_rms_out.data(), backward.ffw_hidden.data(),
      num_tokens, even_odd, grad.gating_einsum_w,
      backward.bf_pre_ffw_rms_out.data(), pool);
  RMSNormVJP(weights.pre_ffw_norm_scale.data(),
             forward.attention_out.data(),
             backward.bf_pre_ffw_rms_out.data(),
             kModelDim, num_tokens,
             grad.pre_ffw_norm_scale.data(),
             backward.attention_out.data(), pool);
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(next_layer_grad, backward.attention_out.data(), kModelDim);
  }
  MatMulVJP<kHeads * kQKVDim, kModelDim>(
        weights.attn_vec_einsum_w, forward.att_out.data(),
        backward.attention_out.data(), num_tokens, even_odd,
        grad.attn_vec_einsum_w, backward.att_out.data(), pool);
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    hwy::ZeroBytes(backward.kv.data() + pos * kHeads * kQKVDim * 2 + kQKVDim,
                   kQKVDim * sizeof(backward.kv[0]));
  }
  const size_t num_tasks = kHeads * num_tokens;
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const float* HWY_RESTRICT f_head_att = forward.att.data() +
                                             head * kSeqLen +
                                             pos * kHeads * kSeqLen;
      const float* HWY_RESTRICT b_att_out = backward.att_out.data() +
                                            head * kQKVDim +
                                            pos * kHeads * kQKVDim;
      float* HWY_RESTRICT b_head_att = backward.att.data() +
                                       head * kSeqLen +
                                       pos * kHeads * kSeqLen;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const float* HWY_RESTRICT f_v2 =
            forward.kv.data() + pos2 * kHeads * kQKVDim * 2 + kQKVDim;
        b_head_att[pos2] = Dot(b_att_out, f_v2, kQKVDim);
      }
      for (size_t pos2 = pos; pos2 < num_tokens; ++pos2) {
        float* HWY_RESTRICT b_v2 =
            backward.kv.data() + pos2 * kHeads * kQKVDim * 2 + kQKVDim;
        MulByConstAndAdd(f_head_att[pos2], b_att_out, b_v2, kQKVDim);
      }
    }
  }
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    const float* HWY_RESTRICT f_head_att = forward.att.data() +
                                           head * kSeqLen +
                                           pos * kHeads * kSeqLen;
    float* HWY_RESTRICT b_head_att = backward.att.data() +
                                     head * kSeqLen +
                                     pos * kHeads * kSeqLen;
    SoftmaxVJP(f_head_att, b_head_att, std::min(pos + 1, kSeqLen));
  });
}

template <size_t kModelDim, size_t kVocabSize, typename ArrayT>
void ComputeLogits(const ArrayT& weights,  // kVocabSize * kModelDim
                   const float* HWY_RESTRICT x,  // num_tokens * kModelDim
                   size_t num_tokens, float* HWY_RESTRICT even_odd,
                   float* HWY_RESTRICT output,  // num_tokens * kVocabSize
                   hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kVocabSize, kModelDim>(
        weights, 0, x + pos * kModelDim, even_odd, output + pos * kVocabSize,
        pool);
  }
}

void ApplySoftcap(const float* HWY_RESTRICT x, float* HWY_RESTRICT output,
                  size_t num_tokens, size_t vocab_size, hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    LogitsSoftCap(30.0f, x + pos * vocab_size, output + pos * vocab_size,
                  vocab_size);
  }
}

void SoftcapVJP(const float* HWY_RESTRICT output, const float* HWY_RESTRICT v,
                size_t num_tokens, size_t vocab_size,
                float* HWY_RESTRICT grad, hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * vocab_size;
    SoftCapGrad(30.0f, output + offset, grad + offset, vocab_size);
    MulBy(v + offset, grad + offset, vocab_size);
  }
}

template<size_t kVocabSize>
float CrossEntropyLoss(const float* HWY_RESTRICT x,
                       const std::vector<int>& prompt,
                       size_t context_size,
                       hwy::ThreadPool& pool) {
  float loss = 0.0f;
  for (size_t pos = 0; pos + 1 < prompt.size(); ++pos) {
    if (pos + 1 < context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = prompt[pos + 1];
    loss += SoftmaxCrossEntropy(x + pos * kVocabSize, kVocabSize, next_token);
  }
  return loss;
}

template<size_t kVocabSize>
void LossGradient(const float* HWY_RESTRICT x, const std::vector<int>& prompt,
                  size_t context_size, float* HWY_RESTRICT grad,
                  hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos + 1 < prompt.size(); ++pos) {
    if (pos + 1 < context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = prompt[pos + 1];
    // TODO(szabadka) This requires that kVocabSize is a multiple of the
    // SIMD lane count.
    Softmax(x + pos * kVocabSize, kVocabSize, kVocabSize,
            grad + pos * kVocabSize);
    grad[pos * kVocabSize + next_token] -= 1.0f;
    MulByConst(1.0 / std::log(2.0), grad + pos * kVocabSize, kVocabSize);
  }
}

template <typename TConfig>
float CrossEntropyLossWithGradUpdate(const std::vector<int>& prompt,
                                     size_t context_size,
                                     const WeightStorageT& weights_u8,
                                     WeightStorageT& forward_u8,
                                     WeightStorageT& grad_u8,
                                     WeightStorageT& backward_u8,
                                     hwy::ThreadPool& pool) {
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kLayers = TConfig::kLayers;
  const float kEmbScaling = EmbeddingScaling<TConfig>();

  using TWeights = Weights<TConfig>;
  const auto& weights = *reinterpret_cast<const TWeights*>(weights_u8.get());
  auto& grad = *reinterpret_cast<TWeights*>(grad_u8.get());

  HWY_DASSERT(context_size > 0);
  HWY_DASSERT(context_size < prompt.size());
  const size_t num_tokens = prompt.size() - 1;

  ForwardPass<TConfig>* forward =
      reinterpret_cast<ForwardPass<TConfig>*>(forward_u8.get());
  ForwardPass<TConfig>* backward =
      reinterpret_cast<ForwardPass<TConfig>*>(backward_u8.get());

#if 0
  InputEmbedding(weights.embedder_input_embedding, prompt, kEmbScaling,
                 forward->layers[0].input.data(), kModelDim);
#endif

  for (size_t layer = 0; layer < kLayers; ++layer) {
    float* HWY_RESTRICT output = layer + 1 < kLayers ?
                                 forward->layers[layer + 1].input.data() :
                                 forward->final_layer_output.data();
    ApplyForwardLayer(*weights.GetLayer(layer), forward->layers[layer],
                      num_tokens, forward->even_odd.data(), output, pool);
  }

  ApplyRMSNorm(weights.final_norm_scale.data(),
               forward->final_layer_output.data(),
               kModelDim, num_tokens, forward->final_norm_output.data(), pool);

  ComputeLogits<kModelDim, kVocabSize>(
      weights.embedder_input_embedding, forward->final_norm_output.data(),
      num_tokens, forward->even_odd.data(), forward->raw_logits.data(), pool);

  ApplySoftcap(forward->raw_logits.data(), forward->logits.data(),
               num_tokens, kVocabSize, pool);

  float loss = CrossEntropyLoss<kVocabSize>(forward->logits.data(), prompt,
                                            context_size, pool);

  LossGradient<kVocabSize>(forward->logits.data(), prompt, context_size,
                           backward->logits.data(), pool);

  SoftcapVJP(forward->logits.data(), backward->logits.data(), num_tokens,
             kVocabSize, backward->raw_logits.data(), pool);

  MatMulVJP<kModelDim, kVocabSize>(
      weights.embedder_input_embedding, forward->final_norm_output.data(),
      backward->raw_logits.data(), num_tokens, forward->even_odd.data(),
      grad.embedder_input_embedding, backward->final_norm_output.data(),
      pool);

  RMSNormVJP(weights.final_norm_scale.data(),
             forward->final_layer_output.data(),
             backward->final_norm_output.data(),
             kModelDim, num_tokens,
             grad.final_norm_scale.data(),
             backward->final_layer_output.data(), pool);

  for (int layer = TConfig::kLayers - 1; layer >= 0; --layer) {
    float* HWY_RESTRICT next_layer_grad =
        layer + 1 < kLayers ? backward->layers[layer + 1].input.data()
                            : backward->final_layer_output.data();
    LayerVJP(*weights.GetLayer(layer), forward->layers[layer], next_layer_grad,
             num_tokens, forward->even_odd.data(),
             *grad.GetLayer(layer), backward->layers[layer], pool);
  }

#if 0
  ImputEmbeddingVJP(weights, prompt, next_layer_grad.data(), grad);
#endif

  return loss;
}

float CrossEntropyLossWithGradUpdateT(const std::vector<int>& prompt,
                                      size_t context_size,
                                      Model model,
                                      const WeightStorageT& weights,
                                      WeightStorageT& forward,
                                      WeightStorageT& grad,
                                      WeightStorageT& backward,
                                      hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return CrossEntropyLossWithGradUpdate<ConfigGemma2B>(
          prompt, context_size, weights, forward, grad, backward, pool);
    case Model::GEMMA_TINY:
      return CrossEntropyLossWithGradUpdate<ConfigGemmaTiny>(
          prompt, context_size, weights, forward, grad, backward, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(AllocateWeightsT);
HWY_EXPORT(AllocateForwardPassT);
HWY_EXPORT(AllocateBackwardPassT);
HWY_EXPORT(LogWeightStatsT);
HWY_EXPORT(InitWeightsT);
HWY_EXPORT(UpdateWeightsT);
HWY_EXPORT(CrossEntropyLossWithGradUpdateT);
HWY_EXPORT(LoadCompressedWeightsT);
HWY_EXPORT(LoadWeightsT);
HWY_EXPORT(CompressWeightsT);
HWY_EXPORT(Generate2B);
HWY_EXPORT(Generate7B);
HWY_EXPORT(GenerateGriffin2B);
HWY_EXPORT(ComputeCrossEntropy2B);
HWY_EXPORT(ComputeCrossEntropy7B);
HWY_EXPORT(ComputeCrossEntropyGriffin2B);

KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len,
                      size_t conv1d_cache_size, size_t rglru_cache_size) {
  KVCache kv_cache = {};
  if (size_cache_pos != 0) {
    kv_cache.kv_cache =
        hwy::AllocateAligned<float>(seq_len * size_cache_pos * 2);
  }
  if (conv1d_cache_size != 0) {
    kv_cache.conv1d_cache = hwy::AllocateAligned<float>(conv1d_cache_size);
    hwy::ZeroBytes(kv_cache.conv1d_cache.get(),
                   conv1d_cache_size * sizeof(kv_cache.conv1d_cache[0]));
  }
  if (rglru_cache_size != 0) {
    kv_cache.rglru_cache = hwy::AllocateAligned<float>(rglru_cache_size);
    hwy::ZeroBytes(kv_cache.rglru_cache.get(),
                   rglru_cache_size * sizeof(kv_cache.rglru_cache[0]));
  }
  return kv_cache;
}

template <class Config>
GemmaImpl<Config>::GemmaImpl(
    std::unique_ptr<sentencepiece::SentencePieceProcessor>& tokenizer,
    hwy::AlignedFreeUniquePtr<uint8_t[]>& weights_u8, hwy::ThreadPool& pool)
    : tokenizer(GemmaTokenizerImpl(std::move(tokenizer))),
      weights_u8(std::move(weights_u8)),
      prefill(hwy::MakeUniqueAligned<Activations<Config, kPrefillBatchSize>>()),
      state(hwy::MakeUniqueAligned<Activations<Config, 1>>()) {}

template <>
void GemmaImpl<ConfigGemma2B>::Generate(
    size_t max_tokens, size_t max_generated_tokens, float temperature,
    const std::vector<int>& prompt, size_t start_pos, KVCache& kv_cache,
    hwy::ThreadPool& pool, const StreamFunc& stream_token,
    const AcceptFunc& accept_token, std::mt19937& gen, int verbosity,
    LayersOutputT* layers_output) {
  HWY_DYNAMIC_DISPATCH(Generate2B)
  (*this, max_tokens, max_generated_tokens, temperature, prompt, start_pos,
   kv_cache, pool, stream_token, accept_token, gen, verbosity,
   layers_output);
}

template <>
void GemmaImpl<ConfigGemma7B>::Generate(
    size_t max_tokens, size_t max_generated_tokens, float temperature,
    const std::vector<int>& prompt, size_t start_pos, KVCache& kv_cache,
    hwy::ThreadPool& pool, const StreamFunc& stream_token,
    const AcceptFunc& accept_token, std::mt19937& gen, int verbosity,
    LayersOutputT* layers_output) {
  HWY_DYNAMIC_DISPATCH(Generate7B)
  (*this, max_tokens, max_generated_tokens, temperature, prompt, start_pos,
   kv_cache, pool, stream_token, accept_token, gen, verbosity, layers_output);
}

template <>
void GemmaImpl<ConfigGriffin2B>::Generate(
    size_t max_tokens, size_t max_generated_tokens, float temperature,
    const std::vector<int>& prompt, size_t start_pos, KVCache& kv_cache,
    hwy::ThreadPool& pool, const StreamFunc& stream_token,
    const AcceptFunc& accept_token, std::mt19937& gen, int verbosity,
    LayersOutputT* layers_output) {
  HWY_DYNAMIC_DISPATCH(GenerateGriffin2B)
  (*this, max_tokens, max_generated_tokens, temperature, prompt, start_pos,
   kv_cache, pool, stream_token, accept_token, gen, verbosity,
   layers_output);
}

template <>
float GemmaImpl<ConfigGemma2B>::ComputeCrossEntropy(
    size_t max_tokens, const std::vector<int>& prompt, KVCache& kv_cache,
    hwy::ThreadPool& pool, int verbosity) {
  return HWY_DYNAMIC_DISPATCH(ComputeCrossEntropy2B)(
      *this, max_tokens, prompt, kv_cache, pool, verbosity);
}

template <>
float GemmaImpl<ConfigGemma7B>::ComputeCrossEntropy(
    size_t max_tokens, const std::vector<int>& prompt, KVCache& kv_cache,
    hwy::ThreadPool& pool, int verbosity) {
  return HWY_DYNAMIC_DISPATCH(ComputeCrossEntropy7B)(
      *this, max_tokens, prompt, kv_cache, pool, verbosity);
}

template <>
float GemmaImpl<ConfigGriffin2B>::ComputeCrossEntropy(
    size_t max_tokens, const std::vector<int>& prompt, KVCache& kv_cache,
    hwy::ThreadPool& pool, int verbosity) {
  return HWY_DYNAMIC_DISPATCH(ComputeCrossEntropyGriffin2B)(
      *this, max_tokens, prompt, kv_cache, pool, verbosity);
}

Gemma::Gemma(const Path& tokenizer_path, const Path& weights, Model model_type,
             hwy::ThreadPool& pool) {
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer;
  {
    PROFILER_ZONE("Startup.tokenizer");
    tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!tokenizer->Load(tokenizer_path.path).ok()) {
      HWY_ABORT("Failed to load the tokenizer file.");
    }
  }

  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8;
  if constexpr (kWeightsAreCompressed) {
    weights_u8 =
        HWY_DYNAMIC_DISPATCH(LoadCompressedWeightsT)(model_type, weights, pool);
  } else {
    weights_u8 = HWY_DYNAMIC_DISPATCH(LoadWeightsT)(model_type, weights, pool);
  }

  switch (model_type) {
    case Model::GEMMA_2B:
      impl_.reset(new GemmaImpl<ConfigGemma2B>(tokenizer, weights_u8, pool));
      break;
    case Model::GEMMA_7B:
      impl_.reset(new GemmaImpl<ConfigGemma7B>(tokenizer, weights_u8, pool));
      break;
    case Model::GRIFFIN_2B:
      impl_.reset(new GemmaImpl<ConfigGriffin2B>(tokenizer, weights_u8, pool));
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model_type));
  }
}

Gemma::~Gemma() = default;  // after GemmaInterface is defined

const GemmaTokenizer* Gemma::Tokenizer() const { return impl_->Tokenizer(); }
const WeightStorageT& Gemma::Weights() const { return impl_->Weights(); }

void GenerateGemma(Gemma& gemma, size_t max_tokens, size_t max_generated_tokens,
                   float temperature, const std::vector<int>& prompt,
                   size_t start_pos, KVCache& kv_cache, hwy::ThreadPool& pool,
                   const StreamFunc& stream_token,
                   const AcceptFunc& accept_token, std::mt19937& gen,
                   int verbosity, LayersOutputT* layers_output) {
  pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
  gemma.impl_->Generate(max_tokens, max_generated_tokens, temperature, prompt,
                        start_pos, kv_cache, pool, stream_token, accept_token,
                        gen, verbosity, layers_output);
  pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
}

void GenerateGemma(Gemma& gemma, RuntimeConfig runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   const StreamFunc& stream_token, std::mt19937& gen) {
  GenerateGemma(
      gemma, runtime_config.max_tokens, runtime_config.max_generated_tokens,
      runtime_config.temperature, prompt, start_pos, kv_cache, pool,
      stream_token, [](int) { return true; }, gen, runtime_config.verbosity,
      /*layers_output=*/nullptr);
}

void CompressWeights(gcpp::Model model, const Path& weights,
                     const Path& compressed_weights, hwy::ThreadPool& pool) {
  HWY_DYNAMIC_DISPATCH(CompressWeightsT)
  (model, weights, compressed_weights, pool);
}

float ComputeCrossEntropy(Gemma& gemma, size_t max_tokens,
                          const std::vector<int>& prompt, KVCache& kv_cache,
                          hwy::ThreadPool& pool, int verbosity) {
  pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
  const float result = gemma.impl_->ComputeCrossEntropy(
      max_tokens, prompt, kv_cache, pool, verbosity);
  pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
  return result;
}

WeightStorageT AllocateWeights(Model model, hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(AllocateWeightsT)(model, pool);
}

WeightStorageT AllocateForwardPass(Model model) {
  return HWY_DYNAMIC_DISPATCH(AllocateForwardPassT)(model);
}

WeightStorageT AllocateBackwardPass(Model model) {
  return HWY_DYNAMIC_DISPATCH(AllocateBackwardPassT)(model);
}

void LogWeightStats(Model model, const WeightStorageT& weights) {
  return HWY_DYNAMIC_DISPATCH(LogWeightStatsT)(model, weights);
}

void InitWeights(Model model, WeightStorageT& weights,
                 InitMode init_mode, hwy::ThreadPool& pool, std::mt19937* gen) {
  return HWY_DYNAMIC_DISPATCH(InitWeightsT)(model, weights, init_mode, gen);
}

void UpdateWeights(Model model, const WeightStorageT& grad, float scale,
                   WeightStorageT& weights, hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(UpdateWeightsT)(model, grad, scale, weights,
                                              pool);
}

float CrossEntropyLossWithGradUpdate(
    const std::vector<int>& prompt, size_t context_size, const Model& model,
    const WeightStorageT& weights, WeightStorageT& forward,
    WeightStorageT& grad, WeightStorageT& backward, hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(CrossEntropyLossWithGradUpdateT)(
      prompt, context_size, model, weights, forward, grad, backward, pool);
}

namespace {
constexpr const char* kModelFlags[] = {"2b-pt", "7b-pt", "gr2b-pt",
                                       "2b-it", "7b-it", "gr2b-it",
                                       "tiny"};
constexpr Model kModelTypes[] = {Model::GEMMA_2B,   Model::GEMMA_7B,
                                 Model::GRIFFIN_2B, Model::GEMMA_2B,
                                 Model::GEMMA_7B,   Model::GRIFFIN_2B,
                                 Model::GEMMA_TINY};
constexpr ModelTraining kModelTraining[] = {
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_PT, ModelTraining::GEMMA_PT,
    ModelTraining::GEMMA_IT, ModelTraining::GEMMA_IT, ModelTraining::GEMMA_IT,
    ModelTraining::GEMMA_IT};
}  // namespace

const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training) {
  constexpr size_t kNum = std::end(kModelFlags) - std::begin(kModelFlags);
  static char kErrorMessageBuffer[kNum * 8 + 1024] =
      "Invalid or missing model flag, need to specify one of ";
  for (size_t i = 0; i + 1 < kNum; i++) {
    strcat(kErrorMessageBuffer, kModelFlags[i]);  // NOLINT
    strcat(kErrorMessageBuffer, ", ");            // NOLINT
  }
  strcat(kErrorMessageBuffer, kModelFlags[kNum - 1]);  // NOLINT
  strcat(kErrorMessageBuffer, ".");                    // NOLINT
  std::string model_type_lc = model_flag;
  std::transform(begin(model_type_lc), end(model_type_lc), begin(model_type_lc),
                 [](unsigned char c) { return std::tolower(c); });
  for (size_t i = 0; i < kNum; i++) {
    if (kModelFlags[i] == model_type_lc) {
      model = kModelTypes[i];
      training = kModelTraining[i];
      return nullptr;
    }
  }
  return kErrorMessageBuffer;
}

}  // namespace gcpp
#endif  // HWY_ONCE
