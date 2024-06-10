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

#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <string>
#include <vector>

#include "backprop/backward.h"
#include "backprop/forward.h"
#include "backprop/prompt.h"
#include "backprop/optimizer.h"
#include "compression/io.h"
#include "gemma/activations.h"
#include "gemma/cross_entropy.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "util/args.h"
#include "hwy/timer.h"

namespace gcpp {

struct Args : public ArgsBase<Args> {
  static constexpr size_t kDefaultNumThreads = ~size_t{0};

  void ChooseNumThreads() {
    if (num_threads == kDefaultNumThreads) {
      // This is a rough heuristic, replace with something better in the future.
      num_threads = static_cast<size_t>(std::clamp(
          static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
    }
  }

 public:
  Args(int argc, char* argv[]) {
    InitAndParse(argc, argv);
    ChooseNumThreads();
  }

  // Returns error string or nullptr if OK.
  const char* Validate() {
    ModelTraining model_training;
    const char* parse_result =
        ParseModelTypeAndTraining(model_type_str, model_type, model_training);
    if (parse_result) return parse_result;
    if (corpus.path.empty()) {
      return "Missing --corpus flag, a file for the training data.";
    }
    if (!corpus.Exists()) {
      return "Can't open file specified with --corpus flag.";
    }
    if (!weights_in.path.empty() && !weights_in.Exists()) {
      return "Can't open file specified with --weights_in flag.";
    }
    return nullptr;
  }

  std::string model_type_str;
  Model model_type;
  Path corpus;
  size_t num_threads;
  Path weights_in;
  Path weights_out;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(model_type_str, "model", std::string(),
            "Model type\n    2b-it = 2B parameters, instruction-tuned\n    "
            "2b-pt = 2B parameters, pretrained\n    7b-it = 7B parameters "
            "instruction-tuned\n    7b-pt = 7B parameters, pretrained\n    "
            "gr2b-it = griffin 2B parameters, instruction-tuned\n    "
            "gr2b-pt = griffin 2B parameters, pretrained\n    "
            "    Required argument.");
    visitor(corpus, "corpus", Path(),
            "Path to (text) training data.\n    "
            "    Required argument.");
    visitor(num_threads, "num_threads",
            kDefaultNumThreads,  // see ChooseNumThreads
            "Number of threads to use.\n    Default = Estimate of the "
            "number of suupported concurrent threads.",
            2);
    visitor(weights_in, "weights_in", Path(),
            "Starting checksum for the training. If missing, training starts "
            "from random weights.");
    visitor(weights_out, "weights_out", Path(),
            "If present, weights will be periodically saved to this file.");
  }
};

Prompt CreatePrompt(const std::vector<uint8_t>& corpus, const size_t seq_len,
                    size_t& pos) {
  Prompt prompt;
  prompt.context_size = 1;
  prompt.tokens.reserve(seq_len);
  for (size_t i = 0; i < seq_len && pos < corpus.size(); ++i) {
    prompt.tokens.push_back(corpus[pos++]);
  }
  return prompt;
}

void ShowHelp(Args& args) {
  std::cerr
      << "Usage:\n./train --model <model type> --corpus <corpus file>\n";
  std::cerr << "\n*Arguments*\n\n";
  args.Help();
  std::cerr << "\n";
}

template <typename TConfig>
struct GetSeqLen {
  int operator()() const { return TConfig::kSeqLen; }
};

int Run(Args& args) {
  hwy::ThreadPool pool(args.num_threads);

  auto corpus_file = OpenFileOrNull(args.corpus, "rb");
  if (!corpus_file) {
    fprintf(stderr, "Failed to open corpus file");
    return EXIT_FAILURE;
  }
  std::vector<uint8_t> corpus_data(corpus_file->FileSize());
  corpus_file->Read(0, corpus_data.size(), corpus_data.data());

  Model model_type = args.model_type;
  Type weight_type = Type::kF32;
  ByteStorageT grad =
      CallForModelAndWeight<AllocateWeightsF>(model_type, weight_type, pool);
  ByteStorageT grad_m =
      CallForModelAndWeight<AllocateWeightsF>(model_type, weight_type, pool);
  ByteStorageT grad_v =
      CallForModelAndWeight<AllocateWeightsF>(model_type, weight_type, pool);
  ByteStorageT forward =
      CallForModelAndWeight<AllocateForwardPass>(model_type, weight_type);
  ByteStorageT backward =
      CallForModelAndWeight<AllocateForwardPass>(model_type, weight_type);
  const size_t seq_len = CallForModel<float, GetSeqLen>(model_type);

  Gemma gemma(GemmaTokenizer(), model_type, weight_type, pool);

  if (!args.weights_in.path.empty()) {
    ByteStorageT weights = LoadRawWeights(args.weights_in, model_type,
                                          weight_type, pool, false);
    gemma.SetWeights(std::move(weights));
  } else {
    std::mt19937 gen(42);
    RandInitWeights(model_type, weight_type, gemma.Weights(), pool, gen);
  }
  printf("Initial weights:\n");
  LogWeightStats(model_type, weight_type, gemma.Weights());

  constexpr size_t kBatchSize = 8;
  float alpha = 0.001;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float epsilon = 1e-8;

  const double time_start = hwy::platform::Now();
  auto time_elapsed = [&time_start]() {
    return (hwy::platform::Now() - time_start);
  };

  CallForModelAndWeight<ZeroInitWeightsF>(model_type, weight_type, grad_m,
                                          pool);
  CallForModelAndWeight<ZeroInitWeightsF>(model_type, weight_type, grad_v,
                                          pool);

  size_t batches = 0;
  size_t update_steps = 0;
  size_t pos = 0;
  float total_bits = 0.0f;
  float epoch_bits = 0.0f;
  size_t epoch_start = 0;
  gcpp::KVCache kv_cache = gcpp::KVCache::Create(model_type);
  while (pos < corpus_data.size()) {
    size_t batch_start = pos;
    for (size_t i = 0; i < kBatchSize && pos < corpus_data.size(); ++i) {
      Prompt prompt = CreatePrompt(corpus_data, seq_len, pos);
      epoch_bits += ComputeCrossEntropy(
          gemma, seq_len, prompt.tokens, kv_cache, 0);
    }
    size_t updates_per_batch = batches < 1000 ? 10 : 1;
    for (size_t iter = 0; iter < updates_per_batch; ++iter) {
      pos = batch_start;
      CallForModelAndWeight<ZeroInitWeightsF>(model_type, weight_type, grad,
                                              pool);
      for (size_t i = 0; i < kBatchSize && pos < corpus_data.size(); ++i) {
        Prompt prompt = CreatePrompt(corpus_data, seq_len, pos);
        CrossEntropyLossForwardPass(
            model_type, prompt, gemma.Weights(), forward, pool);
        CrossEntropyLossBackwardPass(
            model_type, prompt, gemma.Weights(), forward, grad, backward, pool);
      }
      AdamUpdate(model_type, weight_type, grad, alpha, beta1, beta2, epsilon,
                 ++update_steps, gemma.Weights(), grad_m, grad_v, pool);
    }
    ++batches;
    if (batches % 100 == 0 || pos == corpus_data.size()) {
      total_bits += epoch_bits;
      const float speed_kbps = pos * 1.0 / 1024.0 / time_elapsed();
      const size_t epoch_bytes = pos - epoch_start;
      const float epoch_loss = epoch_bits / epoch_bytes;
      const float total_loss = total_bits / pos;
      printf("time: %6.1fs   step: %6zu    pos: %8zu   speed: %6.2f kB/s   "
             "loss: %6.3f (epoch)  %6.3f (total)\n",
             time_elapsed(), batches, pos, speed_kbps, epoch_loss, total_loss);
      epoch_bits = 0.0f;
      epoch_start = pos;
      if (!args.weights_out.path.empty()) {
        SaveRawWeights(gemma.Weights(), args.weights_out, model_type);
      }
    }
  }

  return EXIT_SUCCESS;
}

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::Args args(argc, argv);

  if (gcpp::HasHelp(argc, argv)) {
    ShowHelp(args);
    return 0;
  }

  if (const char* error = args.Validate()) {
    ShowHelp(args);
    fprintf(stderr, "\nInvalid args: %s", error);
    return EXIT_FAILURE;
  }

  return gcpp::Run(args);
}
