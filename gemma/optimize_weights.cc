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

// Command line tool to optimize weights.

#include <iostream>
#include <string>

#include "gemma/backward.h"
#include "gemma/forward.h"
#include "gemma/gemma.h"
#include "gemma/optimizer.h"
#include "gemma/sampler.h"
#include "gemma/weights.h"
#include "util/args.h"

namespace gcpp {

struct Args : public ArgsBase<Args> {
  static constexpr size_t kDefaultNumThreads = ~size_t{0};

  void ChooseNumThreads() {
    if (num_threads == kDefaultNumThreads) {
      // This is a rough heuristic, replace with something better in the future.
      num_threads = static_cast<size_t>(std::clamp(
          static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 64));
    }
  }

 public:
  Args(int argc, char* argv[]) {
    InitAndParse(argc, argv);
    ChooseNumThreads();
  }

  gcpp::Model ModelType() const { return model_type; }

  // Returns error string or nullptr if OK.
  const char* Validate() {
    ModelTraining model_training;
    const char* parse_result =
        ParseModelTypeAndTraining(model_type_str, model_type, model_training);
    if (parse_result) return parse_result;
    return nullptr;
  }

  std::string model_type_str;
  gcpp::Model model_type;
  size_t num_threads;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(model_type_str, "model", std::string(),
            "Model type\n"
            "    Required argument.");
    visitor(num_threads, "num_threads",
            kDefaultNumThreads,  // see ChooseNumThreads
            "Number of threads to use.\n    Default = Estimate of the "
            "number of suupported concurrent threads.",
            2);
  }
};

void ShowHelp(gcpp::Args& args) {
  std::cerr
      << "Usage:\n./optimize_weights --model_type <model type>\n";
  std::cerr << "\n*Arguments*\n\n";
  args.Help();
  std::cerr << "\n";
}

void Run(Args& args) {
  hwy::ThreadPool pool(args.num_threads);
  std::mt19937 gen(42);

  ByteStorageT weights = AllocateWeights(args.model_type, pool);
  ByteStorageT grad = AllocateWeights(args.model_type, pool);
  ByteStorageT grad_m = AllocateWeights(args.model_type, pool);
  ByteStorageT grad_v = AllocateWeights(args.model_type, pool);
  ByteStorageT forward = AllocateForwardPass(args.model_type);
  ByteStorageT backward = AllocateForwardPass(args.model_type);
  ByteStorageT inference = AllocateInferenceState(args.model_type);
  auto kv_cache = CreateKVCache(args.model_type);
  size_t max_tokens = 32;
  size_t max_generated_tokens = 16;
  float temperature = 1.0f;
  int verbosity = 0;
  const auto accept_token = [](int) { return true; };

  const auto generate = [&](const std::vector<int>& prompt) {
    std::vector<int> reply;
    auto stream_token = [&reply](int token, float) {
      reply.push_back(token);
      return token != ReverseSequenceSampler::kEndToken;
    };
    RuntimeConfig runtime = {
      max_tokens, max_generated_tokens, temperature, verbosity, &gen,
      stream_token, accept_token, ReverseSequenceSampler::kEndToken,
    };
    TimingInfo timing_info;
    GenerateGemma(args.model_type, weights, inference, runtime, prompt, 0,
                  kv_cache, pool, timing_info);
    return reply;
  };

  auto verify = [&](const Prompt& prompt) {
    auto context = prompt.context();
    std::vector<int> reply = generate(context);
    bool ok = true;
    for (size_t i = 0; ok && i < prompt.tokens.size(); ++i) {
      if (i >= reply.size() || reply[i] != prompt.tokens[i]) {
        ok = false;
      }
    }
    return ok;
  };

  RandInitWeights(args.model_type, weights, pool, gen);
  ZeroInitWeights(args.model_type, grad_m, pool);
  ZeroInitWeights(args.model_type, grad_v, pool);

  printf("Initial weights:\n");
  LogWeightStats(args.model_type, weights);

  constexpr size_t kBatchSize = 8;
  float learning_rate = 0.0005f;

  ReverseSequenceSampler training_task({
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1});
  size_t steps = 0;
  float prev_loss = std::numeric_limits<float>::max();
  for (; steps < 1000000; ++steps) {
    std::mt19937 sgen(42);
    ZeroInitWeights(args.model_type, grad, pool);
    float total_loss = 0.0f;
    size_t num_ok = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      Prompt prompt = training_task.Sample(sgen);
      total_loss += CrossEntropyLossForwardPass(
          args.model_type, prompt, weights, forward, pool);
      CrossEntropyLossBackwardPass(
          args.model_type, prompt, weights, forward, grad, backward, pool);
      num_ok += verify(prompt) ? 1 : 0;
    }
    total_loss /= kBatchSize;

    const float scale = -learning_rate / kBatchSize;
    UpdateWeights(args.model_type, grad, scale, weights, pool);
    printf("step: %zu  total_loss: %.15f   num_ok: %zu/%zu\n",
           steps, total_loss, num_ok, kBatchSize);
    if (steps % 100 == 0) {
      printf("Batch gradient:\n");
      LogWeightStats(args.model_type, grad);
    }
    if (total_loss < 0.5f) {
      break;
    }
    prev_loss = total_loss;
  }
  printf("Num steps: %zu\n", steps);
  printf("Final weights:\n");
  LogWeightStats(args.model_type, weights);
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
    HWY_ABORT("\nInvalid args: %s", error);
  }

  gcpp::Run(args);

  return 0;
}
