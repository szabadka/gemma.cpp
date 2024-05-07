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

#include "gemma/gemma.h"
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

static constexpr int kStartToken = 10;
static constexpr int kReverseToken = 11;
static constexpr int kEndToken = 12;

class PromptSampler {
 public:
  virtual std::vector<int> Sample(std::mt19937& gen) = 0;
};

class ReverseSequenceSampler : public PromptSampler {
 public:
  explicit ReverseSequenceSampler(size_t len) : dist_(0, 9), len_(len) {}

  std::vector<int> Sample(std::mt19937& gen) override {
    std::vector<int> sample(2 * len_ + 3);
    sample[0] = kStartToken;
    sample[len_ + 1] = kReverseToken;
    sample[2 * len_ + 2] = kEndToken;
    for (size_t i = 0; i < len_; ++i) {
      sample[i + 1] = sample[2 * len_ + 1 - i] = dist_(gen);
    }
    return sample;
  }

 private:
  std::uniform_int_distribution<> dist_;
  size_t len_;
};

void LogPrompt(const std::vector<int>& prompt) {
  static const char* kVocab[] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<", "-->", ">",
  };
  for (int token : prompt) printf(" %s", kVocab[token]);
  printf("\n");
}

void Run(Args& args) {
  hwy::ThreadPool pool(args.num_threads);
  std::mt19937 gen(42);

  WeightStorageT weights = AllocateWeights(args.model_type, pool);
  WeightStorageT grad = AllocateWeights(args.model_type, pool);

  InitWeights(args.model_type, weights, InitMode::RAND_INIT, pool, &gen);

  constexpr size_t kBatchSize = 16;
  float learning_rate = 0.001f;

  ReverseSequenceSampler training_task(10);
  for (;;) {
    InitWeights(args.model_type, grad, InitMode::ZERO_INIT, pool);
    float total_loss = 0.0f;
    for (size_t i = 0; i < kBatchSize; ++i) {
      std::vector<int> prompt = training_task.Sample(gen);
      LogPrompt(prompt);
      total_loss += CrossEntropyLossWithGradUpdate(
          prompt, args.model_type, weights, grad, pool);
    }
    total_loss /= kBatchSize;
    const float scale = learning_rate / kBatchSize;
    UpdateWeights(args.model_type, grad, scale, weights, pool);
    printf("total_loss: %f\n", total_loss);
    if (total_loss < 0.01f) {
      break;
    }
  }
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
