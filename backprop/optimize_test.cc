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

#include <iostream>
#include <string>

#include "backprop/backward.h"
#include "backprop/forward.h"
#include "backprop/optimizer.h"
#include "backprop/sampler.h"
#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "gtest/gtest.h"
#include "hwy/stats.h"
#include "hwy/timer.h"

static constexpr int kLogGradients = 0;

namespace gcpp {

TEST(OptimizeTest, GradientDescent) {
  hwy::ThreadPool pool(0);
  std::mt19937 gen(42);

  Model model_type = Model::GEMMA_TINY;
  ByteStorageT weights = AllocateWeights(model_type, pool);
  ByteStorageT grad = AllocateWeights(model_type, pool);
  ByteStorageT grad_m = AllocateWeights(model_type, pool);
  ByteStorageT grad_v = AllocateWeights(model_type, pool);
  ByteStorageT forward = AllocateForwardPass(model_type);
  ByteStorageT backward = AllocateForwardPass(model_type);
  ByteStorageT inference = AllocateInferenceState(model_type);
  auto kv_cache = CreateKVCache(model_type);
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
    GenerateGemma(model_type, weights, inference, runtime, prompt, 0,
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

  RandInitWeights(model_type, weights, pool, gen);
  ZeroInitWeights(model_type, grad_m, pool);
  ZeroInitWeights(model_type, grad_v, pool);

  printf("Initial weights:\n");
  LogWeightStats(model_type, weights);

  constexpr size_t kBatchSize = 8;
  constexpr float kBatchScale = 1.0f / kBatchSize;
  float alpha = 0.001;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float epsilon = 1e-8;

  ReverseSequenceSampler training_task({
      0, 0, 0, 0, 0, 0, 0, 0, 1});
  size_t steps = 0;
  std::mt19937 sgen(42);
  hwy::Stats num_ok_stats;
  hwy::Stats loss_stats;
  const double time_start = hwy::platform::Now();
  auto time_elapsed = [&time_start]() {
    return (hwy::platform::Now() - time_start);
  };
  for (; steps < 1000000; ++steps) {
    ZeroInitWeights(model_type, grad, pool);
    float loss = 0.0f;
    size_t num_ok = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      Prompt prompt = training_task.Sample(sgen);
      loss += CrossEntropyLossForwardPass(
          model_type, prompt, weights, forward, pool);
      CrossEntropyLossBackwardPass(
          model_type, prompt, weights, forward, grad, backward, pool);
      num_ok += verify(prompt) ? 1 : 0;
    }
    loss *= kBatchScale;
    loss_stats.Notify(loss);
    num_ok_stats.Notify(num_ok);

    AdamUpdate(model_type, grad, alpha, beta1, beta2, epsilon, steps + 1,
               weights, grad_m, grad_v, pool);
    if (steps % 1000 == 0) {
      printf("time: %6.1fs  step: %6zu   loss: %.15f   num_ok: %.2f/%zu\n",
             time_elapsed(), steps, loss_stats.Mean(), num_ok_stats.Mean(),
             kBatchSize);
      if (kLogGradients) {
        printf("Batch gradient:\n");
        LogWeightStats(model_type, grad, kBatchScale);
      }
      if (loss_stats.Mean() < 0.1f) {
        break;
      }
      loss_stats.Reset();
      num_ok_stats.Reset();
    }
  }
  printf("Num steps: %zu\n", steps);
  printf("Final weights:\n");
  LogWeightStats(model_type, weights);
  EXPECT_LT(steps, 60000);
  EXPECT_GT(num_ok_stats.Mean(), 0.95 * kBatchSize);
}

}  // namespace gcpp
