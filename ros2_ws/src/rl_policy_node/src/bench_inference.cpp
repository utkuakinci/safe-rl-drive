// Latency of policy inference + shield in C++, without ROS transport.
//   ros2 run rl_policy_node bench_inference <model.onnx|.engine> <onnxruntime|onnxruntime_cuda|tensorrt> [iters]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "rl_policy_node/inference_backend.hpp"
#include "rl_policy_node/safety_shield.hpp"

using rl_policy_node::make_backend;

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model> <backend> [iters]\n", argv[0]);
    return 1;
  }
  const int iters = argc > 3 ? std::stoi(argv[3]) : 10000;
  auto policy = make_backend(argv[2], argv[1]);
  rl_policy_node::SafetyShield shield;

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<float> obs(policy->obs_size());
  std::vector<float> safety(3 + 5 * 8, 0.f);
  safety[0] = 25.f;
  std::vector<double> t(iters);

  for (int i = 0; i < iters + 500; ++i) {
    for (auto & v : obs) v = u(rng);
    const auto t0 = std::chrono::steady_clock::now();
    const auto & logits = policy->infer(obs);
    const int a = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    volatile auto d = shield.filter(a, safety).final_action;
    (void)d;
    const auto t1 = std::chrono::steady_clock::now();
    if (i >= 500) t[i - 500] = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  std::sort(t.begin(), t.end());
  auto pct = [&](double q) { return t[static_cast<size_t>(q * (t.size() - 1))]; };
  std::printf("%s  recurrent=%d  iters=%d  p50=%.1fus  p90=%.1fus  p99=%.1fus  max=%.1fus\n",
              policy->name().c_str(), policy->recurrent(), iters, pct(0.5), pct(0.9), pct(0.99), t.back());
  return 0;
}
