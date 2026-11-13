// Real-time policy node: observation -> (ONNX Runtime | TensorRT) -> safety shield -> command.
//
// Subscribes  /rl/observation  std_msgs/Float32MultiArray
//               data = [policy_obs (V*F) | safety_obs (3 + 5K) | episode_start (1)]
//               (one message so that both views always belong to the same simulation step)
// Publishes   /rl/command      std_msgs/Int32MultiArray  data = [final_action, policy_action, aeb]
//             /rl/shield       std_msgs/String           human-readable intervention reason
//             /rl/latency_us   std_msgs/Float64          inference + shield time of this step
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <vector>

#include "rl_policy_node/inference_backend.hpp"
#include "rl_policy_node/safety_shield.hpp"

namespace rl_policy_node
{

class PolicyNode : public rclcpp::Node
{
public:
  explicit PolicyNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("rl_policy_node", options)
  {
    const auto model_path = declare_parameter<std::string>("model_path");
    const auto backend = declare_parameter<std::string>("backend", "onnxruntime");
    stats_window_ = declare_parameter<int>("stats_window", 500);

    ShieldConfig sc;
    sc.enabled = declare_parameter<bool>("shield.enabled", sc.enabled);
    sc.ttc_threshold = declare_parameter<double>("shield.ttc_threshold", sc.ttc_threshold);
    sc.standstill_gap = declare_parameter<double>("shield.standstill_gap", sc.standstill_gap);
    sc.time_headway = declare_parameter<double>("shield.time_headway", sc.time_headway);
    sc.vehicle_length = declare_parameter<double>("shield.vehicle_length", sc.vehicle_length);
    sc.lane_width = declare_parameter<double>("shield.lane_width", sc.lane_width);
    sc.lc_front_gap = declare_parameter<double>("shield.lc_front_gap", sc.lc_front_gap);
    sc.lc_rear_gap = declare_parameter<double>("shield.lc_rear_gap", sc.lc_rear_gap);
    sc.lc_ttc = declare_parameter<double>("shield.lc_ttc", sc.lc_ttc);
    sc.cpa_radius = declare_parameter<double>("shield.cpa_radius", sc.cpa_radius);
    sc.cpa_horizon = declare_parameter<double>("shield.cpa_horizon", sc.cpa_horizon);
    sc.aeb_ttc = declare_parameter<double>("shield.aeb_ttc", sc.aeb_ttc);
    shield_ = std::make_unique<SafetyShield>(sc);

    policy_ = make_backend(backend, model_path);
    obs_.resize(policy_->obs_size());
    probs_.resize(policy_->num_actions());
    RCLCPP_INFO(get_logger(), "backend=%s recurrent=%d obs=%zu actions=%zu shield=%s",
                policy_->name().c_str(), policy_->recurrent(), policy_->obs_size(),
                policy_->num_actions(), sc.enabled ? "on" : "off");

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    cmd_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("/rl/command", qos);
    shield_pub_ = create_publisher<std_msgs::msg::String>("/rl/shield", 10);
    lat_pub_ = create_publisher<std_msgs::msg::Float64>("/rl/latency_us", 10);
    obs_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "/rl/observation", qos, [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) { on_obs(*m); });
  }

private:
  void on_obs(const std_msgs::msg::Float32MultiArray & msg)
  {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n_pol = obs_.size();
    if (msg.data.size() < n_pol + kSafetyHeader + 1) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "observation too short: %zu", msg.data.size());
      return;
    }
    const bool episode_start = msg.data.back() > 0.5f;
    if (episode_start) policy_->reset_state();
    std::copy(msg.data.begin(), msg.data.begin() + n_pol, obs_.begin());
    safety_obs_.assign(msg.data.begin() + n_pol, msg.data.end() - 1);

    const auto & logits = policy_->infer(obs_);
    // softmax -> probabilities (used by the shield to pick the most likely safe alternative)
    const float mx = *std::max_element(logits.begin(), logits.end());
    float z = 0.f;
    for (size_t i = 0; i < logits.size(); ++i) z += (probs_[i] = std::exp(logits[i] - mx));
    for (auto & p : probs_) p /= z;
    const int action = static_cast<int>(std::max_element(probs_.begin(), probs_.end()) - probs_.begin());
    const ShieldDecision d = shield_->filter(action, safety_obs_, probs_);

    std_msgs::msg::Int32MultiArray cmd;
    cmd.data = {d.final_action, action, d.aeb ? 1 : 0};
    cmd_pub_->publish(cmd);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

    std_msgs::msg::Float64 lat;
    lat.data = us;
    lat_pub_->publish(lat);
    if (d.intervened) {
      std_msgs::msg::String s;
      s.data = d.reason();
      shield_pub_->publish(s);
      ++interventions_;
    }
    record(us);
  }

  void record(double us)
  {
    window_.push_back(us);
    ++steps_;
    if (static_cast<int>(window_.size()) < stats_window_) return;
    std::sort(window_.begin(), window_.end());
    auto pct = [&](double q) { return window_[static_cast<size_t>(q * (window_.size() - 1))]; };
    RCLCPP_INFO(get_logger(), "latency over %zu steps: p50=%.1fus p99=%.1fus max=%.1fus | shield %.1f%% (%zu/%zu)",
                window_.size(), pct(0.5), pct(0.99), window_.back(), 100.0 * interventions_ / steps_,
                interventions_, steps_);
    window_.clear();
  }

  std::unique_ptr<InferenceBackend> policy_;
  std::unique_ptr<SafetyShield> shield_;
  std::vector<float> obs_, safety_obs_, probs_;
  std::vector<double> window_;
  int stats_window_ = 500;
  size_t steps_ = 0, interventions_ = 0;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr obs_sub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr shield_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr lat_pub_;
};

}  // namespace rl_policy_node

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(rl_policy_node::PolicyNode)
