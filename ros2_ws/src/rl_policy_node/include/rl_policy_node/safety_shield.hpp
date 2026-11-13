// C++ port of src/safe_rl_drive/safety/shield.py. Behaviour is pinned by test/shield_vectors.csv,
// which is generated from the Python implementation (scripts/gen_shield_vectors.py).
#pragma once

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace rl_policy_node
{

enum Action : int { LANE_LEFT = 0, IDLE = 1, LANE_RIGHT = 2, FASTER = 3, SLOWER = 4, NUM_ACTIONS = 5 };

// Rule bit flags (one bit per rule, OR-ed per action).
enum Rule : unsigned {
  FRONT_TTC = 1u << 0,
  FRONT_GAP = 1u << 1,
  CROSSING_CONFLICT = 1u << 2,
  LC_GAP = 1u << 3,
  LC_TTC = 1u << 4,
};

struct ShieldConfig
{
  bool enabled = true;
  double ttc_threshold = 2.5;
  double standstill_gap = 4.0;
  double time_headway = 0.6;
  double vehicle_length = 5.0;
  double lane_width = 4.0;
  double lc_front_gap = 5.0;
  double lc_rear_gap = 5.0;
  double lc_ttc = 2.0;
  double cpa_radius = 3.0;
  double cpa_horizon = 2.5;
  double aeb_ttc = 1.5;
  std::vector<int> fallback_order = {IDLE, SLOWER, LANE_RIGHT, LANE_LEFT, FASTER};
};

struct ShieldDecision
{
  int original = IDLE;
  int final_action = IDLE;
  bool intervened = false;
  bool aeb = false;
  std::array<unsigned, NUM_ACTIONS> unsafe{};  // rule mask per action, 0 = safe
  double front_ttc = std::numeric_limits<double>::infinity();
  double front_gap = std::numeric_limits<double>::infinity();
  std::string reason() const;
};

// Safety observation layout: [ego_speed, left_ok, right_ok, (presence, s, d, vs, vd) * K]
constexpr int kSafetyHeader = 3;
constexpr int kSafetyFeats = 5;

class SafetyShield
{
public:
  explicit SafetyShield(ShieldConfig cfg = {}) : cfg_(std::move(cfg)) {}

  // probs may be empty -> fixed fallback order.
  ShieldDecision filter(int action, const std::vector<float> & safety_obs,
                        const std::vector<float> & probs = {}) const;
  void assess(const std::vector<float> & safety_obs, ShieldDecision & out) const;
  const ShieldConfig & config() const { return cfg_; }

private:
  ShieldConfig cfg_;
};

const char * action_name(int a);
std::string rule_names(unsigned mask);

}  // namespace rl_policy_node
