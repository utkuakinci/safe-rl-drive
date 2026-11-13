#include "rl_policy_node/safety_shield.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace rl_policy_node
{

const char * action_name(int a)
{
  static const char * names[] = {"LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"};
  return (a >= 0 && a < NUM_ACTIONS) ? names[a] : "UNKNOWN";
}

std::string rule_names(unsigned mask)
{
  // alphabetical, like the Python implementation (sorted rule names)
  static const std::pair<unsigned, const char *> rules[] = {
    {CROSSING_CONFLICT, "crossing_conflict"}, {FRONT_GAP, "front_gap"}, {FRONT_TTC, "front_ttc"},
    {LC_GAP, "lc_gap"}, {LC_TTC, "lc_ttc"}};
  std::string out;
  for (const auto & [bit, name] : rules) {
    if (mask & bit) {
      if (!out.empty()) out += ",";
      out += name;
    }
  }
  return out;
}

std::string ShieldDecision::reason() const
{
  if (!intervened) return "";
  std::ostringstream ss;
  ss << action_name(original) << "->" << action_name(final_action) << " (" << rule_names(unsafe[original]) << ")";
  if (aeb) ss << " +AEB";
  return ss.str();
}

void SafetyShield::assess(const std::vector<float> & x, ShieldDecision & d) const
{
  const auto & c = cfg_;
  const double inf = std::numeric_limits<double>::infinity();
  const double ego_speed = x.at(0);
  const bool left_ok = x.at(1) > 0.5f, right_ok = x.at(2) > 0.5f;
  const double half_w = c.lane_width / 2.0;
  d.unsafe.fill(0u);
  double front_ttc = inf, front_gap = inf;

  const size_t k = (x.size() - kSafetyHeader) / kSafetyFeats;
  for (size_t i = 0; i < k; ++i) {
    const float * v = &x[kSafetyHeader + i * kSafetyFeats];
    if (v[0] < 0.5f) continue;
    const double s = v[1], dl = v[2], vs = v[3], vd = v[4];
    const double gap = std::abs(s) - c.vehicle_length;
    if (std::abs(dl) < half_w && s > 0) {  // ego corridor leader
      const double closing = -vs;
      const double ttc = closing > 1e-6 ? gap / closing : inf;
      front_ttc = std::min(front_ttc, std::max(ttc, 0.0));
      front_gap = std::min(front_gap, gap);
    } else if (s > 0) {  // crossing / merging traffic ahead
      const double v2 = vs * vs + vd * vd;
      const double t_star = v2 < 1e-9 ? 0.0 : std::clamp(-(s * vs + dl * vd) / v2, 0.0, c.cpa_horizon);
      const double d_min = std::hypot(s + vs * t_star, dl + vd * t_star);
      if (d_min < c.cpa_radius && t_star < c.cpa_horizon) {
        d.unsafe[FASTER] |= CROSSING_CONFLICT;
        d.unsafe[IDLE] |= CROSSING_CONFLICT;
      }
    }
    // target-lane checks
    const std::pair<double, int> sides[] = {{-1.0, LANE_LEFT}, {1.0, LANE_RIGHT}};
    for (const auto & [side, action] : sides) {
      if (!(half_w <= side * dl && side * dl < 3 * half_w)) continue;
      if (-(c.vehicle_length + c.lc_rear_gap) < s && s < (c.vehicle_length + c.lc_front_gap)) {
        d.unsafe[action] |= LC_GAP;
      }
      const double closing = s > 0 ? -vs : vs;
      if (closing > 1e-6 && gap / closing < c.lc_ttc) d.unsafe[action] |= LC_TTC;
    }
  }

  if (front_ttc < c.ttc_threshold) {
    d.unsafe[FASTER] |= FRONT_TTC;
    d.unsafe[IDLE] |= FRONT_TTC;
  }
  if (front_gap < c.standstill_gap + c.time_headway * ego_speed) d.unsafe[FASTER] |= FRONT_GAP;
  if (front_gap < c.standstill_gap) d.unsafe[IDLE] |= FRONT_GAP;
  // lane change without a target lane behaves like IDLE
  if (!left_ok) d.unsafe[LANE_LEFT] |= d.unsafe[IDLE];
  if (!right_ok) d.unsafe[LANE_RIGHT] |= d.unsafe[IDLE];
  d.unsafe[SLOWER] = 0u;  // minimal-risk manoeuvre is never blocked
  d.front_ttc = front_ttc;
  d.front_gap = front_gap;
}

ShieldDecision SafetyShield::filter(int action, const std::vector<float> & safety_obs,
                                    const std::vector<float> & probs) const
{
  ShieldDecision d;
  d.original = d.final_action = action;
  if (!cfg_.enabled) return d;
  assess(safety_obs, d);
  if (d.unsafe[action] != 0u) {
    std::vector<int> order;
    if (!probs.empty()) {
      order.resize(probs.size());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });
    } else {
      order = cfg_.fallback_order;
    }
    d.final_action = SLOWER;
    for (int a : order) {
      if (a >= 0 && a < NUM_ACTIONS && d.unsafe[a] == 0u) {
        d.final_action = a;
        break;
      }
    }
  }
  d.aeb = d.front_ttc < cfg_.aeb_ttc;
  if (d.aeb) d.final_action = SLOWER;
  d.intervened = d.final_action != action || d.aeb;
  return d;
}

}  // namespace rl_policy_node
