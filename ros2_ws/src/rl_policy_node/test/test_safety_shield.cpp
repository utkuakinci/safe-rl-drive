#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rl_policy_node/safety_shield.hpp"

using namespace rl_policy_node;

namespace
{
std::vector<float> empty_road(float ego_speed = 25.f, bool left = true, bool right = true)
{
  std::vector<float> x(kSafetyHeader + 8 * kSafetyFeats, 0.f);
  x[0] = ego_speed;
  x[1] = left ? 1.f : 0.f;
  x[2] = right ? 1.f : 0.f;
  return x;
}

void put(std::vector<float> & x, int i, float s, float d, float vs, float vd = 0.f)
{
  float * v = &x[kSafetyHeader + i * kSafetyFeats];
  v[0] = 1.f, v[1] = s, v[2] = d, v[3] = vs, v[4] = vd;
}
}  // namespace

TEST(SafetyShield, EmptyRoadAllowsEverything)
{
  SafetyShield sh;
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    auto d = sh.filter(a, empty_road());
    EXPECT_EQ(d.final_action, a);
    EXPECT_FALSE(d.intervened);
  }
}

TEST(SafetyShield, ClosingLeaderBlocksFaster)
{
  SafetyShield sh;
  auto x = empty_road();
  put(x, 0, 20.f, 0.f, -8.f);  // gap 15 m, closing 8 m/s -> TTC 1.9 s
  auto d = sh.filter(FASTER, x);
  EXPECT_TRUE(d.intervened);
  EXPECT_NE(d.final_action, FASTER);
  EXPECT_NE(d.final_action, IDLE);
  EXPECT_TRUE(d.unsafe[FASTER] & FRONT_TTC);
}

TEST(SafetyShield, VeryLowTtcTriggersAeb)
{
  SafetyShield sh;
  auto x = empty_road();
  put(x, 0, 12.f, 0.f, -10.f);  // TTC 0.7 s
  auto d = sh.filter(IDLE, x);
  EXPECT_TRUE(d.aeb);
  EXPECT_EQ(d.final_action, SLOWER);
}

TEST(SafetyShield, OccupiedTargetLaneBlocksLaneChange)
{
  SafetyShield sh;
  auto x = empty_road();
  put(x, 0, 2.f, -4.f, 0.f);  // car alongside on the left
  auto d = sh.filter(LANE_LEFT, x);
  EXPECT_TRUE(d.unsafe[LANE_LEFT] & LC_GAP);
  EXPECT_EQ(d.final_action, IDLE);
  EXPECT_FALSE(sh.filter(LANE_RIGHT, x).intervened);
}

TEST(SafetyShield, MostProbableSafeActionIsChosen)
{
  SafetyShield sh;
  auto x = empty_road();
  put(x, 0, 20.f, 0.f, -8.f);
  std::vector<float> probs = {0.05f, 0.1f, 0.3f, 0.5f, 0.05f};  // FASTER > LANE_RIGHT > ...
  EXPECT_EQ(sh.filter(FASTER, x, probs).final_action, LANE_RIGHT);
}

TEST(SafetyShield, DisabledShieldIsTransparent)
{
  ShieldConfig c;
  c.enabled = false;
  SafetyShield sh(c);
  auto x = empty_road();
  put(x, 0, 6.f, 0.f, -10.f);
  EXPECT_EQ(sh.filter(FASTER, x).final_action, FASTER);
}

// Cross-language conformance: vectors generated from the Python shield.
// Row: action, expected_final, expected_aeb, safety_obs...
TEST(SafetyShield, MatchesPythonReferenceVectors)
{
  std::ifstream f(SHIELD_VECTORS);
  ASSERT_TRUE(f.good()) << "missing " << SHIELD_VECTORS;
  SafetyShield sh;
  std::string line;
  int n = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> row;
    while (std::getline(ss, cell, ',')) row.push_back(std::stof(cell));
    const int action = static_cast<int>(row[0]);
    const int expected = static_cast<int>(row[1]);
    const bool expected_aeb = row[2] > 0.5f;
    std::vector<float> obs(row.begin() + 3, row.end());
    auto d = sh.filter(action, obs);
    EXPECT_EQ(d.final_action, expected) << "row " << n << ": " << d.reason();
    EXPECT_EQ(d.aeb, expected_aeb) << "row " << n;
    ++n;
  }
  EXPECT_GT(n, 100);
}
