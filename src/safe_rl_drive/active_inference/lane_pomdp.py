"""A small lane-choice POMDP shared by the active-inference agent (as its generative model) and by
PPO / RecurrentPPO (as a Gymnasium environment).

Story: the ego car drives on a 3-lane road towards a stalled vehicle in an unknown lane. A noisy
detector reports the obstacle's lane every step; it gets more reliable as the distance shrinks.
The agent can change lane (left / stay / right) once per step and must not be in the obstacle's
lane when the distance reaches zero.

Hidden state factors:   ego lane (3, controllable) x obstacle lane (3) x distance (D+1)
Observation modalities: ego lane (3, exact) | detector (3, noisy) | outcome (neutral, passed, crash)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

N_LANES = 3
D_MAX = 5
ACTIONS = ("LEFT", "STAY", "RIGHT")
OUTCOMES = ("neutral", "passed", "crash")
# Detector accuracy vs distance (index = distance); 1/3 is chance. Even at distance 1 a single reading is
# wrong 25% of the time, so evidence must be accumulated over steps (memory / belief state matters).
DEFAULT_ACCURACY = (1.0, 0.75, 0.7, 0.62, 0.55, 0.5)
REWARD = {"passed": 1.0, "crash": -4.0, "neutral": 0.0}


def detector_likelihood(accuracy=DEFAULT_ACCURACY) -> np.ndarray:
    """P(detector = o | obstacle = s, distance = d) -> array [o, s, d]."""
    L = np.zeros((N_LANES, N_LANES, D_MAX + 1))
    for d, acc in enumerate(accuracy):
        L[:, :, d] = (1 - acc) / (N_LANES - 1)
        np.fill_diagonal(L[:, :, d], acc)
    return L


class LaneObstacleEnv(gym.Env):
    """Generative process. Observation for model-free agents: one-hot(ego) | one-hot(detector) | one-hot(distance)."""

    metadata = {"render_modes": []}

    def __init__(self, accuracy=DEFAULT_ACCURACY, lane_change_cost: float = 0.0):
        self.accuracy = tuple(accuracy)
        self.lik = detector_likelihood(self.accuracy)
        self.lane_change_cost = lane_change_cost
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(2 * N_LANES + D_MAX + 1,), dtype=np.float32)

    # raw observation tuple, used by the active-inference agent
    def _raw_obs(self, outcome: int) -> list[int]:
        det = int(self.np_random.choice(N_LANES, p=self.lik[:, self.obstacle, self.distance]))
        return [self.ego, det, outcome]

    def _vec(self, raw: list[int]) -> np.ndarray:
        v = np.zeros(self.observation_space.shape, dtype=np.float32)
        v[raw[0]] = 1
        v[N_LANES + raw[1]] = 1
        v[2 * N_LANES + self.distance] = 1
        return v

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ego, self.distance = 1, D_MAX
        self.obstacle = int(self.np_random.integers(N_LANES))
        self.lane_changes = 0
        self.raw = self._raw_obs(0)
        return self._vec(self.raw), {"raw": self.raw}

    def step(self, action):
        new_ego = int(np.clip(self.ego + int(action) - 1, 0, N_LANES - 1))
        self.lane_changes += int(new_ego != self.ego)
        cost = self.lane_change_cost * (new_ego != self.ego)
        self.ego = new_ego
        self.distance -= 1
        outcome = 0 if self.distance > 0 else (2 if self.ego == self.obstacle else 1)
        self.raw = self._raw_obs(outcome)
        reward = REWARD[OUTCOMES[outcome]] - cost
        done = self.distance == 0
        info = {"raw": self.raw, "outcome": OUTCOMES[outcome], "lane_changes": self.lane_changes}
        return self._vec(self.raw), reward, done, False, info
