"""Gym wrapper that routes every action through the safety shield."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from safe_rl_drive.safety.observer import safety_observation
from safe_rl_drive.safety.shield import SafetyShield


def apply_emergency_brake(base_env, safety_obs: np.ndarray, margin: float = 3.0):
    """Cut the ego target speed below the closest in-lane leader's speed (AEB actuation in sim)."""
    from safe_rl_drive.safety.observer import SAFETY_FEATS, SAFETY_HEADER

    ego = base_env.vehicle
    others = np.asarray(safety_obs[SAFETY_HEADER:]).reshape(-1, SAFETY_FEATS)
    ahead = others[(others[:, 0] > 0.5) & (others[:, 1] > 0) & (np.abs(others[:, 2]) < 2.0)]
    leader_speed = float(ego.speed + ahead[np.argmin(ahead[:, 1]), 3]) if len(ahead) else 0.0
    target = max(0.0, leader_speed - margin)
    if hasattr(ego, "target_speed"):
        ego.target_speed = min(float(ego.target_speed), target)


class ShieldWrapper(gym.Wrapper):
    """Filters actions with a :class:`SafetyShield`.

    * ``info["shield"]`` carries the decision of the last step (original/final action, rules).
    * ``info["episode_metrics"]`` (from :class:`DrivingMetricsWrapper`) is extended with the number
      of interventions in the episode.
    * Call :meth:`set_action_probs` before ``step`` to let the shield fall back to the most probable
      safe action instead of the fixed preference order.
    """

    def __init__(self, env: gym.Env, shield: SafetyShield):
        super().__init__(env)
        self.shield = shield
        self._probs: np.ndarray | None = None
        self._episode_interventions = 0
        self._episode_steps = 0

    def set_action_probs(self, probs: np.ndarray | None):
        self._probs = probs

    def reset(self, **kwargs):
        self._episode_interventions = 0
        self._episode_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        safety_obs = safety_observation(self.env.unwrapped)
        dec = self.shield.filter(int(action), safety_obs, self._probs)
        self._probs = None
        env_action = dec.final
        if dec.aeb:
            # SLOWER would re-derive the target from the discrete speed set; instead cut the target
            # speed directly and send IDLE so the low-level controller tracks the AEB target.
            apply_emergency_brake(self.env.unwrapped, safety_obs)
            env_action = self.shield.index.get("IDLE", 1)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self._episode_steps += 1
        self._episode_interventions += int(dec.intervened)
        info["shield"] = dec.to_dict()
        if "episode_metrics" in info:
            info["episode_metrics"]["shield_interventions"] = float(self._episode_interventions)
            info["episode_metrics"]["shield_rate"] = self._episode_interventions / max(self._episode_steps, 1)
        return obs, reward, terminated, truncated, info
