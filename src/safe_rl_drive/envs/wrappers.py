"""Gymnasium wrappers for metrics, sensor corruption and actuation latency."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class DrivingMetricsWrapper(gym.Wrapper):
    """Collects per-episode driving KPIs and puts them into ``info["episode_metrics"]``.

    KPIs: success, collision, off-road, mean speed, RMS / max jerk (comfort), lane changes.
    Jerk is the finite difference of the planar acceleration vector, sampled at policy frequency.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_buffers()

    def _reset_buffers(self):
        self._speeds: list[float] = []
        self._vel: list[np.ndarray] = []
        self._lane_changes = 0
        self._last_lane = None
        self._return = 0.0
        self._length = 0

    @property
    def dt(self) -> float:
        return 1.0 / float(self.env.unwrapped.config.get("policy_frequency", 1))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_buffers()
        self._record()
        return obs, info

    def _record(self):
        ego = self.env.unwrapped.vehicle
        self._speeds.append(float(ego.speed))
        self._vel.append(np.asarray(ego.velocity, dtype=np.float64).copy())
        lane = getattr(ego, "lane_index", None)
        lane_id = lane[2] if lane is not None else None
        if self._last_lane is not None and lane_id is not None and lane_id != self._last_lane:
            self._lane_changes += 1
        self._last_lane = lane_id

    def _success(self, crashed: bool) -> bool:
        base = self.env.unwrapped
        ego = base.vehicle
        if crashed or not ego.on_road:
            return False
        if hasattr(base, "has_arrived"):  # intersection / goal-directed scenarios
            try:
                return bool(base.has_arrived(ego))
            except TypeError:
                return bool(base.has_arrived(ego, exit_distance=25))
        return True  # survival scenarios: reaching the time limit without incident

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record()
        self._return += float(reward)
        self._length += 1
        if terminated or truncated:
            info["episode_metrics"] = self.summary(info)
        return obs, reward, terminated, truncated, info

    def summary(self, info: dict | None = None) -> dict:
        ego = self.env.unwrapped.vehicle
        crashed = bool(ego.crashed or (info or {}).get("crashed", False))
        vel = np.stack(self._vel) if len(self._vel) > 0 else np.zeros((1, 2))
        acc = np.diff(vel, axis=0) / self.dt
        jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1) / self.dt if len(acc) > 1 else np.zeros(1)
        return {
            "success": float(self._success(crashed)),
            "collision": float(crashed),
            "offroad": float(not ego.on_road),
            "mean_speed": float(np.mean(self._speeds)),
            "jerk_rms": float(np.sqrt(np.mean(jerk**2))),
            "jerk_max": float(np.max(jerk)),
            "lane_changes": float(self._lane_changes),
            "return": self._return,
            "length": float(self._length),
        }


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Sensor corruption on the normalised Kinematics observation.

    * additive Gaussian noise (std in normalised units) on all features of *other* present vehicles
      and on the ego speed;
    * per-vehicle dropout: a detected vehicle disappears from the observation (missed detection).
    """

    def __init__(self, env: gym.Env, std: float = 0.0, dropout: float = 0.0, seed: int | None = None):
        super().__init__(env)
        self.std = float(std)
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32, copy=True)
        others = obs[1:]
        present = others[:, 0] > 0.5
        if self.std > 0:
            noise = self.rng.normal(0.0, self.std, size=others[:, 1:].shape).astype(np.float32)
            others[:, 1:] += noise * present[:, None]
            obs[0, 3:5] += self.rng.normal(0.0, self.std, size=2).astype(np.float32)
        if self.dropout > 0:
            drop = present & (self.rng.random(len(others)) < self.dropout)
            others[drop] = 0.0
        obs[1:] = others
        return np.clip(obs, -1.0, 1.0)


class ActionDelayWrapper(gym.Wrapper):
    """Applies the action chosen ``delay`` policy steps ago (actuation / compute latency)."""

    def __init__(self, env: gym.Env, delay: int = 1, noop_action: int = 1):
        super().__init__(env)
        self.delay = int(delay)
        self.noop = noop_action
        self._queue: deque = deque()

    def reset(self, **kwargs):
        self._queue = deque([self.noop] * self.delay)
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.delay == 0:
            return self.env.step(action)
        self._queue.append(action)
        return self.env.step(self._queue.popleft())


class MaskFeaturesWrapper(gym.ObservationWrapper):
    """Zeroes selected features (e.g. ``vx, vy``) to create a partially observable variant.

    The observation *shape* is unchanged, so the same export / ROS2 interface is used. Without
    velocities, a feed-forward policy cannot infer closing speeds from one frame - a recurrent
    policy can, by integrating positions over time (the PPO vs RecurrentPPO comparison).
    """

    def __init__(self, env: gym.Env, features: list[str] | tuple[str, ...]):
        super().__init__(env)
        from safe_rl_drive.envs.spec import FEATURES

        self.cols = [FEATURES.index(f) for f in features]

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32, copy=True)
        obs[:, self.cols] = 0.0
        return obs
