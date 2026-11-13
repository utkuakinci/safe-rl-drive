"""Environment factory used by training, evaluation, the explainer and the ROS2 bridge."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway-env environments)

from safe_rl_drive.envs.scenarios import scenario_config
from safe_rl_drive.envs.wrappers import (
    ActionDelayWrapper,
    DrivingMetricsWrapper,
    MaskFeaturesWrapper,
    ObservationNoiseWrapper,
)


@dataclass
class EnvCondition:
    """A (possibly perturbed) evaluation condition."""

    name: str = "nominal"
    scenario: str = "highway"
    env_overrides: dict = field(default_factory=dict)
    obs_noise_std: float = 0.0
    obs_dropout: float = 0.0
    action_delay: int = 0
    mask_features: tuple[str, ...] = ()


def make_env(
    scenario: str = "highway",
    seed: int | None = None,
    env_overrides: dict | None = None,
    obs_noise_std: float = 0.0,
    obs_dropout: float = 0.0,
    action_delay: int = 0,
    mask_features: tuple[str, ...] | list[str] = (),
    render_mode: str | None = None,
    shield=None,
) -> gym.Env:
    """Build a wrapped highway-env environment.

    Wrapper order (outermost first): delay -> shield -> noise -> mask -> metrics -> highway-env.
    The shield sits *inside* the delay, i.e. directly in front of actuation, so it vets the action that
    is actually executed. Noise / masking only corrupt the policy observation; the shield reads its
    own safety channel from the simulator.
    """
    gym_id, cfg = scenario_config(scenario, env_overrides)
    env = gym.make(gym_id, config=cfg, render_mode=render_mode)
    env = DrivingMetricsWrapper(env)
    if mask_features:
        env = MaskFeaturesWrapper(env, mask_features)
    if obs_noise_std > 0 or obs_dropout > 0:
        env = ObservationNoiseWrapper(env, std=obs_noise_std, dropout=obs_dropout, seed=seed)
    if shield is not None:
        from safe_rl_drive.safety.wrapper import ShieldWrapper

        env = ShieldWrapper(env, shield)
    if action_delay > 0:
        env = ActionDelayWrapper(env, delay=action_delay)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_condition_env(cond: EnvCondition, seed: int | None = None, **kwargs) -> gym.Env:
    return make_env(
        scenario=cond.scenario,
        seed=seed,
        env_overrides=cond.env_overrides,
        obs_noise_std=cond.obs_noise_std,
        obs_dropout=cond.obs_dropout,
        action_delay=cond.action_delay,
        mask_features=cond.mask_features,
        **kwargs,
    )


def find_wrapper(env: gym.Env, cls) -> gym.Env | None:
    """Return the first wrapper of type ``cls`` in the wrapper chain (or None)."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, cls):
            return env
        env = env.env
    return None
