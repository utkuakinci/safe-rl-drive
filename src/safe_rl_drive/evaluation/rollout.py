"""Episode rollouts with optional shield, used by training callbacks and offline evaluation."""

from __future__ import annotations

import time

import numpy as np

from safe_rl_drive.envs.factory import EnvCondition, find_wrapper, make_condition_env
from safe_rl_drive.policy import PolicyRunner
from safe_rl_drive.safety.shield import SafetyShield, ShieldConfig
from safe_rl_drive.safety.wrapper import ShieldWrapper


def run_episodes(
    model,
    cond: EnvCondition,
    n_episodes: int,
    seed: int = 10_000,
    shield_config: ShieldConfig | dict | None = None,
    deterministic: bool = True,
) -> tuple[list[dict], dict]:
    """Roll out ``n_episodes``. Returns ``(episode_metrics, shield_stats)``.

    Evaluation seeds start at 10_000 so they never overlap with training seeds.
    """
    shield = None
    if shield_config is not None:
        cfg = (
            shield_config
            if isinstance(shield_config, ShieldConfig)
            else ShieldConfig.from_dict(shield_config)
        )
        shield = SafetyShield(cfg)
    env = make_condition_env(cond, seed=seed, shield=shield)
    shield_env = find_wrapper(env, ShieldWrapper)
    runner = PolicyRunner(model, deterministic=deterministic, seed=seed)
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        runner.reset()
        done, t_policy, n = False, 0.0, 0
        info: dict = {}
        while not done:
            t0 = time.perf_counter()
            action, probs = runner.act(obs)
            t_policy += time.perf_counter() - t0
            n += 1
            if shield_env is not None:
                shield_env.set_action_probs(probs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        m = dict(info.get("episode_metrics", {}))
        m.setdefault("shield_interventions", 0.0)
        m.setdefault("shield_rate", 0.0)
        m["episode"] = ep
        m["policy_ms"] = 1000.0 * t_policy / max(n, 1)
        episodes.append(m)
    env.close()
    return episodes, (shield.stats.to_dict() if shield else {})


def summarize(episodes: list[dict]) -> dict:
    keys = [k for k in episodes[0] if k != "episode"] if episodes else []
    return {k: float(np.mean([e[k] for e in episodes])) for k in keys}
