"""Rule-based safety shield placed in front of the learned policy.

The shield is deliberately simple, deterministic and auditable (no learned components), so that
its behaviour can be specified, tested and argued about in a safety case. The C++ port in
``ros2_ws/src/rl_policy_node/src/safety_shield.cpp`` implements exactly the same rules and is
covered by the same test vectors.

Rules (each marks a set of meta-actions as unsafe for the current step):

* ``front_ttc``       - time-to-collision to the leader in the ego corridor below ``ttc_threshold``
                        -> FASTER and IDLE unsafe.
* ``front_gap``       - bumper gap below ``standstill_gap + time_headway * v_ego`` -> FASTER unsafe;
                        below ``standstill_gap`` -> IDLE unsafe as well.
* ``crossing_conflict`` - constant-velocity closest point of approach with a vehicle ahead but
                        outside the corridor (cross / merging traffic) closer than ``cpa_radius``
                        within ``cpa_horizon`` -> FASTER and IDLE unsafe.
* ``lc_gap`` / ``lc_ttc`` - lane change into a target lane with insufficient gap or a closing
                        vehicle (front or rear) -> that lane change unsafe.
* A lane change with no target lane degenerates to IDLE in highway-env, so it inherits IDLE's status.

If the policy's action is unsafe, the shield picks the most probable safe action (if action
probabilities are given) or the first safe action from a fixed preference order. SLOWER is the
always-available fallback (minimal-risk manoeuvre). If the leader TTC drops below ``aeb_ttc`` the
shield additionally requests an emergency brake (AEB), which the actuation layer executes outside
the discrete meta-action space (in simulation: the ego target speed is cut to below the leader's).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from safe_rl_drive.envs.spec import ACTION_INDEX, ACTIONS
from safe_rl_drive.safety.observer import SAFETY_FEATS, SAFETY_HEADER

INF = float("inf")


@dataclass
class ShieldConfig:
    enabled: bool = True
    ttc_threshold: float = 2.5  # s
    standstill_gap: float = 4.0  # m, bumper to bumper
    time_headway: float = 0.6  # s
    vehicle_length: float = 5.0  # m
    lane_width: float = 4.0  # m
    lc_front_gap: float = 5.0  # m, required free space in target lane ahead
    lc_rear_gap: float = 5.0  # m, required free space in target lane behind
    lc_ttc: float = 2.0  # s
    cpa_radius: float = 3.0  # m
    cpa_horizon: float = 2.5  # s
    aeb_ttc: float = 1.5  # s, below this the minimal-risk manoeuvre becomes an emergency brake
    fallback_order: tuple[str, ...] = ("IDLE", "SLOWER", "LANE_RIGHT", "LANE_LEFT", "FASTER")

    @classmethod
    def from_dict(cls, d: dict | None) -> ShieldConfig:
        d = dict(d or {})
        if "fallback_order" in d:
            d["fallback_order"] = tuple(d["fallback_order"])
        return cls(**d)


@dataclass
class ShieldDecision:
    original: int
    final: int
    intervened: bool
    unsafe: dict[str, list[str]] = field(default_factory=dict)  # action name -> violated rules
    front_ttc: float = INF
    front_gap: float = INF
    aeb: bool = False  # emergency brake requested (beyond the meta-action space)

    @property
    def reason(self) -> str:
        if not self.intervened:
            return ""
        rules = ",".join(self.unsafe.get(ACTIONS[self.original], []))
        tag = " +AEB" if self.aeb else ""
        return f"{ACTIONS[self.original]}->{ACTIONS[self.final]} ({rules}){tag}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reason"] = self.reason
        return d


class SafetyShield:
    def __init__(self, config: ShieldConfig | None = None, actions: dict[int, str] | None = None):
        self.cfg = config or ShieldConfig()
        self.actions = dict(actions or ACTIONS)
        self.index = {v: k for k, v in self.actions.items()}
        self.stats = ShieldStats()

    # ------------------------------------------------------------------ analysis
    def assess(self, safety_obs: np.ndarray) -> tuple[dict[str, list[str]], float, float]:
        """Return ``(unsafe_actions -> rules, front_ttc, front_gap)`` for one safety observation."""
        c = self.cfg
        x = np.asarray(safety_obs, dtype=np.float64)
        ego_speed, left_ok, right_ok = x[0], x[1] > 0.5, x[2] > 0.5
        others = x[SAFETY_HEADER:].reshape(-1, SAFETY_FEATS)
        half_w = c.lane_width / 2.0
        unsafe: dict[str, set[str]] = {}

        def mark(action: str, rule: str):
            unsafe.setdefault(action, set()).add(rule)

        front_ttc, front_gap = INF, INF
        for presence, s, d, vs, vd in others:
            if presence < 0.5:
                continue
            gap = abs(s) - c.vehicle_length
            # --- ego corridor: leader ---
            if abs(d) < half_w and s > 0:
                closing = -vs
                ttc = gap / closing if closing > 1e-6 else INF
                front_ttc, front_gap = min(front_ttc, max(ttc, 0.0)), min(front_gap, gap)
            # --- crossing / merging traffic ahead ---
            elif s > 0:
                v2 = vs * vs + vd * vd
                t_star = 0.0 if v2 < 1e-9 else min(max(-(s * vs + d * vd) / v2, 0.0), c.cpa_horizon)
                d_min = math.hypot(s + vs * t_star, d + vd * t_star)
                if d_min < c.cpa_radius and t_star < c.cpa_horizon:
                    mark("FASTER", "crossing_conflict")
                    mark("IDLE", "crossing_conflict")
            # --- target-lane checks for lane changes ---
            for side, action in ((-1.0, "LANE_LEFT"), (1.0, "LANE_RIGHT")):
                if not (half_w <= side * d < 3 * half_w):
                    continue
                if -(c.vehicle_length + c.lc_rear_gap) < s < (c.vehicle_length + c.lc_front_gap):
                    mark(action, "lc_gap")
                closing = -vs if s > 0 else vs  # leader approaching us / follower approaching us
                if closing > 1e-6 and gap / closing < c.lc_ttc:
                    mark(action, "lc_ttc")

        if front_ttc < c.ttc_threshold:
            mark("FASTER", "front_ttc")
            mark("IDLE", "front_ttc")
        if front_gap < c.standstill_gap + c.time_headway * ego_speed:
            mark("FASTER", "front_gap")
        if front_gap < c.standstill_gap:
            mark("IDLE", "front_gap")
        # A lane change without a target lane behaves like IDLE.
        for ok, action in ((left_ok, "LANE_LEFT"), (right_ok, "LANE_RIGHT")):
            if not ok and "IDLE" in unsafe:
                for rule in unsafe["IDLE"]:
                    mark(action, rule)
        unsafe.pop("SLOWER", None)  # minimal-risk manoeuvre is never blocked
        return {k: sorted(v) for k, v in unsafe.items()}, front_ttc, front_gap

    # ------------------------------------------------------------------ filtering
    def filter(
        self, action: int, safety_obs: np.ndarray, action_probs: np.ndarray | None = None
    ) -> ShieldDecision:
        action = int(action)
        if not self.cfg.enabled:
            dec = ShieldDecision(action, action, False)
            self.stats.update(dec)
            return dec
        unsafe, ttc, gap = self.assess(safety_obs)
        final = action
        if self.actions[action] in unsafe:
            if action_probs is not None:
                order = [self.actions[int(i)] for i in np.argsort(-np.asarray(action_probs), kind="stable")]
            else:
                order = list(self.cfg.fallback_order)
            final = next(
                (self.index[a] for a in order if a in self.index and a not in unsafe),
                self.index.get("SLOWER", ACTION_INDEX["SLOWER"]),
            )
        aeb = ttc < self.cfg.aeb_ttc
        if aeb:
            final = self.index.get("SLOWER", ACTION_INDEX["SLOWER"])
        dec = ShieldDecision(action, final, final != action or aeb, unsafe, ttc, gap, aeb)
        self.stats.update(dec)
        return dec


class ShieldStats:
    """Intervention bookkeeping (logged to W&B and the evaluation tables)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.interventions = 0
        self.by_rule: Counter = Counter()
        self.by_transition: Counter = Counter()
        self.aeb = 0

    def update(self, dec: ShieldDecision):
        self.steps += 1
        if dec.intervened:
            self.interventions += 1
            for rule in dec.unsafe.get(ACTIONS.get(dec.original, str(dec.original)), []):
                self.by_rule[rule] += 1
            self.by_transition[f"{ACTIONS[dec.original]}->{ACTIONS[dec.final]}"] += 1
            self.aeb += int(dec.aeb)

    @property
    def rate(self) -> float:
        return self.interventions / max(self.steps, 1)

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "interventions": self.interventions,
            "intervention_rate": self.rate,
            "aeb": self.aeb,
            **{f"rule/{k}": v for k, v in self.by_rule.items()},
            **{f"transition/{k}": v for k, v in self.by_transition.items()},
        }
