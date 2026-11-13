"""MetaDrive sim-to-sim transfer test: the highway-env policy (ONNX) + safety shield, no retraining.

What changes between the training simulator and this one:
  * vehicle dynamics: MetaDrive uses a Bullet rigid-body car driven by steering / throttle-brake commands,
    highway-env a kinematic bicycle with its own low-level controllers;
  * traffic: MetaDrive's IDM traffic with its own lane-change logic and spawn pattern;
  * road: 3.5 m lanes instead of 4 m, procedurally generated straight blocks.

What is kept: the policy observation contract (``safe_rl_drive.envs.spec``), rebuilt here from MetaDrive
ground truth, the 2 Hz meta-action interface and the shield. Lateral *positions* are expressed in
highway-env lane units (1 lane = 4 m), so the policy sees the lane geometry it was trained on; the
shield gets metric values and the real lane width.

    python sim2sim_demo/metadrive_demo.py --onnx models/rppo_highway.onnx --episodes 20 \
        --out results/metadrive
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from safe_rl_drive.envs.spec import ACTIONS, FEATURES, OBS_SHAPE, normalize
from safe_rl_drive.safety.observer import SAFETY_FEATS, SAFETY_HEADER, SAFETY_K
from safe_rl_drive.safety.shield import SafetyShield, ShieldConfig

TARGET_SPEEDS = np.array([10.0, 15.0, 20.0, 25.0, 30.0])  # highway-env training config
HW_LANE_WIDTH = 4.0
STEP_DT = 0.1  # MetaDrive: 0.02 s physics x decision_repeat 5
POLICY_EVERY = 5  # 2 Hz decisions, as in training
HORIZON_S = 40.0  # highway-env episode length


def make_env(
    n_episodes: int,
    start_seed: int,
    density: float,
    traffic_kmh: float,
    lanes: int,
    traffic_mode: str = "trigger",
):
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy

    # MetaDrive is tuned for urban driving: every vehicle type is capped at 80 km/h and IDM traffic
    # cruises at 30 km/h. The policy was trained with target speeds up to 30 m/s among ~20-25 m/s
    # traffic, so lift the cap and speed traffic up (traffic speed stays controlled by IDM).
    BaseVehicle.max_speed_km_h = property(lambda self: 130.0)
    BaseVehicle.max_speed_m_s = property(lambda self: 130.0 / 3.6)
    IDMPolicy.NORMAL_SPEED = traffic_kmh
    IDMPolicy.MAX_SPEED = max(IDMPolicy.MAX_SPEED, traffic_kmh + 20)
    return MetaDriveEnv(
        dict(
            map="S" * 24,  # ~1.4 km of straight blocks: 40 s at 30 m/s fits
            map_config={"lane_num": lanes, "lane_width": 3.5},
            random_lane_width=False,
            random_lane_num=False,
            random_agent_model=False,
            traffic_density=density,
            # "trigger" (MetaDrive default): traffic is spread along the whole road and starts driving when
            # the ego comes close. "respawn" only spawns at the start of the map, so a fast ego soon drives alone.
            traffic_mode=traffic_mode,
            num_scenarios=n_episodes,
            start_seed=start_seed,
            horizon=int(HORIZON_S / STEP_DT),
            use_render=False,
            log_level=50,
            vehicle_config={"spawn_velocity": [20.0, 0.0]},
        )
    )


def _others(env):
    """Every traffic vehicle that physically exists on the road this episode.

    * driving: ``traffic_manager.traffic_vehicles``;
    * parked, not yet triggered (trigger mode spawns each block's cars at reset and starts them when the
      ego enters the preceding block) - they are real obstacles, so they must be observed too.
    ``engine.get_objects()`` is *not* used: it also returns pooled vehicles left over from earlier episodes.
    """
    tm, ego = env.engine.traffic_manager, env.agent
    pending = [name for block in getattr(tm, "block_triggered_vehicles", []) for name in block.vehicles]
    parked = list(env.engine.get_objects(pending).values()) if pending else []
    return [v for v in [*tm.traffic_vehicles, *parked] if v is not ego]


class MetaDriveInterface:
    """Policy + safety observations from MetaDrive ground truth, in the frame of the leftmost lane."""

    def __init__(self, env):
        self.env = env

    def observations(self):
        ego = self.env.agent
        lanes = ego.navigation.current_ref_lanes
        ref, width = lanes[0], lanes[0].width

        def frame(pos):  # (longitudinal, lateral) along the leftmost lane; lateral grows with lane index
            return np.asarray(ref.local_coordinates(np.asarray(pos)), dtype=np.float64)

        def frame_vel(pos, vel):
            return frame(np.asarray(pos) + np.asarray(vel)) - frame(pos)

        e_pos, e_vel = frame(ego.position), frame_vel(ego.position, ego.velocity)
        ego_lane = int(np.clip(round(e_pos[1] / width), 0, len(lanes) - 1))
        ego_yaw = (ego.heading_theta - ref.heading_theta_at(e_pos[0]) + math.pi) % (2 * math.pi) - math.pi
        to_hw = HW_LANE_WIDTH / width

        raw = np.zeros(OBS_SHAPE, dtype=np.float32)
        # highway-env's ego row is absolute: x saturates at the clip range, y = lane index * 4 m
        raw[0] = (1.0, 100.0, e_pos[1] * to_hw, e_vel[0], e_vel[1], math.cos(ego_yaw), math.sin(ego_yaw))

        others = []
        for v in _others(self.env):
            p = frame(v.position)
            if not (-width < p[1] < len(lanes) * width):  # other carriageway / off the road
                continue
            dp = p - e_pos
            if np.hypot(*dp) > 100.0:
                continue
            dv = frame_vel(v.position, v.velocity) - e_vel
            yaw = v.heading_theta - ref.heading_theta_at(p[0])
            others.append((np.hypot(*dp), dp, dv, yaw))
        others.sort(key=lambda o: o[0])
        for i, (_, dp, dv, yaw) in enumerate(others[: OBS_SHAPE[0] - 1], start=1):
            raw[i] = (1.0, dp[0], dp[1] * to_hw, dv[0], dv[1], math.cos(yaw), math.sin(yaw))

        safety = np.zeros(SAFETY_HEADER + SAFETY_K * SAFETY_FEATS, dtype=np.float32)
        safety[:3] = (float(ego.speed), float(ego_lane > 0), float(ego_lane < len(lanes) - 1))
        for i, (_, dp, dv, _) in enumerate(others[:SAFETY_K]):
            b = SAFETY_HEADER + i * SAFETY_FEATS
            safety[b : b + 5] = (1.0, dp[0], dp[1], dv[0], dv[1])
        return normalize(raw), safety, len(lanes), ego_lane, e_pos[1], ego_yaw, width


class MetaActionController:
    """Meta-action -> (target lane, target speed) -> heading-tracking lateral + PI speed control on MetaDrive's
    normalised [steering, throttle_brake] (positive steering turns towards lower lane indices)."""

    # Lateral: lane-offset error -> desired heading (|yaw| <= 0.15 rad) -> heading tracking, gain reduced
    # with speed. A plain PD on the offset made the car slide at highway speed (back-to-back lane changes
    # ended off-road); this version is stable for single, double and zig-zag lane changes at 15-30 m/s.
    K_OFFSET, K_HEADING, MAX_YAW, MAX_STEER = 0.8, 2.0, 0.15, 0.3

    def __init__(self, speed: float):
        self.speed_idx = int(np.argmin(np.abs(TARGET_SPEEDS - speed)))
        self.target_lane: int | None = None
        self.integral = 0.0
        self.aeb, self.aeb_speed = False, 0.0

    def command(self, action: str, n_lanes: int, ego_lane: int, aeb: bool, leader_speed: float | None):
        # Same semantics as highway-env's ControlledVehicle: lane changes are relative to the current
        # *target* lane, IDLE / FASTER / SLOWER keep it (an ongoing lane change is not aborted).
        if self.target_lane is None:
            self.target_lane = ego_lane
        if action == "FASTER":
            self.speed_idx = min(self.speed_idx + 1, len(TARGET_SPEEDS) - 1)
        elif action == "SLOWER":
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif action == "LANE_LEFT":
            self.target_lane = max(self.target_lane - 1, 0)
        elif action == "LANE_RIGHT":
            self.target_lane = min(self.target_lane + 1, n_lanes - 1)
        self.aeb = aeb
        self.aeb_speed = max(0.0, (leader_speed or 0.0) - 3.0)

    def control(self, speed: float, lat: float, yaw: float, width: float) -> list[float]:
        target = round(lat / width) if self.target_lane is None else self.target_lane
        err = target * width - lat
        v = max(speed, 5.0)
        yaw_ref = -float(np.clip(np.arctan(self.K_OFFSET * err / v), -self.MAX_YAW, self.MAX_YAW))
        gain = self.K_HEADING * 15.0 / max(v, 15.0)
        steer = float(np.clip(gain * (yaw_ref - yaw), -self.MAX_STEER, self.MAX_STEER))
        v_ref = (
            min(self.aeb_speed, TARGET_SPEEDS[self.speed_idx]) if self.aeb else TARGET_SPEEDS[self.speed_idx]
        )
        e = v_ref - speed
        self.integral = float(np.clip(self.integral + e * STEP_DT, -10, 10))
        u = -1.0 if self.aeb and e < 0 else 0.3 * e + 0.05 * self.integral
        return [steer, float(np.clip(u, -1.0, 1.0))]


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 16 * len(lines) + 8), fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((8, 4 + 16 * i), t, fill=(255, 90, 90) if t.startswith("SHIELD") else (255, 255, 255))
    return np.asarray(img)


def draw_scene(env, n_lanes: int, width: float, intervened: bool, px: float = 6.0, size=(900, 260)):
    """Ego-following bird's-eye view drawn from ground truth (MetaDrive's own top-down renderer fits the
    whole map into one canvas, which shrinks a 1.4 km straight road to < 1 px per metre)."""
    from PIL import Image, ImageDraw

    ego = env.agent
    ref = ego.navigation.current_ref_lanes[0]
    e_s, _ = ref.local_coordinates(np.asarray(ego.position))
    road_mid = (n_lanes - 1) * width / 2

    def pix(s, lat):
        return ((s - e_s) * px + size[0] / 3, (lat - road_mid) * px + size[1] / 2 + 20)

    img = Image.new("RGB", size, (232, 232, 232))
    d = ImageDraw.Draw(img)
    top, bottom = pix(0, -width / 2)[1], pix(0, (n_lanes - 0.5) * width)[1]
    d.rectangle((0, top, size[0], bottom), fill=(90, 90, 90))
    for k in range(n_lanes + 1):
        y = pix(0, (k - 0.5) * width)[1]
        if k in (0, n_lanes):
            d.line((0, y, size[0], y), fill=(255, 255, 255), width=2)
        else:  # dashed, anchored to the road so the dashes move
            x0 = -((e_s * px) % 60)
            for x in np.arange(x0, size[0], 60):
                d.line((x, y, x + 30, y), fill=(230, 230, 230), width=1)

    vehicles = [*_others(env), ego]
    for v in vehicles:
        s, lat = ref.local_coordinates(np.asarray(v.position))
        if abs(s - e_s) > 160:
            continue
        yaw = v.heading_theta - ref.heading_theta_at(s)
        length, w = getattr(v, "LENGTH", 4.5), getattr(v, "WIDTH", 1.9)
        c, sn = np.cos(yaw), np.sin(yaw)
        corners = [(dl * length / 2, dw * w / 2) for dl, dw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        poly = [pix(s + a * c - b * sn, lat + a * sn + b * c) for a, b in corners]
        colour = ((230, 60, 60) if intervened else (40, 200, 90)) if v is ego else (70, 140, 230)
        d.polygon(poly, fill=colour, outline=(20, 20, 20))
    d.text(
        (8, size[1] - 16),
        "green = ego (red while the shield overrides), blue = MetaDrive IDM traffic",
        fill=(60, 60, 60),
    )
    return np.asarray(img)


class OnnxPolicy:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.meta = json.loads(Path(path).with_suffix(".json").read_text())
        assert self.meta["features"] == list(FEATURES), "ONNX model uses a different observation spec"
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.recurrent = bool(self.meta["recurrent"])
        self.reset()

    def reset(self):
        if self.recurrent:
            self.h = np.zeros((self.meta["lstm_layers"], 1, self.meta["lstm_hidden"]), np.float32)
            self.c = self.h.copy()

    def probs(self, obs: np.ndarray) -> np.ndarray:
        feeds = {"obs": obs[None].astype(np.float32)}
        if self.recurrent:
            feeds.update(h_in=self.h, c_in=self.c)
        out = self.sess.run(None, feeds)
        if self.recurrent:
            self.h, self.c = out[1], out[2]
        logits = out[0][0]
        p = np.exp(logits - logits.max())
        return p / p.sum()


def run_episode(env, seed: int, policy: OnnxPolicy, shield: SafetyShield | None, record: bool):
    env.reset(seed=seed)
    policy.reset()
    iface = MetaDriveInterface(env)
    ctrl = MetaActionController(env.agent.speed)
    frames, speeds, text = [], [], []
    decisions = interventions = aebs = 0
    pileup = False  # traffic vehicles crashed into each other within 60 m of the ego (MetaDrive physics)
    info: dict = {}
    for t in range(int(HORIZON_S / STEP_DT)):
        obs, safety, n_lanes, ego_lane, lat, yaw, width = iface.observations()
        if t % POLICY_EVERY == 0:
            p = policy.probs(obs)
            action = int(p.argmax())
            final, aeb, reason = action, False, ""
            if shield is not None:
                d = shield.filter(action, safety, p)
                final, aeb, reason = d.final, d.aeb, d.reason
                interventions += int(d.intervened)
                aebs += int(d.aeb)
            decisions += 1
            o = safety[SAFETY_HEADER:].reshape(-1, SAFETY_FEATS)
            ahead = o[(o[:, 0] > 0) & (o[:, 1] > 0) & (np.abs(o[:, 2]) < width / 2)]
            leader = float(safety[0] + ahead[np.argmin(ahead[:, 1]), 3]) if len(ahead) else None
            ctrl.command(ACTIONS[final], n_lanes, ego_lane, aeb, leader)
            text = [
                f"t={t * STEP_DT:4.1f}s  v={safety[0]:4.1f} m/s  lane={ego_lane}  "
                f"policy={ACTIONS[action]} (p={p[action]:.2f})  executed={ACTIONS[final]}",
                f"shield {'on' if shield else 'off'}: interventions {interventions}/{decisions}",
            ] + ([f"SHIELD: {reason}"] if reason else [])
        _, _, term, trunc, info = env.step(ctrl.control(float(env.agent.speed), lat, yaw, width))
        speeds.append(float(env.agent.speed))
        if not pileup and not (term and info.get("crash")):
            ego_xy = np.asarray(env.agent.position)
            pileup = any(
                getattr(v, "crash_vehicle", False) and np.hypot(*(np.asarray(v.position) - ego_xy)) < 60
                for v in env.engine.traffic_manager.traffic_vehicles
            )
        if record:
            frames.append(
                overlay(
                    draw_scene(env, n_lanes, width, intervened=bool(shield) and "SHIELD" in text[-1]), text
                )
            )
        if term or trunc:
            break
    collision = bool(info.get("crash_vehicle") or info.get("crash"))
    offroad = bool(info.get("out_of_road"))
    return {
        "seed": seed,
        "success": float(not collision and not offroad),
        "collision": float(collision),
        "collision_after_traffic_pileup": float(collision and pileup),
        "offroad": float(offroad),
        "mean_speed": float(np.mean(speeds)),
        "seconds": len(speeds) * STEP_DT,
        "decisions": decisions,
        "shield_rate": interventions / max(decisions, 1),
        "aeb": aebs,
    }, frames


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--density", type=float, default=0.3, help="MetaDrive traffic_density")
    p.add_argument("--traffic-kmh", type=float, default=80.0, help="IDM cruise speed of traffic")
    p.add_argument("--lanes", type=int, default=3)
    p.add_argument("--traffic-mode", choices=["trigger", "respawn", "hybrid"], default="trigger")
    p.add_argument("--shield", choices=["off", "on", "both"], default="both")
    p.add_argument(
        "--video-episode", type=int, default=0, help="episode index recorded (shield on); -1 = none"
    )
    p.add_argument("--out", type=Path, default=Path("results/metadrive"))
    a = p.parse_args(argv)

    import imageio.v2 as imageio

    a.out.mkdir(parents=True, exist_ok=True)
    policy = OnnxPolicy(a.onnx)
    env = make_env(a.episodes, a.seed, a.density, a.traffic_kmh, a.lanes, a.traffic_mode)
    modes = ["off", "on"] if a.shield == "both" else [a.shield]
    rows = []
    try:
        for mode in modes:
            shield = SafetyShield(ShieldConfig(lane_width=3.5)) if mode == "on" else None
            for i in range(a.episodes):
                record = i == a.video_episode and mode == modes[-1]
                ep, frames = run_episode(env, a.seed + i, policy, shield, record)
                ep["shield"] = mode
                rows.append(ep)
                print(
                    f"shield={mode} ep={i:2d} success={ep['success']:.0f} collision={ep['collision']:.0f} "
                    f"pileup={ep['collision_after_traffic_pileup']:.0f} offroad={ep['offroad']:.0f} v={ep['mean_speed']:.1f} t={ep['seconds']:.1f}s "
                    f"shield_rate={ep['shield_rate']:.2f}",
                    flush=True,
                )
                if frames:
                    imageio.mimsave(
                        a.out / "metadrive_demo.mp4", frames, fps=int(1 / STEP_DT), macro_block_size=1
                    )
                    imageio.mimsave(a.out / "metadrive_demo.gif", frames[::2], duration=2 * STEP_DT, loop=0)
    finally:
        env.close()

    summary = {
        "onnx": a.onnx,
        "episodes": a.episodes,
        "density": a.density,
        "traffic_kmh": a.traffic_kmh,
        "traffic_mode": a.traffic_mode,
    }
    for mode in modes:
        eps = [r for r in rows if r["shield"] == mode]
        summary[f"shield_{mode}"] = {
            k: float(np.mean([e[k] for e in eps]))
            for k in (
                "success",
                "collision",
                "collision_after_traffic_pileup",
                "offroad",
                "mean_speed",
                "shield_rate",
            )
        }
    (a.out / "episodes.json").write_text(json.dumps(rows, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
