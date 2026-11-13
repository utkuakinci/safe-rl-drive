"""CARLA demo: drive the exported ONNX policy (+ safety shield) on the Town04 highway and record video.

The policy was trained in highway-env and is *not* retrained here: this script only re-creates the
observation interface (see ``safe_rl_drive.envs.spec``) from CARLA ground truth and maps the
discrete meta-actions to a lane / speed controller. That makes it an honest sim-to-sim transfer test.

Requires a running CARLA 0.9.15 server (``./CarlaUE4.sh -RenderOffScreen``) and ``pip install carla==0.9.15``.

    python carla_demo/run_demo.py --onnx models/rppo_highway.onnx --out results/carla_demo.mp4 \
        --seconds 90 --npc 60 --shield
"""

from __future__ import annotations

import argparse
import json
import math
import queue
from pathlib import Path

import numpy as np

from safe_rl_drive.envs.spec import ACTIONS, FEATURES, OBS_SHAPE, normalize
from safe_rl_drive.safety.observer import SAFETY_FEATS, SAFETY_HEADER, SAFETY_K
from safe_rl_drive.safety.shield import SafetyShield

TARGET_SPEEDS = np.array([10.0, 15.0, 20.0, 25.0, 30.0])  # same as highway-env training config
SIM_DT = 0.05  # 20 Hz physics
POLICY_EVERY = 10  # 2 Hz decisions, as in training
LANE_WIDTH = 4.0


# ------------------------------------------------------------------------------------------------
# geometry helpers
# ------------------------------------------------------------------------------------------------
def to_frame(vec_xy, yaw):
    """World -> frame rotated by yaw (CARLA: x forward, y right, left-handed; matches highway-env y-down)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vec_xy[0] + s * vec_xy[1], -s * vec_xy[0] + c * vec_xy[1]])


def same_direction_lanes(wp):
    """Driving lanes in the ego's direction, ordered left -> right; returns (lanes, ego_index)."""
    import carla

    left, right = [], []
    w = wp.get_left_lane()
    while w is not None and w.lane_type == carla.LaneType.Driving and w.lane_id * wp.lane_id > 0:
        left.insert(0, w)
        w = w.get_left_lane()
    w = wp.get_right_lane()
    while w is not None and w.lane_type == carla.LaneType.Driving and w.lane_id * wp.lane_id > 0:
        right.append(w)
        w = w.get_right_lane()
    return left + [wp] + right, len(left)


class CarlaInterface:
    """Builds policy + safety observations from CARLA ground truth."""

    def __init__(self, world, ego):
        self.world, self.ego, self.map = world, ego, world.get_map()

    def _ego_frame(self):
        wp = self.map.get_waypoint(self.ego.get_location())
        lane_yaw = math.radians(wp.transform.rotation.yaw)  # lane direction = highway-env x axis
        return wp, lane_yaw

    def observations(self):
        wp, lane_yaw = self._ego_frame()
        lanes, ego_lane = same_direction_lanes(wp)
        ego_loc, ego_vel = self.ego.get_location(), self.ego.get_velocity()
        ego_v = to_frame((ego_vel.x, ego_vel.y), lane_yaw)
        ego_yaw = math.radians(self.ego.get_transform().rotation.yaw) - lane_yaw
        lat_in_lane = to_frame(
            (ego_loc.x - wp.transform.location.x, ego_loc.y - wp.transform.location.y), lane_yaw
        )[1]

        # Ego row: highway-env uses absolute coordinates for the ego. x is always clipped to +1 after the
        # first ~100 m, y encodes the lane (lane_index * 4 m) -> reproduce exactly that.
        raw = np.zeros(OBS_SHAPE, dtype=np.float32)
        raw[0] = (
            1.0,
            100.0,
            ego_lane * LANE_WIDTH + lat_in_lane,
            ego_v[0],
            ego_v[1],
            math.cos(ego_yaw),
            math.sin(ego_yaw),
        )

        others = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.ego.id:
                continue
            loc, vel = actor.get_location(), actor.get_velocity()
            dp = to_frame((loc.x - ego_loc.x, loc.y - ego_loc.y), lane_yaw)
            if np.hypot(*dp) > 100.0:
                continue
            dv = to_frame((vel.x - ego_vel.x, vel.y - ego_vel.y), lane_yaw)
            yaw = math.radians(actor.get_transform().rotation.yaw) - lane_yaw
            others.append((np.hypot(*dp), dp, dv, yaw))
        others.sort(key=lambda o: o[0])
        for i, (_, dp, dv, yaw) in enumerate(others[: OBS_SHAPE[0] - 1], start=1):
            raw[i] = (1.0, dp[0], dp[1], dv[0], dv[1], math.cos(yaw), math.sin(yaw))
        policy_obs = normalize(raw)

        # Safety channel: ego-frame metric values (rotation by ego heading ~ lane heading on highway).
        safety = np.zeros(SAFETY_HEADER + SAFETY_K * SAFETY_FEATS, dtype=np.float32)
        safety[0] = float(np.hypot(ego_vel.x, ego_vel.y))
        safety[1] = float(ego_lane > 0)
        safety[2] = float(ego_lane < len(lanes) - 1)
        for i, (_, dp, dv, _) in enumerate(others[:SAFETY_K]):
            b = SAFETY_HEADER + i * SAFETY_FEATS
            safety[b : b + 5] = (1.0, dp[0], dp[1], dv[0], dv[1])
        return policy_obs, safety, lanes, ego_lane


class MetaActionController:
    """Discrete meta-action -> (target lane, target speed) -> pure pursuit + PI speed control."""

    def __init__(self, ego, world):
        self.ego, self.map = ego, world.get_map()
        self.speed_idx = 2
        self.target_lane_wp = None
        self.integral = 0.0
        self.aeb = False

    def command(self, action: str, lanes, ego_lane, aeb: bool, leader_speed: float | None):
        self.aeb = aeb
        if action == "FASTER":
            self.speed_idx = min(self.speed_idx + 1, len(TARGET_SPEEDS) - 1)
        elif action == "SLOWER":
            self.speed_idx = max(self.speed_idx - 1, 0)
        target = ego_lane
        if action == "LANE_LEFT":
            target = max(ego_lane - 1, 0)
        elif action == "LANE_RIGHT":
            target = min(ego_lane + 1, len(lanes) - 1)
        self.target_lane_wp = lanes[target]
        self.aeb_speed = max(0.0, (leader_speed or 0.0) - 3.0)

    def control(self):
        import carla

        tf = self.ego.get_transform()
        v = self.ego.get_velocity()
        speed = math.hypot(v.x, v.y)
        wp = self.target_lane_wp or self.map.get_waypoint(tf.location)
        lookahead = max(8.0, 0.8 * speed)
        tgt = wp.next(lookahead)[0].transform.location
        yaw = math.radians(tf.rotation.yaw)
        local = to_frame((tgt.x - tf.location.x, tgt.y - tf.location.y), yaw)
        curvature = 2.0 * local[1] / max(local[0] ** 2 + local[1] ** 2, 1e-3)
        steer = float(np.clip(math.atan(2.85 * curvature) / 0.7, -1, 1))  # 2.85 m wheelbase, 40 deg max

        v_ref = self.aeb_speed if self.aeb else TARGET_SPEEDS[self.speed_idx]
        err = v_ref - speed
        self.integral = float(np.clip(self.integral + err * SIM_DT, -10, 10))
        u = 0.25 * err + 0.02 * self.integral
        if self.aeb:
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer)
        return carla.VehicleControl(
            throttle=float(np.clip(u, 0, 1)), brake=float(np.clip(-u, 0, 1)), steer=steer
        )


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 22 * len(lines) + 10), fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((10, 6 + 22 * i), t, fill=(255, 80, 80) if t.startswith("SHIELD") else (255, 255, 255))
    return np.asarray(img)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town04")
    p.add_argument("--npc", type=int, default=60)
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument("--shield", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/carla_demo.mp4")
    a = p.parse_args(argv)

    import carla
    import imageio.v2 as imageio
    import onnxruntime as ort

    meta = json.loads(Path(a.onnx).with_suffix(".json").read_text())
    assert meta["features"] == list(FEATURES), "ONNX model was exported with a different observation spec"
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    shield = SafetyShield() if a.shield else None
    rng = np.random.default_rng(a.seed)

    client = carla.Client(a.host, a.port)
    client.set_timeout(30.0)
    world = client.load_world(a.town)
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, SIM_DT
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    # Town04 highway spawn points: pick the first one on a multi-lane road
    ego_tf = next(
        s for s in spawns if len(same_direction_lanes(world.get_map().get_waypoint(s.location))[0]) >= 3
    )
    ego = world.spawn_actor(bp.find("vehicle.tesla.model3"), ego_tf)
    actors = [ego]
    npc_bps = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    for s in spawns[: a.npc]:
        npc = world.try_spawn_actor(npc_bps[int(rng.integers(len(npc_bps)))], s)
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            tm.vehicle_percentage_speed_difference(npc, float(rng.uniform(-20, 30)))
            actors.append(npc)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam = world.spawn_actor(
        cam_bp, carla.Transform(carla.Location(x=-8, z=4), carla.Rotation(pitch=-12)), attach_to=ego
    )
    actors.append(cam)
    frames_q: queue.Queue = queue.Queue()
    cam.listen(frames_q.put)

    collided = []
    col_sensor = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    col_sensor.listen(lambda e: collided.append(e.other_actor.type_id))
    actors.append(col_sensor)

    iface, ctrl = CarlaInterface(world, ego), MetaActionController(ego, world)
    recurrent = meta["recurrent"]
    if recurrent:
        h = np.zeros((meta["lstm_layers"], 1, meta["lstm_hidden"]), np.float32)
        c = h.copy()
    writer = imageio.get_writer(a.out, fps=int(1 / SIM_DT), macro_block_size=1)
    text, interventions, decisions = ["warming up"], 0, 0
    try:
        for tick in range(int(a.seconds / SIM_DT)):
            if tick % POLICY_EVERY == 0:
                obs, safety, lanes, ego_lane = iface.observations()
                feeds = {"obs": obs[None]}
                if recurrent:
                    feeds.update(h_in=h, c_in=c)
                out = sess.run(None, feeds)
                if recurrent:
                    h, c = out[1], out[2]
                logits = out[0][0]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                action = int(probs.argmax())
                final, aeb, reason = action, False, ""
                if shield is not None:
                    d = shield.filter(action, safety, probs)
                    final, aeb, reason = d.final, d.aeb, d.reason
                    interventions += int(d.intervened)
                decisions += 1
                others = safety[SAFETY_HEADER:].reshape(-1, SAFETY_FEATS)
                ahead = others[(others[:, 0] > 0) & (others[:, 1] > 0) & (np.abs(others[:, 2]) < 2)]
                leader = float(safety[0] + ahead[np.argmin(ahead[:, 1]), 3]) if len(ahead) else None
                ctrl.command(ACTIONS[final], lanes, ego_lane, aeb, leader)
                text = [
                    f"t={tick * SIM_DT:5.1f}s  v={safety[0]:4.1f} m/s  lane={ego_lane}  policy={ACTIONS[action]} (p={probs[action]:.2f})",
                    f"executed={ACTIONS[final]}  shield interventions={interventions}/{decisions}  collisions={len(collided)}",
                ] + ([f"SHIELD: {reason}"] if reason else [])
            ego.apply_control(ctrl.control())
            world.tick()
            frame = frames_q.get(timeout=5.0)
            rgb = np.frombuffer(frame.raw_data, dtype=np.uint8).reshape(frame.height, frame.width, 4)[
                ..., 2::-1
            ]
            writer.append_data(overlay(rgb, text))
    finally:
        writer.close()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actors[::-1]:
            actor.destroy()
    summary = {"decisions": decisions, "shield_interventions": interventions, "collisions": len(collided)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {a.out}: {summary}")


if __name__ == "__main__":
    main()
