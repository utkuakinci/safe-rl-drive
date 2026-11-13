"""Lock-step highway-env <-> ROS2 bridge.

Publishes ``/rl/observation`` (policy obs | safety obs | episode_start), waits for ``/rl/command``
([final_action, policy_action, aeb]) from the C++ node, steps the simulator, repeats.
Reports driving KPIs and the round-trip latency (publish -> command received) per episode.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension

from safe_rl_drive.envs.factory import make_env
from safe_rl_drive.envs.spec import OBS_SHAPE
from safe_rl_drive.safety.observer import SAFETY_OBS_SIZE, safety_observation
from safe_rl_drive.safety.wrapper import apply_emergency_brake


class HighwayBridge(Node):
    def __init__(self):
        super().__init__("highway_bridge")
        self.scenario = self.declare_parameter("scenario", "highway").value
        self.episodes = int(self.declare_parameter("episodes", 20).value)
        self.seed = int(self.declare_parameter("seed", 40_000).value)
        self.timeout = float(self.declare_parameter("command_timeout_s", 1.0).value)
        self.report = Path(self.declare_parameter("report", "results/ros2_closed_loop.json").value)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Float32MultiArray, "/rl/observation", qos)
        self.sub = self.create_subscription(Int32MultiArray, "/rl/command", self.on_command, qos)
        self.env = make_env(self.scenario, seed=self.seed)
        self.ep = 0
        self.results: list[dict] = []
        self.rtt_us: list[float] = []
        self.started = False
        self.waiting = False
        self.timer = self.create_timer(0.1, self.tick)

    # -------------------------------------------------------------------------------------------
    def tick(self):
        if not self.started:
            if self.pub.get_subscription_count() == 0:
                return
            self.started = True
            self.obs, _ = self.env.reset(seed=self.seed)
            self.episode_start = True
            self.publish()
        elif self.waiting and time.perf_counter() - self.t_pub > self.timeout:
            self.get_logger().warn("command timeout - republishing observation")
            self.publish()

    def publish(self):
        safety = safety_observation(self.env.unwrapped)
        data = np.concatenate([self.obs.reshape(-1), safety, [1.0 if self.episode_start else 0.0]]).astype(
            np.float32
        )
        msg = Float32MultiArray()
        n_pol = int(np.prod(OBS_SHAPE))
        msg.layout.dim = [
            MultiArrayDimension(label="policy", size=n_pol, stride=n_pol),
            MultiArrayDimension(label="safety", size=SAFETY_OBS_SIZE, stride=SAFETY_OBS_SIZE),
            MultiArrayDimension(label="episode_start", size=1, stride=1),
        ]
        msg.data = data.tolist()
        self._safety = safety
        self.t_pub = time.perf_counter()
        self.waiting = True
        self.pub.publish(msg)

    def on_command(self, msg: Int32MultiArray):
        if not self.waiting:
            return
        self.waiting = False
        self.rtt_us.append((time.perf_counter() - self.t_pub) * 1e6)
        final, _policy_action, aeb = msg.data[0], msg.data[1], msg.data[2]
        action = final
        if aeb:
            apply_emergency_brake(self.env.unwrapped, self._safety)
            action = 1  # IDLE: track the AEB target speed
        self.obs, _, term, trunc, info = self.env.step(action)
        self.episode_start = False
        if term or trunc:
            m = info["episode_metrics"]
            self.results.append(m)
            self.get_logger().info(
                f"episode {self.ep}: success={m['success']:.0f} collision={m['collision']:.0f} "
                f"speed={m['mean_speed']:.1f} rtt_p50={np.percentile(self.rtt_us, 50):.0f}us"
            )
            self.ep += 1
            if self.ep >= self.episodes:
                self.finish()
                return
            self.obs, _ = self.env.reset(seed=self.seed + self.ep)
            self.episode_start = True
        self.publish()

    def finish(self):
        rtt = np.asarray(self.rtt_us)
        summary = {
            "scenario": self.scenario,
            "episodes": len(self.results),
            "success": float(np.mean([r["success"] for r in self.results])),
            "collision": float(np.mean([r["collision"] for r in self.results])),
            "mean_speed": float(np.mean([r["mean_speed"] for r in self.results])),
            "rtt_us": {
                "p50": float(np.percentile(rtt, 50)),
                "p99": float(np.percentile(rtt, 99)),
                "max": float(rtt.max()),
            },
        }
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(json.dumps(summary, indent=2))
        self.get_logger().info(f"closed-loop summary: {summary}")
        raise SystemExit(0)


def main():
    rclpy.init()
    node = HighwayBridge()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
