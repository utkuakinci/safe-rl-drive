"""Closed loop: highway-env bridge <-> C++ policy node.

ros2 launch rl_policy_node drive.launch.py model_path:=/models/policy.onnx backend:=onnxruntime scenario:=highway
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory("rl_policy_node"), "config", "policy.yaml")
    bridge = Node(
        package="rl_sim_bridge",
        executable="highway_bridge",
        name="highway_bridge",
        output="screen",
        parameters=[
            {
                "scenario": LaunchConfiguration("scenario"),
                "episodes": ParameterValue(LaunchConfiguration("episodes"), value_type=int),
            }
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path", default_value="/models/policy.onnx"),
            DeclareLaunchArgument("backend", default_value="onnxruntime"),
            DeclareLaunchArgument("scenario", default_value="highway"),
            DeclareLaunchArgument("episodes", default_value="20"),
            DeclareLaunchArgument("shield", default_value="true"),
            Node(
                package="rl_policy_node",
                executable="policy_node",
                name="rl_policy_node",
                output="screen",
                parameters=[
                    params,
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "backend": LaunchConfiguration("backend"),
                        "shield.enabled": ParameterValue(LaunchConfiguration("shield"), value_type=bool),
                    },
                ],
            ),
            bridge,
            # stop the whole system when the evaluation episodes are done
            RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[EmitEvent(event=Shutdown())])),
        ]
    )
