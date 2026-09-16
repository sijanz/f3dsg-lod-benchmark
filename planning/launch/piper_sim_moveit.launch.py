#!/usr/bin/env python3
"""
Dedicated Headless MoveIt 2 Launch File for Agilex Piper FunGraph3D Simulation.
"""

from moveit_configs_utils import MoveItConfigsBuilder
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("piper", package_name="piper_with_gripper_moveit")
        .robot_description(file_path="config/piper.urdf.xacro")
        .robot_description_semantic(file_path="config/piper.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Move Group Node configured for pure simulation planning
    move_group_params = [
        moveit_config.to_dict(),
        {
            "use_sim_time": False,
            "publish_robot_description_semantic": True,
            "allow_trajectory_execution": False,
            "publish_planning_scene": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
            "monitor_dynamics": False,
        }
    ]

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        arguments=[
            "--ros-args",
            "--log-level", "moveit_core.constraint_samplers:=debug",
            "--log-level", "moveit_kinematic_constraints:=debug",
            "--log-level", "moveit_ros_planning:=debug",
            "--log-level", "moveit_robot_state:=debug",
            "--log-level", "ompl:=debug"
        ],
        parameters=move_group_params,
    )

    return LaunchDescription([
        move_group_node,
    ])
