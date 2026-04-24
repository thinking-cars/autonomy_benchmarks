#!/usr/bin/env python3

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    """Create and return the launch description for the autonomy_benchmarks node."""

    args = [
        DeclareLaunchArgument(
            "benchmark",
            default_value="nuscenes_lidar_object_detection",
            description="benchmark name",
            choices=["waymo_camera_object_detection_2d", "waymo_camera_object_detection_3d", "nuscenes_lidar_object_detection"],
        ),
        DeclareLaunchArgument("name", default_value="autonomy_benchmarks", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument(
            "params",
            default_value=os.path.join(get_package_share_directory("autonomy_benchmarks"), "config", "params.yml"),
            description="path to parameter file",
        ),
        DeclareLaunchArgument(
            "log_level", default_value="info", description="ROS logging level (debug, info, warn, error, fatal)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use simulation clock"),
    ]

    nodes = [
        Node(
            package="autonomy_benchmarks",
            executable="autonomy_benchmarks",
            namespace=LaunchConfiguration("namespace"),
            name=LaunchConfiguration("name"),
            parameters=[
                LaunchConfiguration("params"),
                {"benchmark": LaunchConfiguration("benchmark")},
            ],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            remappings=[
                (
                    "object_list/prediction",
                    "object_list/lidar_01",
                ),  # TODO(unknown): remove remapping once testing with actual models is possible
            ],
            output="screen",
            emulate_tty=True,
        ),
        # dummy nodes to subscribe to required topics published by autonomy_datasets
        # TODO(unknown): remove once testing with actual models is possible
        ExecuteProcess(
            cmd=["ros2", "topic", "hz", "/lidar_01/point_cloud"],
            name="dummy_topic_hz",
            output="log",
            shell=False,
        ),
        ExecuteProcess(
            cmd=["ros2", "topic", "hz", "/ego_data"],
            name="dummy_topic_hz",
            output="log",
            shell=False,
        ),
    ]

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            *nodes,
        ]
    )
