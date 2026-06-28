#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    """Create and return the launch description for the autonomy_benchmarks node."""

    remappable_topics = [
        DeclareLaunchArgument("objects_tracked", default_value="~/objects_tracked"),
        DeclareLaunchArgument("objects_truth", default_value="~/objects_truth"),
    ]

    args = [
        DeclareLaunchArgument(
            "benchmark",
            default_value="nuscenes_object_tracking",
            description="benchmark name",
            choices=["nuscenes_object_tracking"],
        ),
        DeclareLaunchArgument("name", default_value="autonomy_benchmarks", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument(
            "log_level", default_value="info", description="ROS logging level (debug, info, warn, error, fatal)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use simulation clock"),
        *remappable_topics,
    ]

    nodes = [
        Node(
            package="autonomy_benchmarks",
            executable="autonomy_benchmarks",
            namespace=LaunchConfiguration("namespace"),
            name=LaunchConfiguration("name"),
            parameters=[
                {"benchmark": LaunchConfiguration("benchmark")},
            ],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            remappings=[(la.default_value[0].text, LaunchConfiguration(la.name)) for la in remappable_topics],
            output="screen",
            emulate_tty=True,
        ),
    ]

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            *nodes,
        ]
    )
