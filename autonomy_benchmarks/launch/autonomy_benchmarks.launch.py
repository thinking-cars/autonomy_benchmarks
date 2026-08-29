#!/usr/bin/env python3

# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue

# Names of the benchmark's data inputs. Each name must match a key returned by
# the benchmark's ``required_inputs()`` (and thus a ``compute_sample_metrics``
# parameter). This single list is the source of truth: it drives both the
# per-input launch arguments and the topic remappings below, so adding a new
# input (e.g. "map", "radar") is a one-line change here.
_INPUTS = ["prediction", "label"]

# Inputs whose topic is derived from another input's topic instead of getting a
# launch argument of its own, as ``input name -> (source input, topic suffix)``.
# The dataset publishes the meta information of an object list next to it, on
# "<object list topic>/meta_info".
_DERIVED_INPUTS = {"label_meta_info": ("label", "/meta_info")}


def generate_launch_description():
    """Create and return the launch description for the autonomy_benchmarks node."""

    args = [
        DeclareLaunchArgument(
            "benchmark",
            default_value="nuscenes_lidar_object_detection",
            description="benchmark name",
            choices=["nuscenes_lidar_object_detection"],
        ),
        DeclareLaunchArgument("name", default_value="autonomy_benchmarks", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument(
            "log_level", default_value="info", description="ROS logging level (debug, info, warn, error, fatal)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use simulation clock"),
        DeclareLaunchArgument(
            "visualize",
            default_value="false",
            choices=["true", "false"],
            description="publish the per-sample true positives, false positives and false negatives for RViz",
        ),
        # One argument per benchmark input; defaults to the node-relative name so
        # an unset input is a no-op remap. Override with e.g. prediction:=/real/topic.
        *[
            DeclareLaunchArgument(
                name,
                default_value=f"~/{name}",
                description=f"real ROS topic feeding the benchmark's '{name}' input",
            )
            for name in _INPUTS
        ],
    ]

    # Remap each node-relative input name onto the topic its argument resolves to,
    # then onto the topic derived from another input for the derived inputs.
    remappings = [(name, LaunchConfiguration(name)) for name in _INPUTS]
    remappings += [(name, [LaunchConfiguration(source), suffix]) for name, (source, suffix) in _DERIVED_INPUTS.items()]

    node = Node(
        package="autonomy_benchmarks",
        executable="autonomy_benchmarks",
        namespace=LaunchConfiguration("namespace"),
        name=LaunchConfiguration("name"),
        parameters=[
            {"benchmark": LaunchConfiguration("benchmark")},
            {"visualize": ParameterValue(LaunchConfiguration("visualize"), value_type=bool)},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        remappings=remappings,
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            node,
        ]
    )
