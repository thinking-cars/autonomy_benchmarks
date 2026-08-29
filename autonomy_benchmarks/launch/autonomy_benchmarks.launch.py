#!/usr/bin/env python3

# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

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

    # Service of the dataset node the benchmark requests the samples to evaluate from, remapped
    # onto the topic its argument resolves to just like the benchmark's data inputs.
    remappable_topics = [
        DeclareLaunchArgument(
            "request_samples",
            default_value="~/request_samples",
            description="service of the dataset node used to request the samples to evaluate",
        ),
    ]

    args = [
        *remappable_topics,
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
            description="publish the per-sample true positives, false positives and false negatives and open RViz on them",
        ),
        DeclareLaunchArgument(
            "samples_per_request",
            default_value="1",
            description="number of samples to request from the dataset at a time (0 requests all remaining samples at once)",
        ),
        DeclareLaunchArgument(
            "sample_ids",
            default_value="",
            description="comma-separated IDs of the dataset samples to evaluate (all samples if empty)",
        ),
        DeclareLaunchArgument(
            "evaluation_timeout",
            default_value="60.0",
            description="seconds to wait for a published sample to be evaluated before continuing without it",
        ),
        DeclareLaunchArgument(
            "results_path",
            default_value="",
            description="path of the JSON file the benchmark results are written to (results are only logged if empty)",
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
    remappings += [(la.default_value[0].text, LaunchConfiguration(la.name)) for la in remappable_topics]

    node = Node(
        package="autonomy_benchmarks",
        executable="autonomy_benchmarks",
        namespace=LaunchConfiguration("namespace"),
        name=LaunchConfiguration("name"),
        parameters=[
            {"benchmark": LaunchConfiguration("benchmark")},
            {"visualize": ParameterValue(LaunchConfiguration("visualize"), value_type=bool)},
            {"samples_per_request": ParameterValue(LaunchConfiguration("samples_per_request"), value_type=int)},
            {"sample_ids": ParameterValue(LaunchConfiguration("sample_ids"), value_type=str)},
            {"evaluation_timeout": ParameterValue(LaunchConfiguration("evaluation_timeout"), value_type=float)},
            {"results_path": ParameterValue(LaunchConfiguration("results_path"), value_type=str)},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        remappings=remappings,
        output="screen",
        emulate_tty=True,
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            PathJoinSubstitution([FindPackageShare("autonomy_benchmarks"), "config", "conf.rviz"]),
        ],
        condition=IfCondition(LaunchConfiguration("visualize")),
        output="screen",
    )

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            node,
            rviz,
        ]
    )
