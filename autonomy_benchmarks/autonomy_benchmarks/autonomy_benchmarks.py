from typing import Any, Optional, Union

import message_filters
import rclpy
import rclpy.exceptions
import tf2_ros
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class AutonomyBenchmarks(Node):
    """ROS 2 node for benchmarking automated driving tasks."""

    def __init__(self):
        """Constructor"""
        super().__init__("autonomy_benchmarks")

        self.auto_reconfigurable_params: list[str] = []
        self.benchmark = self.declare_and_load_parameter(
            name="benchmark",
            param_type=rclpy.Parameter.Type.STRING,
            description="benchmark name",
            is_required=True,
            add_to_auto_reconfigurable_params=False,
        )

        self.benchmark_handler = None
        self.num_samples_evaluated = 0

        self.setup()

    def declare_and_load_parameter(
        self,
        name: str,
        param_type: rclpy.Parameter.Type,
        description: str,
        default: Optional[Any] = None,
        add_to_auto_reconfigurable_params: bool = True,
        is_required: bool = False,
        read_only: bool = False,
        from_value: Optional[Union[int, float]] = None,
        to_value: Optional[Union[int, float]] = None,
        step_value: Optional[Union[int, float]] = None,
        additional_constraints: str = "",
    ) -> Any:
        """Declares and loads a ROS parameter

        Args:
            name (str): name
            param_type (rclpy.Parameter.Type): parameter type
            description (str): description
            default (Optional[Any], optional): default value
            add_to_auto_reconfigurable_params (bool, optional): enable reconfiguration of parameter
            is_required (bool, optional): whether failure to load parameter will stop node
            read_only (bool, optional): set parameter to read-only
            from_value (Optional[Union[int, float]], optional): parameter range minimum
            to_value (Optional[Union[int, float]], optional): parameter range maximum
            step_value (Optional[Union[int, float]], optional): parameter range step
            additional_constraints (str, optional): additional constraints description

        Returns:
            Any: parameter value
        """

        # declare parameter
        param_desc = ParameterDescriptor()
        param_desc.description = description
        param_desc.additional_constraints = additional_constraints
        param_desc.read_only = read_only
        if from_value is not None and to_value is not None:
            if param_type == rclpy.Parameter.Type.INTEGER:
                value_range = IntegerRange(from_value=from_value, to_value=to_value)
                if step_value is not None:
                    value_range.step = step_value
                param_desc.integer_range = [value_range]
            elif param_type == rclpy.Parameter.Type.DOUBLE:
                value_range = FloatingPointRange(from_value=from_value, to_value=to_value)
                if step_value is not None:
                    value_range.step = step_value
                param_desc.floating_point_range = [value_range]
            else:
                self.get_logger().warn(f"Parameter type of parameter '{name}' does not support specifying a range")
        self.declare_parameter(name, param_type, param_desc)

        # load parameter
        try:
            param = self.get_parameter(name).value
            self.get_logger().info(f"Loaded parameter '{name}': {param}")
        except rclpy.exceptions.ParameterUninitializedException:
            if is_required:
                self.get_logger().fatal(f"Missing required parameter '{name}', exiting")
                raise SystemExit(1)
            else:
                self.get_logger().warn(f"Missing parameter '{name}', using default value: {default}")
                param = default
                self.set_parameters([rclpy.Parameter(name=name, value=param)])

        # add parameter to auto-reconfigurable parameters
        if add_to_auto_reconfigurable_params:
            self.auto_reconfigurable_params.append(name)

        return param

    def parameters_callback(self, parameters: list[rclpy.Parameter]) -> SetParametersResult:
        """Handles reconfiguration when a parameter value is changed

        Args:
            parameters (list[rclpy.Parameter]): parameters

        Returns:
            SetParametersResult: parameter change result
        """

        for param in parameters:
            if param.name in self.auto_reconfigurable_params:
                setattr(self, param.name, param.value)
                self.get_logger().info(f"Reconfigured parameter '{param.name}' to: {param.value}")

        result = SetParametersResult()
        result.successful = True

        return result

    def setup(self):
        """Sets up subscribers, publishers, etc. to configure the node"""

        # callback for dynamic parameter configuration
        self.add_on_set_parameters_callback(self.parameters_callback)

        # TF listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.transform_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.data_subscriptions: dict[str, Optional[message_filters.Subscriber]] = {}

        # get handler for specified benchmark
        if self.benchmark == "nuscenes_object_tracking":
            from autonomy_benchmarks.benchmarks.object_tracking.NuscenesObjectTracking import (
                NuscenesObjectTracking,
            )

            self.benchmark_handler = NuscenesObjectTracking()
        else:
            self.get_logger().fatal(f"Benchmark '{self.benchmark}' not recognized, exiting")
            raise SystemExit(1)

        # create subscriptions for benchmark data inputs
        for msg_topic, msg_type in self.benchmark_handler.required_inputs().items():
            self.data_subscriptions[msg_topic] = message_filters.Subscriber(
                self,
                msg_type,
                f"~/{msg_topic}",
                qos_profile=QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                ),
            )

        self.message_synchronizer = message_filters.TimeSynchronizer(
            list(self.data_subscriptions.values()),
            10,
        )
        self.message_synchronizer.registerCallback(self.evaluate_sample)

    def evaluate_sample(self, *args):
        """Callback to evaluate a single sample when all required input messages have been received"""
        self.get_logger().info("Received synchronized input messages, evaluating sample...")

        if self.benchmark_handler is None:
            self.get_logger().fatal("Benchmark handler not initialized, exiting")
            raise SystemExit(1)

        messages = {topic: msg for topic, msg in zip(self.benchmark_handler.required_inputs().keys(), args)}

        # TODO(RaphvK): add scene_id to messages if available, otherwise use None

        sample_result = self.benchmark_handler.compute_sample_metrics(scene_id=None, **messages)
        self.benchmark_handler._sample_results.append(sample_result)
        self.num_samples_evaluated += 1

        # TODO(RaphvK): call self.benchmark_handler.aggregate_metrics() when all samples have been processed
        if self.num_samples_evaluated % 10 == 0:  # TODO(RaphvK): replace with actual number of samples to evaluate
            aggregated_metrics = self.benchmark_handler.compute_aggregated_metrics(self.benchmark_handler._sample_results)
            self.get_logger().info(f"Aggregated metrics: {aggregated_metrics}")


def main():
    """Initializes ROS, runs the node event loop, and performs shutdown cleanup."""

    rclpy.init()
    node = AutonomyBenchmarks()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
