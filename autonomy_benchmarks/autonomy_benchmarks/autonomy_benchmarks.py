# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

import json
import time
from collections import deque, OrderedDict
from functools import partial
from typing import Any, Callable, Optional, Sequence, Union

import rclpy
import rclpy.exceptions
from autonomy_datasets_msgs.srv import RequestSamples
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.subscription import Subscription
from rclpy.task import Future

# Interval in seconds at which the benchmark checks whether it can request further samples; the
# benchmark also advances whenever a request is answered or a sample has been evaluated
_REQUEST_TIMER_PERIOD_S = 0.5

# Interval in seconds at which waiting for the sample request service of the dataset is logged
_SERVICE_WAIT_LOG_INTERVAL_S = 10.0

# Number of samples whose messages are kept while they wait for the messages of their remaining
# benchmark inputs
_SYNCHRONIZER_QUEUE_SIZE = 10


def parse_sample_ids(sample_ids: str) -> list[int]:
    """Parses the IDs of the dataset samples to evaluate

    Args:
        sample_ids (str): comma-separated sample IDs, e.g. "0,10,20"

    Returns:
        list[int]: parsed sample IDs, empty if no ID is given

    Raises:
        ValueError: if the IDs are not a comma-separated list of integers
    """
    return [int(sample_id) for sample_id in sample_ids.split(",") if sample_id.strip()]


class SampleSynchronizer:
    """Matches the messages of the benchmark inputs that belong to the same dataset sample.

    The dataset stamps all messages of a sample with the recording time of that sample, so the
    messages of a sample are matched by their exact header stamp. Messages of a sample that never
    completes are dropped in the order they arrived, never by comparing their stamps: the dataset
    replays one scene after the other, and a scene can have been recorded days before the scene
    played before it, so the stamp of a message says nothing about how recently it was received.
    (``message_filters.TimeSynchronizer`` drops by stamp instead, and therefore discards the
    messages of a scene that starts before the end of the preceding one.)

    Messages are added from subscription callbacks, which the node executor runs one after
    another, so no locking is needed.
    """

    def __init__(self, topics: Sequence[str], callback: Callable[..., None], queue_size: int = _SYNCHRONIZER_QUEUE_SIZE):
        """Constructor

        Args:
            topics (Sequence[str]): input names to match, in the order their messages are passed
                to the callback
            callback (Callable[..., None]): called with the messages of every completed sample
            queue_size (int, optional): number of samples to keep while they wait for the messages
                of their remaining inputs
        """
        self.topics = list(topics)
        self.callback = callback
        self.queue_size = queue_size
        # messages of the samples that are still missing inputs, by header stamp and in the order
        # the samples were first received on any input
        self.incomplete_samples: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()

    def add(self, topic: str, message: Any):
        """Adds a received message and reports the sample it completes to the callback

        Args:
            topic (str): input name the message was received on
            message (Any): received message, stamped with the time of its sample
        """
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        messages = self.incomplete_samples.setdefault(stamp, {})
        messages[topic] = message

        if len(messages) == len(self.topics):
            del self.incomplete_samples[stamp]
            self.callback(*(messages[topic] for topic in self.topics))
            return

        # give up on the sample that has been waiting for its remaining inputs the longest
        while len(self.incomplete_samples) > self.queue_size:
            self.incomplete_samples.popitem(last=False)


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
            default="nuscenes_lidar_object_detection",
        )

        self.visualize = self.declare_and_load_parameter(
            name="visualize",
            param_type=rclpy.Parameter.Type.BOOL,
            description="publish the per-sample true positives, false positives and false negatives for RViz",
            default=False,
        )

        self.samples_per_request = self.declare_and_load_parameter(
            name="samples_per_request",
            param_type=rclpy.Parameter.Type.INTEGER,
            description="number of samples to request from the dataset at a time; 0 requests all remaining "
            "samples at once, 1 evaluates every sample before the next one is published",
            default=1,
            from_value=0,
            to_value=100000,
        )

        self.sample_ids = self.declare_and_load_parameter(
            name="sample_ids",
            param_type=rclpy.Parameter.Type.STRING,
            description="comma-separated IDs of the dataset samples to evaluate (e.g. '0,10,20'); "
            "if empty, all samples of the dataset are evaluated",
            default="",
            add_to_auto_reconfigurable_params=False,
            read_only=True,
        )
        try:
            self.requested_sample_ids = parse_sample_ids(self.sample_ids)
        except ValueError:
            self.get_logger().fatal(f"Parameter 'sample_ids' is not a comma-separated list of sample IDs: '{self.sample_ids}'")
            raise SystemExit(1)

        self.evaluation_timeout = self.declare_and_load_parameter(
            name="evaluation_timeout",
            param_type=rclpy.Parameter.Type.DOUBLE,
            description="seconds to wait for a published sample to be evaluated before continuing without it",
            default=60.0,
            from_value=0.0,
            to_value=3600.0,
        )

        self.results_path = self.declare_and_load_parameter(
            name="results_path",
            param_type=rclpy.Parameter.Type.STRING,
            description="path of the JSON file the benchmark results are written to; results are only logged if empty",
            default="",
        )

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

        self.data_subscriptions: dict[str, Subscription] = {}

        # get handler for specified benchmark
        benchmark_handler = None
        if self.benchmark == "nuscenes_lidar_object_detection":
            from autonomy_benchmarks.benchmarks.lidar_object_detection.NuscenesLidarObjectDetection import (
                NuscenesLidarObjectDetection,
            )

            benchmark_handler = NuscenesLidarObjectDetection()
        else:
            self.get_logger().fatal(f"Benchmark '{self.benchmark}' not recognized, exiting")
            raise SystemExit(1)

        # create subscriptions for benchmark data inputs, whose messages are matched into the
        # samples to evaluate by their header stamp
        self.message_synchronizer = SampleSynchronizer(
            topics=list(benchmark_handler.required_inputs()),
            callback=self.evaluate_sample,
            queue_size=_SYNCHRONIZER_QUEUE_SIZE,
        )
        for msg_topic, msg_type in benchmark_handler.required_inputs().items():
            self.data_subscriptions[msg_topic] = self.create_subscription(
                msg_type,
                msg_topic,
                partial(self.message_synchronizer.add, msg_topic),
                qos_profile=QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                ),
            )

        # create publishers visualizing the benchmark's per-sample matching outcome
        self.visualization_publishers: dict[str, Publisher] = {}
        if self.visualize:
            for msg_topic, msg_type in benchmark_handler.visualization_outputs().items():
                self.visualization_publishers[msg_topic] = self.create_publisher(
                    msg_type,
                    f"~/{msg_topic}",
                    qos_profile=QoSProfile(
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=10,
                    ),
                )
            self.get_logger().info(f"Visualizing benchmark results on: {sorted(self.visualization_publishers)}")

        # store handler and ordered topic list for use in evaluate_sample
        self.benchmark_handler = benchmark_handler
        self.input_topics: list = list(benchmark_handler.required_inputs().keys())

        self.sample_request_client = self.create_client(RequestSamples, "~/request_samples")
        self.published_sample_ids: list[int] = []
        # A sample may already be evaluated before the dataset node answers the request that
        # published it, so published samples and evaluated samples are matched in publishing
        # order: whichever of the two arrives first waits here for its counterpart, which makes
        # at most one of both queues non-empty at a time.
        self.scenes_awaiting_sample: deque = deque()
        self.samples_awaiting_scene: deque = deque()
        self.pending_request: Optional[Future] = None
        self.evaluation_deadline: Optional[float] = None
        self.publishing_finished = False
        self.benchmark_finished = False
        self.num_evaluated_samples = 0
        # driven by a steady clock, so that the benchmark also advances while the simulation clock
        # of the dataset stands still, i.e. while no sample is being published
        self.request_timer = self.create_timer(
            _REQUEST_TIMER_PERIOD_S,
            self.advance_benchmark,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(f"Requesting samples to evaluate from '{self.sample_request_client.srv_name}'")

    def advance_benchmark(self):
        """Requests the next samples to evaluate, or finalizes the benchmark once all were published

        Called periodically as well as whenever a request has been answered or a sample has been
        evaluated, and does nothing while the benchmark is waiting for one of those.
        """
        if self.benchmark_finished or self.pending_request is not None:
            return
        if not self.sample_request_client.service_is_ready():
            self.get_logger().warn(
                f"Waiting for service '{self.sample_request_client.srv_name}' to request samples of the dataset...",
                throttle_duration_sec=_SERVICE_WAIT_LOG_INTERVAL_S,
            )
            return
        if self.awaiting_evaluations():
            return
        if self.publishing_finished:
            self.finalize_benchmark()
            self.shutdown()
            return
        self.request_samples()

    def awaiting_evaluations(self) -> bool:
        """Reports whether published samples are still waiting to be evaluated

        A sample is evaluated once all benchmark inputs have been received for it, which happens
        once the system under test has processed the sample the dataset published. Samples that
        are not evaluated within 'evaluation_timeout' seconds are given up on, so that a system
        under test which skips samples does not stall the benchmark.

        Returns:
            bool: whether the benchmark waits for published samples to be evaluated
        """
        outstanding_evaluations = len(self.scenes_awaiting_sample) - len(self.samples_awaiting_scene)
        if outstanding_evaluations <= 0:
            return False
        if self.evaluation_deadline is not None and time.monotonic() < self.evaluation_deadline:
            return True
        missing_samples = ", ".join(str(sample_id) for sample_id in self.published_sample_ids[-outstanding_evaluations:])
        self.get_logger().warn(
            f"Sample(s) {missing_samples} were not evaluated within {self.evaluation_timeout} s, continuing without them"
        )
        # Only samples waiting to be evaluated are left in the queue, as evaluated samples are
        # matched with a scene as soon as one is published. Dropping their scenes keeps the
        # following samples matched with the scene they were published from; only the evaluation
        # of a given up sample that still arrives later shifts the matching by one sample.
        self.scenes_awaiting_sample.clear()
        return False

    def request_samples(self):
        """Requests the next samples to evaluate from the dataset node"""
        request = RequestSamples.Request()
        if self.requested_sample_ids:
            request.mode = RequestSamples.Request.MODE_SAMPLE_IDS
            request.sample_ids = self.requested_sample_ids
            requested_samples = f"the samples {self.sample_ids}"
        elif self.samples_per_request > 0:
            request.mode = RequestSamples.Request.MODE_NEXT_SAMPLES
            request.num_samples = self.samples_per_request
            requested_samples = f"the next {self.samples_per_request} sample(s)"
        else:
            request.mode = RequestSamples.Request.MODE_ALL_SAMPLES
            requested_samples = "all remaining samples"

        self.get_logger().debug(f"Requesting {requested_samples} of the dataset for evaluation")
        self.pending_request = self.sample_request_client.call_async(request)
        self.pending_request.add_done_callback(self.samples_published_callback)

    def samples_published_callback(self, future: Future):
        """Records the samples the dataset node has published and continues the benchmark

        Args:
            future (Future): future of the request, holding the response of the dataset node
        """
        self.pending_request = None
        try:
            response = future.result()
        except Exception as exception:
            self.get_logger().error(f"Requesting samples of the dataset failed: {exception}")
            self.publishing_finished = True
            self.advance_benchmark()
            return

        published_sample_ids = [int(sample_id) for sample_id in response.published_sample_ids]
        self.published_sample_ids.extend(published_sample_ids)
        self.assign_scenes(response.published_scene_ids)
        self.evaluation_deadline = time.monotonic() + self.evaluation_timeout

        published_samples = ", ".join(str(sample_id) for sample_id in published_sample_ids)
        if response.success:
            self.get_logger().debug(f"Dataset published sample(s) {published_samples}: {response.message}")
        else:
            self.get_logger().warn(f"Dataset did not publish all requested samples: {response.message}")

        # A request for a fixed set of samples is answered once all of them have been published,
        # any other request is repeated until the dataset has published its last sample.
        self.publishing_finished = (
            response.end_of_dataset or not response.success or bool(self.requested_sample_ids) or self.samples_per_request <= 0
        )
        self.advance_benchmark()

    def assign_scenes(self, published_scene_ids: Sequence[str]):
        """Attributes the scenes of published samples to the samples that are evaluated for them

        The benchmark aggregates the metrics of the samples of a scene, so every evaluated sample
        needs the scene the dataset published it from. Samples are evaluated in the order the
        dataset published them, so both are matched in that order; a scene whose sample has not
        been evaluated yet waits for it, and vice versa.

        Args:
            published_scene_ids (Sequence[str]): scenes of the published samples, in the order
                the samples were published
        """
        for scene_id in published_scene_ids:
            if self.samples_awaiting_scene:
                self.samples_awaiting_scene.popleft()["scene_id"] = str(scene_id)
            else:
                self.scenes_awaiting_sample.append(str(scene_id))

    def finalize_benchmark(self):
        """Aggregates the metrics of the evaluated samples per scene and over the whole benchmark

        The results hold the metrics of every single sample, of the samples of each scene, and of
        all evaluated samples, of which the metrics over all samples are logged.
        """
        self.benchmark_finished = True
        self.request_timer.cancel()

        if not self.num_evaluated_samples:
            self.get_logger().warn(f"Benchmark '{self.benchmark}' evaluated no sample, no metrics are aggregated")
            return

        results = self.benchmark_handler.finalize()
        aggregated_metrics = json.dumps(results["aggregated_metrics"], indent=2, default=str)
        self.get_logger().info(
            f"Benchmark '{self.benchmark}' finished after {results['num_samples']} evaluated sample(s) "
            f"of {results['num_scenes']} scene(s)."
        )

        if self.results_path:
            try:
                results_path = self.benchmark_handler.save_results(self.results_path, results=results)
                self.get_logger().info(f"Wrote benchmark results to '{results_path}'")
            except OSError as exception:
                self.get_logger().error(f"Failed to write benchmark results to '{self.results_path}': {exception}")
        else:
            self.get_logger().info(f"Aggregated dataset metrics:\n{aggregated_metrics}")

    def shutdown(self):
        """Stops the node, as there is nothing left to evaluate once the benchmark has finished

        Shutting down the ROS context ends the spinning of the node in 'main', which lets the
        process exit with the benchmark results reported.
        """
        self.get_logger().info("Nothing left to evaluate, shutting down")
        rclpy.try_shutdown()

    def evaluate_sample(self, *args):
        """Callback to evaluate a single sample when all required input messages have been received.

        The positional *args* are the synchronized ROS messages in the same order
        as the input names (dict keys) returned by
        ``benchmark_handler.required_inputs()``.  Those keys match the
        ``compute_sample_metrics`` parameter names, so the raw ``ObjectList``
        messages are forwarded to ``benchmark_handler.record_sample`` by keyword;
        the benchmark extracts the fields it needs inside
        ``compute_sample_metrics``.

        Samples are identified by the ROS header stamp of their messages, which is the stamp the
        dataset recorded them with, and are attributed to the scene the dataset reported for them,
        which can arrive after the sample has been evaluated.
        """
        self.get_logger().debug("Received synchronized input messages, evaluating sample...")

        if len(args) != len(self.input_topics):
            self.get_logger().error(f"Expected {len(self.input_topics)} messages but received {len(args)}; skipping sample.")
            return

        # Map each synchronized message to its required-input name (which matches
        # the compute_sample_metrics parameter names).
        messages = dict(zip(self.input_topics, args))

        # Use the ROS header stamp of the first message as sample ID.
        stamp = args[0].header.stamp
        sample_id = f"{stamp.sec}.{stamp.nanosec:09d}"
        self.get_logger().debug(f"Sample ID: '{sample_id}'")

        result = self.benchmark_handler.record_sample(sample_id=sample_id, **messages)
        self.num_evaluated_samples += 1
        # attribute the sample to the scene the dataset published it from, which the dataset may
        # only report after the sample has been evaluated
        if self.scenes_awaiting_sample:
            result["scene_id"] = self.scenes_awaiting_sample.popleft()
        else:
            self.samples_awaiting_scene.append(result)
        # a system under test that needs longer for some samples must not run into the timeout,
        # which therefore restarts with every evaluated sample
        self.evaluation_deadline = time.monotonic() + self.evaluation_timeout
        if not self.results_path:
            self.get_logger().debug(f"Sample '{sample_id}' result: {result}")

        # publish the sample's matching outcome for inspection in RViz
        if self.visualization_publishers:
            visualization = self.benchmark_handler.visualize_sample(sample_id=sample_id, **messages)
            for msg_topic, publisher in self.visualization_publishers.items():
                publisher.publish(visualization[msg_topic])

        # request the next samples, or aggregate the dataset metrics if this was the last one
        self.advance_benchmark()


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
