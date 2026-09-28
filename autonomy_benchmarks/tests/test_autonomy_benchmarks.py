# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the helpers of the autonomy_benchmarks node.

The node itself drives the evaluation via the ``request_samples`` service of the dataset and is
covered by running it against a dataset. Tested here are the parsing of the samples to evaluate,
as an unparsable value stops the node, the matching of the received input messages into the
samples to evaluate, which has to hold up when the dataset continues with a scene that was
recorded before the scene played before it, the evaluation of a sample, which requests the next
samples unless the user controls the playback manually, and the finalization of the results,
which reports the samples of an interrupted evaluation as incomplete.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
from autonomy_benchmarks.autonomy_benchmarks import AutonomyBenchmarks, parse_sample_ids, SampleSynchronizer

_TOPICS = ["prediction", "label", "label_meta_info"]

# stamps of the last sample of a scene and of the first samples of the scene the dataset continues
# with, which nuScenes recorded years earlier
_PREVIOUS_SCENE = (1537853053, 397270000)
_NEXT_SCENE = [(1531885320, 49418000), (1531885320, 548742000), (1531885321, 48634000)]


def _message(stamp: tuple[int, int]) -> SimpleNamespace:
    """Fake an input message stamped with the recording time of its sample."""
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp[0], nanosec=stamp[1])))


def _synchronizer(queue_size: int = 10) -> tuple[SampleSynchronizer, list]:
    """Create a synchronizer of the benchmark inputs next to the list of the samples it matched."""
    matched_samples: list = []
    synchronizer = SampleSynchronizer(_TOPICS, callback=lambda *msgs: matched_samples.append(msgs), queue_size=queue_size)
    return synchronizer, matched_samples


def _publish_sample(synchronizer: SampleSynchronizer, stamp: tuple[int, int], topics=None) -> dict:
    """Add one message per given input (all of them by default), all stamped with the same time."""
    messages = {topic: _message(stamp) for topic in topics or _TOPICS}
    for topic, message in messages.items():
        synchronizer.add(topic, message)
    return messages


class TestParseSampleIds:
    """Tests parsing the 'sample_ids' parameter into the sample IDs to request."""

    def test_parses_comma_separated_ids(self):
        """Sample IDs are parsed in the given order."""
        assert parse_sample_ids("0,10,20") == [0, 10, 20]

    def test_parses_single_id(self):
        """A single ID is a valid request."""
        assert parse_sample_ids("7") == [7]

    def test_ignores_surrounding_whitespace(self):
        """Sample IDs separated by ', ' are parsed like IDs separated by ','."""
        assert parse_sample_ids(" 1, 2 ,3 ") == [1, 2, 3]

    @pytest.mark.parametrize("sample_ids", ["", " ", ","])
    def test_no_ids_evaluate_the_whole_dataset(self, sample_ids):
        """An empty value requests no specific samples, so the whole dataset is evaluated."""
        assert parse_sample_ids(sample_ids) == []

    @pytest.mark.parametrize("sample_ids", ["1;2", "first", "1.5", "1-2"])
    def test_rejects_values_that_are_no_ids(self, sample_ids):
        """A value that is no comma-separated list of IDs is rejected."""
        with pytest.raises(ValueError):
            parse_sample_ids(sample_ids)


class TestSampleSynchronizer:
    """Tests matching the messages of the benchmark inputs into the samples to evaluate."""

    def test_matches_the_messages_of_a_sample_in_input_order(self):
        """A sample is reported once every input has been received, in the order of the inputs."""
        synchronizer, matched_samples = _synchronizer()

        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=list(reversed(_TOPICS)))

        assert matched_samples == [tuple(messages[topic] for topic in _TOPICS)]
        assert not synchronizer.incomplete_samples

    def test_waits_for_the_missing_inputs_of_a_sample(self):
        """A sample of which an input is missing is not reported yet."""
        synchronizer, matched_samples = _synchronizer()

        _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["label", "label_meta_info"])

        assert matched_samples == []

    def test_matches_samples_of_a_scene_recorded_before_the_previous_scene(self):
        """The dataset continues with an older scene, whose samples are matched all the same."""
        synchronizer, matched_samples = _synchronizer()

        _publish_sample(synchronizer, _PREVIOUS_SCENE)
        messages = _publish_sample(synchronizer, _NEXT_SCENE[0])

        assert len(matched_samples) == 2
        assert matched_samples[-1] == tuple(messages[topic] for topic in _TOPICS)

    def test_keeps_a_waiting_sample_older_than_a_matched_one(self):
        """A sample of a new, older scene is not dropped by a late sample of the previous scene."""
        synchronizer, matched_samples = _synchronizer()
        # the last sample of the previous scene still waits for the system under test, while the
        # first sample of the next scene, recorded years earlier, is published already
        pending = _publish_sample(synchronizer, _PREVIOUS_SCENE, topics=["label", "label_meta_info"])
        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["label", "label_meta_info"])

        synchronizer.add("prediction", _message(_PREVIOUS_SCENE))
        synchronizer.add("prediction", _message(_NEXT_SCENE[0]))

        assert len(matched_samples) == 2
        assert matched_samples[0][1:] == (pending["label"], pending["label_meta_info"])
        assert matched_samples[1][1:] == (messages["label"], messages["label_meta_info"])

    def test_gives_up_on_the_sample_waiting_the_longest(self):
        """Samples are dropped in the order they arrived, not by their stamp."""
        synchronizer, matched_samples = _synchronizer(queue_size=2)

        # a sample of the previous scene waits first, followed by two samples of the older scene
        for stamp in [_PREVIOUS_SCENE, *_NEXT_SCENE[:2]]:
            _publish_sample(synchronizer, stamp, topics=["label"])

        # the queue size is exceeded, so the sample that has been waiting the longest is given up
        # on, even though the samples kept for evaluation were recorded years before it
        assert list(synchronizer.incomplete_samples) == _NEXT_SCENE[:2]

        _publish_sample(synchronizer, _NEXT_SCENE[2], topics=["label"])

        assert list(synchronizer.incomplete_samples) == _NEXT_SCENE[1:]
        assert matched_samples == []


class _FakeLogger:
    """Collect the messages the node logs instead of publishing them to ROS."""

    def __init__(self):
        """Start with an empty log."""
        self.messages: list = []

    def info(self, message: str, **kwargs) -> None:
        """Record a logged message, whatever its severity."""
        self.messages.append(message)

    debug = warn = error = info


class _FakeBenchmarkHandler:
    """Stand in for the benchmark whose results the node aggregates and writes."""

    def __init__(self):
        """Start without finalized or written results."""
        self.finalized_complete = None
        self.written_results = None

    def finalize(self, complete: bool = True) -> dict:
        """Report results that are marked the way the node asked for."""
        self.finalized_complete = complete
        return {"num_samples": 2, "num_scenes": 1, "complete": complete, "aggregated_metrics": {}}

    def save_results(self, output_path: str, results: dict = None) -> str:
        """Keep the results instead of writing them to a file."""
        self.written_results = results
        return output_path


class _RecordingBenchmarkHandler:
    """Stand in for the benchmark that records the samples the node evaluates."""

    def __init__(self):
        """Start without recorded samples."""
        self.recorded_samples: list = []

    def record_sample(self, sample_id: str, **messages) -> dict:
        """Record a sample the way the benchmark stores it, still without a scene."""
        entry = {"sample_id": sample_id, "scene_id": None, "metrics": {}}
        self.recorded_samples.append(entry)
        return entry


def _evaluating_node(manual_playback: bool) -> SimpleNamespace:
    """Stub the node state that evaluating a sample reads, counting the attempts to continue the benchmark."""
    node = SimpleNamespace(
        manual_playback=manual_playback,
        input_topics=_TOPICS,
        benchmark_handler=_RecordingBenchmarkHandler(),
        num_evaluated_samples=0,
        scenes_awaiting_sample=deque(),
        samples_awaiting_scene=deque(),
        evaluation_timeout=60.0,
        evaluation_deadline=None,
        results_path="",
        visualization_publishers={},
        num_advances=0,
        get_logger=lambda logger=_FakeLogger(): logger,
    )
    node.advance_benchmark = lambda: setattr(node, "num_advances", node.num_advances + 1)
    return node


def _sample_messages(stamp: tuple[int, int]) -> tuple:
    """Fake the synchronized input messages of one sample, in the order of the inputs."""
    return tuple(_message(stamp) for _ in _TOPICS)


class TestEvaluateSample:
    """Tests evaluating a sample with the playback driven by the benchmark or by the user in RViz."""

    def test_benchmark_requests_the_next_samples_after_evaluating_one(self):
        """Without manual playback, the benchmark continues with the next samples on its own."""
        node = _evaluating_node(manual_playback=False)

        AutonomyBenchmarks.evaluate_sample(node, *_sample_messages(_NEXT_SCENE[0]))

        assert node.num_advances == 1
        # the dataset reports the scene of the sample with the response to the request
        assert list(node.samples_awaiting_scene) == node.benchmark_handler.recorded_samples

    def test_manual_playback_leaves_requesting_samples_to_the_user(self):
        """With manual playback, samples are evaluated as they arrive, without requesting further ones."""
        node = _evaluating_node(manual_playback=True)

        for stamp in _NEXT_SCENE:
            AutonomyBenchmarks.evaluate_sample(node, *_sample_messages(stamp))

        assert node.num_evaluated_samples == len(_NEXT_SCENE)
        assert node.num_advances == 0
        # the scenes are only reported to the playback panel, so no sample waits for one
        assert not node.samples_awaiting_scene


def _node(num_evaluated_samples: int = 2, results_path: str = "/results/benchmark.json") -> SimpleNamespace:
    """Stub the node state that finalizing a benchmark reads, without initializing ROS."""
    return SimpleNamespace(
        benchmark="counting",
        benchmark_finished=False,
        benchmark_handler=_FakeBenchmarkHandler(),
        num_evaluated_samples=num_evaluated_samples,
        results_path=results_path,
        request_timer=SimpleNamespace(cancel=lambda: None),
        get_logger=lambda logger=_FakeLogger(): logger,
    )


class TestFinalizeBenchmark:
    """Tests reporting the results of a benchmark that ran to its end or was interrupted."""

    def test_finished_benchmark_writes_complete_results(self):
        """A benchmark that evaluated all its samples reports complete results."""
        node = _node()

        AutonomyBenchmarks.finalize_benchmark(node)

        assert node.benchmark_handler.finalized_complete is True
        assert node.benchmark_handler.written_results["complete"] is True

    def test_interrupted_benchmark_writes_incomplete_results(self):
        """An evaluation stopped before its last sample, e.g. with Ctrl-C, still writes its results."""
        node = _node()

        AutonomyBenchmarks.finalize_benchmark(node, complete=False)

        assert node.benchmark_handler.finalized_complete is False
        assert node.benchmark_handler.written_results["complete"] is False

    def test_finished_benchmark_is_not_finalized_again_on_shutdown(self):
        """Shutting down after the last sample must not overwrite the results with incomplete ones."""
        node = _node()
        AutonomyBenchmarks.finalize_benchmark(node)

        AutonomyBenchmarks.finalize_benchmark(node, complete=False)

        assert node.benchmark_handler.finalized_complete is True
        assert node.benchmark_handler.written_results["complete"] is True

    def test_manual_playback_writes_its_results_when_stopped(self):
        """With manual playback, which runs no request timer, the results are written once the node is stopped."""
        node = _node()
        node.request_timer = None

        AutonomyBenchmarks.finalize_benchmark(node, complete=False)

        assert node.benchmark_handler.written_results["complete"] is False

    def test_interrupted_benchmark_without_samples_writes_nothing(self):
        """An evaluation interrupted before its first sample has no results to write."""
        node = _node(num_evaluated_samples=0)

        AutonomyBenchmarks.finalize_benchmark(node, complete=False)

        assert node.benchmark_handler.written_results is None

    def test_results_are_only_logged_without_a_results_path(self):
        """Without 'results_path' the interrupted results are logged instead of written."""
        node = _node(results_path="")

        AutonomyBenchmarks.finalize_benchmark(node, complete=False)

        assert node.benchmark_handler.finalized_complete is False
        assert node.benchmark_handler.written_results is None
