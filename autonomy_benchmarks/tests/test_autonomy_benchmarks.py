# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the helpers of the autonomy_benchmarks node.

The node itself drives the evaluation via the ``request_samples`` service of the dataset and is
covered by running it against a dataset. Tested here are the parsing of the samples to evaluate,
as an unparsable value stops the node, and the matching of the received input messages into the
samples to evaluate, which has to hold up when the dataset continues with a scene that was
recorded before the scene played before it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from autonomy_benchmarks.autonomy_benchmarks import parse_sample_ids, SampleSynchronizer

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
