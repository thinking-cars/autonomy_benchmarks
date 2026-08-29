# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the result store of the AutonomyBenchmark base class.

Metrics are reported on three levels: for every single sample, aggregated over the samples of each
scene of the dataset, and aggregated over all evaluated samples. A minimal benchmark whose metrics
are trivial to predict is used, so that the tests cover the grouping and not a metric definition.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from autonomy_benchmarks.benchmarks.AutonomyBenchmark import AutonomyBenchmark


class CountingBenchmark(AutonomyBenchmark):
    """Minimal benchmark counting the objects of a sample and summing them up when aggregating."""

    def __init__(self) -> None:
        """Name the benchmark."""
        super().__init__(name="counting", description="counts objects")

    def required_inputs(self) -> Dict[str, Any]:
        """Declare the inputs, which this benchmark does not read from ROS messages."""
        return {"prediction": object, "label": object}

    def compute_sample_metrics(self, prediction: Any, label: Any, sample_id: str = None) -> Dict[str, Any]:
        """Report the given prediction and label counts of a single sample."""
        return {"num_predictions": prediction, "num_labels": label}

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sum the counts of the given samples."""
        return {
            "num_predictions": sum(entry["metrics"]["num_predictions"] for entry in sample_results),
            "num_labels": sum(entry["metrics"]["num_labels"] for entry in sample_results),
        }


def _benchmark_of(samples) -> CountingBenchmark:
    """Record ``(sample_id, scene_id, num_predictions, num_labels)`` samples in a benchmark."""
    benchmark = CountingBenchmark()
    for sample_id, scene_id, num_predictions, num_labels in samples:
        benchmark.record_sample(prediction=num_predictions, label=num_labels, sample_id=sample_id, scene_id=scene_id)
    return benchmark


class TestSampleResults:
    """Tests recording samples with the scene of the dataset they belong to."""

    def test_records_sample_with_its_scene(self):
        """A recorded sample keeps its ID, its scene and its metrics."""
        benchmark = _benchmark_of([("0", "scene_a", 2, 3)])

        assert benchmark.finalize()["sample_results"] == [
            {"sample_id": "0", "scene_id": "scene_a", "metrics": {"num_predictions": 2, "num_labels": 3}}
        ]

    def test_scene_can_be_set_after_the_sample_was_recorded(self):
        """An evaluation loop that learns the scene late sets it on the returned entry."""
        benchmark = CountingBenchmark()

        entry = benchmark.record_sample(prediction=1, label=1, sample_id="0")
        assert entry["scene_id"] is None
        entry["scene_id"] = "scene_a"

        assert benchmark.sample_results_by_scene() == {"scene_a": [entry]}


class TestFinalize:
    """Tests aggregating the recorded samples per scene and over the whole benchmark."""

    def test_aggregates_per_sample_scene_and_benchmark(self):
        """Metrics are reported for every sample, every scene and all samples together."""
        results = _benchmark_of(
            [
                ("0", "scene_a", 1, 1),
                ("1", "scene_a", 2, 3),
                ("2", "scene_b", 4, 5),
            ]
        ).finalize()

        assert results["num_samples"] == 3
        assert results["num_scenes"] == 2
        assert results["aggregated_metrics"] == {"num_predictions": 7, "num_labels": 9}
        assert results["scene_results"]["scene_a"] == {
            "num_samples": 2,
            "sample_ids": ["0", "1"],
            "aggregated_metrics": {"num_predictions": 3, "num_labels": 4},
        }
        assert results["scene_results"]["scene_b"]["aggregated_metrics"] == {"num_predictions": 4, "num_labels": 5}
        assert [entry["metrics"] for entry in results["sample_results"]] == [
            {"num_predictions": 1, "num_labels": 1},
            {"num_predictions": 2, "num_labels": 3},
            {"num_predictions": 4, "num_labels": 5},
        ]

    def test_groups_samples_of_a_scene_that_are_not_recorded_consecutively(self):
        """Samples are grouped by their scene, not by the order they were recorded in."""
        results = _benchmark_of(
            [
                ("0", "scene_a", 1, 0),
                ("1", "scene_b", 2, 0),
                ("2", "scene_a", 4, 0),
            ]
        ).finalize()

        assert results["scene_results"]["scene_a"]["sample_ids"] == ["0", "2"]
        assert results["scene_results"]["scene_a"]["aggregated_metrics"]["num_predictions"] == 5
        assert results["scene_results"]["scene_b"]["sample_ids"] == ["1"]

    def test_samples_without_a_scene_are_only_aggregated_over_the_benchmark(self):
        """A sample that cannot be attributed to a scene still counts for the whole benchmark."""
        results = _benchmark_of([("0", "scene_a", 1, 0), ("1", None, 2, 0)]).finalize()

        assert results["num_samples"] == 2
        assert results["num_scenes"] == 1
        assert results["aggregated_metrics"]["num_predictions"] == 3
        assert results["scene_results"]["scene_a"]["aggregated_metrics"]["num_predictions"] == 1

    def test_reports_no_scene_without_recorded_scenes(self):
        """Samples recorded without a scene aggregate to no scene results at all."""
        results = _benchmark_of([("0", None, 1, 0)]).finalize()

        assert results["num_scenes"] == 0
        assert results["scene_results"] == {}


class TestSaveResults:
    """Tests writing the results of all three levels to a JSON file."""

    def test_writes_sample_scene_and_benchmark_metrics(self, tmp_path):
        """The stored results hold the metrics of every sample, every scene and the benchmark."""
        benchmark = _benchmark_of([("0", "scene_a", 1, 1), ("1", "scene_b", 2, 2)])

        output_path = benchmark.save_results(str(tmp_path / "results" / "counting.json"))

        stored = json.loads(open(output_path).read())
        assert stored["aggregated_metrics"] == {"num_predictions": 3, "num_labels": 3}
        assert sorted(stored["scene_results"]) == ["scene_a", "scene_b"]
        assert [entry["sample_id"] for entry in stored["sample_results"]] == ["0", "1"]

    def test_writes_previously_computed_results(self, tmp_path):
        """Results that have already been computed are written as they are."""
        benchmark = _benchmark_of([("0", "scene_a", 1, 1)])
        results = benchmark.finalize()

        output_path = benchmark.save_results(str(tmp_path / "counting.json"), results=results)

        assert json.loads(open(output_path).read())["aggregated_metrics"] == results["aggregated_metrics"]
