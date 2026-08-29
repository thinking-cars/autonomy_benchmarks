# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for all AutonomyHub benchmarks.

Each benchmark defines how to compute per-sample and aggregated metrics for a
specific perception task (e.g. 2-D / 3-D object detection).  Concrete
subclasses must override the abstract methods.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AutonomyBenchmark(ABC):
    """Meta-class (abstract base class) for AutonomyHub benchmarks.

    A benchmark is responsible for:
    * extracting the model input from a dataset sample,
    * computing per-sample metrics from a prediction and the ground-truth label,
    * aggregating per-sample metrics into scene-level and dataset-level metrics,
    * persisting results to JSON.
    """

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize a benchmark definition and empty result store."""
        self.name: str = name
        self.description: str = description
        self._sample_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface – must be implemented by every concrete benchmark
    # ------------------------------------------------------------------

    @abstractmethod
    def required_inputs(self) -> Dict[str, Any]:
        """Define expected input ROS message types.

        Returns
        -------
        A dictionary mapping topic names to their expected message types.
        """

    @abstractmethod
    def compute_sample_metrics(self, prediction: Any, label: Any, sample_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute metrics for a single (prediction, label) pair.

        Parameters
        ----------
        prediction:
            The model's output for one sample.
        label:
            The ground-truth for the same sample.
        sample_id:
            An optional identifier for the sample.

        Returns
        -------
        A dictionary mapping metric names to their values.
        """

    @abstractmethod
    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-sample metrics over the full dataset.

        Parameters
        ----------
        sample_results:
            A list of dictionaries, each returned by
            :meth:`compute_sample_metrics`.

        Returns
        -------
        A dictionary mapping aggregated metric names to their values (e.g.
        mean-AP, precision, recall).
        """

    # ------------------------------------------------------------------
    # Optional visualization interface
    # ------------------------------------------------------------------

    def visualization_outputs(self) -> Dict[str, Any]:
        """Define the ROS message types this benchmark publishes for visualization.

        Returns
        -------
        A dictionary mapping output names to their ROS message types, empty for
        a benchmark that offers no visualization.  The names are node-relative
        topics and match the keys of :meth:`visualize_sample`.
        """
        return {}

    def visualize_sample(self, prediction: Any, label: Any, sample_id: Optional[str] = None, **auxiliary: Any) -> Dict[str, Any]:
        """Build the visualization messages for a single (prediction, label) pair.

        Called once per sample while visualization is enabled, with the same
        arguments as :meth:`record_sample`.

        Returns
        -------
        A dictionary mapping the names of :meth:`visualization_outputs` to ready
        ROS messages, empty for a benchmark that offers no visualization.
        """
        return {}

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def record_sample(
        self,
        prediction: Any,
        label: Any,
        sample_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        **auxiliary: Any,
    ) -> Dict[str, Any]:
        """Compute and store per-sample metrics.

        This is the main entry point used by the evaluation loop.  Any
        keyword arguments in *auxiliary* are forwarded verbatim to
        :meth:`compute_sample_metrics`, allowing benchmarks to receive
        auxiliary per-sample data (e.g. static-map annotations) without
        changing the abstract interface.

        Parameters
        ----------
        scene_id:
            The scene of the dataset the sample belongs to, which
            :meth:`finalize` aggregates the samples by.  An evaluation loop
            that learns the scene only after the sample has been evaluated may
            set it on the returned entry instead of passing it here.

        Returns
        -------
        The stored entry of the sample, as ``{"sample_id", "scene_id",
        "metrics"}``.
        """
        metrics = self.compute_sample_metrics(prediction, label, sample_id, **auxiliary)
        entry: Dict[str, Any] = {"sample_id": sample_id, "scene_id": scene_id, "metrics": metrics}
        self._sample_results.append(entry)
        return entry

    def sample_results_by_scene(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group the recorded samples by the scene of the dataset they belong to.

        Returns
        -------
        A dictionary mapping each scene to its recorded samples, in the order
        the samples were recorded.  Samples recorded without a scene are left
        out, as they cannot be attributed to one.
        """
        scenes: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._sample_results:
            scene_id = entry.get("scene_id")
            if scene_id is not None:
                scenes.setdefault(str(scene_id), []).append(entry)
        return scenes

    def finalize(self) -> Dict[str, Any]:
        """Compute aggregated metrics and return the full results payload.

        Metrics are reported on three levels: ``aggregated_metrics`` over all
        evaluated samples, ``scene_results`` over the samples of each scene, and
        ``sample_results`` for every single sample.
        """
        aggregated = self.compute_aggregated_metrics(self._sample_results)
        scenes = self.sample_results_by_scene()
        return {
            "benchmark": self.name,
            "description": self.description,
            "num_samples": len(self._sample_results),
            "num_scenes": len(scenes),
            "aggregated_metrics": aggregated,
            "scene_results": {
                scene_id: {
                    "num_samples": len(entries),
                    "sample_ids": [entry["sample_id"] for entry in entries],
                    "aggregated_metrics": self.compute_aggregated_metrics(entries),
                }
                for scene_id, entries in scenes.items()
            },
            # "sample_results": self._sample_results,
        }

    def save_results(self, output_path: str, results: Optional[Dict[str, Any]] = None) -> str:
        """Finalize and write results to a JSON file.

        Parameters
        ----------
        output_path:
            Path to the output JSON file.
        results:
            A results payload previously obtained from :meth:`finalize`, which
            is computed here when omitted.

        Returns
        -------
        The absolute path of the written file.
        """
        results = self.finalize() if results is None else results
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        return os.path.abspath(output_path)

    def reset(self) -> None:
        """Clear accumulated sample results."""
        self._sample_results.clear()
