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


class AutohubBenchmark(ABC):
    """Meta-class (abstract base class) for AutonomyHub benchmarks.

    A benchmark is responsible for:
    * extracting the model input from a dataset sample,
    * computing per-sample metrics from a prediction and the ground-truth label,
    * aggregating per-sample metrics into dataset-level metrics,
    * persisting results to JSON.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name: str = name
        self.description: str = description
        self._sample_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface – must be implemented by every concrete benchmark
    # ------------------------------------------------------------------

    @abstractmethod
    def get_input(self, sample: Dict[str, Any]) -> Any:
        """Extract the model input tensor / data from a dataset *sample*.

        Parameters
        ----------
        sample:
            A single example from the ``tensorflow_datasets`` dataset,
            typically a dictionary of tensors.

        Returns
        -------
        The data that should be fed to the model adapter's ``predict`` method.
        """

    @abstractmethod
    def get_label(self, sample: Dict[str, Any]) -> Any:
        """Extract the ground-truth label from a dataset *sample*.

        Parameters
        ----------
        sample:
            A single example from the ``tensorflow_datasets`` dataset.

        Returns
        -------
        The ground-truth annotation(s) corresponding to the sample.
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
    # Concrete helpers
    # ------------------------------------------------------------------

    def record_sample(
        self, prediction: Any, label: Any, sample_id: Optional[str] = None, **auxiliary: Any
    ) -> Dict[str, Any]:
        """Compute and store per-sample metrics.

        This is the main entry point used by the evaluation loop.  Any
        keyword arguments in *auxiliary* are forwarded verbatim to
        :meth:`compute_sample_metrics`, allowing benchmarks to receive
        auxiliary per-sample data (e.g. static-map annotations) without
        changing the abstract interface.
        """
        metrics = self.compute_sample_metrics(prediction, label, sample_id, **auxiliary)
        entry: Dict[str, Any] = {"sample_id": sample_id, "metrics": metrics}
        self._sample_results.append(entry)
        return entry

    def finalize(self) -> Dict[str, Any]:
        """Compute aggregated metrics and return the full results payload."""
        aggregated = self.compute_aggregated_metrics(self._sample_results)
        return {
            "benchmark": self.name,
            "description": self.description,
            "num_samples": len(self._sample_results),
            "aggregated_metrics": aggregated,
            "sample_results": self._sample_results,
        }

    def save_results(self, output_path: str) -> str:
        """Finalize and write results to a JSON file.

        Parameters
        ----------
        output_path:
            Path to the output JSON file.

        Returns
        -------
        The absolute path of the written file.
        """
        results = self.finalize()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        return os.path.abspath(output_path)

    def reset(self) -> None:
        """Clear accumulated sample results."""
        self._sample_results.clear()
