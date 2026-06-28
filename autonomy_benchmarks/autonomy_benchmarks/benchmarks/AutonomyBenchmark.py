"""Abstract base class for all AutonomyHub benchmarks.

Each benchmark defines how to compute per-sample and aggregated metrics for a
specific perception task (e.g. 2-D / 3-D object detection).  Concrete
subclasses must override the abstract methods.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AutonomyBenchmark(ABC):
    """Meta-class (abstract base class) for AutonomyHub benchmarks.

    A benchmark is responsible for:
    * extracting the model input from a dataset sample,
    * computing per-sample metrics from a prediction and the ground-truth label,
    * aggregating per-sample metrics into dataset-level metrics,
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
    def compute_sample_metrics(self, **kwargs) -> Dict[str, Any]:
        """Compute metrics for a single sample.

        Parameters
        ----------
        **kwargs:
            Arbitrary keyword arguments representing the model input and
            ground-truth label for a single sample.

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
        A dictionary mapping aggregated metric names to their values.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def save_results(self, results: Dict[str, Any], output_path: str) -> str:
        """Finalize and write results to a JSON file.

        Parameters
        ----------
        results:
            The results dictionary to save.
        output_path:
            Path to the output JSON file.

        Returns
        -------
        The absolute path of the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        return os.path.abspath(output_path)

    def reset(self) -> None:
        """Clear accumulated sample results."""
        self._sample_results.clear()
