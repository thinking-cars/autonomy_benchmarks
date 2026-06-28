"""nuScenes - Object tracking benchmark."""

from __future__ import annotations

from typing import Any, Dict, List

# import perception_msgs_utils as pmu

from autonomy_benchmarks.benchmarks.AutonomyBenchmark import AutonomyBenchmark
from perception_msgs.msg import ObjectList


class NuscenesObjectTracking(AutonomyBenchmark):
    """Benchmark for object tracking on the nuScenes dataset."""

    def __init__(self) -> None:
        """Configure benchmark."""
        super().__init__(
            name="nuscenes_object_tracking",
            description=("3D bounding-box object tracking benchmark using the nuScenes dataset."),
        )

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def required_inputs(self) -> Dict[str, Any]:
        """Define expected input ROS message types."""

        return {
            "objects_tracked": ObjectList,
            "objects_truth": ObjectList,
        }

    def compute_sample_metrics(self, **kwargs) -> Dict[str, Any]:
        """Compute metrics per sample

        Returns:
            Dict[str, Any]: metrics computed for the sample, with metric names as keys and metric values as values
        """

        object_list_tracked: ObjectList = kwargs["objects_tracked"]
        object_list_ground_truth: ObjectList = kwargs["objects_truth"]

        # TODO(RaphvK): Implement the actual object tracking evaluation logic here.

        return {
            "sample_prediction_num": len(object_list_tracked.objects),
            "sample_ground_truth_num": len(object_list_ground_truth.objects),
        }

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregated metrics over all samples

        Args:
            sample_results (List[Dict[str, Any]]): List of metrics computed for each sample,
                with metric names as keys and metric values as values

        Returns:
            Dict[str, Any]: Aggregated metrics over all samples, with metric names as keys and aggregated metric values as values
        """

        # TODO(RaphvK): Implement the actual aggregation logic here.

        return {
            "num_samples": len(sample_results),
        }
