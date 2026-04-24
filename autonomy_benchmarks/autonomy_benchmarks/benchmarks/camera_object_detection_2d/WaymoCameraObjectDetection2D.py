"""Waymo Open Dataset - 2D camera object detection benchmark.

Evaluates 2D bounding-box predictions from camera images against Waymo
ground-truth labels using the official Waymo Open Dataset evaluation protocol.

Key evaluation settings
-----------------------
- Matching:    2D IoU ≥ threshold (class-specific).
- Thresholds:  Vehicle: 0.7, Pedestrian: 0.5, Cyclist: 0.5 (Sign excluded).
- Ranges:      [0, 35), [35, 50), [50, ∞) m BEV distance from vehicle origin.
- Levels:      l1 (LEVEL_1 GTs only), l2 (LEVEL_1 + LEVEL_2 GTs).
- Metrics:     (m)AP - (mean) average precision (trapezoidal integration).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autohub_benchmarks.benchmarks.AutohubBenchmark import AutohubBenchmark
from autohub_benchmarks.utils.BoundingBox2D import (
    BoundingBox2D,
)
from autohub_benchmarks.utils.BoundingBox2DUtils import BoundingBox2DUtils
from autohub_benchmarks.utils.ObjectDetectionUtils import (
    CumulativeResultsTypeNonUnifiedThreshold,
    ObjectDetectionUtils,
)


class WaymoCameraObjectDetection2D(AutohubBenchmark):
    """Benchmark for 2D camera object detection on the Waymo Open Dataset.

    See the module docstring for full evaluation settings. Class IDs and their
    IoU thresholds are:

    ============  ========  =============
    class_name    class_id  IoU_threshold
    ============  ========  =============
    Vehicle       1         0.7
    Pedestrian    2         0.5
    Sign          3         excluded
    Cyclist       4         0.5
    ============  ========  =============
    """

    def __init__(self) -> None:
        """Configure Waymo 2D thresholds, ranges, and score aggregation."""
        super().__init__(
            name="waymo_camera_object_detection_2d",
            description=("2D camera object detection benchmark using the Waymo Open Dataset."),
        )

        self.matching_thresholds: List[Dict[int, float]] = [
            {
                1: 0.7,  # Vehicle
                2: 0.5,  # Pedestrian
                4: 0.5,  # Cyclist
                # class_id 3 (Sign) is intentionally excluded from evaluation
            }
        ]
        self.distance_ranges: Dict[str, Tuple[float, float]] = {
            "00_35": (0.0, 35.0),
            "35_50": (35.0, 50.0),
            "50_inf": (50.0, float("inf")),
        }
        self.levels: List[str] = ["l1", "l2"]
        self.compute_ap_func = ObjectDetectionUtils.compute_ap_trapezoidal

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_input(self, sample: Dict[str, Any]) -> Any:
        """Return the camera image from *sample*."""
        return sample.get("image_front")

    def get_label(self, sample: Dict[str, Any]) -> Any:
        """Return 2D bounding-box annotations from *sample*."""
        return sample.get("objects")

    def compute_sample_metrics(
        self,
        prediction: List[BoundingBox2D],
        label: List[BoundingBox2D],
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pass 1 - match predictions to ground truth for a single frame.

        Groups boxes by difficulty level and distance range, then greedily
        matches each prediction to the highest-IoU unmatched GT per image and
        class. Records ``is_tp``, ``tp_weight``, ``iou``, and
        ``confidence_score`` for every prediction.

        Parameters
        ----------
        prediction:
            Predicted 2D bounding boxes for this frame.
        label:
            Ground-truth 2D bounding boxes for this frame.
        sample_id:
            Optional frame identifier (unused in computation).

        Returns
        -------
        A dict with keys:

        - ``sample_prediction_num``, ``sample_ground_truth_num``
        - ``match_records``: ``lev → range → thresholds_str → class_id → {pred_entries, gt_count}``
        """
        sample_prediction_num = len(prediction)
        sample_ground_truth_num = len(label)

        # rl: distance range, difficulty level.
        pred_boxes_grouped_rl, gt_boxes_grouped_rl = self._prepare_pred_gt_boxes(prediction, label)
        match_records = self._run_matching(pred_boxes_grouped_rl, gt_boxes_grouped_rl)

        return {
            "sample_prediction_num": sample_prediction_num,
            "sample_ground_truth_num": sample_ground_truth_num,
            "match_records": match_records,
        }

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 2 - compute global P-R curve and AP from all per-frame match records.

        Merges the ``match_records`` from each frame into one global ranked list
        per class, then computes AP via trapezoidal integration.

        Parameters
        ----------
        sample_results:
            List of ``{"sample_id": ..., "metrics": {..., "match_records": ...}}``
            entries produced by :meth:`record_sample`.

        Returns
        -------
        A dict with keys:

        - ``aggregated_metrics`` → ``{"threshold_metrics": ..., "benchmark_score": ...}``
        - ``threshold_metrics``: per-level / per-range AP and mAP.
        - ``benchmark_score``: per-class AP and overall mAP for each level.
        """
        # Collect per-frame match records from sample_results.
        all_match_records = [entry["metrics"]["match_records"] for entry in sample_results]
        merged_records = self._merge_match_records(all_match_records)

        # rl: distance range, difficulty level
        # Compute cumulative P-R stats globally for AP from the merged records.
        threshold_results_grouped_rl: Dict[str, Dict[str, CumulativeResultsTypeNonUnifiedThreshold]] = {}
        threshold_metrics_grouped_rl: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for lev in self.levels:
            threshold_results_grouped_rl[lev] = {}
            threshold_metrics_grouped_rl[lev] = {}
            for r in self.distance_ranges:
                threshold_results_grouped_rl[lev][r] = self._compute_global_cumulative_stats(merged_records[lev][r])
                threshold_metrics_grouped_rl[lev][r] = ObjectDetectionUtils.compute_ap_thresholds(
                    threshold_results_grouped_rl[lev][r], self.compute_ap_func
                )

        benchmark_score = self._compute_benchmark_score(threshold_metrics_grouped_rl)

        return {
            "aggregated_metrics": {
                "threshold_metrics": threshold_metrics_grouped_rl,
                "benchmark_score": benchmark_score,
            },
        }

    # ------------------------------------------------------------------
    # Helper functions for metric computation
    # ------------------------------------------------------------------

    def _prepare_pred_gt_boxes(self, pred_boxes: List[BoundingBox2D], gt_boxes: List[BoundingBox2D]) -> Tuple[
        Dict[str, Dict[str, List[BoundingBox2D]]],
        Dict[str, Dict[str, List[BoundingBox2D]]],
    ]:
        """Prepare prediction and ground-truth boxes for matching.

        GT boxes are split by difficulty level (l1/l2) and distance range.
        For l1 evaluation only LEVEL_1 GTs are included; for l2 both LEVEL_1
        and LEVEL_2 GTs are included. The difficulty level is taken directly
        from ``box.group``.

        Predictions are not range-filtered here: distance to a 2D bounding box
        in camera space is not directly observable, so the same prediction set
        is used for all distance ranges and difficulty levels. Matching against
        range-filtered GTs ensures predictions only score TPs within each range.

        Raises
        ------
        ValueError
            If any GT box has ``distance=None``.  A valid BEV distance must be
            pre-populated by the dataset loader (derived from the corresponding
            3D annotation's world-space coordinates) — it cannot be inferred
            from 2D pixel coordinates alone.

        Returns
        -------
        Tuple ``(pred_boxes_grouped_rl, gt_boxes_grouped_rl)`` where each is a nested dict
        ``level → range_key → List[BoundingBox2D]``.
        """
        # rl: distance range, difficulty level
        # Predictions are the same for every level/range combination.
        pred_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox2D]]] = {}
        for lev in self.levels:
            pred_boxes_grouped_rl[lev] = {}
            for r in self.distance_ranges:
                pred_boxes_grouped_rl[lev][r] = pred_boxes if pred_boxes is not None else []

        # GT boxes are filtered by difficulty level and distance range.
        gt_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox2D]]] = {}
        for lev in self.levels:
            gt_boxes_grouped_rl[lev] = {}
            # l1 includes only l1 ground truths; l2 includes both l1 and l2.
            included_groups = {"l1": ["l1"], "l2": ["l1", "l2"]}[lev]
            gt_boxes_level = [box for box in gt_boxes if box.group in included_groups] if gt_boxes is not None else []
            for r, (start, end) in self.distance_ranges.items():
                for box in gt_boxes_level:
                    if box.distance is None:
                        raise ValueError(
                            f"GT box (class_id={box.class_id}, image_id={box.image_id}) has distance=None. "
                            "A valid BEV distance is required for range-based evaluation in the Waymo benchmark. "
                            "Ensure the dataset loader populates BoundingBox2D.distance with the BEV distance "
                            "derived from the corresponding 3D annotation."
                        )
                gt_boxes_grouped_rl[lev][r] = [box for box in gt_boxes_level if start <= box.distance < end]

        return pred_boxes_grouped_rl, gt_boxes_grouped_rl

    def _run_matching(
        self,
        pred_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox2D]]],
        gt_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox2D]]],
    ) -> Dict:
        """Pass 1 - match predictions to GTs per image for all levels, ranges, thresholds, and classes.

        Returns match_records structured as:
        ``lev → range → thresholds_str → class_id → {pred_entries, gt_count}``

        Each entry in ``pred_entries`` contains:

        - ``confidence_score``: prediction confidence.
        - ``iou``: IoU with matched GT (0.0 if unmatched).
        - ``is_tp``: whether the prediction was matched to a GT.
        - ``tp_weight``: ``1.0`` if TP else ``0.0`` (used for AP).
        """
        match_records: Dict = {}
        for lev in self.levels:
            match_records[lev] = {}
            for r in self.distance_ranges:
                match_records[lev][r] = {}
                pred_boxes = pred_boxes_grouped_rl[lev][r]
                gt_boxes = gt_boxes_grouped_rl[lev][r]

                gt_by_image_and_class: Dict[int, Dict[int, List[BoundingBox2D]]] = {}
                for gt in gt_boxes:
                    gt_by_image_and_class.setdefault(gt.image_id, {}).setdefault(gt.class_id, []).append(gt)
                pred_by_image_and_class: Dict[int, Dict[int, List[BoundingBox2D]]] = {}
                for pred in pred_boxes:
                    pred_by_image_and_class.setdefault(pred.image_id, {}).setdefault(pred.class_id, []).append(pred)

                all_class_ids: set = set()
                for d in list(gt_by_image_and_class.values()) + list(pred_by_image_and_class.values()):
                    all_class_ids.update(d.keys())
                all_image_ids = set(gt_by_image_and_class) | set(pred_by_image_and_class)

                for threshold_dict in self.matching_thresholds:
                    threshold_dict_str = "_".join(f"{cls}:{thresh:.2f}" for cls, thresh in threshold_dict.items())
                    match_records[lev][r][threshold_dict_str] = {}
                    for class_id in all_class_ids:
                        # Classes labeled in the dataset but excluded from the benchmark are skipped.
                        if class_id not in threshold_dict:
                            continue
                        threshold = threshold_dict[class_id]
                        pred_entries: List[Dict] = []
                        gt_count = 0

                        for image_id in all_image_ids:
                            gts = gt_by_image_and_class.get(image_id, {}).get(class_id, [])
                            preds = pred_by_image_and_class.get(image_id, {}).get(class_id, [])
                            gt_count += len(gts)
                            preds = sorted(preds, key=lambda p: -p.confidence_score)
                            matched_gt_indices: set = set()
                            for pred in preds:
                                best_iou = 0.0
                                best_gt_idx = -1
                                # Match to the highest-IoU unmatched GT that meets the threshold.
                                for gt_idx, gt in enumerate(gts):
                                    if gt_idx in matched_gt_indices:
                                        continue
                                    iou = BoundingBox2DUtils.compute_iou(pred, gt)
                                    if iou >= threshold and iou > best_iou:
                                        best_iou = iou
                                        best_gt_idx = gt_idx
                                if best_gt_idx != -1:
                                    matched_gt_indices.add(best_gt_idx)
                                    is_tp = True
                                    tp_weight = 1.0
                                else:
                                    is_tp = False
                                    tp_weight = 0.0
                                pred_entries.append(
                                    {
                                        "confidence_score": pred.confidence_score,
                                        "iou": best_iou,
                                        "is_tp": is_tp,
                                        "tp_weight": tp_weight,
                                    }
                                )

                        match_records[lev][r][threshold_dict_str][class_id] = {
                            "pred_entries": pred_entries,
                            "gt_count": gt_count,
                        }

        return match_records

    def _merge_match_records(self, all_match_records: List[Dict]) -> Dict:
        """Merge per-frame match records into a single dataset-level structure.

        Concatenates ``pred_entries`` lists and sums ``gt_count`` across all
        frames, grouped by level / distance range / threshold string / class ID.
        """
        merged: Dict = {}
        for frame_records in all_match_records:
            for lev, ranges in frame_records.items():
                merged.setdefault(lev, {})
                for r, threshold_records in ranges.items():
                    merged[lev].setdefault(r, {})
                    for thresholds_str, class_records in threshold_records.items():
                        merged[lev][r].setdefault(thresholds_str, {})
                        for class_id, data in class_records.items():
                            if class_id not in merged[lev][r][thresholds_str]:
                                merged[lev][r][thresholds_str][class_id] = {
                                    "pred_entries": [],
                                    "gt_count": 0,
                                }
                            merged[lev][r][thresholds_str][class_id]["pred_entries"].extend(data["pred_entries"])
                            merged[lev][r][thresholds_str][class_id]["gt_count"] += data["gt_count"]
        return merged

    def _compute_global_cumulative_stats(
        self,
        range_records: Dict[str, Dict[int, Dict[str, Any]]],
    ) -> CumulativeResultsTypeNonUnifiedThreshold:
        """Pass 2 - compute global cumulative P-R statistics from merged match records.

        Sorts all predictions globally by confidence descending and accumulates
        cumulative TP, FP, FN, precision, and recall across the entire dataset.
        Each TP contributes ``1.0`` (standard AP; no heading weight in 2D).
        """
        cumulative_results: CumulativeResultsTypeNonUnifiedThreshold = {}
        for thresholds_str, class_records in range_records.items():
            threshold_results = cumulative_results.setdefault(thresholds_str, {})
            for class_id, data in class_records.items():
                pred_entries = sorted(data["pred_entries"], key=lambda x: -x["confidence_score"])
                total_gt_count = data["gt_count"]
                class_results = threshold_results.setdefault(class_id, {})
                cum_tp, cum_fp = 0.0, 0.0
                num_matched = 0
                for pred_idx, entry in enumerate(pred_entries):
                    if entry["is_tp"]:
                        cum_tp += entry["tp_weight"]
                        num_matched += 1
                    else:
                        cum_fp += 1.0
                    cum_fn = float(total_gt_count - num_matched)
                    cum_precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
                    cum_recall = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0
                    class_results[pred_idx] = {
                        "confidence_score": entry["confidence_score"],
                        "iou": entry["iou"],
                        "cum_tp": cum_tp,
                        "cum_fp": cum_fp,
                        "cum_fn": cum_fn,
                        "cum_precision": cum_precision,
                        "cum_recall": cum_recall,
                    }
        return cumulative_results

    def _compute_benchmark_score(
        self,
        threshold_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute the final benchmark score from per-level, per-range metrics.

        Averages mAP across all distance ranges for each difficulty level and
        extracts per-class AP for Vehicle, Pedestrian, and Cyclist.

        Returns
        -------
        Flat dict with keys such as ``"ap_l1_vehicle"``, ``"ap_l2_pedestrian"``,
        ``"ap_l1_all-ns"`` (overall l1 mAP), ``"ap_l2_all-ns"`` (overall l2 mAP).
        """

        def compute_score_case(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
            overalls: List[float] = []
            per_class: Dict[int, List[float]] = {}
            for metrics in metrics_list:
                overalls.append(metrics["overall_map"])
                for metrics_threshold in metrics["thresholds"].values():
                    for class_id, class_metric in metrics_threshold["per_class_metrics"].items():
                        per_class.setdefault(class_id, []).append(class_metric["ap"])
            return {
                "overall": round(float(np.mean(overalls)), 4) if overalls else 0.0,
                "per_class": {class_id: round(float(np.mean(vals)), 4) for class_id, vals in sorted(per_class.items())},
            }

        l1_scores = compute_score_case(list(threshold_metrics["l1"].values()))
        l2_scores = compute_score_case(list(threshold_metrics["l2"].values()))

        vehicle_id, pedestrian_id, cyclist_id = 1, 2, 4
        benchmark_score: Dict[str, float] = {}
        benchmark_score["ap_l1_vehicle"] = l1_scores["per_class"].get(vehicle_id)
        benchmark_score["ap_l1_pedestrian"] = l1_scores["per_class"].get(pedestrian_id)
        benchmark_score["ap_l1_cyclist"] = l1_scores["per_class"].get(cyclist_id)
        benchmark_score["ap_l2_vehicle"] = l2_scores["per_class"].get(vehicle_id)
        benchmark_score["ap_l2_pedestrian"] = l2_scores["per_class"].get(pedestrian_id)
        benchmark_score["ap_l2_cyclist"] = l2_scores["per_class"].get(cyclist_id)
        benchmark_score["ap_l1_all-ns"] = l1_scores["overall"]
        benchmark_score["ap_l2_all-ns"] = l2_scores["overall"]

        return benchmark_score
