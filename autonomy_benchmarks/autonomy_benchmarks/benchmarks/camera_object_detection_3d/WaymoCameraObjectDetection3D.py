"""Waymo Open Dataset camera object detection benchmark.

Evaluates 3D bounding-box predictions from camera images against Waymo
ground-truth labels using the official Waymo Open Dataset evaluation protocol.

Key evaluation settings
-----------------------
- Matching:    3D IoU ≥ threshold (class-specific).
- Thresholds:  Vehicle: 0.7, Pedestrian: 0.5, Cyclist: 0.5 (Sign excluded).
- Ranges:      [0, 35), [35, 50), [50, ∞) m BEV distance from vehicle origin.
- Levels:      l1 (LEVEL_1 GTs only), l2 (LEVEL_1 + LEVEL_2 GTs).
- Difficulty:  resolved per box via lidar point count and explicit group label.
- Metrics:     (m)AP  - (mean) average precision (trapezoidal integration).
               (m)APH - (mean) average precision weighted by heading accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_benchmarks.benchmarks.AutonomyBenchmark import AutonomyBenchmark
from autonomy_benchmarks.utils.BoundingBox3D import BoundingBox3D
from autonomy_benchmarks.utils.BoundingBox3DUtils import BoundingBox3DUtils
from autonomy_benchmarks.utils.ObjectDetectionUtils import CumulativeResultsTypeNonUnifiedThreshold, ObjectDetectionUtils
from perception_msgs.msg import ObjectList


class WaymoCameraObjectDetection3D(AutonomyBenchmark):
    """Benchmark for 3D camera object detection on the Waymo Open Dataset.

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
        """Configure Waymo 3D thresholds, ranges, and weighted metrics."""
        super().__init__(
            name="waymo_camera_object_detection_3d",
            description=("3D bounding-box object detection benchmark using the Waymo Open Dataset."),
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
        self.compute_iou_func = BoundingBox3DUtils.compute_iou_3d
        self.compute_dist_func = BoundingBox3DUtils.compute_dist_bev
        self.compute_ap_func = ObjectDetectionUtils.compute_ap_trapezoidal

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def required_inputs(self) -> Dict[str, Any]:
        """Define expected input ROS message types."""

        return {
            "/object_list_label": ObjectList,
            "/object_list_prediction": ObjectList,
        }

    def compute_sample_metrics(
        self, prediction: List[BoundingBox3D], label: List[BoundingBox3D], sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Pass 1 - match predictions to ground truth for a single frame.

        Filters boxes by distance range and difficulty level, then greedily
        matches each prediction to the highest-IoU unmatched GT per image and
        class. Records ``is_tp``, ``tp_weight_ap``, ``tp_weight_aph``, ``iou``,
        and ``confidence_score`` for every prediction.

        Parameters
        ----------
        prediction:
            Predicted 3D bounding boxes for this frame.
        label:
            Ground-truth 3D bounding boxes for this frame.
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
        """Pass 2 - compute global P-R curve and AP/APH from all per-frame match records.

        Merges the ``match_records`` from each frame into one global ranked list
        per class, then computes AP and APH via trapezoidal integration.

        Parameters
        ----------
        sample_results:
            List of ``{"sample_id": ..., "metrics": {..., "match_records": ...}}``
            entries produced by :meth:`record_sample`.

        Returns
        -------
        A dict with keys:

        - ``aggregated_metrics`` → ``{"threshold_metrics", "threshold_metrics_with_heading", "benchmark_score"}``
        - ``threshold_metrics``: per-level / per-range AP and mAP.
        - ``threshold_metrics_with_heading``: same structure, weighted by heading accuracy (APH).
        - ``benchmark_score``: per-class AP / APH and overall mAP / mAPH for each level.
        """
        # Collect per-frame match records from sample_results.
        all_match_records = [entry["metrics"]["match_records"] for entry in sample_results]
        merged_records = self._merge_match_records(all_match_records)

        # rl: distance range, difficulty level
        # Compute cumulative P-R stats globally for AP and APH from the merged records.
        Type = CumulativeResultsTypeNonUnifiedThreshold
        threshold_results_grouped_rl: Dict[str, Dict[str, Type]] = {}
        threshold_metrics_grouped_rl: Dict[str, Dict[str, Dict[str, Any]]] = {}
        threshold_results_grouped_rl_wh: Dict[str, Dict[str, Type]] = {}
        threshold_metrics_grouped_rl_wh: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for lev in self.levels:
            threshold_results_grouped_rl[lev] = {}
            threshold_metrics_grouped_rl[lev] = {}
            threshold_results_grouped_rl_wh[lev] = {}
            threshold_metrics_grouped_rl_wh[lev] = {}
            for r in self.distance_ranges:
                # AP: each TP contributes 1.0.
                threshold_results_grouped_rl[lev][r] = self._compute_global_cumulative_stats(
                    merged_records[lev][r], weight_key="tp_weight_ap"
                )
                # APH: each TP is weighted by heading accuracy in [0, 1].
                threshold_results_grouped_rl_wh[lev][r] = self._compute_global_cumulative_stats(
                    merged_records[lev][r], weight_key="tp_weight_aph"
                )
                threshold_metrics_grouped_rl[lev][r] = ObjectDetectionUtils.compute_ap_thresholds(
                    threshold_results_grouped_rl[lev][r], self.compute_ap_func
                )
                threshold_metrics_grouped_rl_wh[lev][r] = ObjectDetectionUtils.compute_ap_thresholds(
                    threshold_results_grouped_rl_wh[lev][r], self.compute_ap_func
                )

        benchmark_score = self._compute_benchmark_score(threshold_metrics_grouped_rl, threshold_metrics_grouped_rl_wh)

        return {
            "aggregated_metrics": {
                "threshold_metrics": threshold_metrics_grouped_rl,
                "threshold_metrics_with_heading": threshold_metrics_grouped_rl_wh,
                "benchmark_score": benchmark_score,
            },
        }

    # ------------------------------------------------------------------
    # Helper functions for metric computation
    # ------------------------------------------------------------------

    def _prepare_pred_gt_boxes(self, pred_boxes: List[BoundingBox3D], gt_boxes: List[BoundingBox3D]) -> Tuple[
        Dict[str, Dict[str, List[BoundingBox3D]]],
        Dict[str, Dict[str, List[BoundingBox3D]]],
    ]:
        """Prepare prediction and ground-truth boxes for matching.

        Separates boxes into the three distance ranges [0, 35), [35, 50),
        [50, ∞) using BEV distance to the vehicle origin. GT boxes are
        additionally split by difficulty level (l1/l2) via
        :func:`_resolve_difficulty`; for l1 evaluation only LEVEL_1 GTs are
        included, for l2 both LEVEL_1 and LEVEL_2 GTs are included.

        Returns
        -------
        Tuple ``(pred_boxes_rl, gt_boxes_rl)`` where each is a nested dict
        ``level → range_key → List[BoundingBox3D]``.
        """

        def _resolve_difficulty(box: BoundingBox3D) -> Optional[str]:
            """Resolve the difficulty level of a ground truth box.

            Follows the Waymo difficulty definition:

            - **LEVEL_2** (``"l2"``): ``number_of_lidar_points >= 1 and <= 5``, **or**
              explicitly marked as LEVEL_2 in the released data (``group="l2"``).
              An explicit LEVEL_2 label always takes priority, regardless of the
              point count.
            - **LEVEL_1** (``"l1"``): ``number_of_lidar_points > 5`` **and** not marked as
              LEVEL_2 in the released data. A ``group="l1"`` label does not
              override the point-count rule — the count still determines the final
              level.

            Returns ``None`` when the difficulty cannot be determined (no usable
            point count and not marked LEVEL_2), so that the caller can exclude
            the box.
            """
            # Explicitly marked LEVEL_2 in the released data — always LEVEL_2.
            if box.group == "l2":
                return "l2"
            # Apply the point-count rule (also when group="l1" or group is unset).
            if box.number_of_lidar_points is not None:
                if box.number_of_lidar_points > 5:
                    return "l1"
                if box.number_of_lidar_points >= 1:
                    return "l2"
            return None

        # rl: distance range, difficulty level
        pred_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox3D]]] = {}
        for lev in self.levels:
            pred_boxes_grouped_rl[lev] = {}
            for r, (start, end) in self.distance_ranges.items():
                pred_boxes_grouped_rl[lev][r] = BoundingBox3DUtils.boxes_filter(
                    pred_boxes, start, end, self.compute_dist_func, include_start=True, include_end=False
                )
        gt_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox3D]]] = {}
        for lev in self.levels:
            gt_boxes_grouped_rl[lev] = {}
            # l1 includes only l1 ground truths; l2 includes both l1 and l2 ground truths.
            included_groups = {"l1": ["l1"], "l2": ["l1", "l2"]}[lev]
            gt_boxes_levels = [box for box in gt_boxes if _resolve_difficulty(box) in included_groups]
            for r, (start, end) in self.distance_ranges.items():
                gt_boxes_grouped_rl[lev][r] = BoundingBox3DUtils.boxes_filter(
                    gt_boxes_levels, start, end, self.compute_dist_func, include_start=True, include_end=False
                )

        return pred_boxes_grouped_rl, gt_boxes_grouped_rl

    def _run_matching(
        self,
        pred_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox3D]]],
        gt_boxes_grouped_rl: Dict[str, Dict[str, List[BoundingBox3D]]],
    ) -> Dict:
        """Pass 1 - match predictions to GTs per image for all levels, ranges, thresholds, and classes.

        Returns match_records structured as:
        ``lev → range → thresholds_str → class_id → {pred_entries, gt_count}``

        Each entry in ``pred_entries`` contains:

        - ``confidence_score``: prediction confidence.
        - ``iou``: IoU with matched GT (0.0 if unmatched).
        - ``is_tp``: whether the prediction was matched to a GT.
        - ``tp_weight_ap``: ``1.0`` if TP else ``0.0`` (used for AP).
        - ``tp_weight_aph``: heading accuracy in ``[0, 1]`` if TP else ``0.0`` (used for APH).

        AP and APH weights are both computed in a single matching pass so
        that :meth:`_compute_global_cumulative_stats` can handle both without
        repeating the matching.
        """
        match_records: Dict = {}
        for lev in self.levels:
            match_records[lev] = {}
            for r in self.distance_ranges:
                match_records[lev][r] = {}
                pred_boxes = pred_boxes_grouped_rl[lev][r]
                gt_boxes = gt_boxes_grouped_rl[lev][r]

                gt_by_image_and_class: Dict[int, Dict[int, List[BoundingBox3D]]] = {}
                for gt in gt_boxes:
                    gt_by_image_and_class.setdefault(gt.image_id, {}).setdefault(gt.class_id, []).append(gt)
                pred_by_image_and_class: Dict[int, Dict[int, List[BoundingBox3D]]] = {}
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
                                    iou = self.compute_iou_func(pred, gt)
                                    if iou >= threshold and iou > best_iou:
                                        best_iou = iou
                                        best_gt_idx = gt_idx
                                if best_gt_idx != -1:
                                    matched_gt = gts[best_gt_idx]
                                    heading_diff = abs(pred.yaw - matched_gt.yaw)
                                    min_heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                                    matched_gt_indices.add(best_gt_idx)
                                    is_tp = True
                                    tp_weight_ap = 1.0
                                    tp_weight_aph = 1.0 - min_heading_diff / np.pi  # in [0, 1]
                                else:
                                    is_tp = False
                                    tp_weight_ap = 0.0
                                    tp_weight_aph = 0.0
                                pred_entries.append(
                                    {
                                        "confidence_score": pred.confidence_score,
                                        "iou": best_iou,
                                        "is_tp": is_tp,
                                        "tp_weight_ap": tp_weight_ap,
                                        "tp_weight_aph": tp_weight_aph,
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
        weight_key: str,
    ) -> CumulativeResultsTypeNonUnifiedThreshold:
        """Pass 2 - compute global cumulative P-R statistics from merged match records.

        Sorts all predictions globally by confidence descending and accumulates
        cumulative TP, FP, FN, precision, and recall across the entire dataset.
        ``weight_key`` selects which TP weight to accumulate:

        - ``"tp_weight_ap"``: each TP contributes ``1.0`` → standard AP.
        - ``"tp_weight_aph"``: each TP contributes heading accuracy in ``[0, 1]`` → APH.

        FP and FN always count as ``1.0`` regardless of heading.
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
                        cum_tp += entry[weight_key]
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
        threshold_metrics_with_heading: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute the benchmark scores (AP and APH) for all level / class combinations.

        Aggregates per-distance-range mAP and mAPH values into overall and per-class
        scores for both difficulty levels (l1, l2).  The final dict keys follow the
        convention ``{metric}_{level}_{class}`` (e.g. ``"ap_l1_vehicle"``) and
        ``{metric}_{level}_all-ns`` for the macro average across classes.
        """

        def compute_score_case(label: str, metrics_list: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
            overalls = []
            per_class: Dict[int, List[float]] = {}
            for metrics in metrics_list:
                overalls.append(metrics["overall_map"])
                for metrics_threshold in metrics["thresholds"].values():
                    for class_id, class_metric in metrics_threshold["per_class_metrics"].items():
                        per_class.setdefault(class_id, []).append(class_metric["ap"])
            return label, {
                "overall": round(float(np.mean(overalls)), 4) if overalls else 0.0,
                "per_class": {class_id: round(float(np.mean(vals)), 4) for class_id, vals in sorted(per_class.items())},
            }

        # Construct the benchmark score.
        scores: Dict[str, Dict[str, Any]] = {}
        # mAP cases
        map_cases = [
            ("l1_ap", [m for m in threshold_metrics["l1"].values()]),
            ("l2_ap", [m for m in threshold_metrics["l2"].values()]),
        ]
        # mAPH cases
        maph_cases = [
            ("l1_aph", [m for m in threshold_metrics_with_heading["l1"].values()]),
            ("l2_aph", [m for m in threshold_metrics_with_heading["l2"].values()]),
        ]
        for label, metrics_list in map_cases + maph_cases:
            case_label, case_result = compute_score_case(label, metrics_list)
            scores[case_label] = case_result

        # Transform to designed format for the final benchmark score.
        l1_ap_class_scores = scores["l1_ap"]["per_class"]
        l2_ap_class_scores = scores["l2_ap"]["per_class"]
        l1_aph_class_scores = scores["l1_aph"]["per_class"]
        l2_aph_class_scores = scores["l2_aph"]["per_class"]
        vehicle_id, pedestrian_id, cyclist_id = 1, 2, 4
        benchmark_score: Dict[str, float] = {}
        benchmark_score["ap_l1_vehicle"] = l1_ap_class_scores.get(vehicle_id)
        benchmark_score["ap_l1_pedestrian"] = l1_ap_class_scores.get(pedestrian_id)
        benchmark_score["ap_l1_cyclist"] = l1_ap_class_scores.get(cyclist_id)
        benchmark_score["ap_l2_vehicle"] = l2_ap_class_scores.get(vehicle_id)
        benchmark_score["ap_l2_pedestrian"] = l2_ap_class_scores.get(pedestrian_id)
        benchmark_score["ap_l2_cyclist"] = l2_ap_class_scores.get(cyclist_id)
        benchmark_score["ap_l1_all-ns"] = scores["l1_ap"]["overall"]
        benchmark_score["ap_l2_all-ns"] = scores["l2_ap"]["overall"]
        benchmark_score["aph_l1_vehicle"] = l1_aph_class_scores.get(vehicle_id)
        benchmark_score["aph_l1_pedestrian"] = l1_aph_class_scores.get(pedestrian_id)
        benchmark_score["aph_l1_cyclist"] = l1_aph_class_scores.get(cyclist_id)
        benchmark_score["aph_l2_vehicle"] = l2_aph_class_scores.get(vehicle_id)
        benchmark_score["aph_l2_pedestrian"] = l2_aph_class_scores.get(pedestrian_id)
        benchmark_score["aph_l2_cyclist"] = l2_aph_class_scores.get(cyclist_id)
        benchmark_score["aph_l1_all-ns"] = scores["l1_aph"]["overall"]
        benchmark_score["aph_l2_all-ns"] = scores["l2_aph"]["overall"]

        return benchmark_score
