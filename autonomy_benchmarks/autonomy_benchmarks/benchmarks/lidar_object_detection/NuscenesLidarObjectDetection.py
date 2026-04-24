"""nuScenes - Lidar object detection benchmark.

Evaluates 3D bounding-box predictions from lidar point cloud input against
nuScenes ground-truth labels using the official nuScenes evaluation protocol.

Key evaluation settings
-----------------------
- Matching:   BEV center-point Euclidean distance ≤ threshold.
- Thresholds: 0.5, 1.0, 2.0, 4.0 m (applied uniformly to all classes).
- Ranges:     Per-class max detection distance (barrier/cone ≤ 30 m,
              bicycle/motorcycle/pedestrian ≤ 40 m, others ≤ 50 m).
- GT filter:  GT boxes with 0 lidar and 0 radar points are excluded.
- Bike-rack:  Bicycle (class 8) and motorcycle (class 7) boxes whose BEV
              center falls inside a bike-rack annotation are excluded from
              both predictions and GT before matching.
- AP filter:  P-R points with cum_precision <= 0.1 or cum_recall <= 0.1
              are excluded from the trapezoidal integration.
- TP metrics: ATE, ASE, AOE, AVE, AAE measured at the 2.0 m threshold.
              AOE for traffic_cone is ignored (set to 0); for barrier
              it is capped at π (180°). AVE and AAE for barrier/traffic_cone
              are ignored (set to 0 / not evaluated). AAE requires
              BoundingBox3D.attribute to be populated by the dataset loader;
              when attribute data is absent, AAE is skipped per TP entry.
- NDS:        nuScenes Detection Score combining mAP, mATE, mASE, mAOE,
              mAVE, mAAE with weights 5-1-1-1-1-1, normalised by 10.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autohub_benchmarks.benchmarks.AutohubBenchmark import AutohubBenchmark
from autohub_benchmarks.utils.BoundingBox3D import BoundingBox3D
from autohub_benchmarks.utils.BoundingBox3DUtils import BoundingBox3DUtils
from autohub_benchmarks.utils.ObjectDetectionUtils import ObjectDetectionUtils
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import box as shapely_box
from shapely.geometry import Point


class NuscenesLidarObjectDetection(AutohubBenchmark):
    """Benchmark for lidar 3D object detection on the nuScenes dataset.

    See the module docstring for full evaluation settings. Class IDs evaluated
    with a uniform BEV distance threshold (0.5 / 1.0 / 2.0 / 4.0 m) are:

    ====================  ========
    class_name            class_id
    ====================  ========
    car                   1
    truck                 2
    bus                   3
    trailer               4
    construction_vehicle  5
    pedestrian            6
    motorcycle            7
    bicycle               8
    traffic_cone          9
    barrier               10
    ====================  ========
    """

    def __init__(self) -> None:
        """Configure nuScenes thresholds, class ranges, and TP metric rules."""
        super().__init__(
            name="nuscenes_lidar_object_detection",
            description=("3D bounding-box object detection benchmark from lidar point clouds using the nuScenes dataset."),
        )

        # BEV distance thresholds used for AP and TP-metric computation.
        self.matching_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
        # Distance threshold used exclusively for TP error metrics.
        self.tp_metric_threshold: float = 2.0
        # Minimum precision required for a P-R point to enter AP integration.
        self.min_precision: float = 0.1
        # Minimum recall required for a P-R point to enter AP integration.
        self.min_recall: float = 0.1
        # Per-class maximum detection distance (BEV). Pred and GT boxes beyond
        # the class-specific limit are excluded before matching.
        self.per_class_detection_ranges: Dict[int, Tuple[float, float]] = {
            1: (0.0, 50.0),  # car
            2: (0.0, 50.0),  # truck
            3: (0.0, 50.0),  # bus
            4: (0.0, 50.0),  # trailer
            5: (0.0, 50.0),  # construction_vehicle
            6: (0.0, 40.0),  # pedestrian
            7: (0.0, 40.0),  # motorcycle
            8: (0.0, 40.0),  # bicycle
            9: (0.0, 30.0),  # traffic_cone
            10: (0.0, 30.0),  # barrier
        }
        # Class IDs for which orientation error is ignored (set to 0).
        self.aoe_ignored_classes: set = {9}  # traffic_cone
        # Class IDs for which orientation error is capped at π (180° symmetry).
        self.aoe_pi_classes: set = {10}  # barrier
        # Class IDs for which velocity error is ignored (set to 0).
        self.ave_ignored_classes: set = {9, 10}  # traffic_cone, barrier
        # Class IDs for which attribute error is ignored (void attribute in spec).
        self.aae_ignored_classes: set = {9, 10}  # traffic_cone, barrier
        # Class IDs filtered out when their BEV center falls inside a bike-rack.
        self.bike_rack_filtered_classes: frozenset = frozenset({7, 8})  # motorcycle=7, bicycle=8
        # Class ID of static bike-rack annotations in the GT object list.
        self.bike_rack_class_id: int = 12
        self.compute_dist_func = BoundingBox3DUtils.compute_dist_bev
        self.compute_ap_func = ObjectDetectionUtils.compute_ap_trapezoidal

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_input(self, sample: Dict[str, Any]) -> Any:
        """Return the lidar point cloud from *sample*."""
        return sample.get("point_cloud")

    def get_label(self, sample: Dict[str, Any]) -> Any:
        """Return 3D bounding-box annotations from *sample*."""
        return sample.get("objects")

    def compute_sample_metrics(
        self,
        prediction: List[BoundingBox3D],
        label: List[BoundingBox3D],
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pass 1 - match predictions to ground truth for a single frame.

        Uses greedy BEV-distance matching (highest-confidence predictions
        first).  For each prediction records:

        - ``confidence_score``
        - ``dist`` - BEV distance to matched GT (``inf`` if unmatched).
        - ``is_tp`` - whether the prediction was matched.
        - TP error metrics (``ate``, ``ase``, ``aoe``, ``ave``):
          non-zero only for TPs.

        Parameters
        ----------
        prediction:
            Predicted 3D bounding boxes for this frame.
        label:
            Ground-truth 3D bounding boxes for this frame.  Bike-rack boxes
            (``class_id == self.bike_rack_class_id``) are extracted from this
            list and used to filter bicycle and motorcycle boxes whose BEV
            center falls inside a bike-rack region, per the official nuScenes
            protocol.
        sample_id:
            Optional frame identifier (unused in computation).

        Returns
        -------
        A dict with keys:

        - ``sample_prediction_num``, ``sample_ground_truth_num``
        - ``match_records``: ``threshold → class_id → {pred_entries, gt_count}``
        """
        sample_prediction_num = len(prediction)
        sample_ground_truth_num = len(label)

        pred_boxes, gt_boxes = self._prepare_pred_gt_boxes(prediction, label)
        match_records = self._run_matching(pred_boxes, gt_boxes)

        return {
            "sample_prediction_num": sample_prediction_num,
            "sample_ground_truth_num": sample_ground_truth_num,
            "match_records": match_records,
        }

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 2 - aggregate per-frame match records into dataset-level metrics.

        Parameters
        ----------
        sample_results:
            List of ``{"sample_id": ..., "metrics": {..., "match_records": ...}}``
            entries produced by :meth:`record_sample`.

        Returns
        -------
        A dict containing:

        - ``threshold_metrics``: per-threshold AP / mAP (with P-R filter).
        - ``tp_metrics``: per-class ATE, ASE, AOE, AVE, AAE at 2 m threshold.
        - ``benchmark_score``: mAP and NDS summary.
        """
        all_match_records = [entry["metrics"]["match_records"] for entry in sample_results]
        merged_records = self._merge_match_records(all_match_records)

        # Compute cumulative P-R statistics for AP.
        cumulative_results = self._compute_global_cumulative_stats(merged_records)
        # Compute per-threshold AP with nuScenes min-precision/recall gating.
        threshold_metrics = self._compute_threshold_metrics(cumulative_results)
        # Compute TP error metrics at the 2 m threshold.
        tp_metrics = self._compute_tp_metrics(merged_records)
        # Compute benchmark score (mAP and NDS).
        benchmark_score = self._compute_benchmark_score(threshold_metrics, tp_metrics)

        return {
            "threshold_metrics": threshold_metrics,
            "tp_metrics": tp_metrics,
            "benchmark_score": benchmark_score,
        }

    # ------------------------------------------------------------------
    # Helper functions for metric computation
    # ------------------------------------------------------------------

    def _prepare_pred_gt_boxes(
        self,
        pred_boxes: List[BoundingBox3D],
        gt_boxes: List[BoundingBox3D],
    ) -> Tuple[List[BoundingBox3D], List[BoundingBox3D]]:
        """Prepare prediction and ground-truth boxes before matching.

        Applies the three nuScenes pre-matching filters in order:

        1. **GT point filter**: ground-truth boxes with zero lidar *and* zero
           radar points are excluded (they are not reliably annotated).
        2. **Bike-rack filter**: bicycle (class 8) and motorcycle (class 7)
           boxes — both prediction and GT — whose BEV center falls inside a
           bike-rack annotation are excluded.  Bike-rack boxes are identified
           directly from ``gt_boxes`` by ``class_id == self.bike_rack_class_id``
           and are always removed from GT before matching.
        3. **Per-class range filter**: both prediction and GT boxes beyond the
           class-specific BEV distance limit are excluded
           (barrier/cone ≤ 30 m, bicycle/motorcycle/pedestrian ≤ 40 m,
           car/truck/bus/trailer/construction_vehicle ≤ 50 m).

        Returns
        -------
        Filtered ``(pred_boxes, gt_boxes)`` as flat lists ready for matching.
        """

        def _build_bev_polygons(rack_boxes: List[BoundingBox3D]) -> list:
            """Return a Shapely rotated-rectangle polygon for each rack box in BEV."""
            polygons = []
            for box in rack_boxes:
                half_w = (box.width or 0.0) / 2.0
                half_l = (box.length or 0.0) / 2.0
                rect = shapely_box(
                    box.x - half_w,
                    box.y - half_l,
                    box.x + half_w,
                    box.y + half_l,
                )
                if box.yaw:
                    rect = shapely_rotate(rect, box.yaw, origin=(box.x, box.y), use_radians=True)
                polygons.append(rect)
            return polygons

        def _filter_bike_rack_overlaps(
            boxes: List[BoundingBox3D],
            rack_polygons: list,
            bike_rack_filtered_classes: frozenset,
        ) -> List[BoundingBox3D]:
            """Drop bicycle/motorcycle boxes whose BEV center falls inside a rack."""
            filtered: List[BoundingBox3D] = []
            for box in boxes:
                if box.class_id not in bike_rack_filtered_classes:
                    filtered.append(box)
                    continue
                center = Point(box.x, box.y)
                if any(poly.contains(center) for poly in rack_polygons):
                    continue  # drop — inside a bike-rack
                filtered.append(box)
            return filtered

        # Step 1 – exclude GT boxes with no sensor evidence.
        gt_boxes = [
            gt
            for gt in gt_boxes
            if not (
                (gt.number_of_lidar_points is not None and gt.number_of_lidar_points < 1)
                and (gt.number_of_radar_points is not None and gt.number_of_radar_points < 1)
            )
        ]
        # Step 2 – extract bike-rack boxes from GT and exclude bicycles/motorcycles
        # inside bike-rack regions.  Bike-rack boxes themselves are never evaluated.
        bike_rack_boxes = [gt for gt in gt_boxes if gt.class_id == self.bike_rack_class_id]
        gt_boxes = [gt for gt in gt_boxes if gt.class_id != self.bike_rack_class_id]
        if bike_rack_boxes:
            rack_polygons = _build_bev_polygons(bike_rack_boxes)
            pred_boxes = _filter_bike_rack_overlaps(pred_boxes, rack_polygons, self.bike_rack_filtered_classes)
            gt_boxes = _filter_bike_rack_overlaps(gt_boxes, rack_polygons, self.bike_rack_filtered_classes)
        # Step 3 – apply per-class detection range.
        pred_boxes = BoundingBox3DUtils.boxes_filter_per_class(
            pred_boxes, self.per_class_detection_ranges, self.compute_dist_func, include_end=True
        )
        gt_boxes = BoundingBox3DUtils.boxes_filter_per_class(
            gt_boxes, self.per_class_detection_ranges, self.compute_dist_func, include_end=True
        )
        return pred_boxes, gt_boxes

    def _run_matching(
        self,
        pred_boxes: List[BoundingBox3D],
        gt_boxes: List[BoundingBox3D],
    ) -> Dict[float, Dict[int, Dict[str, Any]]]:
        """Greedy BEV-distance matching for one frame across all thresholds.

        Returns
        -------
        ``match_records``:
            ``threshold → class_id → {"pred_entries": [...], "gt_count": int}``

        Each entry in ``pred_entries``:

        - ``confidence_score``, ``dist``, ``is_tp``
        - ``ate``, ``ase``, ``aoe``, ``ave`` (zero when FP)
        - ``aae``: ``float`` (0.0 correct, 1.0 wrong) for TPs whose GT has an
          attribute and the class is not in ``aae_ignored_classes``; ``None``
          when attribute data is absent or the class ignores attributes; always
          ``None`` for FPs.
        """
        # Index boxes by class.
        gt_by_class: Dict[int, List[BoundingBox3D]] = {}
        for gt in gt_boxes:
            gt_by_class.setdefault(gt.class_id, []).append(gt)
        pred_by_class: Dict[int, List[BoundingBox3D]] = {}
        for pred in pred_boxes:
            pred_by_class.setdefault(pred.class_id, []).append(pred)

        all_class_ids = set(gt_by_class) | set(pred_by_class)

        match_records: Dict[float, Dict[int, Dict[str, Any]]] = {}
        for threshold in self.matching_thresholds:
            match_records[threshold] = {}
            for class_id in all_class_ids:
                gts = gt_by_class.get(class_id, [])
                preds = sorted(
                    pred_by_class.get(class_id, []),
                    key=lambda p: -(p.confidence_score or 0.0),
                )
                gt_count = len(gts)
                matched_gt_indices: set = set()
                pred_entries: List[Dict[str, Any]] = []

                for pred in preds:
                    best_dist = float("inf")
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(gts):
                        if gt_idx in matched_gt_indices:
                            continue
                        dist = self.compute_dist_func(pred, gt)
                        if dist <= threshold and dist < best_dist:
                            best_dist = dist
                            best_gt_idx = gt_idx

                    if best_gt_idx != -1:
                        matched_gt = gts[best_gt_idx]
                        matched_gt_indices.add(best_gt_idx)
                        # ATE: BEV Euclidean translation error.
                        ate = math.sqrt((pred.x - matched_gt.x) ** 2 + (pred.y - matched_gt.y) ** 2)
                        # ASE: 1 - 3D IoU after aligning centers and orientation.
                        inter = (
                            min(pred.width, matched_gt.width)
                            * min(pred.height, matched_gt.height)
                            * min(pred.length, matched_gt.length)
                        )
                        vol_pred = pred.width * pred.height * pred.length
                        vol_gt = matched_gt.width * matched_gt.height * matched_gt.length
                        union = vol_pred + vol_gt - inter
                        ase = 1.0 - (inter / union if union > 0 else 0.0)
                        # AOE: smallest yaw angle difference (class-specific rules).
                        heading_diff = abs(pred.yaw - matched_gt.yaw)
                        if class_id in self.aoe_ignored_classes:
                            aoe = 0.0  # traffic_cone: orientation error ignored.
                        elif class_id in self.aoe_pi_classes:
                            # barrier: π-symmetric — wrap into [0, π) then take symmetric minimum.
                            heading_diff_mod = heading_diff % math.pi
                            aoe = min(heading_diff_mod, math.pi - heading_diff_mod)
                        else:
                            aoe = min(heading_diff, 2 * math.pi - heading_diff)
                        # AVE: velocity error in m/s (ignored for barrier and traffic_cone).
                        if class_id in self.ave_ignored_classes:
                            ave = 0.0
                        else:
                            ave = math.sqrt((pred.vx - matched_gt.vx) ** 2 + (pred.vy - matched_gt.vy) ** 2)
                        # AAE: 1 - attribute accuracy (ignored for barrier and traffic_cone).
                        # None when GT attribute is missing — excluded from mean in that case.
                        if class_id in self.aae_ignored_classes:
                            aae: Optional[float] = 0.0  # attribute not evaluated for this class
                        elif matched_gt.attribute is not None:
                            aae = 0.0 if pred.attribute == matched_gt.attribute else 1.0
                        else:
                            aae = None  # GT attribute missing — skip this TP for AAE
                        pred_entries.append(
                            {
                                "confidence_score": pred.confidence_score or 0.0,
                                "dist": best_dist,
                                "is_tp": True,
                                "ate": ate,
                                "ase": ase,
                                "aoe": aoe,
                                "ave": ave,
                                "aae": aae,
                            }
                        )
                    else:
                        pred_entries.append(
                            {
                                "confidence_score": pred.confidence_score or 0.0,
                                "dist": float("inf"),
                                "is_tp": False,
                                "ate": 0.0,
                                "ase": 0.0,
                                "aoe": 0.0,
                                "ave": 0.0,
                                "aae": None,
                            }
                        )

                match_records[threshold][class_id] = {
                    "pred_entries": pred_entries,
                    "gt_count": gt_count,
                }

        return match_records

    def _merge_match_records(
        self, all_match_records: List[Dict[float, Dict[int, Dict[str, Any]]]]
    ) -> Dict[float, Dict[int, Dict[str, Any]]]:
        """Concatenate per-frame match records into one dataset-level structure.

        Concatenates ``pred_entries`` and sums ``gt_count`` across all frames,
        grouped by threshold / class ID.
        """
        merged: Dict[float, Dict[int, Dict[str, Any]]] = {}
        for frame_records in all_match_records:
            for threshold, class_records in frame_records.items():
                merged.setdefault(threshold, {})
                for class_id, data in class_records.items():
                    if class_id not in merged[threshold]:
                        merged[threshold][class_id] = {"pred_entries": [], "gt_count": 0}
                    merged[threshold][class_id]["pred_entries"].extend(data["pred_entries"])
                    merged[threshold][class_id]["gt_count"] += data["gt_count"]
        return merged

    def _compute_global_cumulative_stats(
        self,
        merged: Dict[float, Dict[int, Dict[str, Any]]],
    ) -> Dict[float, Dict[int, Dict[int, Dict[str, float]]]]:
        """Build cumulative P-R statistics from the merged match records.

        Sorts all predictions globally by confidence descending and accumulates
        cumulative TP, FP, FN, precision, and recall across the full dataset.

        Returns
        -------
        ``threshold → class_id → pred_idx → {confidence_score, dist,
        cum_tp, cum_fp, cum_fn, cum_precision, cum_recall}``
        """
        cumulative_results: Dict[float, Dict[int, Dict[int, Dict[str, float]]]] = {}
        for threshold, class_records in merged.items():
            cumulative_results[threshold] = {}
            for class_id, data in class_records.items():
                pred_entries = sorted(data["pred_entries"], key=lambda x: -x["confidence_score"])
                total_gt = data["gt_count"]
                class_results: Dict[int, Dict[str, float]] = {}
                cum_tp, cum_fp = 0.0, 0.0
                num_matched = 0
                for pred_idx, entry in enumerate(pred_entries):
                    if entry["is_tp"]:
                        cum_tp += 1.0
                        num_matched += 1
                    else:
                        cum_fp += 1.0
                    cum_fn = float(total_gt - num_matched)
                    cum_precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
                    cum_recall = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0
                    class_results[pred_idx] = {
                        "confidence_score": entry["confidence_score"],
                        "dist": entry["dist"],
                        "cum_tp": cum_tp,
                        "cum_fp": cum_fp,
                        "cum_fn": cum_fn,
                        "cum_precision": cum_precision,
                        "cum_recall": cum_recall,
                    }
                cumulative_results[threshold][class_id] = class_results
        return cumulative_results

    def _compute_threshold_metrics(
        self,
        cumulative_results: Dict[float, Dict[int, Dict[int, Dict[str, float]]]],
    ) -> Dict[str, Any]:
        """Compute per-threshold and overall AP with nuScenes P-R gating.

        Points where ``cum_precision <= min_precision`` or
        ``cum_recall <= min_recall`` are excluded from the trapezoidal
        integration, matching the nuScenes evaluation protocol.

        Returns
        -------
        ``{"thresholds": {threshold: {"per_class_metrics", "map"}}, "overall_map"}``
        """
        threshold_metrics: Dict[str, Any] = {"thresholds": {}}
        maps = []
        for threshold, class_results in cumulative_results.items():
            per_class_metrics: Dict[int, Dict[str, float]] = {}
            aps = []
            for class_id, predictions in class_results.items():
                sorted_preds = sorted(predictions.values(), key=lambda x: -x["confidence_score"])
                pr_pairs: List[Tuple[float, float]] = []
                for entry in sorted_preds:
                    recall = entry["cum_recall"]
                    precision = entry["cum_precision"]
                    if recall > self.min_recall and precision > self.min_precision:
                        pr_pairs.append((recall, precision))
                ap = self.compute_ap_func(pr_pairs) if pr_pairs else 0.0
                per_class_metrics[class_id] = {"ap": ap}
                aps.append(ap)
            map_val = float(np.mean(aps)) if aps else 0.0
            threshold_metrics["thresholds"][threshold] = {
                "per_class_metrics": per_class_metrics,
                "map": map_val,
            }
            maps.append(map_val)
        threshold_metrics["overall_map"] = float(np.mean(maps)) if maps else 0.0
        return threshold_metrics

    def _compute_tp_metrics(
        self,
        merged: Dict[float, Dict[int, Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        """Compute mean TP error metrics per class at the 2 m threshold.

        Averages ATE, ASE, AOE, AVE, and AAE over true-positive predictions
        matched at :attr:`tp_metric_threshold`. AAE is averaged only over TPs
        where the GT attribute is available (non-``None``); if none are
        available the per-class ``aae`` is ``None``.

        Returns
        -------
        ``class_id → {"ate", "ase", "aoe", "ave", "aae"}``
        where ``aae`` is a ``float`` or ``None``.
        """
        threshold_data = merged.get(self.tp_metric_threshold, {})
        tp_metrics: Dict[int, Dict[str, Any]] = {}
        for class_id, data in threshold_data.items():
            tp_entries = [e for e in data["pred_entries"] if e["is_tp"]]
            if tp_entries:
                aae_vals = [e["aae"] for e in tp_entries if e["aae"] is not None]
                tp_metrics[class_id] = {
                    "ate": float(np.mean([e["ate"] for e in tp_entries])),
                    "ase": float(np.mean([e["ase"] for e in tp_entries])),
                    "aoe": float(np.mean([e["aoe"] for e in tp_entries])),
                    "ave": float(np.mean([e["ave"] for e in tp_entries])),
                    "aae": float(np.mean(aae_vals)) if aae_vals else None,
                }
            else:
                # No TPs → worst-case error (capped at 1.0 in NDS).
                tp_metrics[class_id] = {
                    "ate": 1.0,
                    "ase": 1.0,
                    "aoe": 1.0,
                    "ave": 1.0,
                    "aae": None,  # cannot determine worst-case for missing attribute data
                }
        return tp_metrics

    def _compute_benchmark_score(
        self,
        threshold_metrics: Dict[str, Any],
        tp_metrics: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute the flat benchmark score dict matching the nuScenes metric naming convention.

        Produces per-threshold per-class AP keys (``ap_{threshold}_{class}``),
        per-class mean AP keys (``map_{class}``), the overall ``map``,
        per-class TP error keys (``{metric}_2.0_{class}``), mean TP error keys
        (``m{metric}_2.0``), and the ``nds`` score.

        nuScenes Detection Score (NDS) per the official spec::

            NDS = (5·mAP + score(mATE) + score(mASE) + score(mAOE)
                         + score(mAVE) + score(mAAE)) / 10

        where ``score(x) = max(1 - x, 0)``.  Denominator is always 10.

        ``aae_2.0_{class}`` and ``maae_2.0`` are ``None`` when attribute data
        is unavailable.
        """
        class_names: Dict[int, str] = {
            1: "car",
            2: "truck",
            3: "bus",
            4: "trailer",
            5: "construction_vehicle",
            6: "pedestrian",
            7: "motorcycle",
            8: "bicycle",
            9: "traffic_cone",
            10: "barrier",
        }

        score: Dict[str, Any] = {}

        # Overall mAP.
        map_val = threshold_metrics["overall_map"]
        score["map"] = round(map_val, 4)

        # Per-threshold per-class AP and per-class mean AP.
        per_class_aps: Dict[int, List[float]] = {}
        for threshold, thr_data in threshold_metrics["thresholds"].items():
            for class_id, class_metric in thr_data["per_class_metrics"].items():
                class_name = class_names.get(class_id, f"class_{class_id}")
                ap = class_metric["ap"]
                score[f"ap_{threshold}_{class_name}"] = round(ap, 4)
                per_class_aps.setdefault(class_id, []).append(ap)
        for class_id, aps in per_class_aps.items():
            class_name = class_names.get(class_id, f"class_{class_id}")
            score[f"map_{class_name}"] = round(float(np.mean(aps)), 4)

        # Macro mean TP error metrics across all classes.
        def _mean_tp_metric(key: str) -> float:
            vals = [m[key] for m in tp_metrics.values() if m.get(key) is not None]
            return float(np.mean(vals)) if vals else 1.0

        mate = _mean_tp_metric("ate")
        mase = _mean_tp_metric("ase")
        maoe = _mean_tp_metric("aoe")
        mave = _mean_tp_metric("ave")
        maae = _mean_tp_metric("aae")

        score["mate_2.0"] = round(mate, 4)
        score["mase_2.0"] = round(mase, 4)
        score["maoe_2.0"] = round(maoe, 4)
        score["mave_2.0"] = round(mave, 4)
        score["maae_2.0"] = round(maae, 4)

        # Per-class TP error metrics at the 2 m threshold.
        for class_id, tm in tp_metrics.items():
            class_name = class_names.get(class_id, f"class_{class_id}")
            score[f"ate_2.0_{class_name}"] = round(tm["ate"], 4)
            score[f"ase_2.0_{class_name}"] = round(tm["ase"], 4)
            score[f"aoe_2.0_{class_name}"] = round(tm["aoe"], 4)
            score[f"ave_2.0_{class_name}"] = round(tm["ave"], 4)
            score[f"aae_2.0_{class_name}"] = round(tm["aae"], 4) if tm["aae"] is not None else None

        # NDS combines mAP and TP metrics with weights 5-1-1-1-1-1, normalised by 10.
        tp_scores = (
            max(1.0 - mate, 0.0) + max(1.0 - mase, 0.0) + max(1.0 - maoe, 0.0) + max(1.0 - mave, 0.0) + max(1.0 - maae, 0.0)
        )
        nds = (5.0 * map_val + tp_scores) / 10.0
        score["nds"] = round(nds, 4)

        return score
