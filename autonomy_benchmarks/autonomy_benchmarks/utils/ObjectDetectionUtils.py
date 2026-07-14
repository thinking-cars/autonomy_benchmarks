# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for object-detection benchmarks.

Contains reusable functions shared across all benchmark implementations:

- **AP integration** (``compute_ap_101_point``): 101-point recall interpolation.
- **TP metric integration** (``compute_tp_101_point``): average TP metric over a
  recall interval using 101-point interpolation.
- **Per-threshold aggregation** (``compute_ap_thresholds``): AP and mAP
  across a single or multiple matching thresholds.
- **Geometry helpers** operating on plain ``dict`` object records:
  ``compute_iou_2d`` (axis-aligned 2D IoU), ``get_bev_poly`` /
  ``compute_iou_3d`` (rotated BEV footprint and volumetric 3D IoU),
  ``compute_dist_bev`` (BEV center distance), and ``boxes_filter``
  (keep records within a BEV distance range of the sensor origin).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import box as shapely_box

CumulativeResultsTypeUnifiedThreshold = Dict[float, Dict[int, Dict[int, Dict[str, float]]]]
CumulativeResultsTypeNonUnifiedThreshold = Dict[str, Dict[int, Dict[int, Dict[str, float]]]]


class ObjectDetectionUtils:
    """Static helpers shared across the object-detection benchmarks.

    Groups two kinds of stateless utilities used by every benchmark:

    * Geometry on plain ``dict`` object records: 2D and volumetric 3D IoU,
      the rotated BEV footprint polygon, BEV center distance, and BEV
      distance-range filtering against the sensor origin.
    * Metric integration: 101-point AP and TP-metric recall interpolation
      and per-threshold AP/mAP aggregation over a full evaluation run.
    """

    @staticmethod
    def compute_iou_2d(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
        """2D Intersection over Union of two axis-aligned boxes.

        Args:
            box1: Object record with ``x1, y1, x2, y2`` corner coordinates.
            box2: Object record in the same corner format.

        Returns:
            IoU in ``[0, 1]``; 0.0 when the boxes are disjoint or degenerate.
        """
        xi1, yi1 = max(box1["x1"], box2["x1"]), max(box1["y1"], box2["y1"])
        xi2, yi2 = min(box1["x2"], box2["x2"]), min(box1["y2"], box2["y2"])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def get_bev_poly(box: Dict[str, Any]):
        """Build the box's rotated bird's-eye-view (BEV) footprint as a Shapely polygon.

        A ``width`` × ``length`` rectangle centered at the origin is rotated by
        ``-yaw`` (radians) and translated to the box center ``(x, y)``.

        Args:
            box: Object record with ``x, y, width, length, yaw``.

        Returns:
            A Shapely polygon of the footprint in the ground plane.
        """
        rect = shapely_box(-box["width"] / 2, -box["length"] / 2, box["width"] / 2, box["length"] / 2)
        rotated = rotate(rect, -box["yaw"], origin=(0, 0), use_radians=True)
        return translate(rotated, box["x"], box["y"])

    @staticmethod
    def compute_iou_3d(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
        """True volumetric 3D IoU of two boxes.

        Intersects the rotated BEV footprints and multiplies by the vertical
        (height) overlap to get the intersection volume, divided by the union
        of the two box volumes — the exact volumetric IoU, not the
        ``BEV-IoU × height-IoU`` approximation.

        Args:
            box1: Object record with ``x, y, z, width, length, height, yaw``.
            box2: Object record in the same format.

        Returns:
            3D IoU in ``[0, 1]``; 0.0 when the union volume is zero.
        """
        poly1 = ObjectDetectionUtils.get_bev_poly(box1)
        poly2 = ObjectDetectionUtils.get_bev_poly(box2)
        bev_inter = poly1.intersection(poly2).area

        min_z1, max_z1 = box1["z"] - box1["height"] / 2, box1["z"] + box1["height"] / 2
        min_z2, max_z2 = box2["z"] - box2["height"] / 2, box2["z"] + box2["height"] / 2
        z_inter = max(0.0, min(max_z1, max_z2) - max(min_z1, min_z2))

        vol3d_inter = bev_inter * z_inter
        vol1 = poly1.area * box1["height"]
        vol2 = poly2.area * box2["height"]
        vol3d_union = vol1 + vol2 - vol3d_inter
        return vol3d_inter / vol3d_union if vol3d_union > 0 else 0.0

    @staticmethod
    def compute_dist_bev(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
        """Euclidean distance between two object centers in the BEV (XY) plane.

        Args:
            box1: Object record with ``x, y`` (``z`` is ignored).
            box2: Object record with ``x, y``.

        Returns:
            Non-negative BEV distance.
        """
        dx = box1["x"] - box2["x"]
        dy = box1["y"] - box2["y"]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def boxes_filter(
        boxes: List[Dict[str, Any]],
        range_start: float,
        range_end: float,
        compute_dist_func,
        origin: Dict[str, Any] = {"x": 0.0, "y": 0.0},
    ) -> List[Dict[str, Any]]:
        """Keep records whose BEV distance to ``origin`` lies in ``[range_start, range_end)``.

        Args:
            boxes: Object records to filter.
            range_start: Inclusive lower distance bound.
            range_end: Exclusive upper distance bound.
            compute_dist_func: Callable ``(box, origin) -> float`` giving the BEV
                distance between a record and the origin (e.g. :meth:`compute_dist_bev`).
            origin: Reference point with ``"x"``/``"y"`` keys; defaults to the sensor origin.

        Returns:
            The subset of ``boxes`` within the distance range, order preserved.
        """
        result: List[Dict[str, Any]] = []
        for box in boxes:
            dist = compute_dist_func(box, origin)
            if range_start <= dist < range_end:
                result.append(box)
        return result

    @staticmethod
    def compute_ap_101_point(
        rec: "np.ndarray",
        prec: "np.ndarray",
        min_recall: float = 0.1,
        min_precision: float = 0.1,
    ) -> float:
        """Compute Average Precision using 101-point recall interpolation.

        The precision-recall curve is interpolated onto 101 uniformly spaced
        recall points ``[0.00, 0.01, ..., 1.00]``. Bins below ``min_recall``
        are dropped, ``min_precision`` is subtracted with negatives clipped to
        zero, and the mean is normalised by ``(1 - min_precision)`` so that a
        perfect detector yields AP = 1.0.

        Args:
            rec: Recall values in ascending order (one per prediction, sorted
                by confidence descending before calling).
            prec: Precision values aligned with ``rec``.
            min_recall: Recall threshold below which P-R points are ignored.
            min_precision: Precision offset subtracted before normalisation.

        Returns:
            Average precision as a float in ``[0, 1]``.
        """
        rec_interp = np.linspace(0, 1, 101)
        first_ind = round(100 * min_recall) + 1
        prec_interp = np.interp(rec_interp, rec, prec, right=0)
        prec_valid = np.maximum(prec_interp[first_ind:] - min_precision, 0.0)
        return min(float(np.mean(prec_valid)) / (1.0 - min_precision), 1.0)

    @staticmethod
    def compute_tp_101_point(
        rec_arr: "np.ndarray",
        cm_vals: List[float],
        min_recall: float = 0.1,
    ) -> Optional[float]:
        """Average a cumulative-mean TP metric curve over ``[min_recall, max_recall]``.

        The curve is interpolated onto 101 uniformly spaced recall points. The
        averaged window runs from the first bin above ``min_recall`` to the bin
        for ``max_recall`` (the last element of ``rec_arr``). Returns ``None``
        when ``max_recall <= min_recall``.

        Args:
            rec_arr: Recall values in ascending order, one per prediction
                (sorted by confidence descending before calling).
            cm_vals: Cumulative-mean metric values aligned with ``rec_arr``.
                Must be the same length and contain no ``None`` entries
                (forward-fill ``None`` values before calling).
            min_recall: Lower recall bound; points at or below it are excluded.

        Returns:
            Mean metric value as a float, or ``None`` if ``max_recall <= min_recall``.
        """
        first_ind = round(100 * min_recall) + 1
        last_ind = round(100 * float(rec_arr[-1]))
        if last_ind < first_ind:
            return None
        rec_interp = np.linspace(0, 1, 101)
        interped = np.interp(rec_interp, rec_arr, np.array(cm_vals, dtype=float), right=cm_vals[-1])
        return float(np.mean(interped[first_ind : last_ind + 1]))

    @staticmethod
    def compute_ap_thresholds(
        cumulative_results: Dict[Any, Dict[int, Dict[int, Dict[str, float]]]],
        ap_computation_func,
    ) -> Dict[str, Any]:
        """Compute AP per matching threshold and the overall mAP.

        For each threshold, per-class predictions are ranked by confidence and
        their cumulative recall/precision are integrated into an AP; the mAP is
        the mean AP over classes, and ``overall_map`` the mean mAP over thresholds.

        Args:
            cumulative_results: Nested dict keyed by threshold -> class_id ->
                pred_idx -> ``{"cum_recall", "cum_precision", "confidence_score", ...}``.
                Accepts both unified (``float``-keyed,
                :data:`CumulativeResultsTypeUnifiedThreshold`) and non-unified
                (``str``-keyed, :data:`CumulativeResultsTypeNonUnifiedThreshold`) results.
            ap_computation_func: Callable ``(rec, prec) -> float`` integrating the
                P-R curve (e.g. :meth:`compute_ap_101_point`).

        Returns:
            Dict with structure::

                {
                    "thresholds": {
                        threshold_key: {
                            "per_class_metrics": {class_id: {"ap": float}},
                            "map": float,
                        },
                        ...
                    },
                    "overall_map": float,
                }
        """
        threshold_metrics: Dict[str, Any] = {"thresholds": {}}

        maps = []
        for threshold_key, class_results in cumulative_results.items():
            per_class_metrics: Dict[int, Dict[str, float]] = {}
            aps = []
            for class_id, predictions in class_results.items():
                # Sort predictions for the class by confidence score descending.
                sorted_preds = sorted(predictions.values(), key=lambda x: -x.get("confidence_score", 0.0))
                # Build rec/prec arrays sorted by confidence descending.
                rec_list: List[float] = []
                prec_list: List[float] = []
                for entry in sorted_preds:
                    recall = entry.get("cum_recall")
                    precision = entry.get("cum_precision")
                    if recall is not None and precision is not None:
                        rec_list.append(recall)
                        prec_list.append(precision)
                # Compute AP using provided function.
                if rec_list:
                    ap = ap_computation_func(np.array(rec_list), np.array(prec_list))
                else:
                    ap = 0.0
                per_class_metrics[class_id] = {"ap": ap}
                aps.append(ap)
            # Compute mean AP across all classes.
            map_val = float(np.mean(aps)) if aps else 0.0
            threshold_metrics["thresholds"][threshold_key] = {
                "per_class_metrics": per_class_metrics,
                "map": map_val,
            }
            maps.append(map_val)
        # Compute overall mAP across all matching thresholds.
        threshold_metrics["overall_map"] = float(np.mean(maps)) if maps else 0.0

        return threshold_metrics
