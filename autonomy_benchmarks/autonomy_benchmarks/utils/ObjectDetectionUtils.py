"""Utility helpers for object-detection benchmarks.

Contains reusable functions shared across all benchmark implementations:

- **Precision-recall computation**: cumulative TP/FP/FN accumulation over
  globally ranked predictions.
- **AP integration** (``compute_ap_trapezoidal``, ``compute_ap_11_point``):
  trapezoidal and 11-point interpolation methods.
- **Per-threshold aggregation** (``compute_ap_thresholds``): AP and mAP
  across a single or multiple matching thresholds.
- **Interval aggregation** (``compute_ap_intervals``): AP and mAP across
  threshold intervals with configurable weights.

Matching, box filtering, and class grouping are implemented in the
individual benchmark classes that use these helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

CumulativeResultsTypeUnifiedThreshold = Dict[float, Dict[int, Dict[int, Dict[str, float]]]]
CumulativeResultsTypeNonUnifiedThreshold = Dict[str, Dict[int, Dict[int, Dict[str, float]]]]


class ObjectDetectionUtils:
    """Static utility methods for object-detection tasks.

    Provides precision-recall helpers and AP integration methods
    (trapezoidal and 11-point), as well as aggregation of per-class
    and per-threshold AP across a full evaluation run.
    """

    @staticmethod
    def compute_ap_trapezoidal(
        precision_recall_pairs: List[Tuple[float, float]],
    ) -> float:
        """Compute Average Precision (AP) using trapezoidal integration.

        Parameters
        ----------
        precision_recall_pairs:
            List of ``(recall, precision)`` pairs sampled from the P-R curve.
            May be in any order — sorted internally by recall ascending.

        Returns
        -------
        AP as a float in ``[0, 1]``.
        """

        # Sort by recall in ascending order.
        precision_recall_pairs = sorted(precision_recall_pairs, key=lambda x: x[0])
        recalls, precisions = zip(*precision_recall_pairs)
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        # Numerical integration (trapezoidal rule).
        ap = np.trapz(precisions, recalls)
        return ap

    @staticmethod
    def compute_ap_11_point(
        precision_recall_pairs: List[Tuple[float, float]],
    ) -> float:
        """Compute Average Precision (AP) using 11-point interpolation.

        Parameters
        ----------
        precision_recall_pairs:
            List of ``(recall, precision)`` pairs sampled from the P-R curve.
            May be in any order — sorted internally by recall ascending.

        Returns
        -------
        AP as a float in ``[0, 1]``.
        """

        # Sort by recall in ascending order.
        precision_recall_pairs = sorted(precision_recall_pairs, key=lambda x: x[0])
        recalls, precisions = zip(*precision_recall_pairs)
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_t = precisions[recalls >= t]
            if prec_at_t.size > 0:
                ap += np.max(prec_at_t)
        ap /= 11.0
        return ap

    @staticmethod
    def compute_ap_thresholds(
        cumulative_results: Dict[Any, Dict[int, Dict[int, Dict[str, float]]]],
        ap_computation_func,
    ) -> Dict[str, Any]:
        """Compute AP per matching threshold and the overall mAP.

        Parameters
        ----------
        cumulative_results:
            Nested dict keyed by threshold → class_id → pred_idx →
            ``{"cum_recall", "cum_precision", "confidence_score", ...}``.
            Accepts both unified (``float``-keyed,
            :data:`CumulativeResultsTypeUnifiedThreshold`) and non-unified
            (``str``-keyed, :data:`CumulativeResultsTypeNonUnifiedThreshold`)
            cumulative results.
        ap_computation_func:
            Callable ``(List[Tuple[recall, precision]]) -> float`` used to
            integrate the P-R curve (e.g. :meth:`compute_ap_trapezoidal`).

        Returns
        -------
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
                # Construct list of (recall, precision) pairs.
                pr_pairs: List[Tuple[float, float]] = []
                for entry in sorted_preds:
                    recall = entry.get("cum_recall")
                    precision = entry.get("cum_precision")
                    if recall is not None and precision is not None:
                        pr_pairs.append((recall, precision))
                # Compute AP using provided function (e.g., trapezoidal or 11-point).
                ap = ap_computation_func(pr_pairs) if pr_pairs else 0.0
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

    @staticmethod
    def compute_ap_intervals(
        interval_keys: List[str],
        interval_map: Dict[str, Tuple[float, float, float]],
        cumulative_results: Dict[Any, Dict[int, Dict[int, Dict[str, float]]]],
        ap_computation_func,
    ) -> Dict[str, Any]:
        """Compute AP per threshold interval and the overall mAP across intervals.

        Parameters
        ----------
        interval_keys:
            Ordered list of interval names (e.g. ``["00_35", "35_50", "50_inf"]``).
        interval_map:
            Mapping from interval name to ``(start, end, weight)`` where
            *start* and *end* are the matching-threshold bounds that define
            which entries of *cumulative_results* belong to the interval, and
            *weight* is reserved for future use.
        cumulative_results:
            Nested dict keyed by threshold → class_id → pred_idx →
            ``{"cum_recall", "cum_precision", "confidence_score", ...}``.
            Accepts both unified (``float``-keyed) and non-unified (``str``-keyed)
            cumulative results.
        ap_computation_func:
            Callable ``(List[Tuple[recall, precision]]) -> float`` used to
            integrate the P-R curve (e.g. :meth:`compute_ap_trapezoidal`).

        Returns
        -------
        Dict with structure::

            {
                "intervals": {
                    interval_key: {
                        "per_class_metrics": {class_id: {"ap": float}},
                        "map": float,
                        "thresholds": List[float],
                    },
                    ...
                },
                "overall_map": float,
            }
        """

        interval_metrics: Dict[str, Any] = {"intervals": {}}

        maps = []
        for interval_key in interval_keys:
            start, end, _ = interval_map[interval_key]
            # Get all matching thresholds within the interval
            relevant_thresholds = [t for t in cumulative_results if start <= t <= end]
            per_class_predictions: Dict[int, List[Dict[str, float]]] = {}
            # Aggregate predictions from all thresholds in this interval
            for t in relevant_thresholds:
                class_results = cumulative_results.get(t, {})
                for class_id, preds in class_results.items():
                    if class_id not in per_class_predictions:
                        per_class_predictions[class_id] = []
                    per_class_predictions[class_id].extend(preds.values())
            per_class_metrics: Dict[int, Dict[str, float]] = {}
            aps = []
            for class_id, predictions in per_class_predictions.items():
                # Sort predictions for the class by confidence score descending
                sorted_preds = sorted(predictions, key=lambda x: -x.get("confidence_score", 0.0))
                # Construct list of (recall, precision) pairs
                pr_pairs: List[Tuple[float, float]] = []
                for entry in sorted_preds:
                    recall = entry.get("cum_recall")
                    precision = entry.get("cum_precision")
                    if recall is not None and precision is not None:
                        pr_pairs.append((recall, precision))
                # Compute AP using provided function (e.g., trapezoidal or 11-point)
                ap = ap_computation_func(pr_pairs) if pr_pairs else 0.0
                # Store per-class AP.
                per_class_metrics[class_id] = {"ap": ap}
                # Accumulate for overall metrics across classes
                aps.append(ap)
            # Compute mean metrics across all classes and store results
            map = float(np.mean(aps)) if aps else 0.0
            interval_metrics["intervals"][interval_key] = {
                "per_class_metrics": per_class_metrics,
                "map": map,
                "thresholds": relevant_thresholds,
            }
            # Accumulate for overall metrics across matching intervals
            maps.append(map)
        # Compute overall results for all matching intervals and store results
        overall_map = float(np.mean(maps)) if maps else 0.0
        interval_metrics["overall_map"] = overall_map

        return interval_metrics
