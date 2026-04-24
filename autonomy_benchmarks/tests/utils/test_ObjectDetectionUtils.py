"""Tests for ObjectDetectionUtils."""

from __future__ import annotations

import numpy as np
import pytest

from autohub_benchmarks.utils.ObjectDetectionUtils import ObjectDetectionUtils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cumulative_unified(threshold: float, n_tp: int = 1, n_fp: int = 0):
    """Build a minimal CumulativeResultsTypeUnifiedThreshold for one class."""
    preds = {}
    for i in range(n_tp):
        preds[i] = {
            "confidence_score": 1.0 - i * 0.1,
            "cum_precision": 1.0,
            "cum_recall": (i + 1) / max(n_tp, 1),
        }
    for j in range(n_fp):
        preds[n_tp + j] = {
            "confidence_score": 0.5 - j * 0.1,
            "cum_precision": n_tp / (n_tp + j + 1),
            "cum_recall": 1.0,
        }
    return {threshold: {1: preds}}  # class_id = 1


def _make_cumulative_nonunified(threshold_key: str):
    """Build a minimal CumulativeResultsTypeNonUnifiedThreshold for one class."""
    preds = {
        0: {"confidence_score": 0.9, "cum_precision": 1.0, "cum_recall": 0.5},
        1: {"confidence_score": 0.8, "cum_precision": 1.0, "cum_recall": 1.0},
    }
    return {threshold_key: {1: preds}}


def _build_interval_inputs():
    interval_keys = ["0.50-0.95"]
    interval_map = {"0.50-0.95": (0.5, 0.95, 0.05)}
    thresholds = [round(t, 2) for t in np.arange(0.5, 0.95 + 1e-6, 0.05)]
    cumulative = {}
    for t in thresholds:
        cumulative.update(_make_cumulative_unified(t, n_tp=2))
    return interval_keys, interval_map, cumulative


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestObjectDetectionUtils:
    def test_compute_ap_trapezoidal(self) -> None:
        pr_pairs = [(0.0, 1.0), (0.5, 0.8), (1.0, 0.6)]
        expected = float(np.trapz(np.array([1.0, 0.8, 0.6]), np.array([0.0, 0.5, 1.0])))
        assert ObjectDetectionUtils.compute_ap_trapezoidal(pr_pairs) == pytest.approx(expected)

    def test_compute_ap_11_point(self) -> None:
        pr_pairs = [(r / 10, 1.0) for r in range(11)]
        assert ObjectDetectionUtils.compute_ap_11_point(pr_pairs) == pytest.approx(1.0)

    def test_compute_ap_thresholds_unified(self) -> None:
        cumulative = _make_cumulative_unified(0.5, n_tp=2, n_fp=1)
        result = ObjectDetectionUtils.compute_ap_thresholds(cumulative, ObjectDetectionUtils.compute_ap_trapezoidal)
        assert 0.5 in result["thresholds"]
        per_class = result["thresholds"][0.5]["per_class_metrics"]
        assert "ap" in per_class[1]
        assert 0.0 <= result["overall_map"] <= 1.0

    def test_compute_ap_thresholds_nonunified(self) -> None:
        key = "Vehicle_0.7_Pedestrian_0.5_Cyclist_0.5"
        cumulative = _make_cumulative_nonunified(key)
        result = ObjectDetectionUtils.compute_ap_thresholds(cumulative, ObjectDetectionUtils.compute_ap_trapezoidal)
        assert key in result["thresholds"]
        per_class = result["thresholds"][key]["per_class_metrics"]
        assert "ap" in per_class[1]
        assert 0.0 <= result["overall_map"] <= 1.0

    def test_compute_ap_intervals_unified(self) -> None:
        interval_keys, interval_map, cumulative = _build_interval_inputs()
        result = ObjectDetectionUtils.compute_ap_intervals(
            interval_keys,
            interval_map,
            cumulative,
            ObjectDetectionUtils.compute_ap_trapezoidal,
        )
        assert "0.50-0.95" in result["intervals"]
        per_class = result["intervals"]["0.50-0.95"]["per_class_metrics"]
        assert "ap" in per_class[1]
        assert 0.0 <= result["overall_map"] <= 1.0
