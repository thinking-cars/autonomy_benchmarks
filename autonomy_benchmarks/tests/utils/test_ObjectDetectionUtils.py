# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectDetectionUtils."""

from __future__ import annotations

import numpy as np
import pytest
from autonomy_benchmarks.utils.ObjectDetectionUtils import ObjectDetectionUtils


def _box2d(x1, y1, x2, y2):
    """Build a plain 2D object record (corner format)."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _box3d(x=0.0, y=0.0, z=0.0, width=2.0, length=2.0, height=2.0, yaw=0.0):
    """Build a plain 3D object record."""
    return {"x": x, "y": y, "z": z, "width": width, "length": length, "height": height, "yaw": yaw}


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


class TestObjectDetectionUtils:
    """Tests for AP helpers and aggregation utilities."""

    def test_compute_ap_thresholds_unified(self) -> None:
        """Verify AP threshold aggregation works for unified thresholds."""
        cumulative = _make_cumulative_unified(0.5, n_tp=2, n_fp=1)
        result = ObjectDetectionUtils.compute_ap_thresholds(cumulative, ObjectDetectionUtils.compute_ap_101_point)
        assert 0.5 in result["thresholds"]
        per_class = result["thresholds"][0.5]["per_class_metrics"]
        assert "ap" in per_class[1]
        assert 0.0 <= result["overall_map"] <= 1.0

    def test_compute_ap_thresholds_nonunified(self) -> None:
        """Verify AP threshold aggregation works for keyed threshold sets."""
        key = "Vehicle_0.7_Pedestrian_0.5_Cyclist_0.5"
        cumulative = _make_cumulative_nonunified(key)
        result = ObjectDetectionUtils.compute_ap_thresholds(cumulative, ObjectDetectionUtils.compute_ap_101_point)
        assert key in result["thresholds"]
        per_class = result["thresholds"][key]["per_class_metrics"]
        assert "ap" in per_class[1]
        assert 0.0 <= result["overall_map"] <= 1.0


class TestComputeIou2d:
    """Tests for axis-aligned 2D intersection-over-union."""

    def test_identical_boxes(self) -> None:
        """Verify identical boxes yield an IoU of one."""
        box = _box2d(0.0, 0.0, 10.0, 10.0)
        assert ObjectDetectionUtils.compute_iou_2d(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Verify disjoint boxes yield an IoU of zero."""
        a = _box2d(0.0, 0.0, 1.0, 1.0)
        b = _box2d(2.0, 2.0, 3.0, 3.0)
        assert ObjectDetectionUtils.compute_iou_2d(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Verify partially overlapping boxes yield the expected IoU."""
        a = _box2d(0.0, 0.0, 2.0, 2.0)
        b = _box2d(1.0, 1.0, 3.0, 3.0)
        # intersection = 1×1 = 1, union = 4 + 4 - 1 = 7
        assert ObjectDetectionUtils.compute_iou_2d(a, b) == pytest.approx(1.0 / 7.0)


class TestBevGeometry:
    """Tests for BEV footprint polygon and volumetric 3D IoU."""

    def test_bev_poly_area_matches_box_footprint(self) -> None:
        """Verify polygon area matches the box footprint area."""
        poly = ObjectDetectionUtils.get_bev_poly(_box3d(width=2.0, length=3.0))
        assert poly.area == pytest.approx(6.0, rel=1e-5)

    def test_bev_poly_center_matches_box_xy(self) -> None:
        """Verify polygon centroid matches the box x and y coordinates."""
        poly = ObjectDetectionUtils.get_bev_poly(_box3d(x=5.0, y=7.0, width=2.0, length=2.0))
        assert poly.centroid.x == pytest.approx(5.0, rel=1e-5)
        assert poly.centroid.y == pytest.approx(7.0, rel=1e-5)

    def test_iou_3d_identical_boxes(self) -> None:
        """Verify identical boxes yield a 3D IoU of one."""
        box = _box3d(width=2.0, length=2.0, height=2.0)
        assert ObjectDetectionUtils.compute_iou_3d(box, box) == pytest.approx(1.0)

    def test_iou_3d_no_overlap(self) -> None:
        """Verify separated boxes yield a 3D IoU of zero."""
        a = _box3d(x=0.0, y=0.0, z=0.0, width=1.0, length=1.0, height=1.0)
        b = _box3d(x=10.0, y=10.0, z=10.0, width=1.0, length=1.0, height=1.0)
        assert ObjectDetectionUtils.compute_iou_3d(a, b) == pytest.approx(0.0)

    def test_iou_3d_partial_overlap_volumetric(self) -> None:
        """Verify the true volumetric 3D IoU formula (not BEV_IoU * Z_IoU).

        Two axis-aligned 2*2*2 boxes offset by 1 m in X and 1 m in Z:
          BEV intersection = 1*2 = 2,  BEV area each = 4,  BEV union = 6
          Z intersection   = 1,        Z span = 3
          3D intersection  = 2 * 1 = 2
          vol1 = vol2 = 4*2 = 8
          3D union = 8 + 8 - 2 = 14
          3D IoU   = 2/14 ≈ 0.1429
        """
        a = _box3d(x=0.0, y=0.0, z=0.0, width=2.0, length=2.0, height=2.0)
        b = _box3d(x=1.0, y=0.0, z=1.0, width=2.0, length=2.0, height=2.0)
        assert ObjectDetectionUtils.compute_iou_3d(a, b) == pytest.approx(2.0 / 14.0, rel=1e-5)


class TestComputeDistBev:
    """Tests for BEV (XY-plane) center distance."""

    @pytest.mark.parametrize(
        "box_a,box_b,expected",
        [
            (_box3d(x=1.0, y=2.0), _box3d(x=1.0, y=2.0), 0.0),  # identical
            (_box3d(x=0.0, y=0.0), _box3d(x=1.0, y=0.0), 1.0),  # unit step
            (_box3d(x=0.0, y=0.0), _box3d(x=3.0, y=4.0), 5.0),  # Pythagorean
            (_box3d(x=0.0, y=0.0, z=0.0), _box3d(x=0.0, y=0.0, z=100.0), 0.0),  # z ignored
        ],
    )
    def test_dist_bev(self, box_a, box_b, expected) -> None:
        """Verify BEV distance computation covers key cases."""
        assert ObjectDetectionUtils.compute_dist_bev(box_a, box_b) == pytest.approx(expected)


class TestBoxesFilter:
    """Tests for BEV distance-range filtering against the sensor origin."""

    def test_within_range(self) -> None:
        """Verify boxes within range are retained."""
        boxes = [_box3d(x=5.0, y=0.0)]
        result = ObjectDetectionUtils.boxes_filter(boxes, 0.0, 10.0, ObjectDetectionUtils.compute_dist_bev)
        assert len(result) == 1

    def test_beyond_range(self) -> None:
        """Verify boxes beyond the upper bound are removed."""
        boxes = [_box3d(x=15.0, y=0.0)]
        result = ObjectDetectionUtils.boxes_filter(boxes, 0.0, 10.0, ObjectDetectionUtils.compute_dist_bev)
        assert result == []

    def test_lower_boundary_inclusive(self) -> None:
        """Verify the lower bound is inclusive (``range_start <= dist``)."""
        boxes = [_box3d(x=0.0, y=0.0)]
        result = ObjectDetectionUtils.boxes_filter(boxes, 0.0, 10.0, ObjectDetectionUtils.compute_dist_bev)
        assert len(result) == 1

    def test_upper_boundary_exclusive(self) -> None:
        """Verify the upper bound is exclusive (``dist < range_end``)."""
        boxes = [_box3d(x=10.0, y=0.0)]
        result = ObjectDetectionUtils.boxes_filter(boxes, 0.0, 10.0, ObjectDetectionUtils.compute_dist_bev)
        assert result == []

    def test_empty_input(self) -> None:
        """Verify empty input lists stay empty after filtering."""
        assert ObjectDetectionUtils.boxes_filter([], 0.0, 50.0, ObjectDetectionUtils.compute_dist_bev) == []

    def test_multiple_boxes(self) -> None:
        """Verify filtering keeps only boxes inside the requested range."""
        boxes = [_box3d(x=5.0), _box3d(x=15.0), _box3d(x=25.0)]
        result = ObjectDetectionUtils.boxes_filter(boxes, 0.0, 20.0, ObjectDetectionUtils.compute_dist_bev)
        assert len(result) == 2  # 5 and 15 kept, 25 excluded

    def test_custom_origin(self) -> None:
        """Verify distance is measured from the supplied origin, not (0, 0)."""
        boxes = [_box3d(x=12.0, y=0.0)]
        origin = {"x": 10.0, "y": 0.0}  # box is 2 m from this origin
        result = ObjectDetectionUtils.boxes_filter(boxes, 0.0, 5.0, ObjectDetectionUtils.compute_dist_bev, origin=origin)
        assert len(result) == 1


class TestComputeAp101Point:
    """Direct tests for the 101-point AP interpolation (default min_recall=0.1, min_precision=0.1)."""

    def test_perfect_precision_yields_one(self) -> None:
        """Precision 1.0 across all recall integrates (and normalizes) to AP = 1.0."""
        rec = np.array([0.0, 1.0])
        prec = np.array([1.0, 1.0])
        assert ObjectDetectionUtils.compute_ap_101_point(rec, prec) == pytest.approx(1.0)

    def test_zero_precision_yields_zero(self) -> None:
        """Precision 0.0 everywhere yields AP = 0.0."""
        rec = np.array([0.0, 1.0])
        prec = np.array([0.0, 0.0])
        assert ObjectDetectionUtils.compute_ap_101_point(rec, prec) == pytest.approx(0.0)

    def test_constant_precision_is_min_precision_normalized(self) -> None:
        """Constant precision 0.5 → (0.5 - 0.1) / (1 - 0.1) = 0.4/0.9 after normalization."""
        rec = np.array([0.0, 1.0])
        prec = np.array([0.5, 0.5])
        assert ObjectDetectionUtils.compute_ap_101_point(rec, prec) == pytest.approx(0.4 / 0.9)

    def test_precision_below_min_is_clipped_to_zero(self) -> None:
        """Precision 0.05 < min_precision 0.1 → clipped to 0 → AP = 0.0."""
        rec = np.array([0.0, 1.0])
        prec = np.array([0.05, 0.05])
        assert ObjectDetectionUtils.compute_ap_101_point(rec, prec) == pytest.approx(0.0)

    def test_extrapolates_beyond_max_recall_as_zero(self) -> None:
        """Precision 1.0 up to recall 0.5, then ``right=0`` extrapolation.

        prec_interp = 1.0 on recall (0.11..0.50) → 40 bins at 0.9 after subtracting
        min_precision; 0 on recall (0.51..1.00) → 50 bins at 0. Mean over the 90
        evaluated bins = 36/90 = 0.4, normalized by 0.9 → 0.4/0.9.
        """
        rec = np.array([0.0, 0.5])
        prec = np.array([1.0, 1.0])
        assert ObjectDetectionUtils.compute_ap_101_point(rec, prec) == pytest.approx(0.4 / 0.9)


class TestComputeTp101Point:
    """Direct tests for the 101-point TP-metric interpolation (default min_recall=0.1)."""

    def test_returns_none_when_max_recall_below_min(self) -> None:
        """max_recall 0.05 < min_recall 0.1 → the window is empty → None."""
        rec_arr = np.array([0.05])
        assert ObjectDetectionUtils.compute_tp_101_point(rec_arr, [0.5]) is None

    def test_constant_metric_returns_that_value(self) -> None:
        """A flat metric curve averages to its constant value over the window."""
        rec_arr = np.array([0.0, 1.0])
        assert ObjectDetectionUtils.compute_tp_101_point(rec_arr, [0.3, 0.3]) == pytest.approx(0.3)

    def test_linear_metric_averages_over_window(self) -> None:
        """Metric rising 0→1 with recall averages over bins 0.11..1.00 = (0.11+1.00)/2 = 0.555."""
        rec_arr = np.array([0.0, 1.0])
        assert ObjectDetectionUtils.compute_tp_101_point(rec_arr, [0.0, 1.0]) == pytest.approx(0.555)
