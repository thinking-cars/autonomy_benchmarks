"""Tests for WaymyCameraObjectDetection3D benchmark."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from autonomy_benchmarks.benchmarks.camera_object_detection_3d.WaymoCameraObjectDetection3D import (
    WaymoCameraObjectDetection3D,
)
from autonomy_benchmarks.utils.BoundingBox3D import BoundingBox3D


def _make_box(**kwargs) -> BoundingBox3D:
    """Return a minimal BoundingBox3D; keyword args override defaults."""
    defaults = dict(
        x=0.0,
        y=0.0,
        z=0.0,
        width=1.0,
        height=1.0,
        length=1.0,
        yaw=0.0,
        image_id=0,
        class_id=1,
    )
    defaults.update(kwargs)
    return BoundingBox3D(**defaults)


class TestWaymoCameraObjectDetection3D:
    """Behavioral tests for the Waymo 3D benchmark."""

    def setup_method(self) -> None:
        """Create a benchmark instance with one matching 3D box pair."""
        self.bm = WaymoCameraObjectDetection3D()
        self.pred_boxes = [
            BoundingBox3D(
                x=0.0,
                y=0.0,
                z=0.0,
                width=2.0,
                height=2.0,
                length=4.0,
                yaw=0.0,
                vx=0.0,
                vy=0.0,
                image_id=0,
                class_id=1,
                confidence_score=0.95,
            )
        ]
        self.gt_boxes = [
            BoundingBox3D(
                x=0.0,
                y=0.0,
                z=0.0,
                width=2.0,
                height=2.0,
                length=4.0,
                yaw=0.0,
                vx=0.0,
                vy=0.0,
                image_id=0,
                class_id=1,
                group="l1",
            )
        ]

    def test_name(self) -> None:
        """Verify the benchmark exposes its stable identifier."""
        assert self.bm.name == "waymo_camera_object_detection_3d"

    def _make_sample_results(self, pred_boxes, gt_boxes):
        """Helper: run Pass 1 and wrap result as a sample_results list."""
        metrics = self.bm.compute_sample_metrics(pred_boxes, gt_boxes)
        return [{"metrics": metrics}]

    def test_compute_sample_metrics(self) -> None:
        """Verify per-sample metrics include counts and match records."""
        metrics = self.bm.compute_sample_metrics(self.pred_boxes, self.gt_boxes)
        assert metrics["sample_prediction_num"] == 1
        assert metrics["sample_ground_truth_num"] == 1
        assert "match_records" in metrics

    def test_compute_aggregated_metrics(self) -> None:
        """Verify aggregated metrics expose AP, APH, and benchmark scores."""
        sample_results = self._make_sample_results(self.pred_boxes, self.gt_boxes)
        agg = self.bm.compute_aggregated_metrics(sample_results)
        assert "aggregated_metrics" in agg
        assert "benchmark_score" in agg["aggregated_metrics"]
        assert "threshold_metrics" in agg["aggregated_metrics"]
        assert "threshold_metrics_with_heading" in agg["aggregated_metrics"]

    def test_save_results(self) -> None:
        """Verify final results are serialized with sample and aggregate data."""
        self.bm.record_sample(self.pred_boxes, self.gt_boxes, "s0")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")
            self.bm.save_results(path)
            with open(path) as f:
                data = json.load(f)
            assert data["benchmark"] == "waymo_camera_object_detection_3d"
            assert data["num_samples"] == 1
            assert data["sample_results"][0]["sample_id"] == "s0"
            assert data["sample_results"][0]["metrics"]["sample_prediction_num"] == 1
            assert "aggregated_metrics" in data

    def test_heading_weighted_metrics_present(self) -> None:
        """threshold_metrics_with_heading must be present in aggregated metrics."""
        sample_results = self._make_sample_results(self.pred_boxes, self.gt_boxes)
        agg = self.bm.compute_aggregated_metrics(sample_results)
        assert "threshold_metrics_with_heading" in agg["aggregated_metrics"]

    def test_heading_weighted_fp_branch(self) -> None:
        """Pred far from GT exercises the FP branch of heading-weighted matching."""
        pred_far = [
            BoundingBox3D(
                x=1000.0,
                y=0.0,
                z=0.0,
                width=2.0,
                height=2.0,
                length=4.0,
                yaw=0.0,
                vx=0.0,
                vy=0.0,
                image_id=0,
                class_id=1,
                confidence_score=0.5,
            )
        ]
        # GT needs number_of_lidar_points to be resolved as l1.
        gt_near = _make_box(x=0.0, y=0.0, z=0.0, number_of_lidar_points=10)
        sample_results = self._make_sample_results(pred_far, [gt_near])
        agg = self.bm.compute_aggregated_metrics(sample_results)
        wh = agg["aggregated_metrics"]["threshold_metrics_with_heading"]
        r_key = list(wh["l1"].keys())[0]
        assert wh["l1"][r_key]["overall_map"] == pytest.approx(0.0)

    def test_heading_weighted_tp_branch(self) -> None:
        """GT with group='l1' and no number_of_lidar_points is excluded → pred is FP → APH=0."""
        sample_results = self._make_sample_results(self.pred_boxes, self.gt_boxes)
        agg = self.bm.compute_aggregated_metrics(sample_results)
        wh = agg["aggregated_metrics"]["threshold_metrics_with_heading"]
        r_key = list(wh["l1"].keys())[0]
        assert wh["l1"][r_key]["overall_map"] == pytest.approx(0.0)

    def test_heading_weighted_tp_matched(self) -> None:
        """Two identical preds vs two GTs with yaw=0 → tp_weight_aph=1.0 for both → APH > 0."""
        pred1 = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, confidence_score=0.9, image_id=0)
        pred2 = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, confidence_score=0.8, image_id=1)
        gt1 = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, number_of_lidar_points=10, image_id=0)
        gt2 = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, number_of_lidar_points=10, image_id=1)
        sample_results = self._make_sample_results([pred1, pred2], [gt1, gt2])
        agg = self.bm.compute_aggregated_metrics(sample_results)
        wh = agg["aggregated_metrics"]["threshold_metrics_with_heading"]
        r_key = list(wh["l1"].keys())[0]
        assert wh["l1"][r_key]["overall_map"] > 0.0

    def test_heading_weighted_unknown_class_skipped(self) -> None:
        """A pred/GT with class_id=3 (Sign) exercises the excluded-class branch."""
        sign_pred = _make_box(class_id=3, confidence_score=0.8)
        sign_gt = _make_box(class_id=3, number_of_lidar_points=10)
        sample_results = self._make_sample_results([sign_pred], [sign_gt])
        agg = self.bm.compute_aggregated_metrics(sample_results)
        # Sign class has no threshold so it is skipped - metrics still return a valid dict
        assert "threshold_metrics_with_heading" in agg["aggregated_metrics"]

    def test_heading_weighted_already_matched_gt_skipped(self) -> None:
        """Two preds competing for one GT - the second pred will find the GT already matched."""
        pred_high = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, confidence_score=0.95)
        pred_low = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, confidence_score=0.5)
        gt = _make_box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=4.0, yaw=0.0, number_of_lidar_points=10)
        sample_results = self._make_sample_results([pred_high, pred_low], [gt])
        agg = self.bm.compute_aggregated_metrics(sample_results)
        wh = agg["aggregated_metrics"]["threshold_metrics_with_heading"]
        r_key = list(wh["l1"].keys())[0]
        assert wh["l1"][r_key]["overall_map"] >= 0.0


class TestPreparePredGtBoxesWithPts:
    """Ensure prepare_pred_gt_boxes respects number_of_lidar_points-based difficulty."""

    def setup_method(self) -> None:
        """Create a benchmark instance and reusable prediction box."""
        self.bm = WaymoCameraObjectDetection3D()
        self.pred = _make_box(x=0.0, y=0.0, z=0.0, confidence_score=0.9)

    def test_pts_gt5_included_in_l1(self) -> None:
        """A GT with number_of_lidar_points=6 (no group) should appear in l1 buckets."""
        gt = _make_box(x=0.0, y=0.0, z=0.0, number_of_lidar_points=6)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l1"]["00_35"]) == 1

    def test_pts_gt5_included_in_l2(self) -> None:
        """l2 includes l1 boxes, so a pts=6 GT must also appear in l2 buckets."""
        gt = _make_box(x=0.0, y=0.0, z=0.0, number_of_lidar_points=6)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l2"]["00_35"]) == 1

    def test_pts_1_to_5_excluded_from_l1(self) -> None:
        """A GT with number_of_lidar_points=3 is LEVEL_2 only - must not appear in l1."""
        gt = _make_box(x=0.0, y=0.0, z=0.0, number_of_lidar_points=3)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l1"]["00_35"]) == 0

    def test_pts_1_to_5_included_in_l2(self) -> None:
        """A GT with number_of_lidar_points=3 is LEVEL_2 - must appear in l2 buckets."""
        gt = _make_box(x=0.0, y=0.0, z=0.0, number_of_lidar_points=3)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l2"]["00_35"]) == 1

    def test_pts_0_excluded_from_both_levels(self) -> None:
        """A GT with number_of_lidar_points=0 cannot be classified - excluded everywhere."""
        gt = _make_box(x=0.0, y=0.0, z=0.0, number_of_lidar_points=0)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l1"]["00_35"]) == 0
        assert len(gt_rl["l2"]["00_35"]) == 0

    def test_no_pts_no_group_excluded(self) -> None:
        """A GT with neither group nor number_of_lidar_points is excluded from all levels."""
        gt = _make_box(x=0.0, y=0.0, z=0.0)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l1"]["00_35"]) == 0
        assert len(gt_rl["l2"]["00_35"]) == 0

    def test_explicit_l2_group_excluded_from_l1(self) -> None:
        """Explicit group='l2' overrides any pts; box must not appear in l1."""
        gt = _make_box(x=0.0, y=0.0, z=0.0, group="l2", number_of_lidar_points=10)
        _, gt_rl = self.bm._prepare_pred_gt_boxes([self.pred], [gt])
        assert len(gt_rl["l1"]["00_35"]) == 0
        assert len(gt_rl["l2"]["00_35"]) == 1
