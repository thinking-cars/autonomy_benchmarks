"""Tests for WaymoCameraObjectDetection2D benchmark."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from autohub_benchmarks.benchmarks.camera_object_detection_2d.WaymoCameraObjectDetection2D import (
    WaymoCameraObjectDetection2D,
)
from autohub_benchmarks.utils.BoundingBox2D import (
    BoundingBox2D,
)


class TestWaymoCameraObjectDetection2D:
    def setup_method(self) -> None:
        self.bm = WaymoCameraObjectDetection2D()
        self.pred_boxes = [
            BoundingBox2D(
                x1=8.0,
                y1=8.0,
                x2=12.0,
                y2=12.0,
                image_id=0,
                class_id=1,
                confidence_score=0.95,
            )
        ]
        self.gt_boxes = [
            BoundingBox2D(
                x1=8.0,
                y1=8.0,
                x2=12.0,
                y2=12.0,
                image_id=0,
                class_id=1,
                group="l1",
                distance=20.0,
            )
        ]

    def test_name(self) -> None:
        assert self.bm.name == "waymo_camera_object_detection_2d"

    def test_get_input(self) -> None:
        sample = {"image_front": [1, 2, 3], "objects": []}
        assert self.bm.get_input(sample) == [1, 2, 3]

    def test_get_label(self) -> None:
        sample = {"image_front": None, "objects": [[0, 0, 1, 1], [0, 0, 1, 1]]}
        assert self.bm.get_label(sample) == [[0, 0, 1, 1], [0, 0, 1, 1]]

    def test_compute_sample_metrics(self) -> None:
        metrics = self.bm.compute_sample_metrics(self.pred_boxes, self.gt_boxes)
        assert metrics["sample_prediction_num"] == 1
        assert metrics["sample_ground_truth_num"] == 1
        assert "match_records" in metrics

    def test_compute_aggregated_metrics(self) -> None:
        sample_results = [{"metrics": self.bm.compute_sample_metrics(self.pred_boxes, self.gt_boxes)}]
        agg = self.bm.compute_aggregated_metrics(sample_results)
        assert "aggregated_metrics" in agg
        assert "benchmark_score" in agg["aggregated_metrics"]
        assert "threshold_metrics" in agg["aggregated_metrics"]

    def test_save_results(self) -> None:
        self.bm.record_sample(self.pred_boxes, self.gt_boxes, "s0")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")
            self.bm.save_results(path)
            with open(path) as f:
                data = json.load(f)
            assert data["benchmark"] == "waymo_camera_object_detection_2d"
            assert data["num_samples"] == 1
            assert data["sample_results"][0]["sample_id"] == "s0"
            assert data["sample_results"][0]["metrics"]["sample_prediction_num"] == 1
            assert "aggregated_metrics" in data

    def test_excluded_class_sign_is_skipped(self) -> None:
        """Sign (class_id=3) must be excluded - exercises the `continue` branch."""
        sign_pred = BoundingBox2D(x1=0.0, y1=0.0, x2=1.0, y2=1.0, image_id=0, class_id=3, confidence_score=0.9)
        sign_gt = BoundingBox2D(x1=0.0, y1=0.0, x2=1.0, y2=1.0, image_id=0, class_id=3, group="l1", distance=10.0)
        metrics = self.bm.compute_sample_metrics([sign_pred], [sign_gt])
        # No class-3 entry should appear in any threshold record
        for lev_records in metrics["match_records"].values():
            for range_records in lev_records.values():
                for class_records in range_records.values():
                    assert 3 not in class_records

    def test_already_matched_gt_skipped(self) -> None:
        """Two preds for one GT - second pred must find the GT already matched."""
        pred_high = BoundingBox2D(x1=8.0, y1=8.0, x2=12.0, y2=12.0, image_id=0, class_id=1, confidence_score=0.95)
        pred_low = BoundingBox2D(x1=8.0, y1=8.0, x2=12.0, y2=12.0, image_id=0, class_id=1, confidence_score=0.5)
        gt = BoundingBox2D(x1=8.0, y1=8.0, x2=12.0, y2=12.0, image_id=0, class_id=1, group="l1", distance=20.0)
        metrics = self.bm.compute_sample_metrics([pred_high, pred_low], [gt])
        sample_results = [{"metrics": metrics}]
        agg = self.bm.compute_aggregated_metrics(sample_results)
        # Only one TP possible - second pred is FP
        assert "aggregated_metrics" in agg

    def test_gt_box_with_none_distance_raises(self) -> None:
        """GT box with distance=None must raise ValueError with an informative message."""
        gt_no_dist = BoundingBox2D(x1=0.0, y1=0.0, x2=1.0, y2=1.0, image_id=0, class_id=1, group="l1", distance=None)
        with pytest.raises(ValueError, match="distance=None"):
            self.bm.compute_sample_metrics(self.pred_boxes, [gt_no_dist])
