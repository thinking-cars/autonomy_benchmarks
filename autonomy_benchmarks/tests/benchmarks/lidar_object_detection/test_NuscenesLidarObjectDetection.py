"""Tests for NuscenesLidarObjectDetection benchmark."""

from __future__ import annotations

import math

import pytest

from autohub_benchmarks.benchmarks.lidar_object_detection.NuscenesLidarObjectDetection import (
    NuscenesLidarObjectDetection,
)
from autohub_benchmarks.utils.BoundingBox3D import BoundingBox3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(
    x=0.0,
    y=0.0,
    z=0.0,
    width=1.0,
    height=1.0,
    length=1.0,
    yaw=0.0,
    vx=0.0,
    vy=0.0,
    class_id=1,
    confidence_score=0.9,
    lidar_pts=None,
    radar_pts=None,
) -> BoundingBox3D:
    return BoundingBox3D(
        x=x,
        y=y,
        z=z,
        width=width,
        height=height,
        length=length,
        yaw=yaw,
        vx=vx,
        vy=vy,
        class_id=class_id,
        confidence_score=confidence_score,
        number_of_lidar_points=lidar_pts,
        number_of_radar_points=radar_pts,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_name(self):
        bm = NuscenesLidarObjectDetection()
        assert bm.name == "nuscenes_lidar_object_detection"

    def test_matching_thresholds(self):
        bm = NuscenesLidarObjectDetection()
        assert bm.matching_thresholds == [0.5, 1.0, 2.0, 4.0]

    def test_tp_metric_threshold(self):
        bm = NuscenesLidarObjectDetection()
        assert bm.tp_metric_threshold == 2.0

    def test_per_class_detection_ranges(self):
        bm = NuscenesLidarObjectDetection()
        for cid in [1, 2, 3, 4, 5]:
            assert bm.per_class_detection_ranges[cid] == (0.0, 50.0)
        for cid in [6, 7, 8]:
            assert bm.per_class_detection_ranges[cid] == (0.0, 40.0)
        for cid in [9, 10]:
            assert bm.per_class_detection_ranges[cid] == (0.0, 30.0)

    def test_aoe_special_class_sets(self):
        bm = NuscenesLidarObjectDetection()
        assert 9 in bm.aoe_ignored_classes
        assert 10 in bm.aoe_pi_classes

    def test_ave_ignored_classes(self):
        bm = NuscenesLidarObjectDetection()
        assert 9 in bm.ave_ignored_classes
        assert 10 in bm.ave_ignored_classes


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestInterface:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_get_input(self):
        sample = {"point_cloud": "pc_data", "objects": []}
        assert self.bm.get_input(sample) == "pc_data"

    def test_get_label(self):
        sample = {"point_cloud": None, "objects": ["box1"]}
        assert self.bm.get_label(sample) == ["box1"]

    def test_compute_sample_metrics_counts(self):
        pred = [_box(x=0.0), _box(x=1.0)]
        gt = [_box(x=0.0), _box(x=1.0), _box(x=2.0)]
        result = self.bm.compute_sample_metrics(pred, gt)
        assert result["sample_prediction_num"] == 2
        assert result["sample_ground_truth_num"] == 3

    def test_compute_sample_metrics_empty_inputs(self):
        result = self.bm.compute_sample_metrics([], [])
        assert result["sample_prediction_num"] == 0
        assert result["sample_ground_truth_num"] == 0
        # Implementation always creates an entry per threshold; no class records inside.
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert thr in result["match_records"]
            assert result["match_records"][thr] == {}

    def test_compute_sample_metrics_returns_match_records_for_all_thresholds(self):
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0)]
        result = self.bm.compute_sample_metrics(pred, gt)
        assert "match_records" in result
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert thr in result["match_records"]


# ---------------------------------------------------------------------------
# GT point filter
# ---------------------------------------------------------------------------


class TestGTPointFilter:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_gt_with_zero_lidar_and_zero_radar_excluded(self):
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt_bad = _box(x=0.0, lidar_pts=0, radar_pts=0)
        result = self.bm.compute_sample_metrics(pred, [gt_bad])
        for thr, class_records in result["match_records"].items():
            for class_id, data in class_records.items():
                assert data["gt_count"] == 0
                assert all(not e["is_tp"] for e in data["pred_entries"])

    def test_gt_with_lidar_points_kept(self):
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt_ok = _box(x=0.0, lidar_pts=5, radar_pts=0)
        result = self.bm.compute_sample_metrics(pred, [gt_ok])
        for thr, class_records in result["match_records"].items():
            for class_id, data in class_records.items():
                assert data["gt_count"] == 1

    def test_gt_with_radar_points_kept(self):
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt_ok = _box(x=0.0, lidar_pts=0, radar_pts=3)
        result = self.bm.compute_sample_metrics(pred, [gt_ok])
        for thr, class_records in result["match_records"].items():
            for class_id, data in class_records.items():
                assert data["gt_count"] == 1

    def test_gt_with_none_points_not_excluded(self):
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt_none = _box(x=0.0, lidar_pts=None, radar_pts=None)
        result = self.bm.compute_sample_metrics(pred, [gt_none])
        for thr, class_records in result["match_records"].items():
            for class_id, data in class_records.items():
                assert data["gt_count"] == 1


# ---------------------------------------------------------------------------
# Per-class detection range filter
# ---------------------------------------------------------------------------


class TestPerClassRangeFilter:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_cone_beyond_30m_excluded(self):
        pred = [_box(x=31.0, y=0.0, class_id=9, confidence_score=0.9)]
        gt = [_box(x=31.0, y=0.0, class_id=9, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for thr, class_records in result["match_records"].items():
            assert 9 not in class_records

    def test_cone_at_30m_included(self):
        pred = [_box(x=30.0, y=0.0, class_id=9, confidence_score=0.9)]
        gt = [_box(x=30.0, y=0.0, class_id=9, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for thr, class_records in result["match_records"].items():
            assert 9 in class_records

    def test_pedestrian_beyond_40m_excluded(self):
        pred = [_box(x=45.0, y=0.0, class_id=6, confidence_score=0.9)]
        gt = [_box(x=45.0, y=0.0, class_id=6, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for thr, class_records in result["match_records"].items():
            assert 6 not in class_records

    def test_car_at_50m_included(self):
        pred = [_box(x=50.0, y=0.0, class_id=1, confidence_score=0.9)]
        gt = [_box(x=50.0, y=0.0, class_id=1, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for thr, class_records in result["match_records"].items():
            assert 1 in class_records

    def test_car_beyond_50m_excluded(self):
        pred = [_box(x=51.0, y=0.0, class_id=1, confidence_score=0.9)]
        gt = [_box(x=51.0, y=0.0, class_id=1, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for thr, class_records in result["match_records"].items():
            assert 1 not in class_records


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


class TestMatching:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_perfect_match_is_tp(self):
        pred = [_box(x=0.0, y=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, y=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for thr in [0.5, 1.0, 2.0, 4.0]:
            entries = result["match_records"][thr][1]["pred_entries"]
            assert entries[0]["is_tp"] is True

    def test_distant_pred_is_fp_at_tight_threshold(self):
        pred = [_box(x=10.0, y=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, y=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entries_05 = result["match_records"][0.5][1]["pred_entries"]
        assert entries_05[0]["is_tp"] is False

    def test_gt_matched_once(self):
        pred = [
            _box(x=0.0, y=0.0, confidence_score=0.9),
            _box(x=0.1, y=0.0, confidence_score=0.8),
        ]
        gt = [_box(x=0.0, y=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entries = result["match_records"][0.5][1]["pred_entries"]
        tp_count = sum(1 for e in entries if e["is_tp"])
        assert tp_count == 1

    def test_match_selects_closest(self):
        pred = [_box(x=0.3, y=0.0, confidence_score=0.9)]
        gt_near = _box(x=0.2, y=0.0, class_id=1, lidar_pts=5)
        gt_far = _box(x=0.8, y=0.0, class_id=1, lidar_pts=5)
        result = self.bm.compute_sample_metrics(pred, [gt_far, gt_near])
        entries = result["match_records"][0.5][1]["pred_entries"]
        assert entries[0]["dist"] == pytest.approx(0.1, abs=1e-6)

    def test_gt_count_correct(self):
        gt = [_box(x=float(i), lidar_pts=5) for i in range(3)]
        result = self.bm.compute_sample_metrics([], gt)
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert result["match_records"][thr][1]["gt_count"] == 3

    def test_no_pred_yields_empty_pred_entries(self):
        gt = [_box(x=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics([], gt)
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert result["match_records"][thr][1]["pred_entries"] == []
            assert result["match_records"][thr][1]["gt_count"] == 1

    def test_threshold_selectivity(self):
        pred = [_box(x=0.8, y=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, y=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        assert result["match_records"][0.5][1]["pred_entries"][0]["is_tp"] is False
        assert result["match_records"][1.0][1]["pred_entries"][0]["is_tp"] is True
        assert result["match_records"][2.0][1]["pred_entries"][0]["is_tp"] is True
        assert result["match_records"][4.0][1]["pred_entries"][0]["is_tp"] is True


# ---------------------------------------------------------------------------
# TP error metrics
# ---------------------------------------------------------------------------


class TestTPMetrics:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_ate_correct(self):
        # pred at (3, 0), gt at (0, 0) → BEV dist = 3.0, within 4 m threshold.
        pred = [_box(x=3.0, y=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, y=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][4.0][1]["pred_entries"][0]
        assert entry["is_tp"] is True
        assert entry["ate"] == pytest.approx(3.0, abs=1e-6)

    def test_ase_perfect_match_is_zero(self):
        pred = [_box(width=2.0, height=1.5, length=4.0, confidence_score=0.9)]
        gt = [_box(width=2.0, height=1.5, length=4.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][1]["pred_entries"][0]
        assert entry["is_tp"] is True
        assert entry["ase"] == pytest.approx(0.0, abs=1e-6)

    def test_ase_formula(self):
        """Pred 2x2x2, GT 4x4x4: inter=8, union=64+8-8=64, IoU=1/8, ASE=7/8."""
        pred = [_box(width=2.0, height=2.0, length=2.0, confidence_score=0.9)]
        gt = [_box(width=4.0, height=4.0, length=4.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][1]["pred_entries"][0]
        assert entry["is_tp"] is True
        assert entry["ase"] == pytest.approx(0.875, abs=1e-6)

    def test_aoe_normal_class(self):
        pred = [_box(yaw=0.3, confidence_score=0.9, class_id=1)]
        gt = [_box(yaw=0.0, class_id=1, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][1]["pred_entries"][0]
        expected = min(0.3, 2 * math.pi - 0.3)
        assert entry["aoe"] == pytest.approx(expected, abs=1e-6)

    def test_aoe_traffic_cone_is_zero(self):
        pred = [_box(yaw=1.0, class_id=9, confidence_score=0.9)]
        gt = [_box(yaw=0.0, class_id=9, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][9]["pred_entries"][0]
        assert entry["is_tp"] is True
        assert entry["aoe"] == pytest.approx(0.0, abs=1e-6)

    def test_aoe_barrier_capped_at_pi(self):
        """AOE for barrier (class 10) uses π-symmetry: wrap into [0,π) then min with complement.
        With heading_diff = π+0.5: mod=0.5, aoe = min(0.5, π-0.5) = 0.5, not π+0.5."""
        heading_diff_raw = math.pi + 0.5
        pred = [_box(yaw=heading_diff_raw, class_id=10, confidence_score=0.9)]
        gt = [_box(yaw=0.0, class_id=10, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][10]["pred_entries"][0]
        assert entry["is_tp"] is True
        heading_diff_mod = heading_diff_raw % math.pi
        expected = min(heading_diff_mod, math.pi - heading_diff_mod)
        assert entry["aoe"] == pytest.approx(expected, abs=1e-6)
        assert entry["aoe"] <= math.pi / 2 + 1e-6  # barrier AOE is always ≤ π/2

    def test_ave_normal_class(self):
        pred = [_box(vx=3.0, vy=4.0, confidence_score=0.9, class_id=1)]
        gt = [_box(vx=0.0, vy=0.0, class_id=1, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][1]["pred_entries"][0]
        assert entry["ave"] == pytest.approx(5.0, abs=1e-6)

    def test_ave_traffic_cone_is_zero(self):
        pred = [_box(vx=10.0, vy=10.0, class_id=9, confidence_score=0.9)]
        gt = [_box(vx=0.0, vy=0.0, class_id=9, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][9]["pred_entries"][0]
        assert entry["is_tp"] is True
        assert entry["ave"] == pytest.approx(0.0, abs=1e-6)

    def test_ave_barrier_is_zero(self):
        pred = [_box(vx=10.0, vy=10.0, class_id=10, confidence_score=0.9)]
        gt = [_box(vx=0.0, vy=0.0, class_id=10, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][10]["pred_entries"][0]
        assert entry["is_tp"] is True
        assert entry["ave"] == pytest.approx(0.0, abs=1e-6)

    def test_fp_has_zero_tp_metrics(self):
        # pred at x=10 (within 50 m range), gt at x=0 → dist=10 > all thresholds → FP.
        pred = [_box(x=10.0, confidence_score=0.9)]
        gt = [_box(x=0.0, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][1]["pred_entries"][0]
        assert entry["is_tp"] is False
        for key in ["ate", "ase", "aoe", "ave"]:
            assert entry[key] == 0.0
        assert entry["aae"] is None  # FP always has None for aae

    def test_aae_in_pred_entries_is_none_without_attribute_data(self):
        """When GT has no attribute, aae must be None (not evaluable)."""
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, lidar_pts=5)]  # attribute=None by default
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][0.5][1]["pred_entries"][0]
        assert "aae" in entry
        assert entry["aae"] is None

    def test_aae_correct_attribute_match(self):
        """TP where pred.attribute == gt.attribute → aae = 0.0."""
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, lidar_pts=5)]
        pred[0].attribute = "vehicle.moving"
        gt[0].attribute = "vehicle.moving"
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][2.0][1]["pred_entries"][0]
        assert entry["aae"] == pytest.approx(0.0)

    def test_aae_wrong_attribute_match(self):
        """TP where pred.attribute != gt.attribute → aae = 1.0."""
        pred = [_box(x=0.0, confidence_score=0.9)]
        gt = [_box(x=0.0, lidar_pts=5)]
        pred[0].attribute = "vehicle.parked"
        gt[0].attribute = "vehicle.moving"
        result = self.bm.compute_sample_metrics(pred, gt)
        entry = result["match_records"][2.0][1]["pred_entries"][0]
        assert entry["aae"] == pytest.approx(1.0)

    def test_aae_ignored_for_barrier_and_cone(self):
        """Barrier (10) and traffic_cone (9) always get aae=0.0 (ignored)."""
        for class_id, x in [(10, 0.0), (9, 0.0)]:
            pred_box = _box(x=x, confidence_score=0.9, class_id=class_id)
            gt_box = _box(x=x, lidar_pts=5, class_id=class_id)
            # Give them different attributes - should still be 0.0
            pred_box.attribute = "some_attr"
            gt_box.attribute = "other_attr"
            result = self.bm.compute_sample_metrics([pred_box], [gt_box])
            for thr_records in result["match_records"].values():
                if class_id in thr_records:
                    for e in thr_records[class_id]["pred_entries"]:
                        if e["is_tp"]:
                            assert e["aae"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Merge match records
# ---------------------------------------------------------------------------


class TestMergeMatchRecords:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_merge_gt_count_sums(self):
        frame1 = {0.5: {1: {"pred_entries": [], "gt_count": 3}}}
        frame2 = {0.5: {1: {"pred_entries": [], "gt_count": 2}}}
        merged = self.bm._merge_match_records([frame1, frame2])
        assert merged[0.5][1]["gt_count"] == 5

    def test_merge_pred_entries_concatenated(self):
        e1 = {
            "confidence_score": 0.9,
            "is_tp": True,
            "dist": 0.1,
            "ate": 0.1,
            "ase": 0.0,
            "aoe": 0.0,
            "ave": 0.0,
            "aae": None,
        }
        e2 = {
            "confidence_score": 0.7,
            "is_tp": False,
            "dist": float("inf"),
            "ate": 0.0,
            "ase": 0.0,
            "aoe": 0.0,
            "ave": 0.0,
            "aae": None,
        }
        frame1 = {0.5: {1: {"pred_entries": [e1], "gt_count": 1}}}
        frame2 = {0.5: {1: {"pred_entries": [e2], "gt_count": 0}}}
        merged = self.bm._merge_match_records([frame1, frame2])
        assert len(merged[0.5][1]["pred_entries"]) == 2


# ---------------------------------------------------------------------------
# AP filter (>= inclusive boundary)
# ---------------------------------------------------------------------------


class TestAPFilter:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def test_ap_filter_boundary_exclusive(self):
        """P-R points exactly at 0.1 must be excluded (> not >=) per the
        official nuScenes spec which states recalls and precisions > 0.1.
        A single remaining point above the boundary produces AP == 0 via
        trapezoidal integration (needs at least two points for non-zero area)."""
        cumulative_results = {
            0.5: {
                1: {
                    0: {
                        "confidence_score": 0.9,
                        "dist": 0.0,
                        "cum_tp": 1.0,
                        "cum_fp": 9.0,
                        "cum_fn": 9.0,
                        "cum_precision": 0.1,  # exactly at boundary - excluded
                        "cum_recall": 0.1,  # exactly at boundary - excluded
                    },
                    1: {
                        "confidence_score": 0.8,
                        "dist": 0.0,
                        "cum_tp": 2.0,
                        "cum_fp": 9.0,
                        "cum_fn": 8.0,
                        "cum_precision": 0.18,
                        "cum_recall": 0.2,
                    },
                }
            }
        }
        metrics = self.bm._compute_threshold_metrics(cumulative_results)
        ap = metrics["thresholds"][0.5]["per_class_metrics"][1]["ap"]
        # Only the second point passes the filter; a single point gives AP == 0.
        assert ap == 0.0

    def test_below_min_precision_excluded(self):
        cumulative_results = {
            0.5: {
                1: {
                    0: {
                        "confidence_score": 0.9,
                        "dist": 0.0,
                        "cum_tp": 1.0,
                        "cum_fp": 19.0,
                        "cum_fn": 9.0,
                        "cum_precision": 0.05,
                        "cum_recall": 0.1,
                    }
                }
            }
        }
        metrics = self.bm._compute_threshold_metrics(cumulative_results)
        ap = metrics["thresholds"][0.5]["per_class_metrics"][1]["ap"]
        assert ap == 0.0

    def test_below_min_recall_excluded(self):
        cumulative_results = {
            0.5: {
                1: {
                    0: {
                        "confidence_score": 0.9,
                        "dist": 0.0,
                        "cum_tp": 1.0,
                        "cum_fp": 9.0,
                        "cum_fn": 99.0,
                        "cum_precision": 0.1,
                        "cum_recall": 0.05,
                    }
                }
            }
        }
        metrics = self.bm._compute_threshold_metrics(cumulative_results)
        ap = metrics["thresholds"][0.5]["per_class_metrics"][1]["ap"]
        assert ap == 0.0


# ---------------------------------------------------------------------------
# Aggregated metrics (Pass 2)
# ---------------------------------------------------------------------------


class TestAggregatedMetrics:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    def _make_sample_result(self, pred_boxes, gt_boxes):
        metrics = self.bm.compute_sample_metrics(pred_boxes, gt_boxes)
        return {"metrics": metrics}

    def test_perfect_single_class_map_is_one(self):
        # All preds perfectly match their gt (dist=0). Boxes beyond 50 m are
        # range-filtered for class 1, so use x in [0, 10]. Trapezoidal AP
        # equals (n-1)/n for n matched pairs; with n=10, AP ≈ 0.9.
        n = 10
        preds = [_box(x=float(i), confidence_score=1.0 - i * 0.01) for i in range(n)]
        gts = [_box(x=float(i), lidar_pts=5) for i in range(n)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        # With perfect matching the mAP must be well above 0 (near (n-1)/n).
        assert result["benchmark_score"]["map"] >= 0.8

    def test_no_pred_map_is_zero(self):
        gts = [_box(x=0.0, lidar_pts=5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result([], gts)])
        assert result["benchmark_score"]["map"] == 0.0

    def test_all_fp_map_is_zero(self):
        preds = [_box(x=float(i), confidence_score=0.9) for i in range(5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, [])])
        assert result["benchmark_score"]["map"] == 0.0

    def test_multi_frame_gt_count_accumulates(self):
        r1 = self._make_sample_result([_box(x=0.0, confidence_score=0.9)], [_box(x=0.0, lidar_pts=5)])
        r2 = self._make_sample_result([_box(x=0.0, confidence_score=0.8)], [_box(x=0.0, lidar_pts=5)])
        merged = self.bm._merge_match_records([r1["metrics"]["match_records"], r2["metrics"]["match_records"]])
        assert merged[0.5][1]["gt_count"] == 2

    def test_benchmark_score_keys(self):
        preds = [_box(x=0.0, confidence_score=0.9)]
        gts = [_box(x=0.0, lidar_pts=5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        score = result["benchmark_score"]
        for key in ["map", "mate_2.0", "mase_2.0", "maoe_2.0", "mave_2.0", "maae_2.0", "nds"]:
            assert key in score
        # maae_2.0 is always a float; with the current loader (attribute not
        # populated) only ignored classes 9/10 contribute aae=0.0, so _mean_tp_metric
        # falls back to 1.0 when no TP of class 9/10 exists.
        assert isinstance(score["maae_2.0"], float)

    def test_tp_metrics_keys_include_aae(self):
        preds = [_box(x=0.0, confidence_score=0.9)]
        gts = [_box(x=0.0, lidar_pts=5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        for class_id, tm in result["tp_metrics"].items():
            for key in ["ate", "ase", "aoe", "ave", "aae"]:
                assert key in tm
            # aae is None when attribute data is absent (default _box has attribute=None)
            assert tm["aae"] is None

    def test_threshold_metrics_structure(self):
        preds = [_box(x=0.0, confidence_score=0.9)]
        gts = [_box(x=0.0, lidar_pts=5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        tm = result["threshold_metrics"]
        assert "thresholds" in tm
        assert "overall_map" in tm
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert thr in tm["thresholds"]


# ---------------------------------------------------------------------------
# NDS formula
# ---------------------------------------------------------------------------


class TestNDS:
    def setup_method(self):
        self.bm = NuscenesLidarObjectDetection()

    # Helper: minimal threshold_metrics structure with the required "thresholds" key.
    @staticmethod
    def _tm(overall_map: float) -> dict:
        return {"overall_map": overall_map, "thresholds": {}}

    def test_nds_formula_with_maae(self):
        """Full 6-term formula (denominator 10) when aae data is available."""
        tp_metrics = {1: {"ate": 0.2, "ase": 0.1, "aoe": 0.3, "ave": 0.4, "aae": 0.05}}
        score = self.bm._compute_benchmark_score(self._tm(0.5), tp_metrics)
        expected = (
            5.0 * 0.5
            + max(1.0 - 0.2, 0.0)
            + max(1.0 - 0.1, 0.0)
            + max(1.0 - 0.3, 0.0)
            + max(1.0 - 0.4, 0.0)
            + max(1.0 - 0.05, 0.0)
        ) / 10.0
        assert score["nds"] == pytest.approx(expected, abs=1e-4)

    def test_nds_missing_aae_penalises_score(self):
        """aae=None → maae falls back to 1.0 (worst case) → score(mAAE)=0 → nds=0.9, not 1.0."""
        tp_with = {1: {"ate": 0.0, "ase": 0.0, "aoe": 0.0, "ave": 0.0, "aae": 0.0}}
        tp_without = {1: {"ate": 0.0, "ase": 0.0, "aoe": 0.0, "ave": 0.0, "aae": None}}
        score_with = self.bm._compute_benchmark_score(self._tm(1.0), tp_with)
        score_without = self.bm._compute_benchmark_score(self._tm(1.0), tp_without)
        assert score_with["nds"] == pytest.approx(1.0, abs=1e-4)
        # maae=1.0 → score(1.0)=0 → (5*1 + 1+1+1+1+0) / 10 = 0.9
        assert score_without["nds"] == pytest.approx(0.9, abs=1e-4)

    def test_nds_clamps_tp_errors_above_one(self):
        tp_metrics = {1: {"ate": 2.0, "ase": 3.0, "aoe": 5.0, "ave": 1.5, "aae": 2.0}}
        score = self.bm._compute_benchmark_score(self._tm(0.0), tp_metrics)
        assert score["nds"] >= 0.0


# ---------------------------------------------------------------------------
# Bike-rack filter
# ---------------------------------------------------------------------------


class TestBikeRackFilter:
    """Tests for bike-rack filtering logic inside _prepare_pred_gt_boxes.

    Bike-rack boxes are identified by class_id == bm.bike_rack_class_id (12).
    They are passed as part of the GT list and are never returned in the
    filtered GT output.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.bm = NuscenesLidarObjectDetection()

    def _rack(self, x=0, y=0, width=4.0, length=4.0, yaw=0.0):
        """Return a bike-rack box using the benchmark's rack class_id."""
        return _box(x=x, y=y, width=width, length=length, yaw=yaw, class_id=self.bm.bike_rack_class_id)

    # -- BEV polygon geometry (via _prepare_pred_gt_boxes) --------------------

    def test_rack_center_removes_bike_at_same_position(self):
        """A bicycle exactly at the rack center must be filtered out."""
        rack = self._rack(x=5, y=5, width=4.0, length=4.0)
        bike = _box(x=5, y=5, class_id=8, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, bike])
        assert gt == []

    def test_rack_far_away_does_not_remove_bike(self):
        """A rack far from the bicycle must leave the bicycle untouched."""
        rack = self._rack(x=100, y=100, width=4.0, length=4.0)
        bike = _box(x=0, y=0, class_id=8, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, bike])
        assert len(gt) == 1

    def test_rack_polygon_has_correct_extent(self):
        """A point well outside the rack footprint must not trigger filtering."""
        rack = self._rack(x=0, y=0, width=2.0, length=2.0)
        bike = _box(x=5, y=5, class_id=8, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, bike])
        assert len(gt) == 1

    # -- class selectivity ----------------------------------------------------

    def test_filter_removes_bicycle_class_inside_rack(self):
        rack = self._rack(x=0, y=0, width=4.0, length=4.0)
        bike = _box(x=0, y=0, class_id=8, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, bike])
        assert gt == []

    def test_filter_removes_motorcycle_class_inside_rack(self):
        rack = self._rack(x=0, y=0, width=4.0, length=4.0)
        moto = _box(x=0.5, y=0.5, class_id=7, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, moto])
        assert gt == []

    def test_filter_keeps_car_inside_rack(self):
        """Non-bike classes must never be filtered by a bike-rack."""
        rack = self._rack(x=0, y=0, width=4.0, length=4.0)
        car = _box(x=0, y=0, class_id=1, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, car])
        assert len(gt) == 1

    def test_filter_applied_to_pred_boxes_too(self):
        """The bike-rack filter must remove matching pred boxes as well."""
        rack = self._rack(x=0, y=0, width=4.0, length=4.0)
        bike_pred = _box(x=0, y=0, class_id=8, confidence_score=0.9)
        pred, gt = self.bm._prepare_pred_gt_boxes([bike_pred], [rack])
        assert pred == []

    def test_no_rack_in_gt_leaves_bikes_unchanged(self):
        """When no rack is present in the GT list, bikes pass through."""
        bike = _box(x=0, y=0, class_id=8, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [bike])
        assert len(gt) == 1

    def test_rack_box_not_present_in_output_gt(self):
        """Rack boxes must be stripped from the returned GT list."""
        rack = self._rack(x=100, y=100)
        car = _box(x=0, y=0, class_id=1, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, car])
        assert all(b.class_id != self.bm.bike_rack_class_id for b in gt)

    # -- integration via compute_sample_metrics --------------------------------

    def test_compute_sample_metrics_bike_inside_rack_excluded(self):
        """Bicycle inside rack must be removed from both pred and GT."""
        rack = self._rack(x=0, y=0, width=4.0, length=4.0)
        pred = [_box(x=0, y=0, class_id=8, confidence_score=0.9)]
        gt = [rack, _box(x=0, y=0, class_id=8, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        for threshold_records in result["match_records"].values():
            assert 8 not in threshold_records

    def test_compute_sample_metrics_bike_outside_rack_kept(self):
        """Bicycle outside rack must still participate in matching."""
        rack = self._rack(x=100, y=100, width=4.0, length=4.0)
        pred = [_box(x=0, y=0, class_id=8, confidence_score=0.9)]
        gt = [rack, _box(x=0, y=0, class_id=8, lidar_pts=5)]
        result = self.bm.compute_sample_metrics(pred, gt)
        found = any(8 in tr for tr in result["match_records"].values())
        assert found

    def test_rack_with_nonzero_yaw_rotates_polygon(self):
        """A bike-rack with yaw != 0 triggers shapely_rotate - exercises rotation path."""
        import math

        rack = self._rack(x=0, y=0, width=4.0, length=4.0, yaw=math.pi / 4)
        bike = _box(x=0, y=0, class_id=8, lidar_pts=5)
        pred, gt = self.bm._prepare_pred_gt_boxes([], [rack, bike])
        # Bike center is at the rack center - must be filtered regardless of rotation.
        assert gt == []
