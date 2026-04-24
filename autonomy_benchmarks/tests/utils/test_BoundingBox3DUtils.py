"""Tests for BoundingBox3DUtils."""

from __future__ import annotations

import pytest
from autohub_benchmarks.utils.BoundingBox3D import BoundingBox3D
from autohub_benchmarks.utils.BoundingBox3DUtils import BoundingBox3DUtils


def _box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=2.0, yaw=0.0, class_id=1):
    return BoundingBox3D(
        x=x,
        y=y,
        z=z,
        width=width,
        height=height,
        length=length,
        yaw=yaw,
        vx=0.0,
        vy=0.0,
        class_id=class_id,
    )


# ---------------------------------------------------------------------------
# get_bev_poly
# ---------------------------------------------------------------------------


class TestGetBevPoly:
    """Tests for BEV polygon generation from 3D boxes."""

    def test_area_matches_box_footprint(self) -> None:
        """Verify polygon area matches the box footprint area."""
        box = _box(width=2.0, length=3.0)
        poly = BoundingBox3DUtils.get_bev_poly(box)
        assert poly.area == pytest.approx(6.0, rel=1e-5)

    def test_center_matches_box_xy(self) -> None:
        """Verify polygon centroid matches the box x and y coordinates."""
        box = _box(x=5.0, y=7.0, width=2.0, length=2.0)
        poly = BoundingBox3DUtils.get_bev_poly(box)
        cx, cy = poly.centroid.x, poly.centroid.y
        assert cx == pytest.approx(5.0, rel=1e-5)
        assert cy == pytest.approx(7.0, rel=1e-5)


# ---------------------------------------------------------------------------
# compute_iou_bev
# ---------------------------------------------------------------------------


class TestComputeIouBev:
    """Tests for bird's-eye-view IoU computation."""

    def test_identical_boxes(self) -> None:
        """Verify identical boxes yield a BEV IoU of one."""
        box = _box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
        assert BoundingBox3DUtils.compute_iou_bev(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Verify disjoint boxes yield a BEV IoU of zero."""
        a = _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = _box(10.0, 10.0, 0.0, 1.0, 1.0, 1.0)
        assert BoundingBox3DUtils.compute_iou_bev(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Verify partially overlapping boxes yield the expected BEV IoU."""
        # Two 2×2 boxes offset by 1 in x → intersection=2, union=6
        a = _box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
        b = _box(1.0, 0.0, 0.0, 2.0, 2.0, 2.0)
        iou = BoundingBox3DUtils.compute_iou_bev(a, b)
        assert iou == pytest.approx(2.0 / 6.0, rel=1e-5)


# ---------------------------------------------------------------------------
# compute_iou_3d
# ---------------------------------------------------------------------------


class TestComputeIou3d:
    """Tests for volumetric 3D IoU computation."""

    def test_identical_boxes(self) -> None:
        """Verify identical boxes yield a 3D IoU of one."""
        box = _box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
        assert BoundingBox3DUtils.compute_iou_3d(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Verify separated boxes yield a 3D IoU of zero."""
        a = _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = _box(10.0, 10.0, 10.0, 1.0, 1.0, 1.0)
        assert BoundingBox3DUtils.compute_iou_3d(a, b) == pytest.approx(0.0)

    def test_partial_overlap_volumetric(self) -> None:
        """Verify the true volumetric 3D IoU formula (not BEV_IoU * Z_IoU).

        Two axis-aligned 2*2*2 boxes offset by 1 m in X and 1 m in Z:
          BEV intersection = 1*2 = 2,  BEV area each = 4,  BEV union = 6
          Z intersection   = 1,        Z span = 3
          3D intersection  = 2 * 1 = 2
          vol1 = vol2 = 4*2 = 8
          3D union = 8 + 8 - 2 = 14
          3D IoU   = 2/14 ≈ 0.1429
        """
        a = _box(x=0.0, y=0.0, z=0.0, width=2.0, height=2.0, length=2.0)
        b = _box(x=1.0, y=0.0, z=1.0, width=2.0, height=2.0, length=2.0)
        assert BoundingBox3DUtils.compute_iou_3d(a, b) == pytest.approx(2.0 / 14.0, rel=1e-5)


# ---------------------------------------------------------------------------
# compute_dist_bev
# ---------------------------------------------------------------------------


class TestComputeDistBev:
    """Tests for bird's-eye-view center distance computation."""

    def test_same_center_is_zero(self) -> None:
        """Verify identical centers have zero BEV distance."""
        box = _box(x=1.0, y=2.0)
        assert BoundingBox3DUtils.compute_dist_bev(box, box) == pytest.approx(0.0)

    def test_unit_step_x(self) -> None:
        """Verify a one-meter x offset yields unit BEV distance."""
        a = _box(x=0.0, y=0.0)
        b = _box(x=1.0, y=0.0)
        assert BoundingBox3DUtils.compute_dist_bev(a, b) == pytest.approx(1.0)

    def test_unit_step_y(self) -> None:
        """Verify a one-meter y offset yields unit BEV distance."""
        a = _box(x=0.0, y=0.0)
        b = _box(x=0.0, y=1.0)
        assert BoundingBox3DUtils.compute_dist_bev(a, b) == pytest.approx(1.0)

    def test_diagonal(self) -> None:
        """Verify BEV distance follows the Pythagorean result."""
        a = _box(x=0.0, y=0.0)
        b = _box(x=3.0, y=4.0)
        assert BoundingBox3DUtils.compute_dist_bev(a, b) == pytest.approx(5.0)

    def test_z_dimension_ignored(self) -> None:
        """Verify BEV distance ignores any z-axis separation."""
        a = _box(x=0.0, y=0.0, z=0.0)
        b = _box(x=0.0, y=0.0, z=100.0)
        assert BoundingBox3DUtils.compute_dist_bev(a, b) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        """Verify BEV distance is symmetric between its operands."""
        a = _box(x=1.0, y=2.0)
        b = _box(x=4.0, y=6.0)
        assert BoundingBox3DUtils.compute_dist_bev(a, b) == pytest.approx(BoundingBox3DUtils.compute_dist_bev(b, a))


# ---------------------------------------------------------------------------
# compute_dist_3d
# ---------------------------------------------------------------------------


class TestComputeDist3d:
    """Tests for full 3D center distance computation."""

    def test_same_center_is_zero(self) -> None:
        """Verify identical centers have zero 3D distance."""
        box = _box(x=1.0, y=2.0, z=3.0)
        assert BoundingBox3DUtils.compute_dist_3d(box, box) == pytest.approx(0.0)

    def test_unit_step_x(self) -> None:
        """Verify a one-meter x offset yields unit 3D distance."""
        a = _box(x=0.0, y=0.0, z=0.0)
        b = _box(x=1.0, y=0.0, z=0.0)
        assert BoundingBox3DUtils.compute_dist_3d(a, b) == pytest.approx(1.0)

    def test_pythagorean(self) -> None:
        """Verify 3D distance follows the Pythagorean result."""
        a = _box(x=0.0, y=0.0, z=0.0)
        b = _box(x=1.0, y=2.0, z=2.0)
        assert BoundingBox3DUtils.compute_dist_3d(a, b) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# boxes_filter
# ---------------------------------------------------------------------------


class TestBoxesFilter:
    """Tests for uniform distance-range filtering."""

    def _dist(self, box1, box2):
        return BoundingBox3DUtils.compute_dist_bev(box1, box2)

    def test_box_within_range_kept(self) -> None:
        """Verify boxes within range are retained."""
        box = _box(x=5.0, y=0.0)
        result = BoundingBox3DUtils.boxes_filter([box], 0.0, 10.0, self._dist)
        assert len(result) == 1

    def test_box_beyond_range_excluded(self) -> None:
        """Verify boxes beyond the upper bound are removed."""
        box = _box(x=15.0, y=0.0)
        result = BoundingBox3DUtils.boxes_filter([box], 0.0, 10.0, self._dist)
        assert result == []

    def test_upper_bound_exclusive(self) -> None:
        """Verify the upper bound can be treated as exclusive."""
        box = _box(x=10.0, y=0.0)
        result = BoundingBox3DUtils.boxes_filter([box], 0.0, 10.0, self._dist, include_end=False)
        assert result == []

    def test_upper_bound_inclusive(self) -> None:
        """Verify the upper bound can be treated as inclusive."""
        box = _box(x=10.0, y=0.0)
        result = BoundingBox3DUtils.boxes_filter([box], 0.0, 10.0, self._dist, include_end=True)
        assert len(result) == 1

    def test_lower_bound_inclusive(self) -> None:
        """Verify the lower bound can be treated as inclusive."""
        box = _box(x=5.0, y=0.0)
        result = BoundingBox3DUtils.boxes_filter([box], 5.0, 10.0, self._dist, include_start=True)
        assert len(result) == 1

    def test_lower_bound_exclusive(self) -> None:
        """Verify the lower bound can be treated as exclusive."""
        box = _box(x=5.0, y=0.0)
        result = BoundingBox3DUtils.boxes_filter([box], 5.0, 10.0, self._dist, include_start=False)
        assert result == []

    def test_empty_input(self) -> None:
        """Verify empty input lists stay empty after filtering."""
        assert BoundingBox3DUtils.boxes_filter([], 0.0, 50.0, self._dist) == []

    def test_multiple_boxes_mixed(self) -> None:
        """Verify filtering keeps only boxes inside the requested range."""
        boxes = [_box(x=5.0), _box(x=15.0), _box(x=25.0)]
        result = BoundingBox3DUtils.boxes_filter(boxes, 0.0, 20.0, self._dist, include_end=True)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# boxes_filter_per_class
# ---------------------------------------------------------------------------


class TestBoxesFilterPerClass:
    """Tests for class-specific distance-range filtering."""

    def _dist(self, box1, box2):
        return BoundingBox3DUtils.compute_dist_bev(box1, box2)

    def _ranges(self):
        return {
            1: (0.0, 50.0),
            8: (0.0, 40.0),
            9: (0.0, 30.0),
        }

    def test_box_within_class_range_kept(self) -> None:
        """Verify boxes inside their class range are retained."""
        box = _box(x=25.0, y=0.0, class_id=1)
        result = BoundingBox3DUtils.boxes_filter_per_class([box], self._ranges(), self._dist, include_end=True)
        assert len(result) == 1

    def test_box_beyond_class_range_excluded(self) -> None:
        """Verify boxes beyond their class range are removed."""
        box = _box(x=45.0, y=0.0, class_id=8)
        result = BoundingBox3DUtils.boxes_filter_per_class([box], self._ranges(), self._dist, include_end=True)
        assert result == []

    def test_boundary_inclusive(self) -> None:
        """Verify class ranges can include the upper boundary."""
        box = _box(x=30.0, y=0.0, class_id=9)
        result = BoundingBox3DUtils.boxes_filter_per_class([box], self._ranges(), self._dist, include_end=True)
        assert len(result) == 1

    def test_boundary_exclusive(self) -> None:
        """Verify class ranges can exclude the upper boundary."""
        box = _box(x=30.0, y=0.0, class_id=9)
        result = BoundingBox3DUtils.boxes_filter_per_class([box], self._ranges(), self._dist, include_end=False)
        assert result == []

    def test_unknown_class_passes_through(self) -> None:
        """Verify boxes with unknown classes pass through unchanged."""
        box = _box(x=999.0, y=0.0, class_id=99)
        result = BoundingBox3DUtils.boxes_filter_per_class([box], self._ranges(), self._dist)
        assert len(result) == 1

    def test_empty_input(self) -> None:
        """Verify empty input lists stay empty after per-class filtering."""
        assert BoundingBox3DUtils.boxes_filter_per_class([], self._ranges(), self._dist) == []

    def test_empty_ranges_all_pass_through(self) -> None:
        """Verify empty class-range maps allow every box through."""
        boxes = [_box(x=100.0, class_id=1), _box(x=200.0, class_id=8)]
        result = BoundingBox3DUtils.boxes_filter_per_class(boxes, {}, self._dist)
        assert len(result) == 2

    def test_multiple_classes_filtered_independently(self) -> None:
        """Verify each class uses its own distance threshold."""
        boxes = [
            _box(x=25.0, class_id=1),  # kept  (≤50 m)
            _box(x=45.0, class_id=8),  # excluded (>40 m)
            _box(x=28.0, class_id=9),  # kept  (≤30 m)
            _box(x=35.0, class_id=9),  # excluded (>30 m)
        ]
        result = BoundingBox3DUtils.boxes_filter_per_class(boxes, self._ranges(), self._dist, include_end=True)
        assert len(result) == 2
