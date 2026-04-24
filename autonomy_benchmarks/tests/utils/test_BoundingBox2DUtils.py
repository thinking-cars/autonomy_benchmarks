"""Tests for BoundingBox2DUtils."""

from __future__ import annotations

import pytest
from autohub_benchmarks.utils.BoundingBox2D import BoundingBox2D
from autohub_benchmarks.utils.BoundingBox2DUtils import BoundingBox2DUtils


def _box(x1, y1, x2, y2, image_id=0, class_id=1, confidence_score=None):
    return BoundingBox2D(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_id=image_id,
        class_id=class_id,
        confidence_score=confidence_score,
    )


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestComputeIou:
    """Tests for 2D intersection-over-union calculations."""

    def test_identical_boxes(self) -> None:
        """Verify identical boxes yield an IoU of one."""
        box = _box(0.0, 0.0, 10.0, 10.0)
        assert BoundingBox2DUtils.compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Verify disjoint boxes yield an IoU of zero."""
        a = _box(0.0, 0.0, 1.0, 1.0)
        b = _box(2.0, 2.0, 3.0, 3.0)
        assert BoundingBox2DUtils.compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Verify partially overlapping boxes yield the expected IoU."""
        a = _box(0.0, 0.0, 2.0, 2.0)
        b = _box(1.0, 1.0, 3.0, 3.0)
        # intersection = 1×1 = 1, union = 4 + 4 - 1 = 7
        assert BoundingBox2DUtils.compute_iou(a, b) == pytest.approx(1.0 / 7.0)
