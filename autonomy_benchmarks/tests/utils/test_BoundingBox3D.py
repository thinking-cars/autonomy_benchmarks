"""Tests for BoundingBox3D helper methods."""

from __future__ import annotations

import pytest
from autonomy_benchmarks.utils.BoundingBox3D import BoundingBox3D

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _box(
    x=1.0,
    y=2.0,
    z=3.0,
    width=4.0,
    height=5.0,
    length=6.0,
    yaw=0.5,
    vx=1.5,
    vy=-0.5,
    image_id=0,
    class_id=1,
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
        image_id=image_id,
        class_id=class_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBoundingBox3D:
    """Tests for BoundingBox3D convenience helpers."""

    # ------------------------------------------------------------------
    # to_list
    # ------------------------------------------------------------------

    def test_to_list_length(self) -> None:
        """Verify the serialized box list always has nine elements."""
        assert len(_box().to_list()) == 9

    def test_to_list_values(self) -> None:
        """Verify to_list preserves the numeric field values."""
        box = _box(x=1.0, y=2.0, z=3.0, width=4.0, height=5.0, length=6.0, yaw=0.5, vx=1.5, vy=-0.5)
        assert box.to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, -0.5])

    def test_to_list_order(self) -> None:
        """Verify to_list returns values in the documented field order."""
        x, y, z, w, h, length, yaw, vx, vy = _box(
            x=1.0, y=2.0, z=3.0, width=4.0, height=5.0, length=6.0, yaw=0.7, vx=0.0, vy=0.0
        ).to_list()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert z == pytest.approx(3.0)
        assert w == pytest.approx(4.0)
        assert h == pytest.approx(5.0)
        assert length == pytest.approx(6.0)
        assert yaw == pytest.approx(0.7)
        assert vx == pytest.approx(0.0)
        assert vy == pytest.approx(0.0)
