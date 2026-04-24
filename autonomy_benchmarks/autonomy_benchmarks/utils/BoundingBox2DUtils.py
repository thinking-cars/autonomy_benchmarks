"""Shared geometry utilities for 2D bounding-box benchmarks.

Contains static methods that are dataset- and modality-agnostic:

- **2D IoU** (``compute_iou``): intersection-over-union of two axis-aligned
  bounding boxes.
"""

from __future__ import annotations

from autohub_benchmarks.utils.BoundingBox2D import BoundingBox2D


class BoundingBox2DUtils:
    """Shared static utilities for 2D bounding-box evaluation."""

    @staticmethod
    def compute_iou(box1: BoundingBox2D, box2: BoundingBox2D) -> float:
        """Compute the 2D Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        box1, box2:
            Axis-aligned 2D bounding boxes in ``[x1, y1, x2, y2]`` format
            (top-left and bottom-right corners).

        Returns
        -------
        IoU in ``[0, 1]``; ``0.0`` if the union area is zero.
        """
        b1 = box1.to_list()
        b2 = box2.to_list()
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
