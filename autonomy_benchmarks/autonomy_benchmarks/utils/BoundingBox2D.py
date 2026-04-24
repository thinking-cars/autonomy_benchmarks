"""Defines the BoundingBox2D class for representing 2D bounding boxes in camera object detection tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoundingBox2D:
    """Represents a 2D bounding box with two edge points, class ID, and optional confidence score."""

    x1: float  # corner point 1 x in meter
    y1: float  # corner point 1 y in meter
    x2: float  # corner point 2 x in meter
    y2: float  # corner point 2 y in meter

    image_id: int = 0
    class_id: int = 0

    confidence_score: Optional[float] = None
    distance: Optional[float] = None
    # "easy", "moderate", "hard", "l1", "l2", ...
    group: Optional[str] = None

    def to_list(self) -> List[float]:
        """Convert bounding box coordinates to list format."""

        return [self.x1, self.y1, self.x2, self.y2]
