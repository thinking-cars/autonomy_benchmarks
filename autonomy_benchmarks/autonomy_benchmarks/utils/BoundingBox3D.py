"""Defines the BoundingBox3D class for representing 3D bounding boxes in camera object detection tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoundingBox3D:
    """Represents a 3D bounding box with center (x, y, z),
    dimensions (width, height, length), orientation (yaw),
    velocity (vx, vy), class ID, attribute, and optional confidence score.
    """

    x: float  # center point x in meter
    y: float  # center point y in meter
    z: float  # center point z in meter
    width: float  # dimensions width in meter
    height: float  # dimensions height in meter
    length: float  # dimensions length in meter
    yaw: float  # orientation angle about z-axis in rad

    vx: float = 0.0  # velocity x in m/s
    vy: float = 0.0  # velocity y in m/s

    image_id: int = 0
    class_id: int = 0

    confidence_score: Optional[float] = None
    attribute: Optional[str] = None
    # "easy", "moderate", "hard", "l1", "l2", ...
    group: Optional[str] = None
    # The number of Lidar points in the bounding box
    number_of_lidar_points: Optional[int] = None
    # The number of radar points in the bounding box
    number_of_radar_points: Optional[int] = None

    def to_list(self) -> List[float]:
        """Convert bounding box coordinates to list format."""

        return [
            self.x,
            self.y,
            self.z,
            self.width,
            self.height,
            self.length,
            self.yaw,
            self.vx,
            self.vy,
        ]
