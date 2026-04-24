"""Shared geometry utilities for 3D bounding-box benchmarks.

Contains static methods that are dataset- and modality-agnostic:

- **BEV polygon helper** (``get_bev_poly``): builds a Shapely rotated-rectangle
  polygon for a box's BEV footprint; used internally by all IoU methods.
- **BEV IoU** (``compute_iou_bev``): IoU of the rotated BEV footprints.
- **True volumetric 3D IoU** (``compute_iou_3d``): correct formula
  ``intersection / (vol_1 + vol_2 - intersection)``.
- **BEV distance** (``compute_dist_bev``): Euclidean distance between two box
  centers on the XY plane.
- **3D distance** (``compute_dist_3d``): Euclidean distance between box centers
  in full 3D space.
- **Uniform range filtering** (``boxes_filter``): keeps boxes within a single
  distance range with configurable boundary inclusion.
- **Per-class range filtering** (``boxes_filter_per_class``): applies
  class-specific distance limits; boxes with unknown class IDs pass through.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import box as shapely_box

from autohub_benchmarks.utils.BoundingBox3D import BoundingBox3D


class BoundingBox3DUtils:
    """Shared static utilities for 3D bounding-box evaluation."""

    @staticmethod
    def get_bev_poly(box: BoundingBox3D):
        """Return a Shapely polygon representing the rotated BEV footprint of *box*.

        Parameters
        ----------
        box:
            3D bounding box with ``x``, ``y``, ``width``, ``length``, and
            ``yaw`` attributes defining its BEV footprint.

        Returns
        -------
        A Shapely ``Polygon`` of the box's rotated rectangular BEV footprint,
        translated to the box's ``(x, y)`` position.
        """
        rect = shapely_box(-box.width / 2, -box.length / 2, box.width / 2, box.length / 2)
        rotated = rotate(rect, -box.yaw, origin=(0, 0), use_radians=True)
        return translate(rotated, box.x, box.y)

    @staticmethod
    def compute_iou_bev(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
        """Compute the IoU of two 3D bounding boxes projected onto the BEV plane.

        Parameters
        ----------
        box1, box2:
            3D bounding boxes with ``x``, ``y``, ``width``, ``length``, and
            ``yaw`` attributes defining their BEV footprint.

        Returns
        -------
        BEV IoU in ``[0, 1]``; ``0.0`` if the union area is zero.
        """
        poly1 = BoundingBox3DUtils.get_bev_poly(box1)
        poly2 = BoundingBox3DUtils.get_bev_poly(box2)
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0

    @staticmethod
    def compute_iou_3d(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
        """Compute the true volumetric 3D IoU of two bounding boxes.

        Uses the correct volumetric formula:

        - 3D intersection = BEV intersection area * height overlap
        - 3D union        = vol_1 + vol_2 - 3D intersection

        Parameters
        ----------
        box1, box2:
            3D bounding boxes with ``x``, ``y``, ``z``, ``width``, ``length``,
            ``height``, and ``yaw`` attributes.

        Returns
        -------
        3D IoU in ``[0, 1]``; ``0.0`` if the 3D union volume is zero.
        """
        poly1 = BoundingBox3DUtils.get_bev_poly(box1)
        poly2 = BoundingBox3DUtils.get_bev_poly(box2)
        bev_inter = poly1.intersection(poly2).area

        min_z1, max_z1 = box1.z - box1.height / 2, box1.z + box1.height / 2
        min_z2, max_z2 = box2.z - box2.height / 2, box2.z + box2.height / 2
        z_inter = max(0.0, min(max_z1, max_z2) - max(min_z1, min_z2))

        vol3d_inter = bev_inter * z_inter
        vol1 = poly1.area * box1.height
        vol2 = poly2.area * box2.height
        vol3d_union = vol1 + vol2 - vol3d_inter
        return vol3d_inter / vol3d_union if vol3d_union > 0 else 0.0

    @staticmethod
    def compute_dist_3d(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
        """Compute the Euclidean distance between two box centers in 3D space.

        Parameters
        ----------
        box1, box2:
            3D bounding boxes whose ``x``, ``y``, and ``z`` attributes define
            their centers in the sensor coordinate frame.

        Returns
        -------
        3D center-point Euclidean distance (same unit as the box coordinates).
        """
        return np.linalg.norm([box1.x - box2.x, box1.y - box2.y, box1.z - box2.z])

    @staticmethod
    def compute_dist_bev(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
        """Compute the Euclidean distance between two box centers in BEV (XY plane).

        Parameters
        ----------
        box1, box2:
            3D bounding boxes whose ``x`` and ``y`` attributes define their
            centers in the sensor coordinate frame.

        Returns
        -------
        BEV center-point Euclidean distance (same unit as the box coordinates).
        """
        dx = box1.x - box2.x
        dy = box1.y - box2.y
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def boxes_filter(
        boxes: List[BoundingBox3D],
        range_start: float,
        range_end: float,
        compute_dist_func: Callable[[BoundingBox3D, BoundingBox3D], float],
        include_start: bool = True,
        include_end: bool = False,
    ) -> List[BoundingBox3D]:
        """Filter bounding boxes by distance to the sensor origin.

        Parameters
        ----------
        boxes:
            Boxes to filter.
        range_start:
            Lower bound of the distance range.
        range_end:
            Upper bound of the distance range.
        compute_dist_func:
            Callable ``(box, origin) -> float`` returning the distance from
            *box* to *origin*.
        include_start:
            If ``True`` (default), the lower bound is inclusive (``>=``).
        include_end:
            If ``True``, the upper bound is inclusive (``<=``). Default ``False``.

        Returns
        -------
        Subset of *boxes* whose distance to the origin falls within
        [*range_start*, *range_end*] (boundary inclusion controlled by
        *include_start* / *include_end*).
        """
        origin = BoundingBox3D(x=0.0, y=0.0, z=0.0, width=0.0, height=0.0, length=0.0, yaw=0.0)
        result: List[BoundingBox3D] = []
        for box in boxes:
            dist = compute_dist_func(box, origin)
            lower_ok = dist >= range_start if include_start else dist > range_start
            upper_ok = dist <= range_end if include_end else dist < range_end
            if lower_ok and upper_ok:
                result.append(box)
        return result

    @staticmethod
    def boxes_filter_per_class(
        boxes: List[BoundingBox3D],
        per_class_ranges: Dict[int, Tuple[float, float]],
        compute_dist_func: Callable[[BoundingBox3D, BoundingBox3D], float],
        include_start: bool = True,
        include_end: bool = False,
    ) -> List[BoundingBox3D]:
        """Filter bounding boxes by per-class distance ranges.

        Boxes whose class ID is not present in *per_class_ranges* are kept
        without filtering (pass-through).

        Parameters
        ----------
        boxes:
            Boxes to filter.
        per_class_ranges:
            Mapping from class ID to ``(range_start, range_end)``.
        compute_dist_func:
            Callable ``(box, origin) -> float`` returning the distance from
            *box* to *origin*.
        include_start:
            If ``True`` (default), the lower bound is inclusive (``>=``).
        include_end:
            If ``True``, the upper bound is inclusive (``<=``). Default ``False``.

        Returns
        -------
        Subset of *boxes* that pass their class-specific range filter, plus
        all boxes whose class ID is absent from *per_class_ranges*.
        """
        origin = BoundingBox3D(x=0.0, y=0.0, z=0.0, width=0.0, height=0.0, length=0.0, yaw=0.0)
        result: List[BoundingBox3D] = []
        for box in boxes:
            if box.class_id not in per_class_ranges:
                result.append(box)
                continue
            range_start, range_end = per_class_ranges[box.class_id]
            dist = compute_dist_func(box, origin)
            lower_ok = dist >= range_start if include_start else dist > range_start
            upper_ok = dist <= range_end if include_end else dist < range_end
            if lower_ok and upper_ok:
                result.append(box)
        return result
