# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""nuScenes - Lidar object detection benchmark.

Evaluates 3D bounding-box predictions from lidar point cloud input against
nuScenes ground-truth labels using the official nuScenes evaluation protocol.

Objects are represented as plain ``dict`` records extracted directly from the
``perception_msgs/ObjectList`` message (no intermediate bounding-box type).
Each record carries the fields the metrics consume: ``x``, ``y``, ``width``,
``length``, ``height``, ``yaw``, ``vx``, ``vy``, ``class_name``,
``confidence_score``, ``attribute``, ``number_of_lidar_points``, and
``number_of_radar_points``.

The dataset annotations that ``perception_msgs/Object`` cannot express
(``original_class``, ``attribute``, ``num_lidar_pts``, ``num_radar_pts``) are
published by ``autonomy_datasets`` on a separate
``<object list topic>/meta_info`` topic as an
``autonomy_datasets_msgs/ObjectListMetaInfo``. It is a synchronized input of its
own (``label_meta_info``) and is correlated with the object list via the header
stamp and the object ID.

Key evaluation settings:

- Matching:   BEV center-point Euclidean distance ≤ threshold.
- Thresholds: 0.5, 1.0, 2.0, 4.0 m (applied uniformly to all classes).
- Ranges:     Per-class max detection distance (barrier/cone ≤ 30 m,
              bicycle/motorcycle/pedestrian ≤ 40 m, others ≤ 50 m).
- GT filter:  GT boxes with 0 lidar and 0 radar points are excluded.
- Bike-rack:  Bicycle and motorcycle boxes whose BEV center falls inside a
              bike-rack annotation are excluded from both predictions and GT
              before matching.
- AP filter:  P-R points with cum_precision <= 0.1 or cum_recall <= 0.1
              are excluded from the 101-point recall interpolation.
- TP metrics: ATE, ASE, AOE, AVE, AAE measured at the 2.0 m threshold.
              AOE for traffic_cone is ignored (set to 0); for barrier
              it is capped at π (180°). AVE and AAE for barrier/traffic_cone
              are ignored (set to 0 / not evaluated). AAE requires the
              ``attribute`` field to be populated by the dataset loader;
              when attribute data is absent, AAE is skipped per TP entry.
- NDS:        nuScenes Detection Score combining mAP, mATE, mASE, mAOE,
              mAVE, mAAE with weights 5-1-1-1-1-1, normalised by 10.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_benchmarks.benchmarks.AutonomyBenchmark import AutonomyBenchmark
from autonomy_benchmarks.utils.ObjectDetectionUtils import ObjectDetectionUtils
from autonomy_datasets_msgs.msg import ObjectListMetaInfo
from perception_msgs.msg import ObjectClassification, ObjectList
from perception_msgs_utils.state_getters import (
    get_height,
    get_length,
    get_vel_lat,
    get_vel_lon,
    get_width,
    get_x,
    get_y,
    get_yaw,
)
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import box as shapely_box
from shapely.geometry import Point

_CLASS_IGNORE = "__ignore__"


def _annotation(annotations: Dict[str, List[str]], key: str) -> Optional[str]:
    """Return the first value annotated under ``key``, or ``None`` if absent.

    Args:
        annotations: One object's key-to-values meta information.
        key: Annotation key to look up.

    Returns:
        The first value published for ``key``, else ``None``.
    """

    values = annotations.get(key)
    return values[0] if values else None


def _annotation_int(annotations: Dict[str, List[str]], key: str) -> Optional[int]:
    """Return the first value annotated under ``key`` as ``int``, or ``None`` if absent.

    Args:
        annotations: One object's key-to-values meta information.
        key: Annotation key to look up.

    Returns:
        The first value published for ``key`` parsed as ``int``, else ``None``.

    Raises:
        ValueError: The annotated value is not a valid integer.
    """

    value = _annotation(annotations, key)
    return None if value is None else int(value)


class DetectionClass:
    """NuScenes detection class names (the values of the class maps).

    ``BIKE_RACK`` is not scored but is kept to filter bicycle/motorcycle
    detections; every other member is an evaluated nuScenes detection class.
    """

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    TRAILER = "trailer"
    CONSTRUCTION_VEHICLE = "construction_vehicle"
    PEDESTRIAN = "pedestrian"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    TRAFFIC_CONE = "traffic_cone"
    BARRIER = "barrier"
    BIKE_RACK = "bike_rack"


_DETECTION_CLASSES: frozenset = frozenset(
    {
        DetectionClass.CAR,
        DetectionClass.TRUCK,
        DetectionClass.BUS,
        DetectionClass.TRAILER,
        DetectionClass.CONSTRUCTION_VEHICLE,
        DetectionClass.PEDESTRIAN,
        DetectionClass.MOTORCYCLE,
        DetectionClass.BICYCLE,
        DetectionClass.TRAFFIC_CONE,
        DetectionClass.BARRIER,
        DetectionClass.BIKE_RACK,
    }
)

# nuScenes category -> detection class (official ``general_to_detection``).
_NUSCENES_CATEGORY_TO_CLASS: Dict[str, str] = {
    "vehicle.car": DetectionClass.CAR,
    "vehicle.truck": DetectionClass.TRUCK,
    "vehicle.bus.bendy": DetectionClass.BUS,
    "vehicle.bus.rigid": DetectionClass.BUS,
    "vehicle.trailer": DetectionClass.TRAILER,
    "vehicle.construction": DetectionClass.CONSTRUCTION_VEHICLE,
    "human.pedestrian.adult": DetectionClass.PEDESTRIAN,
    "human.pedestrian.child": DetectionClass.PEDESTRIAN,
    "human.pedestrian.construction_worker": DetectionClass.PEDESTRIAN,
    "human.pedestrian.police_officer": DetectionClass.PEDESTRIAN,
    "vehicle.motorcycle": DetectionClass.MOTORCYCLE,
    "vehicle.bicycle": DetectionClass.BICYCLE,
    "movable_object.trafficcone": DetectionClass.TRAFFIC_CONE,
    "movable_object.barrier": DetectionClass.BARRIER,
    "static_object.bicycle_rack": DetectionClass.BIKE_RACK,
    "animal": _CLASS_IGNORE,
    "human.pedestrian.personal_mobility": _CLASS_IGNORE,
    "human.pedestrian.stroller": _CLASS_IGNORE,
    "human.pedestrian.wheelchair": _CLASS_IGNORE,
    "movable_object.debris": _CLASS_IGNORE,
    "movable_object.pushable_pullable": _CLASS_IGNORE,
    "vehicle.emergency.ambulance": _CLASS_IGNORE,
    "vehicle.emergency.police": _CLASS_IGNORE,
}

# Perception ObjectClassification enum -> detection class.
_PERCEPTION_TYPE_TO_CLASS: Dict[int, str] = {
    ObjectClassification.CAR: DetectionClass.CAR,
    ObjectClassification.BUS: DetectionClass.BUS,
    ObjectClassification.PEDESTRIAN: DetectionClass.PEDESTRIAN,
    ObjectClassification.BICYCLE: DetectionClass.BICYCLE,
    ObjectClassification.MOTORCYCLE: DetectionClass.MOTORCYCLE,
    ObjectClassification.UTILITY: _CLASS_IGNORE,
    ObjectClassification.ANIMAL: _CLASS_IGNORE,
    ObjectClassification.VRU: _CLASS_IGNORE,
    ObjectClassification.MICRO: _CLASS_IGNORE,
    ObjectClassification.UNCLASSIFIED: _CLASS_IGNORE,
    ObjectClassification.UNKNOWN: _CLASS_IGNORE,
}

# Detection classes expressible via the perception enum (image of the map, minus ignore); others are excluded from mAP/NDS.
_SUPPORTED_CLASSES: frozenset = frozenset(set(_PERCEPTION_TYPE_TO_CLASS.values()) - {_CLASS_IGNORE}) & _DETECTION_CLASSES


class NuscenesLidarObjectDetection(AutonomyBenchmark):
    """Benchmark for lidar 3D object detection on the nuScenes dataset.

    See the module docstring for full evaluation settings. Objects are keyed by
    their canonical nuScenes detection class name. The classes evaluated with a
    uniform BEV distance threshold (0.5 / 1.0 / 2.0 / 4.0 m) are: ``car``,
    ``truck``, ``bus``, ``trailer``, ``construction_vehicle``, ``pedestrian``,
    ``motorcycle``, ``bicycle``, ``traffic_cone``, and ``barrier``. The
    ``bike_rack`` class is not scored but is used to filter bicycle/motorcycle
    detections.
    """

    def __init__(self) -> None:
        """Configure nuScenes thresholds, class ranges, and TP metric rules."""
        super().__init__(
            name="nuscenes_lidar_object_detection",
            description=("3D bounding-box object detection benchmark from lidar point clouds using the nuScenes dataset."),
        )

        # BEV distance thresholds used for AP and TP-metric computation.
        self.matching_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
        # Distance threshold used exclusively for TP error metrics.
        self.tp_metric_threshold: float = 2.0
        # Minimum precision required for a P-R point to enter AP integration.
        self.min_precision: float = 0.1
        # Minimum recall required for a P-R point to enter AP integration.
        self.min_recall: float = 0.1
        # Per-class maximum detection distance (BEV). Pred and GT boxes beyond
        # the class-specific limit are excluded before matching.
        self.per_class_detection_ranges: Dict[str, Tuple[float, float]] = {
            "car": (0.0, 50.0),
            "truck": (0.0, 50.0),
            "bus": (0.0, 50.0),
            "trailer": (0.0, 50.0),
            "construction_vehicle": (0.0, 50.0),
            "pedestrian": (0.0, 40.0),
            "motorcycle": (0.0, 40.0),
            "bicycle": (0.0, 40.0),
            "traffic_cone": (0.0, 30.0),
            "barrier": (0.0, 30.0),
        }
        # Classes for which orientation error is ignored (set to 0).
        self.aoe_ignored_classes: set = {"traffic_cone"}
        # Classes for which orientation error is capped at π (180° symmetry).
        self.aoe_pi_classes: set = {"barrier"}
        # Classes for which velocity error is ignored (set to 0).
        self.ave_ignored_classes: set = {"traffic_cone", "barrier"}
        # Classes for which attribute error is ignored (void attribute in spec).
        self.aae_ignored_classes: set = {"traffic_cone", "barrier"}
        # Classes filtered out when their BEV center falls inside a bike-rack.
        self.bike_rack_filtered_classes: frozenset = frozenset({"motorcycle", "bicycle"})
        # Class name of static bike-rack annotations in the GT object list.
        self.bike_rack_class_name: str = "bike_rack"
        self.compute_dist_func = ObjectDetectionUtils.compute_dist_bev
        self.compute_ap_func = ObjectDetectionUtils.compute_ap_101_point
        # Detection classes the model/interface can express. Only these enter the
        # mAP and NDS aggregates; per-class AP is still reported for all classes.
        self.supported_classes: frozenset = _SUPPORTED_CLASSES

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def required_inputs(self) -> Dict[str, Any]:
        """Define expected input ROS message types.

        ``label_meta_info`` carries the dataset annotations of the ``label``
        object list (``original_class``, ``attribute``, point counts).

        Returns:
            Input name to ROS message type. The keys match the
            ``compute_sample_metrics`` parameters and are remapped to real ROS
            topics in the launch file.
        """

        return {
            "prediction": ObjectList,
            "label": ObjectList,
            "label_meta_info": ObjectListMetaInfo,
        }

    @staticmethod
    def _index_meta_info(meta_info: Any) -> Dict[int, Dict[str, List[str]]]:
        """Index an ``ObjectListMetaInfo`` message by object ID.

        Args:
            meta_info: An ``autonomy_datasets_msgs/ObjectListMetaInfo``, or
                ``None`` when the frame carries no meta information.

        Returns:
            Object ID to that object's key-to-values annotations. A key may be
            published more than once per object (e.g. several attributes), so
            values are lists in publication order. Objects without any meta
            information are absent from the index.
        """

        index: Dict[int, Dict[str, List[str]]] = {}
        if meta_info is None:
            return index
        for entry in meta_info.objects:
            annotations = index.setdefault(entry.id, {})
            for key_value in entry.info:
                annotations.setdefault(key_value.key, []).append(key_value.value)
        return index

    def _extract_objects(self, objects: Any, meta_info: Any = None, is_label: bool = False) -> List[Dict[str, Any]]:
        """Extract per-object metric records from a frame's ``ObjectList``.

        The ``class_name`` source depends on the role: ground-truth labels use
        the dataset ``original_class`` annotation normalised via
        :data:`_NUSCENES_CATEGORY_TO_CLASS`; predictions use the perception
        classification enum via :data:`_PERCEPTION_TYPE_TO_CLASS`. Non-evaluated
        classes are dropped. Position, dimensions, yaw and velocity are read via
        each object's own motion model (selected by ``state.model_id``);
        ``confidence_score`` is the object's ``existence_probability``.
        ``attribute`` and the ``num_lidar_pts`` / ``num_radar_pts`` counts come
        from the object's meta information (all optional).

        Args:
            objects: A ``perception_msgs/ObjectList``.
            meta_info: The matching ``autonomy_datasets_msgs/ObjectListMetaInfo``
                published on ``<object list topic>/meta_info``, or ``None`` when
                the object list has no meta information topic (e.g. model
                predictions). Its entries are matched to objects by ID.
            is_label: ``True`` for ground-truth labels (class from
                ``original_class``), ``False`` for predictions (class from the
                perception enum).

        Returns:
            Object records (dicts); each ``class_name`` is the canonical class name.

        Raises:
            ValueError: a label with a missing or unrecognised ``original_class``.
                Predictions whose perception type has no mapping are dropped, not
                raised.
            UnknownStateEntryError: an object whose ``state.model_id`` is not a
                known motion model (raised by the perception getters).
        """

        meta_index = self._index_meta_info(meta_info)

        result: List[Dict[str, Any]] = []
        for obj in objects.objects:
            annotations = meta_index.get(obj.id, {})
            yaw = get_yaw(obj)
            vel_lon = get_vel_lon(obj)
            vel_lat = get_vel_lat(obj)
            vx = vel_lon * math.cos(yaw) - vel_lat * math.sin(yaw)
            vy = vel_lon * math.sin(yaw) + vel_lat * math.cos(yaw)

            # Labels carry their class as the dataset ``original_class``;
            # predictions carry it in the perception classification enum.
            if is_label:
                raw = _annotation(annotations, "original_class")
                if raw is None:
                    raise ValueError(f"'original_class' not found in meta info of object {obj.id}")
                # Categories fold to a class; detection names pass through.
                class_name: Optional[str] = _NUSCENES_CATEGORY_TO_CLASS.get(raw, raw)
                # Unknown categories pass through the map unchanged; reject them.
                if class_name != _CLASS_IGNORE and class_name not in _DETECTION_CLASSES:
                    raise ValueError(f"Unrecognised original_class '{raw}'")
            else:
                # The map yields only a valid class or _CLASS_IGNORE.
                best = max(obj.state.classifications, key=lambda c: c.probability)
                class_name = _PERCEPTION_TYPE_TO_CLASS.get(best.type, _CLASS_IGNORE)
            if class_name == _CLASS_IGNORE:
                continue  # not an evaluated class — drop

            # Attribute and point counts from the dataset meta information.
            attribute: Optional[str] = _annotation(annotations, "attribute")
            num_lidar_points: Optional[int] = _annotation_int(annotations, "num_lidar_pts")
            num_radar_points: Optional[int] = _annotation_int(annotations, "num_radar_pts")

            result.append(
                {
                    "x": get_x(obj),
                    "y": get_y(obj),
                    "width": get_width(obj),
                    "length": get_length(obj),
                    "height": get_height(obj),
                    "yaw": yaw,
                    "vx": vx,
                    "vy": vy,
                    "class_name": class_name,
                    "confidence_score": float(obj.existence_probability),
                    "attribute": attribute,
                    "number_of_lidar_points": num_lidar_points,
                    "number_of_radar_points": num_radar_points,
                }
            )
        return result

    def compute_sample_metrics(
        self,
        prediction: Any,
        label: Any,
        sample_id: Optional[str] = None,
        label_meta_info: Any = None,
    ) -> Dict[str, Any]:
        """Pass 1 - match predictions to ground truth for one frame.

        Extracts objects, applies the pre-matching filters
        (:meth:`_prepare_pred_gt_boxes`), then greedily matches by BEV distance
        (:meth:`_run_matching`).

        Args:
            prediction: Predicted objects (``perception_msgs/ObjectList``).
            label: Ground-truth objects (``perception_msgs/ObjectList``).
            sample_id: Optional frame identifier (unused).
            label_meta_info: The dataset annotations of ``label``
                (``autonomy_datasets_msgs/ObjectListMetaInfo``), received on
                ``<label topic>/meta_info`` in the same synchronized callback.

        Returns:
            ``{"sample_prediction_num", "sample_ground_truth_num",
            "match_records"}``, with pre-filter object counts and
            ``match_records`` = ``threshold → class → {pred_entries, gt_count}``.
        """
        prediction = self._extract_objects(prediction, is_label=False)
        label = self._extract_objects(label, label_meta_info, is_label=True)

        sample_prediction_num = len(prediction)
        sample_ground_truth_num = len(label)

        pred_boxes, gt_boxes = self._prepare_pred_gt_boxes(prediction, label)
        match_records = self._run_matching(pred_boxes, gt_boxes)

        return {
            "sample_prediction_num": sample_prediction_num,
            "sample_ground_truth_num": sample_ground_truth_num,
            "match_records": match_records,
        }

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 2 - aggregate per-frame match records into dataset-level metrics.

        Args:
            sample_results: List of ``{"sample_id": ..., "metrics": {...,
                "match_records": ...}}`` entries produced by
                :meth:`record_sample`.

        Returns:
            A dict containing ``threshold_metrics`` (per-threshold AP / mAP with
            the P-R filter), ``tp_metrics`` (per-class ATE, ASE, AOE, AVE, AAE
            at the 2 m threshold), and ``benchmark_score`` (flat mAP and NDS
            summary).
        """
        all_match_records = [entry["metrics"]["match_records"] for entry in sample_results]
        merged_records = self._merge_match_records(all_match_records)

        # Compute cumulative P-R statistics for AP.
        cumulative_results = self._compute_global_cumulative_stats(merged_records)
        # Compute per-threshold AP with nuScenes min-precision/recall gating.
        threshold_metrics = self._compute_threshold_metrics(cumulative_results)
        # Compute TP error metrics at the 2 m threshold.
        tp_metrics = self._compute_tp_metrics(merged_records)
        # Compute benchmark score (mAP and NDS).
        benchmark_score = self._compute_benchmark_score(threshold_metrics, tp_metrics)

        return {
            "threshold_metrics": threshold_metrics,
            "tp_metrics": tp_metrics,
            "benchmark_score": benchmark_score,
        }

    # ------------------------------------------------------------------
    # Helper functions for metric computation
    # ------------------------------------------------------------------

    def _boxes_filter_per_class(
        self,
        boxes: List[Dict[str, Any]],
        per_class_ranges: Dict[str, Tuple[float, float]],
        include_start: bool = True,
        include_end: bool = False,
    ) -> List[Dict[str, Any]]:
        """Filter object records by per-class BEV distance to the sensor origin.

        Records whose class is absent from ``per_class_ranges`` are kept
        without filtering (pass-through).

        Args:
            boxes: Object records to filter.
            per_class_ranges: Class name to ``(min, max)`` BEV distance range.
            include_start: Whether the lower bound is inclusive.
            include_end: Whether the upper bound is inclusive.

        Returns:
            The records whose BEV distance lies within their class range.
        """
        origin = {"x": 0.0, "y": 0.0}
        result: List[Dict[str, Any]] = []
        for box in boxes:
            class_name = box["class_name"]
            if class_name not in per_class_ranges:
                result.append(box)
                continue
            range_start, range_end = per_class_ranges[class_name]
            dist = self.compute_dist_func(box, origin)
            lower_ok = dist >= range_start if include_start else dist > range_start
            upper_ok = dist <= range_end if include_end else dist < range_end
            if lower_ok and upper_ok:
                result.append(box)
        return result

    def _prepare_pred_gt_boxes(
        self,
        pred_boxes: List[Dict[str, Any]],
        gt_boxes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply the three nuScenes pre-matching filters, in order.

        1. **GT point filter**: drop GT boxes with zero lidar *and* zero radar
           points.
        2. **Bike-rack filter**: drop bicycle/motorcycle boxes (pred and GT)
           whose BEV center falls inside a bike-rack; the rack boxes (GT
           ``class_name == bike_rack``) are removed from GT afterwards.
        3. **Per-class range filter**: drop boxes beyond their class range
           (:attr:`per_class_detection_ranges`).

        Args:
            pred_boxes: Prediction records for the frame.
            gt_boxes: Ground-truth records for the frame (incl. bike-racks).

        Returns:
            Filtered ``(pred_boxes, gt_boxes)`` ready for matching.
        """

        def _build_bev_polygons(rack_boxes: List[Dict[str, Any]]) -> list:
            """Build a BEV rotated-rectangle polygon for each bike-rack box.

            Args:
                rack_boxes: Bike-rack object records.

            Returns:
                One Shapely polygon per rack box, in BEV.
            """
            polygons = []
            for box in rack_boxes:
                half_w = (box["width"] or 0.0) / 2.0
                half_l = (box["length"] or 0.0) / 2.0
                rect = shapely_box(
                    box["x"] - half_w,
                    box["y"] - half_l,
                    box["x"] + half_w,
                    box["y"] + half_l,
                )
                if box["yaw"]:
                    rect = shapely_rotate(rect, box["yaw"], origin=(box["x"], box["y"]), use_radians=True)
                polygons.append(rect)
            return polygons

        def _filter_bike_rack_overlaps(
            boxes: List[Dict[str, Any]],
            rack_polygons: list,
            bike_rack_filtered_classes: frozenset,
        ) -> List[Dict[str, Any]]:
            """Drop bicycle/motorcycle boxes whose BEV center falls inside a rack.

            Args:
                boxes: Object records to filter.
                rack_polygons: BEV bike-rack polygons.
                bike_rack_filtered_classes: Classes subject to rack filtering.

            Returns:
                The records that survive rack filtering.
            """
            filtered: List[Dict[str, Any]] = []
            for box in boxes:
                if box["class_name"] not in bike_rack_filtered_classes:
                    filtered.append(box)
                    continue
                center = Point(box["x"], box["y"])
                if any(poly.contains(center) for poly in rack_polygons):
                    continue  # drop — inside a bike-rack
                filtered.append(box)
            return filtered

        # Step 1 – exclude GT boxes with no sensor evidence.
        gt_boxes = [
            gt
            for gt in gt_boxes
            if not (
                (gt["number_of_lidar_points"] is not None and gt["number_of_lidar_points"] < 1)
                and (gt["number_of_radar_points"] is not None and gt["number_of_radar_points"] < 1)
            )
        ]
        # Step 2 – extract bike-rack boxes from GT and exclude bicycles/motorcycles
        # inside bike-rack regions.  Bike-rack boxes are never evaluated, so they
        # are removed from both GT and predictions before matching.
        bike_rack_boxes = [gt for gt in gt_boxes if gt["class_name"] == self.bike_rack_class_name]
        gt_boxes = [gt for gt in gt_boxes if gt["class_name"] != self.bike_rack_class_name]
        pred_boxes = [pred for pred in pred_boxes if pred["class_name"] != self.bike_rack_class_name]
        if bike_rack_boxes:
            rack_polygons = _build_bev_polygons(bike_rack_boxes)
            pred_boxes = _filter_bike_rack_overlaps(pred_boxes, rack_polygons, self.bike_rack_filtered_classes)
            gt_boxes = _filter_bike_rack_overlaps(gt_boxes, rack_polygons, self.bike_rack_filtered_classes)
        # Step 3 – apply per-class detection range.
        pred_boxes = self._boxes_filter_per_class(pred_boxes, self.per_class_detection_ranges, include_end=True)
        gt_boxes = self._boxes_filter_per_class(gt_boxes, self.per_class_detection_ranges, include_end=True)
        return pred_boxes, gt_boxes

    def _run_matching(
        self,
        pred_boxes: List[Dict[str, Any]],
        gt_boxes: List[Dict[str, Any]],
    ) -> Dict[float, Dict[str, Dict[str, Any]]]:
        """Greedy per-class BEV-distance matching across all thresholds.

        Within each class, predictions (highest confidence first) each claim
        the nearest unmatched GT within the threshold. TP error metrics use the
        class-specific rules (see :meth:`_compute_tp_metrics`).

        Args:
            pred_boxes: Prediction records for the frame (already filtered).
            gt_boxes: Ground-truth records for the frame (already filtered).

        Returns:
            ``threshold → class → {"pred_entries", "gt_count"}``. Each
            ``pred_entries`` item holds ``confidence_score``, ``dist`` (``inf``
            for FPs), ``is_tp``, the error metrics ``ate``/``ase``/``aoe``/``ave``
            (0.0 for FPs), and ``aae`` (``0.0``/``1.0`` for TPs with a GT
            attribute, else ``None``).
        """
        # Index boxes by class.
        gt_by_class: Dict[str, List[Dict[str, Any]]] = {}
        for gt in gt_boxes:
            gt_by_class.setdefault(gt["class_name"], []).append(gt)
        pred_by_class: Dict[str, List[Dict[str, Any]]] = {}
        for pred in pred_boxes:
            pred_by_class.setdefault(pred["class_name"], []).append(pred)

        all_class_names = set(gt_by_class) | set(pred_by_class)

        match_records: Dict[float, Dict[str, Dict[str, Any]]] = {}
        for threshold in self.matching_thresholds:
            match_records[threshold] = {}
            for class_name in all_class_names:
                gts = gt_by_class.get(class_name, [])
                preds = sorted(
                    pred_by_class.get(class_name, []),
                    key=lambda p: -(p["confidence_score"] or 0.0),
                )
                gt_count = len(gts)
                matched_gt_indices: set = set()
                pred_entries: List[Dict[str, Any]] = []

                for pred in preds:
                    best_dist = float("inf")
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(gts):
                        if gt_idx in matched_gt_indices:
                            continue
                        dist = self.compute_dist_func(pred, gt)
                        if dist <= threshold and dist < best_dist:
                            best_dist = dist
                            best_gt_idx = gt_idx

                    if best_gt_idx != -1:
                        matched_gt = gts[best_gt_idx]
                        matched_gt_indices.add(best_gt_idx)
                        # ATE: BEV Euclidean translation error.
                        ate = math.sqrt((pred["x"] - matched_gt["x"]) ** 2 + (pred["y"] - matched_gt["y"]) ** 2)
                        # ASE: 1 - 3D IoU after aligning centers and orientation.
                        inter = (
                            min(pred["width"], matched_gt["width"])
                            * min(pred["height"], matched_gt["height"])
                            * min(pred["length"], matched_gt["length"])
                        )
                        vol_pred = pred["width"] * pred["height"] * pred["length"]
                        vol_gt = matched_gt["width"] * matched_gt["height"] * matched_gt["length"]
                        union = vol_pred + vol_gt - inter
                        ase = 1.0 - (inter / union if union > 0 else 0.0)
                        # AOE: smallest yaw angle difference (class-specific rules).
                        heading_diff = abs(pred["yaw"] - matched_gt["yaw"])
                        if class_name in self.aoe_ignored_classes:
                            aoe = 0.0  # traffic_cone: orientation error ignored.
                        elif class_name in self.aoe_pi_classes:
                            # barrier: π-symmetric — wrap into [0, π) then take symmetric minimum.
                            heading_diff_mod = heading_diff % math.pi
                            aoe = min(heading_diff_mod, math.pi - heading_diff_mod)
                        else:
                            aoe = min(heading_diff, 2 * math.pi - heading_diff)
                        # AVE: velocity error in m/s (ignored for barrier and traffic_cone).
                        if class_name in self.ave_ignored_classes:
                            ave = 0.0
                        else:
                            ave = math.sqrt((pred["vx"] - matched_gt["vx"]) ** 2 + (pred["vy"] - matched_gt["vy"]) ** 2)
                        # AAE: 1 - attribute accuracy (ignored for barrier and traffic_cone).
                        # None when GT attribute is missing — excluded from mean in that case.
                        if class_name in self.aae_ignored_classes:
                            aae: Optional[float] = 0.0  # attribute not evaluated for this class
                        elif matched_gt["attribute"] is not None:
                            aae = 0.0 if pred["attribute"] == matched_gt["attribute"] else 1.0
                        else:
                            aae = None  # GT attribute missing — skip this TP for AAE
                        pred_entries.append(
                            {
                                "confidence_score": pred["confidence_score"] or 0.0,
                                "dist": best_dist,
                                "is_tp": True,
                                "ate": ate,
                                "ase": ase,
                                "aoe": aoe,
                                "ave": ave,
                                "aae": aae,
                            }
                        )
                    else:
                        pred_entries.append(
                            {
                                "confidence_score": pred["confidence_score"] or 0.0,
                                "dist": float("inf"),
                                "is_tp": False,
                                "ate": 0.0,
                                "ase": 0.0,
                                "aoe": 0.0,
                                "ave": 0.0,
                                "aae": None,
                            }
                        )

                match_records[threshold][class_name] = {
                    "pred_entries": pred_entries,
                    "gt_count": gt_count,
                }

        return match_records

    def _merge_match_records(
        self, all_match_records: List[Dict[float, Dict[str, Dict[str, Any]]]]
    ) -> Dict[float, Dict[str, Dict[str, Any]]]:
        """Concatenate per-frame match records into one dataset-level structure.

        Concatenates ``pred_entries`` and sums ``gt_count`` across all frames,
        grouped by threshold / class.

        Args:
            all_match_records: Per-frame ``match_records`` from
                :meth:`_run_matching`.

        Returns:
            The merged ``threshold → class → {"pred_entries", "gt_count"}``.
        """
        merged: Dict[float, Dict[str, Dict[str, Any]]] = {}
        for frame_records in all_match_records:
            for threshold, class_records in frame_records.items():
                merged.setdefault(threshold, {})
                for class_name, data in class_records.items():
                    if class_name not in merged[threshold]:
                        merged[threshold][class_name] = {"pred_entries": [], "gt_count": 0}
                    merged[threshold][class_name]["pred_entries"].extend(data["pred_entries"])
                    merged[threshold][class_name]["gt_count"] += data["gt_count"]
        return merged

    def _compute_global_cumulative_stats(
        self,
        merged: Dict[float, Dict[str, Dict[str, Any]]],
    ) -> Dict[float, Dict[str, Dict[int, Dict[str, float]]]]:
        """Build cumulative P-R statistics from the merged match records.

        Sorts all predictions globally by confidence descending and accumulates
        cumulative TP, FP, FN, precision, and recall across the full dataset.
        ``cum_fn`` is derived as ``total_gt`` minus the number of TPs seen so
        far.

        Args:
            merged: Dataset-level match records keyed by threshold and class.

        Returns:
            ``threshold → class_name → pred_idx → {confidence_score, dist,
            cum_tp, cum_fp, cum_fn, cum_precision, cum_recall}``.
        """
        cumulative_results: Dict[float, Dict[str, Dict[int, Dict[str, float]]]] = {}
        for threshold, class_records in merged.items():
            cumulative_results[threshold] = {}
            for class_name, data in class_records.items():
                pred_entries = sorted(data["pred_entries"], key=lambda x: -x["confidence_score"])
                total_gt = data["gt_count"]
                class_results: Dict[int, Dict[str, float]] = {}
                cum_tp, cum_fp = 0.0, 0.0
                num_matched = 0
                for pred_idx, entry in enumerate(pred_entries):
                    if entry["is_tp"]:
                        cum_tp += 1.0
                        num_matched += 1
                    else:
                        cum_fp += 1.0
                    cum_fn = float(total_gt - num_matched)
                    cum_precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
                    cum_recall = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0
                    class_results[pred_idx] = {
                        "confidence_score": entry["confidence_score"],
                        "dist": entry["dist"],
                        "cum_tp": cum_tp,
                        "cum_fp": cum_fp,
                        "cum_fn": cum_fn,
                        "cum_precision": cum_precision,
                        "cum_recall": cum_recall,
                    }
                cumulative_results[threshold][class_name] = class_results
        return cumulative_results

    def _compute_threshold_metrics(
        self,
        cumulative_results: Dict[float, Dict[str, Dict[int, Dict[str, float]]]],
    ) -> Dict[str, Any]:
        """Compute per-threshold and overall AP.

        Per class, the P-R curve is interpolated to 101 recall points with the
        ``min_recall`` / ``min_precision`` gating (a perfect detector scores
        AP = 1.0). Per-class AP is reported for every class, but the
        per-threshold mAP (and ``overall_map``) averages only over
        ``supported_classes`` -- classes the perception interface cannot express
        are excluded from the mean rather than counted as 0.

        Args:
            cumulative_results: P-R statistics from
                :meth:`_compute_global_cumulative_stats`.

        Returns:
            ``{"thresholds": {threshold: {"per_class_metrics", "map"}},
            "overall_map"}``.
        """
        threshold_metrics: Dict[str, Any] = {"thresholds": {}}
        maps = []
        for threshold, class_results in cumulative_results.items():
            per_class_metrics: Dict[str, Dict[str, float]] = {}
            for class_name, predictions in class_results.items():
                sorted_preds = sorted(predictions.values(), key=lambda x: -x["confidence_score"])
                rec = np.array([p["cum_recall"] for p in sorted_preds])
                prec = np.array([p["cum_precision"] for p in sorted_preds])
                if rec.size == 0:
                    ap = 0.0
                else:
                    ap = self.compute_ap_func(rec, prec, self.min_recall, self.min_precision)
                per_class_metrics[class_name] = {"ap": ap}
            # mAP averages only over classes the interface can express; classes
            # outside ``supported_classes`` (e.g. truck, traffic_cone) keep their
            # per-class AP but do not dilute the mean.
            supported_aps = [m["ap"] for name, m in per_class_metrics.items() if name in self.supported_classes]
            map_val = float(np.mean(supported_aps)) if supported_aps else 0.0
            threshold_metrics["thresholds"][threshold] = {
                "per_class_metrics": per_class_metrics,
                "map": map_val,
            }
            maps.append(map_val)
        threshold_metrics["overall_map"] = float(np.mean(maps)) if maps else 0.0
        return threshold_metrics

    def _compute_tp_metrics(
        self,
        merged: Dict[float, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute mean TP error metrics per class, at the 2 m threshold.

        Cumulative-mean errors (by descending confidence) are interpolated to
        101 recall points and averaged over ``[min_recall, max_recall]``. A
        class with no GT/predictions falls back to worst-case ``1.0`` errors.
        ``aae`` uses only TP entries with attribute data, and is ``None`` when
        none exist.

        Args:
            merged: Dataset-level match records keyed by threshold and class.

        Returns:
            ``class → {"ate", "ase", "aoe", "ave", "aae"}``; ``aae`` is a
            ``float`` or ``None``.
        """
        threshold_data = merged.get(self.tp_metric_threshold, {})
        tp_metrics: Dict[str, Dict[str, Any]] = {}

        for class_name, data in threshold_data.items():
            all_entries = sorted(data["pred_entries"], key=lambda x: -x["confidence_score"])
            total_gt = data["gt_count"]

            if total_gt == 0 or not all_entries:
                tp_metrics[class_name] = {"ate": 1.0, "ase": 1.0, "aoe": 1.0, "ave": 1.0, "aae": None}
                continue

            cum_tp = 0
            rec_vals: List[float] = []
            cm_ate: List[float] = []
            cm_ase: List[float] = []
            cm_aoe: List[float] = []
            cm_ave: List[float] = []
            cm_aae: List[Optional[float]] = []
            sum_ate = sum_ase = sum_aoe = sum_ave = 0.0
            sum_aae = 0.0
            n_aae = 0

            for entry in all_entries:
                if entry["is_tp"]:
                    cum_tp += 1
                    sum_ate += entry["ate"]
                    sum_ase += entry["ase"]
                    sum_aoe += entry["aoe"]
                    sum_ave += entry["ave"]
                    if entry["aae"] is not None:
                        sum_aae += entry["aae"]
                        n_aae += 1
                rec_vals.append(cum_tp / total_gt)
                cm_ate.append(sum_ate / cum_tp if cum_tp > 0 else 0.0)
                cm_ase.append(sum_ase / cum_tp if cum_tp > 0 else 0.0)
                cm_aoe.append(sum_aoe / cum_tp if cum_tp > 0 else 0.0)
                cm_ave.append(sum_ave / cum_tp if cum_tp > 0 else 0.0)
                cm_aae.append(sum_aae / n_aae if n_aae > 0 else None)

            rec_arr = np.array(rec_vals)

            avg = ObjectDetectionUtils.compute_tp_101_point
            if avg(rec_arr, cm_ate, self.min_recall) is None:
                tp_metrics[class_name] = {"ate": 1.0, "ase": 1.0, "aoe": 1.0, "ave": 1.0, "aae": None}
                continue

            aae_result: Optional[float] = None
            if n_aae > 0:
                aae_filled: List[float] = []
                last_known = 0.0
                for v in cm_aae:
                    if v is not None:
                        last_known = v
                    aae_filled.append(last_known)
                aae_result = avg(rec_arr, aae_filled, self.min_recall)

            tp_metrics[class_name] = {
                "ate": avg(rec_arr, cm_ate, self.min_recall),
                "ase": avg(rec_arr, cm_ase, self.min_recall),
                "aoe": avg(rec_arr, cm_aoe, self.min_recall),
                "ave": avg(rec_arr, cm_ave, self.min_recall),
                "aae": aae_result,
            }
        return tp_metrics

    def _compute_benchmark_score(
        self,
        threshold_metrics: Dict[str, Any],
        tp_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Flatten AP and TP metrics into the nuScenes-named score dict.

        Emits per-threshold/per-class AP (``ap_{threshold}_{class}``), per-class
        and overall mAP (``map_{class}``, ``map``), per-class and mean TP errors
        (``{metric}_2.0_{class}``, ``m{metric}_2.0``), ``nds``, and the
        ``supported_classes`` / ``unsupported_classes`` capability lists::

            NDS = (5·mAP + Σ max(1 - m{metric}, 0)) / 10

        ``map`` and the mean TP errors (hence ``nds``) aggregate only over
        ``supported_classes``; per-class AP / TP errors are still emitted for
        every class. Mean TP errors skip ``None`` per-class values and fall back
        to 1.0 when no supported class has one.

        Args:
            threshold_metrics: Output of :meth:`_compute_threshold_metrics`.
            tp_metrics: Output of :meth:`_compute_tp_metrics`.

        Returns:
            A flat dict of rounded (4 dp) score keys, including ``map``,
            per-class/threshold AP, mean/per-class TP errors, and ``nds``.
        """
        score: Dict[str, Any] = {}

        # Overall mAP (averaged over supported classes only; see
        # ``supported_classes``).
        map_val = threshold_metrics["overall_map"]
        score["map"] = round(map_val, 4)
        # Declare which classes the perception interface can express, so the
        # aggregate mAP / NDS can be read in the right context.
        score["supported_classes"] = sorted(self.supported_classes)
        score["unsupported_classes"] = sorted(_DETECTION_CLASSES - self.supported_classes - {self.bike_rack_class_name})

        # Per-threshold per-class AP and per-class mean AP (class_name is the name).
        per_class_aps: Dict[str, List[float]] = {}
        for threshold, thr_data in threshold_metrics["thresholds"].items():
            for class_name, class_metric in thr_data["per_class_metrics"].items():
                ap = class_metric["ap"]
                score[f"ap_{threshold}_{class_name}"] = round(ap, 4)
                per_class_aps.setdefault(class_name, []).append(ap)
        for class_name, aps in per_class_aps.items():
            score[f"map_{class_name}"] = round(float(np.mean(aps)), 4)

        # Macro mean TP error metrics over supported classes only.
        def _mean_tp_metric(key: str) -> float:
            """Mean of a TP-error metric over supported classes, skipping ``None``.

            Args:
                key: The TP-error metric name (e.g. ``"ate"``).

            Returns:
                The macro mean over ``supported_classes``, or ``1.0`` when no
                supported class has a value.
            """
            vals = [m[key] for name, m in tp_metrics.items() if name in self.supported_classes and m.get(key) is not None]
            return float(np.mean(vals)) if vals else 1.0

        mate = _mean_tp_metric("ate")
        mase = _mean_tp_metric("ase")
        maoe = _mean_tp_metric("aoe")
        mave = _mean_tp_metric("ave")
        maae = _mean_tp_metric("aae")

        score["mate_2.0"] = round(mate, 4)
        score["mase_2.0"] = round(mase, 4)
        score["maoe_2.0"] = round(maoe, 4)
        score["mave_2.0"] = round(mave, 4)
        score["maae_2.0"] = round(maae, 4)

        # Per-class TP error metrics at the 2 m threshold (class_name is the name).
        for class_name, tm in tp_metrics.items():
            score[f"ate_2.0_{class_name}"] = round(tm["ate"], 4)
            score[f"ase_2.0_{class_name}"] = round(tm["ase"], 4)
            score[f"aoe_2.0_{class_name}"] = round(tm["aoe"], 4)
            score[f"ave_2.0_{class_name}"] = round(tm["ave"], 4)
            score[f"aae_2.0_{class_name}"] = round(tm["aae"], 4) if tm["aae"] is not None else None

        # NDS combines mAP and TP metrics with weights 5-1-1-1-1-1, normalised by 10.
        tp_scores = (
            max(1.0 - mate, 0.0) + max(1.0 - mase, 0.0) + max(1.0 - maoe, 0.0) + max(1.0 - mave, 0.0) + max(1.0 - maae, 0.0)
        )
        nds = (5.0 * map_val + tp_scores) / 10.0
        score["nds"] = round(nds, 4)

        return score
