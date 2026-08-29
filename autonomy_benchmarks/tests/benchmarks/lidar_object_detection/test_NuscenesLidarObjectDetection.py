# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for NuscenesLidarObjectDetection benchmark.

Tests focus on public API methods: compute_sample_metrics(),
compute_aggregated_metrics(), and benchmark configuration.

Inputs are real ROS messages (built via the helpers below), matching how
``_extract_objects`` reads them: geometry through the ``perception_msgs_utils``
state getters (which require a real ``ObjectState`` with a valid ``model_id``),
the prediction class from the perception ``ObjectClassification`` enum, and the
label class from the ``original_class`` annotation of the accompanying
``autonomy_datasets_msgs/ObjectListMetaInfo`` message, which the dataset
publishes on ``<label topic>/meta_info`` and which is correlated with the
``ObjectList`` by object ID.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
from autonomy_benchmarks.benchmarks.lidar_object_detection.NuscenesLidarObjectDetection import (
    NuscenesLidarObjectDetection,
)
from autonomy_datasets_msgs.msg import ObjectListMetaInfo, ObjectMetaInfo
from diagnostic_msgs.msg import KeyValue
from perception_msgs.msg import HEXAMOTION, Object, ObjectClassification, ObjectList, ObjectState

# Full HEXAMOTION continuous-state length, so the getters' size sanity-check passes.
_HEXAMOTION_STATE_SIZE = HEXAMOTION.CONTINUOUS_STATE_SIZE

# A ground-truth object paired with its dataset annotations (``None`` when the
# dataset publishes no meta information for it).
GroundTruth = Tuple[Object, Optional[List[Tuple[str, str]]]]


def _state(x, y, width, length, height, yaw, vel_lon, vel_lat, classifications) -> ObjectState:
    """Build a HEXAMOTION ``ObjectState`` with the given geometry/classifications."""
    cs = [0.0] * _HEXAMOTION_STATE_SIZE
    cs[HEXAMOTION.X] = x
    cs[HEXAMOTION.Y] = y
    cs[HEXAMOTION.WIDTH] = width
    cs[HEXAMOTION.LENGTH] = length
    cs[HEXAMOTION.HEIGHT] = height
    cs[HEXAMOTION.YAW] = yaw
    cs[HEXAMOTION.VEL_LON] = vel_lon
    cs[HEXAMOTION.VEL_LAT] = vel_lat
    return ObjectState(
        model_id=HEXAMOTION.MODEL_ID,
        continuous_state=cs,
        classifications=classifications,
    )


def _pred(
    x=0.0,
    y=0.0,
    width=1.0,
    length=1.0,
    height=1.0,
    yaw=0.0,
    vel_lon=0.0,
    vel_lat=0.0,
    class_type=ObjectClassification.CAR,
    confidence=0.9,
    obj_id=0,
) -> Object:
    """Build a prediction ``Object``.

    Class comes from the perception enum and the score from
    ``existence_probability``. Predictions carry no dataset meta information.
    """
    state = _state(
        x,
        y,
        width,
        length,
        height,
        yaw,
        vel_lon,
        vel_lat,
        [ObjectClassification(type=class_type, probability=confidence)],
    )
    return Object(id=obj_id, state=state, existence_probability=confidence)


def _gt(
    x=0.0,
    y=0.0,
    width=1.0,
    length=1.0,
    height=1.0,
    yaw=0.0,
    vel_lon=0.0,
    vel_lat=0.0,
    original_class="vehicle.car",
    num_lidar_pts=None,
    num_radar_pts=None,
    attribute=None,
    include_original_class=True,
) -> GroundTruth:
    """Build a ground-truth ``Object`` and its dataset annotations.

    Class comes from the ``original_class`` annotation; point counts and
    attribute are annotations too. The annotations are turned into an
    ``ObjectMetaInfo`` entry by :func:`_label`; ``None`` means the object has no
    meta info entry at all.
    """
    object_annotations: List[Tuple[str, str]] = []
    if include_original_class:
        object_annotations.append(("original_class", original_class))
    if attribute is not None:
        object_annotations.append(("attribute", attribute))
    if num_lidar_pts is not None:
        object_annotations.append(("num_lidar_pts", str(num_lidar_pts)))
    if num_radar_pts is not None:
        object_annotations.append(("num_radar_pts", str(num_radar_pts)))
    state = _state(x, y, width, length, height, yaw, vel_lon, vel_lat, [])
    return Object(state=state, existence_probability=1.0), object_annotations or None


def _msg(objs, frame_id: str = "base_link") -> ObjectList:
    """Wrap objects in a ``perception_msgs/ObjectList`` message."""
    msg = ObjectList(objects=objs)
    msg.header.frame_id = frame_id
    return msg


def _label(
    gts: List[GroundTruth], reverse_meta: bool = False, frame_id: str = "base_link"
) -> Tuple[ObjectList, ObjectListMetaInfo]:
    """Split ``_gt`` pairs into the label and label meta info messages.

    Object IDs are assigned per frame and are what correlates both messages;
    objects without annotations get no ``ObjectMetaInfo`` entry, as published by
    the dataset.

    Args:
        gts: ``(object, annotations)`` pairs as returned by :func:`_gt`.
        reverse_meta: Publish the meta info entries in the reverse order of the
            objects, to prove the correlation is by ID and not by position.
        frame_id: Frame of both message headers.

    Returns:
        The ``(ObjectList, ObjectListMetaInfo)`` pair for the frame.
    """
    objects: List[Object] = []
    meta_entries: List[ObjectMetaInfo] = []
    for object_id, (obj, object_annotations) in enumerate(gts):
        obj.id = object_id
        objects.append(obj)
        if object_annotations:
            info = [KeyValue(key=key, value=value) for key, value in object_annotations]
            meta_entries.append(ObjectMetaInfo(id=object_id, info=info))
    if reverse_meta:
        meta_entries.reverse()
    label = ObjectList(objects=objects)
    label.header.frame_id = frame_id
    meta_info = ObjectListMetaInfo(objects=meta_entries)
    meta_info.header.frame_id = frame_id
    return label, meta_info


class TestNuscenesLidarObjectDetection:
    """Tests for NuscenesLidarObjectDetection benchmark.

    Covers sample metrics computation (Pass 1), aggregated metrics
    (Pass 2), and NDS score calculation.
    """

    def setup_method(self):
        """Create a fresh benchmark instance for each test."""
        self.bm = NuscenesLidarObjectDetection()

    def _metrics(self, pred_objs, gts, reverse_meta: bool = False) -> dict:
        """Compute sample metrics from prediction objects and ``_gt`` pairs.

        Mirrors the node's synchronized callback, which hands the benchmark the
        label object list and its meta info message together.
        """
        label, label_meta_info = _label(gts, reverse_meta=reverse_meta)
        return self.bm.compute_sample_metrics(_msg(pred_objs), label, label_meta_info=label_meta_info)

    def _visualize(self, pred_objs, gts) -> dict:
        """Build the visualization object lists the way the node's callback does."""
        label, label_meta_info = _label(gts)
        return self.bm.visualize_sample(_msg(pred_objs), label, label_meta_info=label_meta_info)

    def test_empty_inputs(self):
        """Verify empty inputs produce valid match records for all thresholds."""
        result = self._metrics([], [])
        assert result["sample_prediction_num"] == 0
        assert result["sample_ground_truth_num"] == 0
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert thr in result["match_records"]
            assert result["match_records"][thr] == {}

    def test_range_filtering_active(self):
        """Verify per-class distance filtering is applied.

        Cars beyond 50m should be excluded from matching.
        """
        result = self._metrics(
            [_pred(x=51.0, y=0.0, confidence=0.9)],
            [_gt(x=51.0, y=0.0, original_class="vehicle.car", num_lidar_pts=5)],
        )
        for thr, class_records in result["match_records"].items():
            assert "car" not in class_records

    def test_basic_matching_produces_tp(self):
        """Verify identical boxes produce true positives at tight threshold."""
        result = self._metrics([_pred(x=0.0, y=0.0, confidence=0.9)], [_gt(x=0.0, y=0.0, num_lidar_pts=5)])
        entry = result["match_records"][0.5]["car"]["pred_entries"][0]
        assert entry["is_tp"] is True

    def test_distant_pred_is_fp_at_tight_threshold(self):
        """Verify distant predictions are false positives at tight thresholds."""
        result = self._metrics([_pred(x=10.0, y=0.0, confidence=0.9)], [_gt(x=0.0, y=0.0, num_lidar_pts=5)])
        entries_05 = result["match_records"][0.5]["car"]["pred_entries"]
        assert entries_05[0]["is_tp"] is False

    def test_bike_rack_gt_is_never_scored(self):
        """Bike-rack boxes in the GT stream are excluded from matching/scoring.

        (Predictions can never be bike-rack: the perception enum has no such
        class, so only the GT stream can carry one.)
        """
        result = self._metrics(
            [_pred(x=0.0, y=0.0, confidence=0.9)],
            [
                _gt(x=0.0, y=0.0, original_class="vehicle.car", num_lidar_pts=5),
                _gt(x=20.0, y=0.0, original_class="static_object.bicycle_rack", num_lidar_pts=5),
            ],
        )
        for threshold_records in result["match_records"].values():
            assert self.bm.bike_rack_class_name not in threshold_records

    def _make_sample_result(self, pred_objs, gts):
        """Wrap sample metrics in the expected result structure."""
        return {"metrics": self._metrics(pred_objs, gts)}

    # --- message extraction, exercised through the public API ---

    def test_label_class_name_comes_from_original_class_meta(self):
        """A label's class_name is read from the 'original_class' annotation."""
        result = self._metrics([], [_gt(x=0.0, original_class="barrier", num_lidar_pts=5)])
        assert "barrier" in result["match_records"][0.5]
        assert "car" not in result["match_records"][0.5]

    def test_label_missing_original_class_raises(self):
        """A label object without an 'original_class' annotation raises ValueError."""
        with pytest.raises(ValueError):
            self._metrics([], [_gt(include_original_class=False)])

    def test_label_without_meta_info_entry_raises(self):
        """A label object with no meta info entry at all raises ValueError."""
        label = _msg([_gt(x=0.0, num_lidar_pts=5)[0]])
        with pytest.raises(ValueError):
            self.bm.compute_sample_metrics(_msg([]), label, label_meta_info=ObjectListMetaInfo())

    def test_meta_info_is_correlated_by_object_id(self):
        """Annotations are matched to objects by ID, not by list position.

        The meta info entries are published in reverse order. The car sits at
        45 m, inside the 50 m car range but beyond the 40 m pedestrian range, so
        swapping the two annotations would drop it from the match records.
        """
        gts = [
            _gt(x=45.0, original_class="vehicle.car", num_lidar_pts=5),
            _gt(x=0.0, original_class="human.pedestrian.adult", num_lidar_pts=5),
        ]
        result = self._metrics([], gts, reverse_meta=True)
        assert result["sample_ground_truth_num"] == 2
        assert set(result["match_records"][0.5]) == {"car", "pedestrian"}

    def test_prediction_class_comes_from_perception_enum(self):
        """A prediction's class_name is derived from the perception classification enum."""
        result = self._metrics(
            [_pred(x=0.0, class_type=ObjectClassification.PEDESTRIAN, confidence=0.9)],
            [_gt(x=0.0, original_class="human.pedestrian.adult", num_lidar_pts=5)],
        )
        entry = result["match_records"][0.5]["pedestrian"]["pred_entries"][0]
        assert entry["is_tp"] is True

    def test_prediction_unmapped_enum_is_dropped(self):
        """A prediction whose perception type has no class mapping is dropped, not raised."""
        result = self._metrics([_pred(x=0.0, class_type=ObjectClassification.ANIMAL, confidence=0.9)], [])
        assert result["sample_prediction_num"] == 0

    def test_gt_attribute_is_extracted_for_aae(self):
        """The GT attribute comes from the meta info topic and drives AAE.

        Predictions carry no attribute (``perception_msgs`` cannot express one
        and detectors publish no meta info topic), so an annotated GT attribute
        always counts as a mismatch, while a GT without attribute leaves the TP
        out of the AAE average.
        """
        preds = [_pred(x=0.0, class_type=ObjectClassification.CAR)]
        result = self._metrics(preds, [_gt(x=0.0, original_class="vehicle.car", num_lidar_pts=5, attribute="vehicle.moving")])
        assert result["match_records"][0.5]["car"]["pred_entries"][0]["aae"] == 1.0

        result = self._metrics(preds, [_gt(x=0.0, original_class="vehicle.car", num_lidar_pts=5)])
        assert result["match_records"][0.5]["car"]["pred_entries"][0]["aae"] is None

    def test_gt_point_filter_active_through_extraction(self):
        """A GT box with 0 lidar and 0 radar points (from the meta info) is excluded."""
        result = self._metrics(
            [_pred(x=0.0, confidence=0.9)],
            [_gt(x=0.0, original_class="vehicle.car", num_lidar_pts=0, num_radar_pts=0)],
        )
        # The zero-point GT is filtered out, so its class has no GT to match.
        for class_records in result["match_records"].values():
            gt_count = class_records.get("car", {}).get("gt_count", 0)
            assert gt_count == 0

    def test_gt_with_points_survives_filter(self):
        """A GT box with lidar points is retained by the point filter."""
        result = self._metrics(
            [_pred(x=0.0, confidence=0.9)],
            [_gt(x=0.0, original_class="vehicle.car", num_lidar_pts=5, num_radar_pts=0)],
        )
        assert result["match_records"][0.5]["car"]["gt_count"] == 1

    def test_full_nuscenes_category_is_normalised_to_detection_class(self):
        """A GT label's full nuScenes category (e.g. 'vehicle.car') routes to 'car'."""
        result = self._metrics(
            [_pred(x=0.0, y=0.0, class_type=ObjectClassification.CAR, confidence=0.9)],
            [_gt(x=0.0, y=0.0, original_class="vehicle.car", num_lidar_pts=5)],
        )
        assert result["sample_ground_truth_num"] == 1
        # The full-category GT lands in the same 'car' bucket as the prediction.
        entry = result["match_records"][0.5]["car"]["pred_entries"][0]
        assert entry["is_tp"] is True

    def test_ignore_category_objects_are_dropped(self):
        """Objects in non-evaluated nuScenes categories are dropped, not raised."""
        result = self._metrics(
            [],
            [
                _gt(x=0.0, original_class="vehicle.car", num_lidar_pts=5),
                _gt(x=1.0, original_class="movable_object.debris", num_lidar_pts=5),
                _gt(x=2.0, original_class="animal", num_lidar_pts=5),
            ],
        )
        # Only the car survives; the two ignore-category objects are dropped.
        assert result["sample_ground_truth_num"] == 1

    def test_bicycle_rack_category_maps_to_bike_rack(self):
        """'static_object.bicycle_rack' is retained as the bike_rack class."""
        rack = _gt(x=0.0, y=0.0, width=4.0, length=4.0, original_class="static_object.bicycle_rack", num_lidar_pts=5)
        bike = _gt(x=0.0, y=0.0, original_class="vehicle.bicycle", num_lidar_pts=5)
        result = self._metrics([], [rack, bike])
        # The bicycle falls inside the rack and is filtered out by bike-rack logic.
        for threshold_records in result["match_records"].values():
            assert "bicycle" not in threshold_records

    def test_unrecognised_label_category_raises(self):
        """An unknown label original_class value raises ValueError."""
        with pytest.raises(ValueError):
            self._metrics([], [_gt(original_class="totally.made.up")])

    def test_perfect_matching_yields_high_map(self):
        """Verify perfect matching produces high mean average precision."""
        n = 10
        preds = [_pred(x=float(i), confidence=1.0 - i * 0.01) for i in range(n)]
        gts = [_gt(x=float(i), num_lidar_pts=5) for i in range(n)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        assert result["benchmark_score"]["map"] >= 0.8

    def test_no_pred_yields_zero_map(self):
        """Verify missing predictions yield zero mAP."""
        gts = [_gt(x=0.0, num_lidar_pts=5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result([], gts)])
        assert result["benchmark_score"]["map"] == 0.0

    def test_all_fp_yields_zero_map(self):
        """Verify all false positives yield zero mAP."""
        preds = [_pred(x=float(i), confidence=0.9) for i in range(5)]
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, [])])
        assert result["benchmark_score"]["map"] == 0.0

    def test_unsupported_class_gt_excluded_from_map(self):
        """A GT class the interface cannot express (truck) must not dilute mAP.

        Cars match perfectly; a truck GT can never be predicted (no perception
        enum), so its AP=0 is excluded rather than dragging mAP toward 0.5.
        """
        n = 10
        preds = [_pred(x=float(i), confidence=1.0 - i * 0.01) for i in range(n)]
        gts = [_gt(x=float(i), num_lidar_pts=5) for i in range(n)]
        gts.append(_gt(x=5.0, y=20.0, original_class="vehicle.truck", num_lidar_pts=5))
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        # truck is unsupported -> excluded; mAP stays high (car only).
        assert result["benchmark_score"]["map"] >= 0.8
        assert "truck" not in self.bm.supported_classes

    def test_supported_but_undetected_class_still_counts_as_zero(self):
        """A supported class with GT but no prediction (pedestrian) counts as 0.

        Unlike an unsupported class, a mappable class that the model simply
        misses is a real failure and must stay in the mAP denominator.
        """
        n = 10
        preds = [_pred(x=float(i), confidence=1.0 - i * 0.01) for i in range(n)]
        gts = [_gt(x=float(i), num_lidar_pts=5) for i in range(n)]
        gts.append(_gt(x=5.0, y=20.0, original_class="human.pedestrian.adult", num_lidar_pts=5))
        result = self.bm.compute_aggregated_metrics([self._make_sample_result(preds, gts)])
        # pedestrian is supported -> AP 0 pulls the mean well below the car-only case.
        assert result["benchmark_score"]["map"] < 0.6
        assert "pedestrian" in self.bm.supported_classes

    def test_capability_lists_reported(self):
        """The score declares which classes the interface can / cannot express."""
        gts = [_gt(x=0.0, num_lidar_pts=5)]
        score = self.bm.compute_aggregated_metrics([self._make_sample_result([], gts)])["benchmark_score"]
        assert score["supported_classes"] == sorted(self.bm.supported_classes)
        assert "car" in score["supported_classes"]
        assert "truck" in score["unsupported_classes"]
        assert "traffic_cone" in score["unsupported_classes"]
        assert self.bm.bike_rack_class_name not in score["unsupported_classes"]

    def test_multi_frame_accumulation(self):
        """Verify GT counts accumulate correctly across multiple frames."""
        r1 = self._make_sample_result([_pred(x=0.0, confidence=0.9)], [_gt(x=0.0, num_lidar_pts=5)])
        r2 = self._make_sample_result([_pred(x=0.0, confidence=0.8)], [_gt(x=0.0, num_lidar_pts=5)])
        merged = self.bm._merge_match_records([r1["metrics"]["match_records"], r2["metrics"]["match_records"]])
        assert merged[0.5]["car"]["gt_count"] == 2

    # --- visualization of the per-sample matching outcome ---

    def test_visualization_outputs_are_object_lists(self):
        """The benchmark declares one ObjectList output per matching outcome."""
        assert self.bm.visualization_outputs() == {
            "true_positives": ObjectList,
            "false_positives": ObjectList,
            "false_negatives": ObjectList,
        }

    def test_visualization_splits_outcomes_by_source_object(self):
        """A matched prediction is a TP, an unmatched one an FP, a missed GT an FN."""
        preds = [_pred(x=0.0, obj_id=100), _pred(x=15.0, obj_id=101)]
        # _label assigns the GT IDs 0 and 1, in order.
        gts = [_gt(x=0.0, num_lidar_pts=5), _gt(x=30.0, num_lidar_pts=5)]
        outputs = self._visualize(preds, gts)
        assert [obj.id for obj in outputs["true_positives"].objects] == [100]
        assert [obj.id for obj in outputs["false_positives"].objects] == [101]
        assert [obj.id for obj in outputs["false_negatives"].objects] == [1]

    def test_visualization_agrees_with_the_scored_matching(self):
        """The split reproduces what was scored at the TP metric threshold."""
        preds = [_pred(x=0.0, obj_id=100), _pred(x=15.0, obj_id=101)]
        gts = [_gt(x=0.0, num_lidar_pts=5), _gt(x=30.0, num_lidar_pts=5)]
        entries = self._metrics(preds, gts)["match_records"][self.bm.tp_metric_threshold]["car"]["pred_entries"]
        outputs = self._visualize(preds, gts)
        assert sum(entry["is_tp"] for entry in entries) == len(outputs["true_positives"].objects)
        assert sum(not entry["is_tp"] for entry in entries) == len(outputs["false_positives"].objects)

    def test_visualization_keeps_the_source_headers(self):
        """Positives carry the prediction header, negatives the label header."""
        # Distinct frames only to show which header each output copies; a real
        # synchronized frame has the same header on both inputs.
        prediction = _msg([_pred(x=0.0, obj_id=100), _pred(x=15.0, obj_id=101)], frame_id="prediction_frame")
        label, label_meta_info = _label([_gt(x=30.0, num_lidar_pts=5)], frame_id="label_frame")
        outputs = self.bm.visualize_sample(prediction, label, label_meta_info=label_meta_info)
        assert outputs["true_positives"].header.frame_id == "prediction_frame"
        assert outputs["false_positives"].header.frame_id == "prediction_frame"
        assert outputs["false_negatives"].header.frame_id == "label_frame"

    def test_visualization_omits_objects_dropped_before_matching(self):
        """Objects the pre-matching filters remove appear in none of the lists."""
        gts = [
            _gt(x=0.0, original_class="vehicle.car", num_lidar_pts=0, num_radar_pts=0),  # no sensor points
            _gt(x=1.0, original_class="animal", num_lidar_pts=5),  # non-evaluated class
            _gt(x=60.0, original_class="vehicle.car", num_lidar_pts=5),  # beyond the 50 m car range
        ]
        outputs = self._visualize([], gts)
        assert outputs["true_positives"].objects == []
        assert outputs["false_positives"].objects == []
        assert outputs["false_negatives"].objects == []

    @staticmethod
    def _tm(overall_map: float) -> dict:
        """Create minimal threshold_metrics structure for NDS testing."""
        return {"overall_map": overall_map, "thresholds": {}}

    def test_nds_with_valid_aae(self):
        """Verify NDS uses full 6-term formula when AAE data is available."""
        tp_metrics = {"car": {"ate": 0.2, "ase": 0.1, "aoe": 0.3, "ave": 0.4, "aae": 0.05}}
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

    def test_nds_without_aae_is_lower(self):
        """Verify missing AAE data (falls back to 1.0) reduces NDS score."""
        tp_with = {"car": {"ate": 0.0, "ase": 0.0, "aoe": 0.0, "ave": 0.0, "aae": 0.0}}
        tp_without = {"car": {"ate": 0.0, "ase": 0.0, "aoe": 0.0, "ave": 0.0, "aae": None}}
        score_with = self.bm._compute_benchmark_score(self._tm(1.0), tp_with)
        score_without = self.bm._compute_benchmark_score(self._tm(1.0), tp_without)
        assert score_with["nds"] == pytest.approx(1.0, abs=1e-4)
        assert score_without["nds"] == pytest.approx(0.9, abs=1e-4)

    def test_nds_clamps_tp_errors(self):
        """Verify TP errors above 1.0 are clamped in NDS computation."""
        tp_metrics = {"car": {"ate": 2.0, "ase": 3.0, "aoe": 5.0, "ave": 1.5, "aae": 2.0}}
        score = self.bm._compute_benchmark_score(self._tm(0.0), tp_metrics)
        assert score["nds"] >= 0.0
