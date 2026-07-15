"""nuScenes - Object tracking benchmark.

Re-implementation of the nuScenes 3D multi-object tracking challenge
(https://www.nuscenes.org/tracking), which adapts the AB3DMOT evaluation
protocol (https://github.com/xinshuoweng/AB3DMOT) by integrating the classic
CLEAR-MOT metrics over a range of recall thresholds.

Inputs are ``perception_msgs/Object`` messages (carried in an ``ObjectList``):
the lifetime ``id`` provides the track identity, ``existence_probability`` the
tracking confidence, the object ``state`` the BEV position, and the
``classifications`` the object class.

Key evaluation settings
-----------------------
- Classes:    7 tracking classes (car, truck, bus, trailer, pedestrian,
              motorcycle, bicycle).  ``construction_vehicle``, ``traffic_cone``
              and ``barrier`` from the detection task are not tracked.
- Matching:   BEV center-point Euclidean distance, gated at ``dist_th_tp``
              (2.0 m).  Greedy nearest-neighbour assignment that first
              preserves the previous frame's correspondences to avoid spurious
              identity switches (CLEAR-MOT style).
- Recall:     Per class, ``num_thresholds`` (40) confidence thresholds are
              derived so that the achieved recall is spread evenly across
              ``[min_recall, 1.0]`` (min_recall = 0.1), following AB3DMOT.
- MOTAR:      Recall-normalised MOTA evaluated at every threshold::

                  MOTAR = max(0, 1 - (IDS + FP + FN - (1 - r)·P) / (r·P))

              with ``r`` the recall actually achieved at the threshold and
              ``P`` the number of ground-truth objects of the class.
- AMOTA:      Mean MOTAR over all recall thresholds (unachieved thresholds are
              filled with the worst value 0.0 before averaging).
- AMOTP:      Mean MOTP (average matched BEV distance) over all recall
              thresholds (unachieved thresholds filled with ``dist_th_tp``).
- Score:      The benchmark score is the class-mean AMOTA (primary nuScenes
              ranking metric) together with the class-mean AMOTP and the
              legacy CLEAR-MOT summary (MOTA, MOTP, recall, TP, FP, FN, IDS,
              FRAG) reported at the MOTA-maximising threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_benchmarks.benchmarks.AutonomyBenchmark import AutonomyBenchmark
from perception_msgs.msg import ObjectClassification, ObjectList


class NuscenesObjectTracking(AutonomyBenchmark):
    """Benchmark for 3D multi-object tracking on the nuScenes dataset.

    See the module docstring for the full evaluation protocol.  The tracking
    class IDs reuse the nuScenes detection numbering:

    ============  ========
    class_name    class_id
    ============  ========
    car           1
    truck         2
    bus           3
    trailer       4
    pedestrian    6
    motorcycle    7
    bicycle       8
    ============  ========
    """

    def __init__(self) -> None:
        """Configure nuScenes tracking classes, thresholds, and metric rules."""
        super().__init__(
            name="nuscenes_object_tracking",
            description=("3D bounding-box multi-object tracking benchmark using the nuScenes dataset."),
        )

        # Tracking class IDs (subset of the nuScenes detection classes) mapped
        # to their canonical names.  Only these classes are evaluated.
        self.tracking_classes: Dict[int, str] = {
            1: "car",
            2: "truck",
            3: "bus",
            4: "trailer",
            6: "pedestrian",
            7: "motorcycle",
            8: "bicycle",
        }
        # BEV center-distance gate (meters) for a prediction to match a GT object.
        self.dist_th_tp: float = 2.0
        # Lowest recall value included in the AMOTA/AMOTP integration.
        self.min_recall: float = 0.1
        # Number of evenly spaced recall thresholds in [min_recall, 1.0].
        self.num_thresholds: int = 40
        # Recall sampling points used to derive the per-class score thresholds.
        self.recall_thresholds: np.ndarray = np.linspace(self.min_recall, 1.0, self.num_thresholds)
        # Worst-case fill values for recall thresholds a tracker cannot reach.
        self.metric_worst: Dict[str, float] = {"amota": 0.0, "amotp": self.dist_th_tp}
        # perception_msgs ObjectClassification.type -> tracking class ID.  The
        # lossy UTILITY -> truck merge reflects the limited resolution of the
        # message taxonomy, which has no separate truck/trailer classes.
        self.msg_class_to_tracking_id: Dict[int, int] = {
            ObjectClassification.PEDESTRIAN: 6,  # pedestrian
            ObjectClassification.BICYCLE: 8,  # bicycle
            ObjectClassification.MICRO: 8,  # bicycle
            ObjectClassification.MOTORCYCLE: 7,  # motorcycle
            ObjectClassification.CAR: 1,  # car
            ObjectClassification.UTILITY: 2,  # truck
            ObjectClassification.BUS: 3,  # bus
        }

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def required_inputs(self) -> Dict[str, Any]:
        """Define expected input ROS message types."""

        return {
            "objects_tracked": ObjectList,
            "objects_truth": ObjectList,
        }

    def compute_sample_metrics(
        self,
        objects_tracked: ObjectList,
        objects_truth: ObjectList,
        scene_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pass 1 - record one frame of tracked and ground-truth objects.

        Tracking metrics are temporal and depend on a confidence-threshold
        sweep over the whole sequence, so no matching happens here.  Each
        ``perception_msgs/Object`` is reduced to the fields the aggregation
        needs (track ID, class ID, BEV position, and tracking confidence).

        Parameters
        ----------
        objects_tracked:
            Tracker output for the frame as a ``perception_msgs/ObjectList``
            (or a list of ``perception_msgs/Object``).
        objects_truth:
            Ground-truth objects for the frame, same accepted types.
        scene_id:
            Identifier of the sequence the frame belongs to.  Track IDs are only
            assumed unique within a scene; identity switches never cross scene
            boundaries.  Defaults to a single shared scene.

        Returns
        -------
        A dict with ``sample_prediction_num``, ``sample_ground_truth_num`` and a
        ``frame`` record consumed during aggregation.
        """
        predictions = self._extract_objects(objects_tracked)
        ground_truths = self._extract_objects(objects_truth)

        print(f"Received {len(predictions)} predictions and {len(ground_truths)} ground truths for scene '{scene_id}'")

        frame = {
            "scene_id": scene_id if scene_id is not None else "__default__",
            "predictions": predictions,
            "ground_truths": ground_truths,
        }

        return {
            "sample_prediction_num": len(predictions),
            "sample_ground_truth_num": len(ground_truths),
            "frame": frame,
        }

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 2 - aggregate per-frame records into dataset-level tracking metrics.

        Frames are grouped into scenes and, per tracking class, evaluated across
        the ``num_thresholds`` recall thresholds to produce AMOTA and AMOTP plus
        the legacy CLEAR-MOT summary.

        Parameters
        ----------
        sample_results:
            List of per-frame results returned by :meth:`compute_sample_metrics`.

        Returns
        -------
        A dict containing ``tracking_metrics`` (per-class detail) and
        ``benchmark_score`` (class-mean AMOTA/AMOTP and CLEAR-MOT summary).
        """
        scenes = self._group_into_scenes(sample_results)

        per_class: Dict[int, Dict[str, Any]] = {}
        for class_id in self.tracking_classes:
            per_class[class_id] = self._evaluate_class(class_id, scenes)

        benchmark_score = self._compute_benchmark_score(per_class)

        return {
            "tracking_metrics": {"per_class": per_class},
            "benchmark_score": benchmark_score,
        }

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _extract_objects(self, objects: Any) -> List[Dict[str, Any]]:
        """Read the fields needed for tracking from a frame's objects.

        Accepts a ``perception_msgs/ObjectList`` or a list of
        ``perception_msgs/Object`` and returns one record per object with its
        track ID, tracking class ID, BEV position, and confidence score.  Object
        fields are accessed through ``perception_msgs_utils`` getters.
        """
        if objects is None:
            return []
        object_msgs = objects.objects if hasattr(objects, "objects") else list(objects)
        if not object_msgs:
            return []

        from perception_msgs_utils.convenience_state_getters import (
            get_class_with_highest_probability,
            get_x,
            get_y,
        )

        records: List[Dict[str, Any]] = []
        for obj in object_msgs:
            msg_type = get_class_with_highest_probability(obj).type
            records.append(
                {
                    "track_id": int(obj.id),
                    "class_id": self.msg_class_to_tracking_id.get(msg_type, 0),
                    "x": float(get_x(obj)),
                    "y": float(get_y(obj)),
                    "score": float(obj.existence_probability),
                }
            )
        return records

    def _group_into_scenes(self, sample_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group per-frame records into ordered, per-scene frame lists."""
        scenes: Dict[str, List[Dict[str, Any]]] = {}
        scene_order: List[str] = []
        for metrics in sample_results:
            frame = metrics.get("frame")
            if frame is None:
                continue
            scene_id = frame.get("scene_id", "__default__")
            if scene_id not in scenes:
                scenes[scene_id] = []
                scene_order.append(scene_id)
            scenes[scene_id].append(frame)

        ordered_scenes: List[List[Dict[str, Any]]] = []
        for scene_id in scene_order:
            frames = scenes[scene_id]
            ordered_scenes.append(frames)
        return ordered_scenes

    # ------------------------------------------------------------------
    # Per-class tracking evaluation
    # ------------------------------------------------------------------

    def _evaluate_class(self, class_id: int, scenes: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Evaluate AMOTA, AMOTP and the CLEAR-MOT summary for a single class.

        Runs one threshold-free accumulation to map recall to confidence, then
        re-accumulates at every recall threshold to build the MOTAR / MOTP
        curves that are averaged into AMOTA / AMOTP.
        """
        gt_count = self._count_ground_truths(class_id, scenes)

        # No ground truth for this class -> metrics are undefined.
        if gt_count == 0:
            return {
                "gt": 0,
                "amota": None,
                "amotp": None,
                "recall": 0.0,
                "mota": 0.0,
                "motp": None,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "ids": 0,
                "frag": 0,
            }

        # Threshold-free pass: collect the confidence scores of all true
        # positives to translate target recalls into score thresholds.
        base = self._accumulate(class_id, scenes, conf_threshold=None)
        score_thresholds = self._compute_score_thresholds(base["tp_scores"], gt_count)

        motar_curve: List[float] = []
        motp_curve: List[float] = []
        recall_curve: List[float] = []
        best: Optional[Dict[str, Any]] = None

        for threshold in score_thresholds:
            if np.isnan(threshold):
                # Recall level unreachable: penalise with the worst values.
                motar_curve.append(self.metric_worst["amota"])
                motp_curve.append(self.metric_worst["amotp"])
                recall_curve.append(0.0)
                continue

            acc = self._accumulate(class_id, scenes, conf_threshold=threshold)
            recall = acc["matches"] / gt_count
            motar = self._motar(acc["ids"], acc["fp"], acc["fn"], recall, gt_count)
            motp = acc["motp_sum"] / acc["matches"] if acc["matches"] > 0 else self.metric_worst["amotp"]

            motar_curve.append(motar if motar is not None else self.metric_worst["amota"])
            motp_curve.append(motp)
            recall_curve.append(recall)

            # Track the MOTA-maximising operating point for the legacy summary.
            mota = max(0.0, 1.0 - (acc["fp"] + acc["fn"] + acc["ids"]) / gt_count)
            if best is None or mota > best["mota"]:
                best = {
                    "mota": mota,
                    "motp": motp if acc["matches"] > 0 else None,
                    "recall": recall,
                    "tp": acc["matches"],
                    "fp": acc["fp"],
                    "fn": acc["fn"],
                    "ids": acc["ids"],
                    "frag": acc["frag"],
                }

        amota = float(np.mean(motar_curve))
        amotp = float(np.mean(motp_curve))

        summary = (
            best
            if best is not None
            else {
                "mota": 0.0,
                "motp": None,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": gt_count,
                "ids": 0,
                "frag": 0,
            }
        )

        return {
            "gt": gt_count,
            "amota": round(amota, 4),
            "amotp": round(amotp, 4),
            "recall": round(summary["recall"], 4),
            "mota": round(summary["mota"], 4),
            "motp": round(summary["motp"], 4) if summary["motp"] is not None else None,
            "tp": summary["tp"],
            "fp": summary["fp"],
            "fn": summary["fn"],
            "ids": summary["ids"],
            "frag": summary["frag"],
            "curves": {
                "confidence": [round(float(t), 4) if not np.isnan(t) else None for t in score_thresholds],
                "recall": [round(r, 4) for r in recall_curve],
                "motar": [round(m, 4) for m in motar_curve],
                "motp": [round(m, 4) for m in motp_curve],
            },
        }

    def _count_ground_truths(self, class_id: int, scenes: List[List[Dict[str, Any]]]) -> int:
        """Return the total number of ground-truth objects of *class_id*."""
        count = 0
        for frames in scenes:
            for frame in frames:
                count += sum(1 for gt in frame["ground_truths"] if gt["class_id"] == class_id)
        return count

    def _accumulate(
        self,
        class_id: int,
        scenes: List[List[Dict[str, Any]]],
        conf_threshold: Optional[float],
    ) -> Dict[str, Any]:
        """Run CLEAR-MOT accumulation for one class at one confidence threshold.

        Predictions with a score below ``conf_threshold`` are discarded
        (``None`` keeps every prediction).  Matching is greedy on BEV distance
        but first re-uses each ground-truth track's previous correspondence so
        that stable tracks do not generate identity switches.  Track IDs are
        assumed unique within a scene; ``prev_match`` therefore resets per scene
        and persists across frames in which a track is temporarily lost.

        Returns the accumulated counts (matches, fp, fn, ids, frag), the summed
        matched distance (``motp_sum``), and the scores of matched predictions
        (``tp_scores``) for the threshold-free pass.
        """
        matches = fp = fn = ids = frag = 0
        motp_sum = 0.0
        tp_scores: List[float] = []

        for frames in scenes:
            prev_match: Dict[Any, Any] = {}  # gt track id -> last matched pred track id
            gt_last_tracked: Dict[Any, bool] = {}  # gt track id -> tracked in previous frame

            for frame in frames:
                gts = [gt for gt in frame["ground_truths"] if gt["class_id"] == class_id]
                preds = [
                    pred
                    for pred in frame["predictions"]
                    if pred["class_id"] == class_id and (conf_threshold is None or pred["score"] >= conf_threshold)
                ]

                pairs = self._match_frame(gts, preds, prev_match)

                matched_gt_tracks = set()
                matched_gt_idx = set()
                matched_pred_idx = set()
                for gt_idx, pred_idx, dist in pairs:
                    gt = gts[gt_idx]
                    pred = preds[pred_idx]
                    matches += 1
                    motp_sum += dist
                    matched_gt_idx.add(gt_idx)
                    matched_pred_idx.add(pred_idx)
                    matched_gt_tracks.add(gt["track_id"])
                    if conf_threshold is None:
                        tp_scores.append(pred["score"])
                    # Identity switch: this GT was previously matched to a
                    # different hypothesis (even across temporary gaps).
                    if gt["track_id"] in prev_match and prev_match[gt["track_id"]] != pred["track_id"]:
                        ids += 1
                    prev_match[gt["track_id"]] = pred["track_id"]

                fn += len(gts) - len(matched_gt_idx)
                fp += len(preds) - len(matched_pred_idx)

                # Fragmentation: a GT that was tracked is present but lost now.
                for gt in gts:
                    track = gt["track_id"]
                    is_tracked = track in matched_gt_tracks
                    if gt_last_tracked.get(track) is True and not is_tracked:
                        frag += 1
                    gt_last_tracked[track] = is_tracked

        return {
            "matches": matches,
            "fp": fp,
            "fn": fn,
            "ids": ids,
            "frag": frag,
            "motp_sum": motp_sum,
            "tp_scores": tp_scores,
        }

    def _match_frame(
        self,
        gts: List[Dict[str, Any]],
        preds: List[Dict[str, Any]],
        prev_match: Dict[Any, Any],
    ) -> List[Tuple[int, int, float]]:
        """Match GT to predictions for one frame within the distance gate.

        Existing correspondences (``prev_match``) are locked in first when still
        within the gate, then the remaining objects are matched greedily by
        increasing BEV distance.  Returns ``(gt_idx, pred_idx, distance)`` tuples.
        """
        used_gt: set = set()
        used_pred: set = set()
        pairs: List[Tuple[int, int, float]] = []

        pred_idx_by_track = {pred["track_id"]: idx for idx, pred in enumerate(preds)}

        # Step 1 - preserve the previous frame's correspondences.
        for gt_idx, gt in enumerate(gts):
            prev_pred_track = prev_match.get(gt["track_id"])
            if prev_pred_track is None:
                continue
            pred_idx = pred_idx_by_track.get(prev_pred_track)
            if pred_idx is None or pred_idx in used_pred:
                continue
            dist = self._dist(gt, preds[pred_idx])
            if dist < self.dist_th_tp:
                used_gt.add(gt_idx)
                used_pred.add(pred_idx)
                pairs.append((gt_idx, pred_idx, dist))

        # Step 2 - greedy nearest-neighbour matching on the remainder.
        candidates: List[Tuple[float, int, int]] = []
        for gt_idx, gt in enumerate(gts):
            if gt_idx in used_gt:
                continue
            for pred_idx, pred in enumerate(preds):
                if pred_idx in used_pred:
                    continue
                dist = self._dist(gt, pred)
                if dist < self.dist_th_tp:
                    candidates.append((dist, gt_idx, pred_idx))

        candidates.sort(key=lambda c: c[0])
        for dist, gt_idx, pred_idx in candidates:
            if gt_idx in used_gt or pred_idx in used_pred:
                continue
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
            pairs.append((gt_idx, pred_idx, dist))

        return pairs

    @staticmethod
    def _dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """BEV (XY) Euclidean distance between two object records."""
        dx = a["x"] - b["x"]
        dy = a["y"] - b["y"]
        return (dx * dx + dy * dy) ** 0.5

    def _compute_score_thresholds(self, tp_scores: List[float], gt_count: int) -> np.ndarray:
        """Map evenly spaced target recalls to prediction-confidence thresholds.

        Mirrors the AB3DMOT / nuScenes procedure: sort the true-positive scores
        descending, read off the recall reached at each score, then interpolate
        the score threshold for every target recall in ``recall_thresholds``.
        Target recalls beyond the maximum achievable recall are set to ``nan``.
        """
        if len(tp_scores) == 0:
            return np.full(self.num_thresholds, np.nan)

        scores = np.sort(np.array(tp_scores, dtype=float))[::-1]
        recalls = np.arange(1, len(scores) + 1) / gt_count
        max_recall_achieved = recalls[-1]

        # np.interp requires an increasing x; recalls are ascending while the
        # corresponding scores decrease, which is fine for the y values.
        thresholds = np.interp(self.recall_thresholds, recalls, scores, right=0.0)
        thresholds[self.recall_thresholds > max_recall_achieved] = np.nan
        return thresholds

    def _motar(self, ids: int, fp: int, fn: int, recall: float, gt_count: int) -> Optional[float]:
        """Recall-normalised MOTA (MOTAR) for one operating point.

        ``MOTAR = max(0, 1 - (IDS + FP + FN - (1 - recall)·P) / (recall·P))``;
        ``None`` when the recall is zero (denominator vanishes).
        """
        denominator = recall * gt_count
        if denominator == 0:
            return None
        nominator = (ids + fp + fn) - (1.0 - recall) * gt_count
        return max(0.0, 1.0 - nominator / denominator)

    # ------------------------------------------------------------------
    # Benchmark score
    # ------------------------------------------------------------------

    def _compute_benchmark_score(self, per_class: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine per-class results into the flat benchmark score.

        The primary nuScenes ranking metric is the class-mean AMOTA.  The
        class-mean AMOTP and the summed CLEAR-MOT counts (with class-mean MOTA,
        MOTP and recall) are reported alongside it.  Classes without ground
        truth do not contribute to any mean.
        """
        score: Dict[str, Any] = {}

        amotas = [m["amota"] for m in per_class.values() if m["amota"] is not None]
        amotps = [m["amotp"] for m in per_class.values() if m["amotp"] is not None]
        motas = [m["mota"] for m in per_class.values() if m["gt"] > 0]
        motps = [m["motp"] for m in per_class.values() if m["motp"] is not None]
        recalls = [m["recall"] for m in per_class.values() if m["gt"] > 0]

        score["amota"] = round(float(np.mean(amotas)), 4) if amotas else 0.0
        score["amotp"] = round(float(np.mean(amotps)), 4) if amotps else self.metric_worst["amotp"]
        score["mota"] = round(float(np.mean(motas)), 4) if motas else 0.0
        score["motp"] = round(float(np.mean(motps)), 4) if motps else None
        score["recall"] = round(float(np.mean(recalls)), 4) if recalls else 0.0

        for key in ("tp", "fp", "fn", "ids", "frag"):
            score[key] = int(sum(m[key] for m in per_class.values()))

        # Per-class AMOTA / AMOTP for inspection.
        for class_id, class_name in self.tracking_classes.items():
            metrics = per_class[class_id]
            score[f"amota_{class_name}"] = metrics["amota"]
            score[f"amotp_{class_name}"] = metrics["amotp"]

        return score
