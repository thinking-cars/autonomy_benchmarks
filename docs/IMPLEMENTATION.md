# Implementation Details

This repository supports the following evaluations of object detection in automated driving systems:

- [nuScenes Challenge](#nuscenes-challenge): 3D lidar object detection

### nuScenes Challenge

> [nuScenes](https://www.nuscenes.org/)

The nuScenes dataset is not redistributed here. You must download it under the [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) and make it available via the companion [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) package.

#### 3D Lidar Object Detection

Supported Datasets:

- [nuScenes Dataset](https://github.com/thinking-cars/autonomy_datasets/blob/main/docs/IMPLEMENTATION.md#nuscenes-dataset)

> This evaluation uses [nuScenes Dataset](https://github.com/thinking-cars/autonomy_datasets/blob/main/docs/IMPLEMENTATION.md#nuscenes-dataset) via the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package.

[![3D Object Detection Challenge](https://img.shields.io/badge/origin-3D_Object_Detection_Challenge-green)](https://www.nuscenes.org/object-detection) ![2019](https://img.shields.io/badge/published-2019-green)

Metrics are computed based on the following assumptions:

- Ground-truth labels carry the full nuScenes annotation category (e.g. `vehicle.car`); predictions already carry the detection class name. Categories are grouped into the 10 evaluated detection classes following the official [`general_to_detection`](https://www.nuscenes.org/object-detection) mapping, categories outside those classes are dropped and not evaluated, and `static_object.bicycle_rack` is retained (as `bike_rack`) solely for the bike-rack filter below and is never scored.

  <details>
  <summary>Full category → detection class mapping</summary>

  - `vehicle.car`: `car`
  - `vehicle.truck`: `truck`
  - `vehicle.bus.bendy`: `bus`
  - `vehicle.bus.rigid`: `bus`
  - `vehicle.trailer`: `trailer`
  - `vehicle.construction`: `construction_vehicle`
  - `human.pedestrian.adult`: `pedestrian`
  - `human.pedestrian.child`: `pedestrian`
  - `human.pedestrian.construction_worker`: `pedestrian`
  - `human.pedestrian.police_officer`: `pedestrian`
  - `vehicle.motorcycle`: `motorcycle`
  - `vehicle.bicycle`: `bicycle`
  - `movable_object.trafficcone`: `traffic_cone`
  - `movable_object.barrier`: `barrier`
  - `static_object.bicycle_rack`: `bike_rack` (filter only, not scored)
  - `animal`: dropped
  - `human.pedestrian.personal_mobility`: dropped
  - `human.pedestrian.stroller`: dropped
  - `human.pedestrian.wheelchair`: dropped
  - `movable_object.debris`: dropped
  - `movable_object.pushable_pullable`: dropped
  - `vehicle.emergency.ambulance`: dropped
  - `vehicle.emergency.police`: dropped

  </details>
- Labels are excluded only when both lidar and radar point counts are available and both are < 1.
- Labels and predictions are only considered if they fall into a class-specific detection range.

  <details>
  <summary>Per-class detection range</summary>

  - Barrier: `<= 30 m`
  - Traffic Cone: `<= 30 m`
  - Bicycle: `<= 40 m`
  - Motorcycle: `<= 40 m`
  - Pedestrian: `<= 40 m`
  - Car: `<= 50 m`
  - Bus: `<= 50 m`
  - Construction Vehicle: `<= 50 m`
  - Trailer: `<= 50 m`
  - Truck: `<= 50 m`

  </details>
- Bikes and motorcycles are removed from predictions and labels if they fall inside a bike-rack.
- A match of label and prediction is defined based on the 2D center distance on the ground plane. Predictions are matched with labels that have the smallest center-distance up to thresholds of `{0.5, 1.0, 2.0, 4.0} meters`. For each match threshold, average precision (`ap`) is calculated by integrating the recall-precision curve for recalls **and** precisions `> 0.1` (points at or below either threshold are excluded). The mean average precision (`map`) is the average over match thresholds and classes.
- All true positive metrics (`ate`, ...) are calculated using a match threshold of `2.0 m`.

<details>
<summary>Output metrics</summary>

| Metric | Description |
| - | - |
| `ap_0.5_{barrier,traffic_cone,...}` | Average precision for class with maximum match distance of 0.5 meters. |
| `ap_1.0_{barrier,traffic_cone,...}` | Average precision for class with maximum match distance of 1 meter. |
| `ap_2.0_{barrier,traffic_cone,...}` | Average precision for class with maximum match distance of 2 meters. |
| `ap_4.0_{barrier,traffic_cone,...}` | Average precision for class with maximum match distance of 4 meters. |
| `map_{barrier,traffic_cone,...}` | Mean average precision for class over all match distance thresholds. |
| `map` | Mean average precision all match distance thresholds and all classes. |
| `ate_2.0_{barrier,traffic_cone,...}` | Average translation error for class as Euclidean distance in meters. |
| `mate_2.0` | Mean average translation error over all classes. |
| `ase_2.0_{barrier,traffic_cone,...}` | Average scale error for class as `1 - IoU` after aligning centers and orientation. |
| `mase_2.0` | Mean average scale error over all classes. |
| `aoe_2.0_{barrier,traffic_cone,...}` | Average orientation error for class as smallest yaw angle difference between prediction and ground truth in radians. Orientation errors for traffic_cones are ignored and barriers are only evaluated up to 180 degrees. |
| `maoe_2.0` | Mean average orientation error over all classes. |
| `ave_2.0_{barrier,traffic_cone,...}` | Average velocity error for class as absolute velocity error in `m/s`. Velocities for barriers and traffic_cones are ignored. |
| `mave_2.0` | Mean average velocity error over all classes. |
| `aae_2.0_{barrier,traffic_cone,...}` | Average attribute error for class as `1 - attribute_accuracy`. Attribute errors for barriers and traffic_cones are ignored. Returns `null` when the dataset loader does not provide attribute annotations. |
| `maae_2.0` | Mean average attribute error over all classes. In the current implementation this is always a numeric value (defaults to `1.0` if no valid attribute-error values are available). |
| `nds` | nuScenes Detection Score (NDS) combining `map` and five TP error scores with weights 5-1-1-1-1-1, normalised by 10: `NDS = (5·mAP + max(1−mATE,0) + max(1−mASE,0) + max(1−mAOE,0) + max(1−mAVE,0) + max(1−mAAE,0)) / 10`. In the current implementation the denominator is always `10`. |

</details>

### Adding more Evaluations

An evaluation declares the topics it reads and computes metrics from their messages. It may read the topics of the system under test only, e.g. the ego state and the surrounding objects to evaluate a closed-loop planner by its time to collision, or compare them with ground truth, e.g. the predictions of a perception algorithm with the labels of a dataset. The node subscribes to all declared topics, matches their messages into samples by their stamp, and passes the messages of each sample by the names of their topics.

To contribute a new evaluation for a dataset or evaluation protocol:

1. Create a new evaluation class in [autonomy_evaluation/evaluations/](../autonomy_evaluation/autonomy_evaluation/evaluations/) that inherits from `Evaluation`.
2. Declare the topics it reads, each by the name it is configured with, e.g. `prediction:=/topic`, mapped to its ROS message type:
   - `required_inputs()`: the topics of the system under test that are evaluated (required).
   - `required_ground_truth()`: the reference the inputs are compared with, e.g. dataset labels (optional, none by default).
   - `derived_topics()`: topics published next to another one, e.g. meta information on `<label topic>/meta_info`, which follow that topic instead of being configured on their own (optional).
3. Implement `compute_sample_metrics()`, whose parameters are named after the declared topics, and `compute_aggregated_metrics()`. Optionally, implement `visualization_outputs()` and `visualize_sample()` to publish a per-sample visualization for RViz.
4. Configure the evaluation in `__init__` (thresholds, per-class ranges, and metric rules as instance attributes); keep static lookup tables (e.g. category-to-class mappings) as module-level `_CONSTANT_NAME` constants.
5. Register the evaluation in `EVALUATIONS` of [registry.py](../autonomy_evaluation/autonomy_evaluation/evaluations/registry.py) so it can be selected via the `evaluation:=<name>` launch argument, add comprehensive tests in [tests/evaluations/](../autonomy_evaluation/tests/evaluations/) following existing test patterns, and update the documentation with evaluation details, metrics table, dataset requirements, and the topics it reads.
6. Create a [Pull Request](https://github.com/thinking-cars/autonomy_evaluation/pulls) on GitHub and wait for maintainer feedback.

An evaluation that is specific to a system under test can also stay in a package of its own. The node loads it by its module and class, e.g. `evaluation:=my_package.evaluations:TimeToCollision`, as long as the module is importable in the environment of the node:

```python
from typing import Any, Dict, List, Optional

from autonomy_evaluation.evaluations import Evaluation
from perception_msgs.msg import EgoData, ObjectList


class TimeToCollision(Evaluation):
    def __init__(self) -> None:
        super().__init__(name="time_to_collision", description="time to collision of a closed-loop planner")

    def required_inputs(self) -> Dict[str, Any]:
        # inputs only: the metric needs no ground truth
        return {"ego_data": EgoData, "objects": ObjectList}

    def compute_sample_metrics(self, ego_data: EgoData, objects: ObjectList, sample_id: Optional[str] = None) -> Dict[str, Any]:
        return {"ttc": ...}

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"min_ttc": min(entry["metrics"]["ttc"] for entry in sample_results)}
```
