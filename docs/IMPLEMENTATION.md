# Implementation Details

This repository supports the following benchmarks for object detection in automated driving systems:

- [Waymo Open Challenges](#waymo-open-challenges): 2D camera object detection, 3D camera-lidar object detection
- [nuScenes Challenge](#nuscenes-challenge): 3D lidar object detection

### Waymo Open Challenges

[![origin](https://img.shields.io/badge/origin-Waymo_Open_Challenges-green)](https://waymo.com/open/challenges/) 
[![non-commercial](https://img.shields.io/badge/license-non--commercial-red)](https://waymo.com/open/terms)

#### 2D Camera Object Detection

> This benchmark uses [Waymo Open Dataset](https://github.com/thinking-cars/autonomy_datasets/docs/IMPLEMENTATION.md#waymo-open-dataset) via the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package.

[![Waymo 2D Detection Challenge](https://img.shields.io/badge/origin-Waymo_2D_Detection_Challenge-green)](https://waymo.com/open/challenges/2020/2d-detection/) ![2020](https://img.shields.io/badge/published-2020-green)

Estimates are treated as matches if the following **intersection-over-union (IoU) thresholds** are reached:

- Vehicle: `iou >= 0.7`
- Pedestrian: `iou >= 0.5`
- Cyclist: `iou >= 0.5`

Note: the label "Sign" is part of the official 2D Detection Challenge (IoU threshold 0.5) but is **excluded from this implementation**. Sign is not part of the `ALL_NS` primary ranking metric on the Waymo leaderboard (which covers Vehicle, Pedestrian, and Cyclist only).

The evaluation is conducted across three distance-based groups, defined by the Euclidean distance between the object center and the vehicle coordinate frame origin: `[0, 35 m), [35 m, 50 m), and [50 m, +inf)`.

Each label is categorized into one of two difficulty levels, *level 1* (`l1`) or *level 2* (`l2`) in the Waymo Open Dataset. Labels of both levels are considered to compute the *level 2* metrics.

| Metric | Description |
| - | - |
| `ap_l1_vehicle` | Average precision of vehicles with level 1 difficulty. |
| `ap_l1_pedestrian` | Average precision of pedestrians with level 1 difficulty. |
| `ap_l1_cyclist` | Average precision of cyclists with level 1 difficulty. |
| `ap_l1_all-ns` | Average precision of vehicles, pedestrians and cyclists with level 1 difficulty. The `-ns` suffix indicates the Sign class is excluded from the mean. |
| `ap_l2_vehicle` | Average precision of vehicles with level 2 difficulty. |
| `ap_l2_pedestrian` | Average precision of pedestrians with level 2 difficulty. |
| `ap_l2_cyclist` | Average precision of cyclists with level 2 difficulty. |
| `ap_l2_all-ns` | Average precision of vehicles, pedestrians and cyclists with level 2 difficulty. The `-ns` suffix indicates the Sign class is excluded from the mean. |

#### 3D Camera Object Detection

> This benchmark uses [Waymo Open Dataset](https://github.com/thinking-cars/autonomy_datasets/docs/IMPLEMENTATION.md#waymo-open-dataset) via the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package.

[![Waymo 2D Detection Challenge](https://img.shields.io/badge/origin-Waymo_3D_Detection_Challenge-green)](https://waymo.com/open/challenges/2020/3d-detection/) ![2020](https://img.shields.io/badge/published-2020-green)

Metrics are computed based on the following assumptions:

- Estimates are treated as matches if the following **intersection-over-union (IoU) thresholds** are reached:
    - Vehicle: `iou >= 0.7`
    - Pedestrian: `iou >= 0.5`
    - Cyclist: `iou >= 0.5`
- Note: the label "Sign" is included in the Waymo Open Dataset but not included in the 3D Detection Challenge.
- The evaluation is conducted across three distance-based groups, defined by the Euclidean distance between the object center and the vehicle coordinate frame origin: `[0, 35 m), [35 m, 50 m), and [50 m, +inf)`.
- Labeled boxes with zero lidar points are not considered during evaluation.
- Each label is categorized into one of two difficulty levels. Labels of both levels are considered to compute the *level 2* metrics.
    - **Level 1** (`l1`) if not marked as *level 2* in the Waymo Open Dataset AND number of lidar points in bounding box > 5
    - **Level 2** (`l2`) if marked as *level 2* in the Waymo Open Dataset OR marked as *level 1* with number of lidar points in bounding box >=1 and <= 5

| Metric | Description |
| - | - |
| `ap_l1_vehicle` | Average precision of vehicles with level 1 difficulty. |
| `ap_l1_pedestrian` | Average precision of pedestrians with level 1 difficulty. |
| `ap_l1_cyclist` | Average precision of cyclists with level 1 difficulty. |
| `ap_l1_all-ns` | Average precision of vehicles, pedestrians and cyclists with level 1 difficulty. The `-ns` suffix indicates the Sign class is excluded from the mean. |
| `aph_l1_vehicle` | Average precision of vehicles with level 1 difficulty weighted by heading. |
| `aph_l1_pedestrian` | Average precision of pedestrians with level 1 difficulty weighted by heading. |
| `aph_l1_cyclist` | Average precision of cyclists with level 1 difficulty weighted by heading. |
| `aph_l1_all-ns` | Average precision of vehicles, pedestrians and cyclists with level 1 difficulty weighted by heading. The `-ns` suffix indicates the Sign class is excluded from the mean. |
| `ap_l2_vehicle` | Average precision of vehicles with level 2 difficulty. |
| `ap_l2_pedestrian` | Average precision of pedestrians with level 2 difficulty. |
| `ap_l2_cyclist` | Average precision of cyclists with level 2 difficulty. |
| `ap_l2_all-ns` | Average precision of vehicles, pedestrians and cyclists with level 2 difficulty. The `-ns` suffix indicates the Sign class is excluded from the mean. |
| `aph_l2_vehicle` | Average precision of vehicles with level 2 difficulty weighted by heading. |
| `aph_l2_pedestrian` | Average precision of pedestrians with level 2 difficulty weighted by heading. |
| `aph_l2_cyclist` | Average precision of cyclists with level 2 difficulty weighted by heading. |
| `aph_l2_all-ns` | Average precision of vehicles, pedestrians and cyclists with level 2 difficulty weighted by heading. The `-ns` suffix indicates the Sign class is excluded from the mean. |

### nuScenes Challenge

> [nuScenes](https://www.nuscenes.org/)

The nuScenes dataset is not redistributed here. You must download it under the [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) and make it available as a TFDS dataset via the companion [autohub-datasets](https://github.com/thinking-cars/autonomy_datasets) package.

#### 3D Lidar Object Detection

> This benchmark uses [nuScenes Dataset](https://github.com/thinking-cars/autonomy_datasets/docs/IMPLEMENTATION.md#nuscenes-dataset) via the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package.

[![3D Object Detection Challenge](https://img.shields.io/badge/origin-3D_Object_Detection_Challenge-green)](https://www.nuscenes.org/object-detection) ![2019](https://img.shields.io/badge/published-2019-green)

Metrics are computed based on the following assumptions:

- Labels are only considered if number of lidar or radar points inside is >= 1.
- Labels and predictions are only considered if they fall into a class-specific detection range:
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
- Bikes and motorcycles are removed from predictions and labels if they fall inside a bike-rack.
- A match of label and prediction is defined based on the 2D center distance on the ground plane. Predictions are matched with labels that have the smallest center-distance up to thresholds of `{0.5, 1.0, 2.0, 4.0} meters`. For each match threshold, average precision (`ap`) is calculated by integrating the recall-precision curve for recalls **and** precisions `> 0.1` (points at or below either threshold are excluded). The mean average precision (`map`) is the average over match thresholds and classes.
- All true positive metrics (`ate`, ...) are calculated using a match threshold of `2.0 m`.

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
| `maae_2.0` | Mean average attribute error over all classes. Returns `null` when attribute annotations are unavailable. |
| `nds` | nuScenes Detection Score (NDS) combining `map` and five TP error scores with weights 5-1-1-1-1-1, normalised by 10: `NDS = (5·mAP + max(1−mATE,0) + max(1−mASE,0) + max(1−mAOE,0) + max(1−mAVE,0) + max(1−mAAE,0)) / 10`. When `maae` is unavailable (attribute data absent), the denominator falls back to 9. |

## Usage

TODO(RaphvK)

### Adding more Benchmarks

1. Create a new dataset adapter based on the existing files [here](../autonomy_datasets/autonomy_datasets/datasets/).
2. Add documentation for the new dataset to this README and add it to the table in the [top-level README](../README.md).
3. Create a [Pull Request](https://github.com/thinking-cars/autonomy_datasets/pulls) on GitHub and wait for maintainer's feedback.
