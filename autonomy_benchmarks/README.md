# `autonomy_benchmarks`

Benchmarking suite for automated driving tasks

## Nodes

### `autonomy_benchmarks`

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `benchmark` | `string` | `nuscenes_lidar_object_detection` | benchmark name |
| `visualize` | `bool` | `false` | publish the per-sample matching outcome for RViz |

#### Published Topics

Only published while `visualize` is `true`. The objects are the ones the benchmark actually scored, split at the TP metric threshold, so anything a pre-matching filter dropped (non-evaluated class, bike rack, out of class range, GT without sensor points) appears in none of them.

| Topic | Type | Description |
| --- | --- | --- |
| `~/true_positives` | `perception_msgs/ObjectList` | predicted objects that matched a ground-truth object |
| `~/false_positives` | `perception_msgs/ObjectList` | predicted objects that matched nothing |
| `~/false_negatives` | `perception_msgs/ObjectList` | ground-truth objects no prediction matched |

## Launch Files

### [`autonomy_benchmarks.launch.py`](launch/autonomy_benchmarks.launch.py)

| Argument | Default | Description |
| --- | --- | --- |
| `benchmark` | `"nuscenes_lidar_object_detection"` | benchmark to run |
| `name` | `"autonomy_benchmarks"` | node name |
| `namespace` | `""` | node namespace |
| `log_level` | `"info"` | ros logging level |
| `use_sim_time` | `"true"` | use sim time |
| `visualize` | `"false"` | publish the per-sample true positives, false positives and false negatives for RViz |
