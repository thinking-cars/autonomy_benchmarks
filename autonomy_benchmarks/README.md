# `autonomy_benchmarks`

Benchmarking suite for automated driving tasks

## Nodes

### `autonomy_benchmarks`

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `benchmark` | `string` | `nuscenes_lidar_object_detection` | benchmark name |
| `visualize` | `bool` | `false` | publish the per-sample true positives, false positives and false negatives for RViz |

## Launch Files

### [`autonomy_benchmarks.launch.py`](launch/autonomy_benchmarks.launch.py)

| Argument | Default | Description |
| --- | --- | --- |
| `benchmark` | `"nuscenes_lidar_object_detection"` | benchmark to run |
| `name` | `"autonomy_benchmarks"` | node name |
| `namespace` | `""` | node namespace |
| `log_level` | `"info"` | ros logging level |
| `use_sim_time` | `"true"` | use sim time |
| `visualize` | `"false"` | publish the per-sample true positives, false positives and false negatives and open RViz |
