# `autonomy_benchmarks`

Benchmarking suite for automated driving tasks

## Nodes

### `autonomy_benchmarks`

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `benchmark` | `string` | `nuscenes_lidar_object_detection` | benchmark name |

## Launch Files

### [`autonomy_benchmarks.launch.py`](launch/autonomy_benchmarks.launch.py)

| Argument | Default | Description |
| --- | --- | --- |
| `benchmark` | `"nuscenes_lidar_object_detection"` | benchmark to run |
| `name` | `"autonomy_benchmarks"` | node name |
| `namespace` | `""` | node namespace |
| `log_level` | `"info"` | ros logging level |
| `use_sim_time` | `"true"` | use sim time |
| `prediction` | `"~/prediction"` | `perception_msgs/ObjectList` topic with the predicted objects |
| `label` | `"~/label"` | `perception_msgs/ObjectList` topic with the ground-truth objects |
| `label_meta_info` | `"~/label_meta_info"` | `autonomy_datasets_msgs/ObjectListMetaInfo` topic with the dataset annotations of the ground-truth objects, published by the dataset on `<label topic>/meta_info` |
