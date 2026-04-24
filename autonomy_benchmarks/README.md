# `autonomy_benchmarks`

Benchmarking suite for automated driving tasks

- [Container Images](#container-images)
- [autonomy_benchmarks](#autonomy_benchmarks)

### Container Images

| Description | Image:Tag | Default Command |
| --- | --- | -- |
|  |  |  |

## Launch Files

### [`autonomy_benchmarks.launch.py`](launch/autonomy_benchmarks.launch.py)

| Argument | Default | Description |
| --- | --- | --- |
| `benchmark` | `"nuscenes_lidar_object_detection"` | benchmark name |
| `name` | `"autonomy_benchmarks"` | node name |
| `namespace` | `""` | node namespace |
| `params` | `os.path.join(get_package_share_directory("autonomy_benchmarks"), "config", "params.yml")` | node parameter file path |
| `log_level` | `"info"` | ros logging level |
| `use_sim_time` | `"true"` | use simulation time |
