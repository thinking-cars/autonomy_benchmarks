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
| `input_topic` | `"~/input"` | TODO |
| `output_topic` | `"~/output"` | TODO |
| `name` | `"autonomy_benchmarks"` | TODO |
| `namespace` | `""` | TODO |
| `params` | `os.path.join(get_package_share_directory("autonomy_benchmarks"), "config", "params.yml")` | TODO |
| `log_level` | `"info"` | TODO |
| `use_sim_time` | `"false"` | TODO |
