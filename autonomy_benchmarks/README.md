# `autonomy_benchmarks`

Benchmarking suite for automated driving tasks

## Nodes

### `autonomy_benchmarks`

The node requests the samples it evaluates from the dataset, using the `request_samples` service
of [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets), which publishes them
and responds once they have been published. By default one sample is requested at a time, so the
dataset only publishes the next sample once the system under test has delivered its output for
the current one and the benchmark has evaluated it. Increase `samples_per_request` to publish
samples in batches, set it to `0` to publish the whole dataset with a single request, or list the
IDs of individual samples in `sample_ids` to evaluate only those.

Once the dataset reports that all requested samples have been published, the per-sample metrics
are aggregated and reported on three levels: for the whole benchmark (`aggregated_metrics`), for
the samples of each scene the dataset published them from (`scene_results`, matched with the
samples via the `published_scene_ids` of the responses), and for every single sample
(`sample_results`). The dataset metrics are logged, and the results of all three levels are
written to a JSON file if `results_path` is set. Samples that are not evaluated within
`evaluation_timeout` seconds of being published, e.g. because the system under test skipped them,
are left out.

```bash
ros2 launch autonomy_benchmarks autonomy_benchmarks.launch.py \
  prediction:=/object_list/prediction \
  label:=/object_list/lidar_01 \
  request_samples:=/datasets/request_samples \
  results_path:=/results/nuscenes_lidar_object_detection.json
```

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `benchmark` | `string` | `nuscenes_lidar_object_detection` | benchmark name |
| `visualize` | `bool` | `false` | publish the per-sample true positives, false positives and false negatives for RViz |
| `samples_per_request` | `int` | `1` | number of samples to request from the dataset at a time; 0 requests all remaining samples at once, 1 evaluates every sample before the next one is published |
| `sample_ids` | `string` | - | comma-separated IDs of the dataset samples to evaluate (e.g. '0,10,20'); if empty, all samples of the dataset are evaluated |
| `evaluation_timeout` | `float` | `60.0` | seconds to wait for a published sample to be evaluated before continuing without it |
| `results_path` | `string` | - | path of the JSON file the benchmark results are written to; results are only logged if empty |

## Launch Files

### [`autonomy_benchmarks.launch.py`](launch/autonomy_benchmarks.launch.py)

| Argument | Default | Description |
| --- | --- | --- |
| `request_samples` | `"~/request_samples"` | service of the dataset node used to request the samples to evaluate |
| `benchmark` | `"nuscenes_lidar_object_detection"` | benchmark to run |
| `name` | `"autonomy_benchmarks"` | node name |
| `namespace` | `""` | node namespace |
| `log_level` | `"info"` | ros logging level |
| `use_sim_time` | `"true"` | use sim time |
| `visualize` | `"false"` | publish the per-sample true positives, false positives and false negatives and open RViz |
| `samples_per_request` | `"1"` | number of samples to request from the dataset at a time (0 requests all remaining samples at once) |
| `sample_ids` | `""` | comma-separated IDs of the dataset samples to evaluate (all samples if empty) |
| `evaluation_timeout` | `"60.0"` | seconds to wait for a published sample to be evaluated before continuing without it |
| `results_path` | `""` | path of the JSON file the benchmark results are written to (results are only logged if empty) |
