# autonomy_benchmarks

<p align="center">
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/releases/latest"><img src="https://img.shields.io/github/v/release/thinking-cars/autonomy_benchmarks"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thinking-cars/autonomy_benchmarks"/></a>
  <br>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/docker-ros.yml"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/compose-oci.yml"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/compose-oci.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/helm-oci.yml"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/helm-oci.yml/badge.svg"/></a>
  <a href="https://thinking-cars.github.io/autonomy_benchmarks"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/consistency.yml"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

> This repository will be part of the **Autonomy.Hub Ecosystem**

As part of the Autonomy.Hub Ecosystem, **Autonomy.Benchmarks** enables the Automated Driving community to easily benchmark their automated driving building blocks across different tasks and datasets:

- 🔄 **Unified ROS 2 Interface**: Work with multiple datasets using the benefits of the ROS 2 ecosystem
- 📊 **Comprehensive Benchmarks**: Use the provided benchmarks with [Autonomy.Datasets](https://github.com/thinking-cars/autonomy_datasets) to benchmark building blocks across different automated driving tasks
- ⚡ **Efficient Data Pipeline**: Works seamlessly with preprocessed Rosbag files from [Autonomy.Datasets](https://github.com/thinking-cars/autonomy_datasets) for fast execution during development
- 🐳 **Dockerized Environment**: Reproducible setup with all dependencies included
- 🔌 **Modular Architecture**: Easy integration with other ROS 2 packages

## Supported Benchmarks

This repository supports various automated driving evaluation benchmarks.

Detailed metric definitions and computation notes are documented in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

> [**Contributions**](docs/IMPLEMENTATION.md#adding-more-benchmarks) adding more benchmarks are welcome

| Benchmark | Challenge | Dataset | Task |
| --------- | --------- | ------- | ---- |
| [**nuScenes 3D Lidar Object Detection**](docs/IMPLEMENTATION.md#3d-lidar-object-detection) | [![3D Object Detection Challenge](https://img.shields.io/badge/origin-3D_Object_Detection_Challenge-green)](https://www.nuscenes.org/object-detection) | [nuScenes](https://github.com/thinking-cars/autonomy_datasets) | 3D bounding box detection from lidar |

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


## 🚀 Quick Start

Clone [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) and follow its setup instructions to prepare your dataset.

Use the provided [docker-compose.yml](docker-compose.yml) to start the full pipeline — dataset publisher, system-under-test, and benchmark node:

```bash
# enable GUI output from Docker container
xhost +local:

# pull and start Docker containers
export COMPOSE_PROFILES="focalformer3d"  # or 'centerpoint'
docker compose pull
docker compose up -d
# stop containers once finished
docker compose down
```

Configure the benchmark task and dataset via ROS launch arguments in [docker-compose.yml](docker-compose.yml):

```yaml
command: ros2 launch autonomy_benchmarks autonomy_benchmarks.launch.py benchmark:=nuscenes_lidar_object_detection prediction:=$your_prediction_topic label:=$your_label_topic visualize:=true
```

## 💻 Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://github.com/thinking-cars/autonomy_benchmarks.git
    ```
1. Initialize the [`.openads-dev-environment`](https://github.com/openads-project/openads-dev-environment) submodule containing development environment configuration.
    ```bash
    cd autonomy_benchmarks
    git submodule update --init --recursive
    ```
1. Open the repository in [Visual Studio Code](https://code.visualstudio.com).
    ```bash
    code .
    ```
1. Install the recommended VS Code extensions.
    > *Ctrl+Shift+P / Extensions: Show Recommended Extensions / Install Workspace Recommended Extensions (Cloud Download Icon)*
1. Reopen the repository in a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers).
    > *Ctrl+Shift+P / Dev Containers: Rebuild and Reopen in Container*

### Build

> *Ctrl+Shift+B*

```bash
colcon build
```

### Run Tests

> *Ctrl+Shift+P / Tasks: Run Test Task*

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=1
colcon test
colcon test-result --verbose
```


## 📝 Documentation

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://thinking-cars.github.io/autonomy_benchmarks).

| Package | Description |
| --- | --- |
| [autonomy_benchmarks](autonomy_benchmarks/README.md) | Benchmarking suite for automated driving tasks |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

This project is maintained by [Thinking Cars](https://thinking-cars.de). We appreciate contributions and are happy to discuss potential collaborations.
