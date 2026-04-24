# autonomy_benchmarks

<p align="center">
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/releases/latest"><img src="https://img.shields.io/github/v/release/thinking-cars/autonomy_benchmarks"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thinking-cars/autonomy_benchmarks"/></a>
  <br>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/docker-ros.yml"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://thinking-cars.github.io/autonomy_benchmarks"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/consistency.yml"><img src="https://github.com/thinking-cars/autonomy_benchmarks/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

> This repository will be part of the **Autonomy.Hub Ecosystem**

**Autonomy.Hub** enables the Automated Driving research community to easily benchmark their automated driving building blocks across different tasks and datasets:

- 🔄 **Unified ROS 2 Interface**: Work with multiple datasets using the benefits of the ROS 2 ecosystem
- 📊 **Comprehensive Benchmarks**: Use the provided datasets with [Autonomy.Benchmarks](https://github.com/thinking-cars/autonomy_benchmarks) to benchmark building blocks across different automated driving tasks
- ⚡ **Efficient Data Pipeline**: Preprocessed Rosbag files ensure fast execution during development
- 🐳 **Dockerized Environment**: Reproducible setup with all dependencies included
- 🔌 **Modular Architecture**: Easy integration with other ROS 2 packages

## Supported Benchmarks

This repository supports the following benchmarks for object detection in automated driving systems:

- [Waymo Open Challenges](docs/IMPLEMENTATION.md#waymo-open-challenges): 2D camera object detection, 3D camera-lidar object detection
- [nuScenes Challenge](docs/IMPLEMENTATION.md#nuscenes-challenge): 3D lidar object detection

New benchmarks can easily be added by subclassing `AutonomyBenchmark` as described in [Adding more Benchmarks](docs/IMPLEMENTATION.md#adding-more-benchmarks).

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


## 🚀 Quick Start

1. Start a container of the pre-built runtime image.
    ```bash
    docker run --rm -it ghcr.io/thinking-cars/autonomy_benchmarks:latest bash
    ```
1. Inside the container, launch the pre-built nodes.
    ```bash
    ros2 launch autonomy_benchmarks autonomy_benchmarks.launch.py
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

This project is maintained by [Thinking Cars](http://thinking-cars.de). We appreciate contributions and are happy to discuss potential collaborations.
