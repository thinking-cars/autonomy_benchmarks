# openads_ros2_demo_repository

<p align="center">
  <a href="https://github.com/openads-project"><img src="https://img.shields.io/badge/OpenADS-f5ff01"/></a>
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/openads-project/openads_ros2_demo_repository/releases/latest"><img src="https://img.shields.io/github/v/release/openads-project/openads_ros2_demo_repository"/></a>
  <a href="https://github.com/openads-project/openads_ros2_demo_repository/blob/main/LICENSE"><img src="https://img.shields.io/github/license/openads-project/openads_ros2_demo_repository"/></a>
  <br>
  <a href="https://github.com/openads-project/openads_ros2_demo_repository/actions/workflows/docker-ros.yml"><img src="https://github.com/openads-project/openads_ros2_demo_repository/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://openads-project.github.io/openads_ros2_demo_repository"><img src="https://github.com/openads-project/openads_ros2_demo_repository/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/openads_ros2_demo_repository/actions/workflows/consistency.yml"><img src="https://github.com/openads-project/openads_ros2_demo_repository/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

**Demo repository for an OpenADS module**

This repository serves as a demo for an OpenADS module, showcasing the structure and documentation style for OpenADS packages. It includes a simple ROS 2 node that subscribes to a topic, processes the data, and publishes the result. This is a short description of the repository and its purpose.

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/etsi_its_messages/refs/heads/main/assets/teaser.gif" width=800>

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


## 🚀 Quick Start

1. Start a container of the pre-built runtime image.
    ```bash
    docker run --rm -it ghcr.io/openads-project/openads_ros2_demo_repository:latest bash
    ```
1. Inside the container, launch the pre-built nodes.
    ```bash
    ros2 launch ros2_demo_package ros2_demo_node_launch.py
    ```

## 💻 Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://github.com/openads-project/openads_ros2_demo_repository.git
    ```
1. Initialize the [`.openads-dev-environment`](https://github.com/openads-project/openads-dev-environment) submodule containing development environment configuration.
    ```bash
    cd openads_ros2_demo_repository
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

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://openads-project.github.io/openads_ros2_demo_repository).

| Package | Description |
| --- | --- |
| [ros2_demo_package](ros2_demo_package/README.md) | ROS 2 demo package |
| [ros2_demo_package_interfaces](ros2_demo_package_interfaces/README.md) | ROS interface definitions for ros2_demo_package |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).

Development and maintenance of this repository are supported by the following projects. We acknowledge the funding of the respective institutions.

| Project | Funding Institution | Grant Number |
| --- | --- | --- |
| Demo Project | Demo Funding Institution | Demo Grant Number |

<p>
  <img src="https://www.drought.uni-freiburg.de/stressres/images/bmftr-logo/image" height=70>
  <img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_funded_en.jpg" height=70>
</p>

<sub><sup>Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor CINEA can be held responsible for them.</sup></sub>
