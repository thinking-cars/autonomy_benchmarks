# `ros2_demo_package`

ROS 2 demo package

## Nodes

### `ros2_demo_node`

```mermaid
flowchart LR
    NODE("ros2_demo_node")
    S0:::hidden -->|~/input| NODE
    SS0:::hidden o--o|~/service| NODE
    NODE -->|~/output| P0:::hidden
    AS0:::hidden o-.-o|~/action| NODE
    classDef hidden display: none;
```

#### Subscribed Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/input` | `geometry_msgs/msg/PointStamped` | input point |

#### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/output` | `geometry_msgs/msg/PointStamped` | output point |

#### Service Servers

| Service | Type | Description |
| --- | --- | --- |
| `~/service` | `std_srvs/srv/SetBool` | demo service |

#### Action Servers

| Action | Type | Description |
| --- | --- | --- |
| `~/action` | `ros2_demo_package_interfaces/action/Fibonacci` | demo action |

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `param` | `float` | `1.0` | demo parameter |

## Launch Files

### [`ros2_demo_node_launch.py`](launch/ros2_demo_node_launch.py)

| Argument | Default | Description |
| --- | --- | --- |
| `input_topic` | `"~/input"` | input topic |
| `output_topic` | `"~/output"` | output topic |
| `service` | `"~/service"` | service topic |
| `name` | `"ros2_demo_node"` | node name |
| `namespace` | `""` | node namespace |
| `params` | `os.path.join(get_package_share_directory("ros2_demo_package"), "config", "params.yml")` | parameter file |
| `log_level` | `"info"` | ros logging level |
| `use_sim_time` | `"false"` | use simulation time |
