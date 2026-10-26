"""P2：由 UAV 航路轨迹构造局部图链路学习数据。

本模块只使用节点自身、HELLO 可交换的一跳邻居统计和候选链路特征。
训练、验证和测试按时间及网络规模分离，避免把同一快照随机拆到不同集合。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .routing_core import NodeState, build_neighbor_graph, distance, relative_speed


@dataclass(frozen=True)
class LinkDataset:
    """局部图链路样本数组。

    参数：
        source_self: 源节点自身特征。
        source_neighbor: 源节点一跳邻居均值特征。
        target_self: 目的节点自身特征。
        target_neighbor: 目的节点一跳邻居均值特征。
        edge: 端点对称的候选链路特征。
        label: 探测窗口得到的链路成功率标签。
        true_probability: 生成标签前的真实成功概率，仅用于受控评估。
        time_index: 样本所属轨迹时刻。
        first_node: 链路第一端节点编号。
        second_node: 链路第二端节点编号。
    """

    source_self: np.ndarray
    source_neighbor: np.ndarray
    target_self: np.ndarray
    target_neighbor: np.ndarray
    edge: np.ndarray
    label: np.ndarray
    true_probability: np.ndarray
    time_index: np.ndarray
    first_node: np.ndarray
    second_node: np.ndarray

    def __len__(self) -> int:
        """返回链路样本数。"""

        return int(self.label.shape[0])

    def model_inputs(self) -> dict[str, np.ndarray]:
        """返回与 TensorFlow 模型输入名一致的数组字典。"""

        return {
            "source_self": self.source_self,
            "source_neighbor": self.source_neighbor,
            "target_self": self.target_self,
            "target_neighbor": self.target_neighbor,
            "edge_features": self.edge,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """训练、验证和测试三个严格分离的数据集合。"""

    train: LinkDataset
    validation: LinkDataset
    test: LinkDataset


def load_flight_routes(directory: Path, maximum_routes: int | None = None) -> list[list[dict[str, float]]]:
    """按数字文件名加载航路 JSON。

    参数：
        directory: 包含 `0.json`、`1.json` 等文件的目录。
        maximum_routes: 最多读取的航路数；`None` 表示全部读取。
    """

    if not directory.exists():
        raise FileNotFoundError(f"航路目录不存在：{directory}")
    files = sorted(directory.glob("*.json"), key=lambda path: int(path.stem))
    if maximum_routes is not None:
        files = files[:maximum_routes]
    if not files:
        raise ValueError("航路目录中没有 JSON 文件")
    routes: list[list[dict[str, float]]] = []
    expected_length: int | None = None
    for path in files:
        with path.open("r", encoding="utf-8") as input_file:
            points = json.load(input_file)
        if expected_length is None:
            expected_length = len(points)
        if len(points) != expected_length:
            raise ValueError(f"航路长度不一致：{path}")
        routes.append(points)
    return routes


def snapshot_states(routes: Sequence[Sequence[Mapping[str, float]]], time_index: int, node_count: int) -> dict[int, NodeState]:
    """把指定时刻的航路点转换为路由核心使用的节点状态。

    参数：
        routes: 已加载航路。
        time_index: 轨迹时刻索引。
        node_count: 使用前多少条航路作为节点。
    """

    if node_count <= 1 or node_count > len(routes):
        raise ValueError("node_count 必须位于 2 和航路总数之间")
    if not 0 <= time_index < len(routes[0]):
        raise ValueError("time_index 超出航路范围")
    states: dict[int, NodeState] = {}
    for node_id, route in enumerate(routes[:node_count]):
        point = route[time_index]
        energy = max(0.35, 1.0 - 0.0012 * time_index - 0.002 * (node_id % 10))
        states[node_id] = NodeState(
            node_id=node_id,
            x=float(point["x"]),
            y=float(point["y"]),
            velocity_x=float(point["v_x"]),
            velocity_y=float(point["v_y"]),
            energy=energy,
        )
    return states


def normalized_node_feature(
    node_id: int,
    states: Mapping[int, NodeState],
    graph: Mapping[int, Sequence[int]],
    position_scale: float,
    velocity_scale: float,
    degree_scale: float,
) -> np.ndarray:
    """生成节点可从自身状态和 HELLO 邻居表得到的六维特征。"""

    if position_scale <= 0 or velocity_scale <= 0 or degree_scale <= 0:
        raise ValueError("特征归一化尺度必须大于 0")
    state = states[node_id]
    return np.asarray(
        [
            state.x / position_scale,
            state.y / position_scale,
            state.velocity_x / velocity_scale,
            state.velocity_y / velocity_scale,
            state.energy,
            min(len(graph.get(node_id, ())) / degree_scale, 1.5),
        ],
        dtype=np.float32,
    )


def neighbor_mean_feature(
    node_id: int,
    node_features: Mapping[int, np.ndarray],
    graph: Mapping[int, Sequence[int]],
) -> np.ndarray:
    """对一跳邻居特征做顺序无关的均值聚合。"""

    neighbors = graph.get(node_id, ())
    if not neighbors:
        return np.zeros_like(node_features[node_id])
    return np.mean([node_features[neighbor] for neighbor in neighbors], axis=0).astype(np.float32)


def edge_feature_vector(
    first: int,
    second: int,
    states: Mapping[int, NodeState],
    graph: Mapping[int, Sequence[int]],
    communication_range: float,
    velocity_scale: float,
    degree_scale: float,
) -> np.ndarray:
    """生成距离、相对运动、航向一致性和公共邻居五维链路特征。"""

    first_state = states[first]
    second_state = states[second]
    displacement_x = second_state.x - first_state.x
    displacement_y = second_state.y - first_state.y
    link_distance = max(math.hypot(displacement_x, displacement_y), 1e-9)
    relative_velocity_x = second_state.velocity_x - first_state.velocity_x
    relative_velocity_y = second_state.velocity_y - first_state.velocity_y
    separating_speed = (
        displacement_x * relative_velocity_x + displacement_y * relative_velocity_y
    ) / link_distance
    first_speed = math.hypot(first_state.velocity_x, first_state.velocity_y)
    second_speed = math.hypot(second_state.velocity_x, second_state.velocity_y)
    if first_speed > 1e-9 and second_speed > 1e-9:
        heading_alignment = (
            first_state.velocity_x * second_state.velocity_x
            + first_state.velocity_y * second_state.velocity_y
        ) / (first_speed * second_speed)
    else:
        heading_alignment = 0.0
    common_neighbors = len(set(graph.get(first, ())).intersection(graph.get(second, ())))
    return np.asarray(
        [
            link_distance / communication_range,
            relative_speed(first_state, second_state) / velocity_scale,
            max(separating_speed, 0.0) / velocity_scale,
            heading_alignment,
            min(common_neighbors / degree_scale, 1.5),
        ],
        dtype=np.float32,
    )


def deterministic_edge_shadow(first: int, second: int) -> float:
    """返回不可由几何特征直接观测的有界链路固定阴影偏置。"""

    low, high = sorted((first, second))
    pseudo_uniform = (math.sin((low + 1) * 12.9898 + (high + 1) * 78.233) + 1.0) / 2.0
    return 0.9 * (pseudo_uniform - 0.5)


def true_link_probability(
    first: int,
    second: int,
    states: Mapping[int, NodeState],
    graph: Mapping[int, Sequence[int]],
    communication_range: float,
    velocity_scale: float,
    degree_scale: float,
    environment_shift: float = 0.0,
) -> float:
    """构造同时受几何、移动、局部拥塞、能量和阴影影响的链路成功率。"""

    edge = edge_feature_vector(
        first,
        second,
        states,
        graph,
        communication_range,
        velocity_scale,
        degree_scale,
    )
    average_degree = (
        len(graph.get(first, ())) + len(graph.get(second, ()))
    ) / (2.0 * degree_scale)
    minimum_energy = min(states[first].energy, states[second].energy)
    logit = (
        2.35
        - 3.1 * float(edge[0]) ** 2
        - 0.65 * float(edge[1])
        - 1.4 * float(edge[2])
        - 1.25 * average_degree
        + 0.35 * float(edge[3])
        + 0.55 * minimum_energy
        + deterministic_edge_shadow(first, second)
        + environment_shift
    )
    probability = 1.0 / (1.0 + math.exp(-logit))
    return min(0.995, max(0.02, probability))


def _empty_dataset(node_dimension: int = 6, edge_dimension: int = 5) -> LinkDataset:
    """建立用于逐步追加前的空数据结构。"""

    return LinkDataset(
        *(np.empty((0, node_dimension), dtype=np.float32) for _ in range(4)),
        np.empty((0, edge_dimension), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
    )


def build_link_dataset(
    routes: Sequence[Sequence[Mapping[str, float]]],
    time_indices: Sequence[int],
    node_count: int,
    communication_range: float,
    position_scale: float,
    velocity_scale: float,
    degree_scale: float,
    probe_count: int,
    seed: int,
    environment_shift: float = 0.0,
) -> LinkDataset:
    """从多个快照生成有监督局部链路样本。

    参数：
        routes: 已加载航路集合。
        time_indices: 数据集使用的轨迹时刻。
        node_count: 每个快照使用的节点数。
        communication_range: 候选链路通信半径。
        position_scale: 位置特征归一化尺度。
        velocity_scale: 速度特征归一化尺度。
        degree_scale: 邻居度归一化尺度。
        probe_count: 每条链路标签使用的 Bernoulli 探测次数。
        seed: 标签采样随机种子。
        environment_shift: 该数据集额外的不可观测环境偏移。
    """

    if min(
        communication_range,
        position_scale,
        velocity_scale,
        degree_scale,
    ) <= 0.0:
        raise ValueError("communication and normalization scales must be positive")
    if probe_count <= 0:
        raise ValueError("probe_count 必须大于 0")
    generator = np.random.default_rng(seed)
    source_self: list[np.ndarray] = []
    source_neighbor: list[np.ndarray] = []
    target_self: list[np.ndarray] = []
    target_neighbor: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    labels: list[float] = []
    probabilities: list[float] = []
    times: list[int] = []
    first_nodes: list[int] = []
    second_nodes: list[int] = []

    for time_index in time_indices:
        states = snapshot_states(routes, time_index, node_count)
        graph = build_neighbor_graph(states, communication_range)
        node_rows = {
            node_id: normalized_node_feature(
                node_id,
                states,
                graph,
                position_scale,
                velocity_scale,
                degree_scale,
            )
            for node_id in states
        }
        neighbor_rows = {
            node_id: neighbor_mean_feature(node_id, node_rows, graph) for node_id in states
        }
        for first in sorted(graph):
            for second in graph[first]:
                if first >= second:
                    continue
                probability = true_link_probability(
                    first,
                    second,
                    states,
                    graph,
                    communication_range,
                    velocity_scale,
                    degree_scale,
                    environment_shift,
                )
                observed_rate = float(generator.binomial(probe_count, probability) / probe_count)
                source_self.append(node_rows[first])
                source_neighbor.append(neighbor_rows[first])
                target_self.append(node_rows[second])
                target_neighbor.append(neighbor_rows[second])
                edge_rows.append(
                    edge_feature_vector(
                        first,
                        second,
                        states,
                        graph,
                        communication_range,
                        velocity_scale,
                        degree_scale,
                    )
                )
                labels.append(observed_rate)
                probabilities.append(probability)
                times.append(time_index)
                first_nodes.append(first)
                second_nodes.append(second)

    if not labels:
        return _empty_dataset()
    return LinkDataset(
        np.stack(source_self),
        np.stack(source_neighbor),
        np.stack(target_self),
        np.stack(target_neighbor),
        np.stack(edge_rows),
        np.asarray(labels, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
        np.asarray(times, dtype=np.int32),
        np.asarray(first_nodes, dtype=np.int32),
        np.asarray(second_nodes, dtype=np.int32),
    )


def build_default_bundle(routes: Sequence[Sequence[Mapping[str, float]]], seed: int) -> DatasetBundle:
    """建立论文默认的时间外推和规模外推数据划分。"""

    common = {
        "routes": routes,
        "communication_range": 200.0,
        "position_scale": 2000.0,
        "velocity_scale": 25.0,
        "degree_scale": 20.0,
        "probe_count": 20,
    }
    train = build_link_dataset(
        time_indices=range(0, 181, 2),
        node_count=60,
        seed=seed,
        environment_shift=0.0,
        **common,
    )
    validation = build_link_dataset(
        time_indices=range(181, 241, 2),
        node_count=80,
        seed=seed + 1,
        environment_shift=-0.10,
        **common,
    )
    test = build_link_dataset(
        time_indices=range(241, 301),
        node_count=100,
        seed=seed + 2,
        environment_shift=-0.25,
        **common,
    )
    return DatasetBundle(train, validation, test)


def _ramp(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if start < end:
        return np.clip((values - start) / (end - start), 0.0, 1.0)
    return np.clip((start - values) / (start - end), 0.0, 1.0)


def _triangle(values: np.ndarray, left: float, peak: float, right: float) -> np.ndarray:
    ascending = np.clip((values - left) / max(peak - left, 1e-12), 0.0, 1.0)
    descending = np.clip((right - values) / max(right - peak, 1e-12), 0.0, 1.0)
    return np.minimum(ascending, descending)


def fqaodv_rule_surface(
    energy: np.ndarray,
    mobility: np.ndarray,
    link_quality: np.ndarray,
    centroid_points: int = 100,
) -> np.ndarray:
    """Reproduce the 27-rule Mamdani surface in the supplied FQ-AODV source.

    Membership breakpoints, min conjunction, algebraic-product implication,
    max aggregation, and 100-point centroid match ``fq-aodv-fuzzy.cc``.  This
    makes the comparison a source-level fuzzy-logic baseline rather than an
    invented weighted average.
    """

    energy = np.clip(np.asarray(energy, dtype=float), 0.0, 1.0)
    mobility = np.clip(np.asarray(mobility, dtype=float), 0.0, 1.0)
    link_quality = np.clip(np.asarray(link_quality, dtype=float), 0.0, 1.0)
    if not (energy.shape == mobility.shape == link_quality.shape):
        raise ValueError("fuzzy inputs must have identical shapes")

    energy_membership = {
        "high": _ramp(energy, 0.2, 0.4),
        "mid": _triangle(energy, 0.0, 0.2, 0.4),
        "low": _ramp(energy, 0.2, 0.0),
    }
    mobility_membership = {
        "slow": _ramp(mobility, 0.6, 0.8),
        "mid": _triangle(mobility, 0.4, 0.6, 0.8),
        "fast": _ramp(mobility, 0.6, 0.4),
    }
    link_membership = {
        "good": _ramp(link_quality, 0.8, 1.0),
        "mid": _triangle(link_quality, 0.6, 0.8, 1.0),
        "bad": _ramp(link_quality, 0.8, 0.6),
    }
    rules = (
        ("high", "slow", "good", "perfect"),
        ("high", "slow", "mid", "good"),
        ("high", "slow", "bad", "unpreferable"),
        ("high", "mid", "good", "good"),
        ("high", "mid", "mid", "acceptable"),
        ("high", "mid", "bad", "bad"),
        ("high", "fast", "good", "unpreferable"),
        ("high", "fast", "mid", "bad"),
        ("high", "fast", "bad", "veryBad"),
        ("mid", "slow", "good", "good"),
        ("mid", "slow", "mid", "acceptable"),
        ("mid", "slow", "bad", "bad"),
        ("mid", "mid", "good", "acceptable"),
        ("mid", "mid", "mid", "unpreferable"),
        ("mid", "mid", "bad", "bad"),
        ("mid", "fast", "good", "bad"),
        ("mid", "fast", "mid", "bad"),
        ("mid", "fast", "bad", "veryBad"),
        ("low", "slow", "good", "unpreferable"),
        ("low", "slow", "mid", "bad"),
        ("low", "slow", "bad", "veryBad"),
        ("low", "mid", "good", "bad"),
        ("low", "mid", "mid", "bad"),
        ("low", "mid", "bad", "veryBad"),
        ("low", "fast", "good", "bad"),
        ("low", "fast", "mid", "veryBad"),
        ("low", "fast", "bad", "veryBad"),
    )

    output_grid = np.linspace(0.0, 1.0, centroid_points, dtype=float)
    output_membership = {
        "perfect": _ramp(output_grid, 0.8, 1.0),
        "good": _triangle(output_grid, 0.6, 0.8, 1.0),
        "acceptable": _triangle(output_grid, 0.4, 0.6, 0.8),
        "unpreferable": _triangle(output_grid, 0.2, 0.4, 0.6),
        "bad": _triangle(output_grid, 0.0, 0.2, 1.4),
        "veryBad": _ramp(output_grid, 0.2, 0.0),
    }
    aggregate = np.zeros(energy.shape + (centroid_points,), dtype=float)
    for energy_name, mobility_name, link_name, output_name in rules:
        activation = np.minimum.reduce(
            (
                energy_membership[energy_name],
                mobility_membership[mobility_name],
                link_membership[link_name],
            )
        )
        implied = activation[..., None] * output_membership[output_name]
        aggregate = np.maximum(aggregate, implied)
    denominator = np.sum(aggregate, axis=-1)
    centroid = np.sum(aggregate * output_grid, axis=-1) / np.maximum(denominator, 1e-12)
    return np.clip(centroid, 0.0, 1.0).astype(np.float32)


def fqaodv_preference_scores(dataset: LinkDataset) -> np.ndarray:
    """用因果快照代理量计算 FQ-AODV 规则面的路由偏好分数。

    原 NS-3 实现把能量/队列状态、链路持续度和 HELLO 接收率输入 27 条
    Mamdani 规则。受控航迹没有分组队列与 HELLO 接收窗口，因此这里分别用
    端点最小能量、相对运动质量和距离/邻居负载构造同时刻代理量。输出只用于
    路由排序，不解释为经过校准的链路成功概率，也不用于 Brier 概率比较。
    """

    energy = np.minimum(dataset.source_self[:, 4], dataset.target_self[:, 4])
    mobility = np.clip(1.0 - 0.65 * dataset.edge[:, 1] - 0.35 * dataset.edge[:, 2], 0.0, 1.0)
    mean_degree = 0.5 * (dataset.source_self[:, 5] + dataset.target_self[:, 5])
    link_quality = np.clip(1.0 - 0.75 * dataset.edge[:, 0] - 0.25 * mean_degree, 0.0, 1.0)
    return np.clip(fqaodv_rule_surface(energy, mobility, link_quality), 0.02, 0.995)


def save_dataset_metadata(bundle: DatasetBundle, output_path: Path) -> None:
    """保存三个数据划分的样本量、时间范围和节点规模说明。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, dataset, nodes in (
        ("train", bundle.train, 60),
        ("validation", bundle.validation, 80),
        ("test", bundle.test, 100),
    ):
        rows.append(
            {
                "split": name,
                "samples": len(dataset),
                "node_count": nodes,
                "time_min": int(dataset.time_index.min()),
                "time_max": int(dataset.time_index.max()),
                "mean_true_probability": float(dataset.true_probability.mean()),
            }
        )
    import csv

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

