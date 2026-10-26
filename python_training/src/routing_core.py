"""P2：低空动态图路由、局部图评分与反馈校准基础实现。

当前 `LocalMessagePassingEvaluator` 是可测试的 GNN 前向骨架，参数尚未训练。
后续 TensorFlow 训练器只需输出同形状权重，不改变动态图和路由接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import heapq
import math
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class NodeState:
    """无人机节点的局部状态。"""

    node_id: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    energy: float = 1.0


@dataclass(frozen=True)
class AodvRouteCandidate:
    """Minimal AODV state needed to keep learning outside safety invariants.

    The learned accumulated cost is inspected only after validity, destination
    sequence-number freshness, and hop count.  It therefore replaces the fuzzy
    local metric inside an AODV-safe equivalence class and cannot make a stale or
    longer equal-sequence route overwrite the standard route-table invariant.
    """

    next_hop: int
    destination_sequence: int
    valid: bool
    accumulated_cost: float
    hop_count: int


def prefer_aodv_candidate(
    current: AodvRouteCandidate | None,
    candidate: AodvRouteCandidate,
) -> bool:
    """Return whether AODV may replace ``current`` by ``candidate``.

    This lexicographic predicate is the implementation counterpart of the
    loop-freedom theorem: validity and sequence number precede the learned metric.
    Within an equal sequence number, the standard lower-hop predicate is retained.
    The accumulated negative-log GNN cost replaces fuzzy scoring only when sequence
    number and hop count are equal.
    """

    if not candidate.valid:
        return False
    if current is None or not current.valid:
        return True
    if candidate.destination_sequence != current.destination_sequence:
        return candidate.destination_sequence > current.destination_sequence
    if candidate.hop_count != current.hop_count:
        return candidate.hop_count < current.hop_count
    if not math.isclose(candidate.accumulated_cost, current.accumulated_cost):
        return candidate.accumulated_cost < current.accumulated_cost
    return candidate.next_hop < current.next_hop


@dataclass
class FeedbackAdapter:
    """对 GNN 链路评分增加有界在线 logit 残差校准。

    参数：
        learning_rate: 每次 ACK/重传反馈的更新步长。
        maximum_bias: logit 校准偏置的绝对值上限。
    """

    learning_rate: float = 0.2
    maximum_bias: float = 0.3

    def __post_init__(self) -> None:
        self.biases: dict[tuple[int, int], float] = {}

    def adapted_score(self, edge: tuple[int, int], predicted_score: float) -> float:
        """返回经有界 logit 偏置校准后的链路成功概率。"""

        probability = _clip(predicted_score, 1e-7, 1.0 - 1e-7)
        return _sigmoid(_logit(probability) + self.biases.get(edge, 0.0))

    def update(self, edge: tuple[int, int], predicted_score: float, observed_success: float) -> float:
        """用 Bernoulli 对数损失的一步投影梯度更新并返回新偏置。"""

        adapted = self.adapted_score(edge, predicted_score)
        old_bias = self.biases.get(edge, 0.0)
        new_bias = _clip(
            old_bias + self.learning_rate * (observed_success - adapted),
            -self.maximum_bias,
            self.maximum_bias,
        )
        self.biases[edge] = new_bias
        return new_bias


@dataclass(frozen=True)
class RouteDiscoveryResult:
    """一次局部度量洪泛的路径和控制开销结果。

    参数：
        route: 源到目的节点的最低累计代价路径。
        rreq_broadcasts: 节点在度量改善后执行的广播次数。
        metric_improvements: 已有节点标签被更优累计代价更新的次数。
        destination_cost: 目的节点最终累计代价；不可达时为正无穷。
    """

    route: tuple[int, ...]
    rreq_broadcasts: int
    metric_improvements: int
    destination_cost: float


class LocalMessagePassingEvaluator:
    """一层均值聚合的局部 GNN 链路评估器。

    该前向计算只使用节点自身、一跳邻居均值和候选链路特征，因此可以
    中心训练后在每个无人机节点分布式部署。邻居采用均值聚合，顺序不影响输出。
    """

    def __init__(self) -> None:
        """使用确定性初始权重建立 smoke-test 前向网络。"""

        self.self_weights = (
            (0.7, 0.0, 0.2, 0.0, 0.1),
            (0.0, 0.7, 0.0, 0.2, 0.1),
            (0.1, 0.1, -0.2, -0.2, 0.8),
        )
        self.neighbor_weights = (
            (0.2, 0.0, 0.1, 0.0, 0.0),
            (0.0, 0.2, 0.0, 0.1, 0.0),
            (0.0, 0.0, -0.1, -0.1, 0.2),
        )
        self.link_weights = (0.2, 0.2, 0.6, 0.2, 0.2, 0.6, -2.5, -0.8, 0.5)
        self.link_bias = 1.5

    def node_embedding(
        self,
        node_id: int,
        states: Mapping[int, NodeState],
        neighbors: Mapping[int, Sequence[int]],
        position_scale: float,
        velocity_scale: float,
    ) -> tuple[float, float, float]:
        """计算一个节点的局部消息传递嵌入。

        参数：
            node_id: 当前节点编号。
            states: 节点编号到状态的映射。
            neighbors: 每个节点的一跳邻居列表。
            position_scale: 位置归一化尺度。
            velocity_scale: 速度归一化尺度。
        """

        own = node_features(states[node_id], position_scale, velocity_scale)
        neighbor_ids = neighbors.get(node_id, ())
        if neighbor_ids:
            neighbor_vectors = [node_features(states[index], position_scale, velocity_scale) for index in neighbor_ids]
            mean_neighbor = tuple(
                sum(vector[column] for vector in neighbor_vectors) / len(neighbor_vectors)
                for column in range(len(own))
            )
        else:
            mean_neighbor = (0.0,) * len(own)
        return tuple(
            math.tanh(_dot(self_row, own) + _dot(neighbor_row, mean_neighbor))
            for self_row, neighbor_row in zip(self.self_weights, self.neighbor_weights)
        )

    def score(
        self,
        first: int,
        second: int,
        states: Mapping[int, NodeState],
        neighbors: Mapping[int, Sequence[int]],
        communication_range: float,
        velocity_scale: float,
    ) -> float:
        """输出候选链路成功概率评分。"""

        first_embedding = self.node_embedding(
            first, states, neighbors, communication_range, velocity_scale
        )
        second_embedding = self.node_embedding(
            second, states, neighbors, communication_range, velocity_scale
        )
        distance_ratio = distance(states[first], states[second]) / communication_range
        relative_speed_ratio = relative_speed(states[first], states[second]) / velocity_scale
        minimum_energy = min(states[first].energy, states[second].energy)
        feature_vector = (
            *first_embedding,
            *second_embedding,
            distance_ratio,
            relative_speed_ratio,
            minimum_energy,
        )
        return _sigmoid(_dot(self.link_weights, feature_vector) + self.link_bias)


def distance(first: NodeState, second: NodeState) -> float:
    """返回两个节点的欧氏距离。"""

    return math.hypot(first.x - second.x, first.y - second.y)


def relative_speed(first: NodeState, second: NodeState) -> float:
    """返回两个节点的相对速度大小。"""

    return math.hypot(first.velocity_x - second.velocity_x, first.velocity_y - second.velocity_y)


def build_neighbor_graph(
    states: Mapping[int, NodeState], communication_range: float
) -> dict[int, list[int]]:
    """根据通信半径构造无向一跳邻居图。"""

    if communication_range <= 0:
        raise ValueError("communication_range 必须大于 0")
    graph = {node_id: [] for node_id in states}
    node_ids = sorted(states)
    for position, first in enumerate(node_ids):
        for second in node_ids[position + 1 :]:
            if distance(states[first], states[second]) <= communication_range:
                graph[first].append(second)
                graph[second].append(first)
    return graph


def node_features(
    state: NodeState, position_scale: float, velocity_scale: float
) -> tuple[float, float, float, float, float]:
    """把位置、速度和能量归一化为节点特征。"""

    if position_scale <= 0 or velocity_scale <= 0:
        raise ValueError("归一化尺度必须大于 0")
    return (
        state.x / position_scale,
        state.y / position_scale,
        state.velocity_x / velocity_scale,
        state.velocity_y / velocity_scale,
        _clip(state.energy, 0.0, 1.0),
    )


def fuzzy_link_score(
    first: NodeState, second: NodeState, communication_range: float, velocity_scale: float
) -> float:
    """实现无需训练的模糊逻辑风格链路评分基线。"""

    distance_quality = _clip(1.0 - distance(first, second) / communication_range, 0.0, 1.0)
    mobility_quality = _clip(1.0 - relative_speed(first, second) / velocity_scale, 0.0, 1.0)
    energy_quality = min(first.energy, second.energy)
    return _clip(0.5 * distance_quality + 0.3 * mobility_quality + 0.2 * energy_quality, 0.0, 1.0)


def minimum_cost_route(
    graph: Mapping[int, Sequence[int]],
    source: int,
    destination: int,
    score_function: Callable[[int, int], float],
    hop_penalty: float = 0.05,
) -> list[int]:
    """使用 AODV 路由发现可获得的局部链路分数寻找最低代价路径。

    链路代价为 `-log(score)+hop_penalty`，分数越高，路径越优。
    """

    if source not in graph or destination not in graph:
        raise ValueError("源或目的节点不在图中")
    distances = {source: 0.0}
    previous: dict[int, int] = {}
    queue = [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances.get(node):
            continue
        if node == destination:
            break
        for neighbor in graph[node]:
            score = _clip(score_function(node, neighbor), 1e-9, 1.0)
            candidate = cost - math.log(score) + hop_penalty
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if destination not in distances:
        return []
    route = [destination]
    while route[-1] != source:
        route.append(previous[route[-1]])
    route.reverse()
    return route


def metric_route_discovery(
    graph: Mapping[int, Sequence[int]],
    source: int,
    destination: int,
    score_function: Callable[[int, int], float],
    hop_penalty: float = 0.05,
    improvement_tolerance: float = 1e-12,
) -> RouteDiscoveryResult:
    """用只依赖邻居广播的 label-correcting 度量洪泛寻找最低代价路径。

    该函数模拟单次 discovery ID 内的 RREQ 式累计度量传播：节点仅在收到
    更低累计代价时更新反向前驱并再次广播。它不是 RFC 3561 的完整实现，
    但避免用全局 Dijkstra 冒充协议执行，并显式给出广播开销。由于链路代价
    ``-log(score)+hop_penalty`` 非负，异步松弛收敛到与集中式最短路相同的解。
    """

    if source not in graph or destination not in graph:
        raise ValueError("源或目的节点不在图中")
    if hop_penalty < 0 or improvement_tolerance < 0:
        raise ValueError("hop_penalty 和 improvement_tolerance 不能为负")
    best_cost = {source: 0.0}
    predecessor: dict[int, int] = {}
    active = deque([source])
    queued = {source}
    broadcasts = 0
    improvements = 0
    while active:
        node = active.popleft()
        queued.discard(node)
        broadcasts += 1
        for neighbor in graph.get(node, ()):
            score = _clip(score_function(node, neighbor), 1e-9, 1.0)
            candidate = best_cost[node] - math.log(score) + hop_penalty
            if candidate + improvement_tolerance < best_cost.get(neighbor, math.inf):
                if neighbor in best_cost:
                    improvements += 1
                best_cost[neighbor] = candidate
                predecessor[neighbor] = node
                if neighbor not in queued:
                    active.append(neighbor)
                    queued.add(neighbor)
    if destination not in best_cost:
        return RouteDiscoveryResult((), broadcasts, improvements, math.inf)
    route = [destination]
    while route[-1] != source:
        route.append(predecessor[route[-1]])
    route.reverse()
    return RouteDiscoveryResult(tuple(route), broadcasts, improvements, best_cost[destination])


def minimum_hop_metric_route(
    graph: Mapping[int, Sequence[int]],
    source: int,
    destination: int,
    score_function: Callable[[int, int], float],
) -> list[int]:
    """Minimize GNN/fuzzy cost inside the standard minimum-hop path class.

    Breadth-first distances retain AODV's hop-count ordering.  Dynamic
    programming then lets the learned score replace fuzzy logic solely among
    paths with identical freshness and hop count.
    """

    if source not in graph or destination not in graph:
        raise ValueError("source or destination is absent from the graph")

    def distances_from(root: int) -> dict[int, int]:
        distances = {root: 0}
        frontier = deque([root])
        while frontier:
            node = frontier.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    frontier.append(neighbor)
        return distances

    from_source = distances_from(source)
    if destination not in from_source:
        return []
    to_destination = distances_from(destination)
    shortest_hops = from_source[destination]
    costs = {source: 0.0}
    predecessor: dict[int, int] = {}
    ordered = sorted(from_source, key=lambda node: (from_source[node], node))
    for node in ordered:
        if node not in costs:
            continue
        for neighbor in graph.get(node, ()):
            if from_source.get(neighbor) != from_source[node] + 1:
                continue
            if from_source[neighbor] + to_destination.get(neighbor, shortest_hops + 1) != shortest_hops:
                continue
            edge_cost = -math.log(_clip(score_function(node, neighbor), 1e-9, 1.0))
            candidate = costs[node] + edge_cost
            if candidate < costs.get(neighbor, math.inf):
                costs[neighbor] = candidate
                predecessor[neighbor] = node
    if destination not in costs:
        return []
    route = [destination]
    while route[-1] != source:
        route.append(predecessor[route[-1]])
    route.reverse()
    return route


def route_reliability(route: Sequence[int], score_function: Callable[[int, int], float]) -> float:
    """用链路评分乘积计算一条路径的预测成功率。"""

    if len(route) < 2:
        return 1.0
    reliability = 1.0
    for first, second in zip(route, route[1:]):
        reliability *= _clip(score_function(first, second), 0.0, 1.0)
    return reliability


def _dot(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(probability: float) -> float:
    """返回有界概率的对数几率。"""

    probability = _clip(probability, 1e-12, 1.0 - 1e-12)
    return math.log(probability / (1.0 - probability))


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))

