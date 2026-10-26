"""Multi-agent cooperative link perception for embodied low-altitude routing.

Each UAV contributes a compact local state through one-hop HELLO exchange.
Every endpoint combines its own state with a permutation-invariant mean of
neighbor states.  A shared, endpoint-symmetric neural model then predicts only
the one-hop delivery prior; it never selects a next hop or an end-to-end route.

The module intentionally keeps sensing, learning, and routing separate:

* sensing constructs local 3-D motion and neighborhood features;
* a small shared neural model estimates one-hop packet success probability;
* relative link lifetime is computed analytically from endpoint kinematics; and
* the protocol simulator composes positive link costs into a complete route.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .flight_graph_data import (
    DatasetBundle,
    LinkDataset,
    deterministic_edge_shadow,
)
from .mobility_lifetime import (
    build_3d_neighbor_graph,
    link_hold_quality,
    predicted_link_lifetime_s,
)


NODE_DIMENSION = 8
EDGE_DIMENSION = 6
ENDPOINT_PAIR_INPUT_DIMENSION = 2 * NODE_DIMENSION + EDGE_DIMENSION
COOPERATIVE_INPUT_DIMENSION = 4 * NODE_DIMENSION + EDGE_DIMENSION


@dataclass(frozen=True)
class EmbodiedNodeState:
    """Locally sensed 3-D motion and slowly varying node state."""

    node_id: int
    x: float
    y: float
    z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    energy: float

    @property
    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def velocity(self) -> tuple[float, float, float]:
        return self.velocity_x, self.velocity_y, self.velocity_z


def snapshot_embodied_states(
    routes: Sequence[Sequence[Mapping[str, float]]],
    time_index: int,
    node_count: int,
) -> dict[int, EmbodiedNodeState]:
    """Return the 3-D state sensed by every UAV at one trajectory sample."""

    if node_count <= 1 or node_count > len(routes):
        raise ValueError("node_count must lie between 2 and the route count")
    if not 0 <= time_index < len(routes[0]):
        raise ValueError("time_index lies outside the trajectory")
    states: dict[int, EmbodiedNodeState] = {}
    for node_id, route in enumerate(routes[:node_count]):
        point = route[time_index]
        energy = max(0.35, 1.0 - 0.0012 * time_index - 0.002 * (node_id % 10))
        states[node_id] = EmbodiedNodeState(
            node_id=node_id,
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point.get("z", 0.0)),
            velocity_x=float(point["v_x"]),
            velocity_y=float(point["v_y"]),
            velocity_z=float(point.get("v_z", 0.0)),
            energy=energy,
        )
    return states


def build_embodied_neighbor_graph(
    states: Mapping[int, EmbodiedNodeState],
    communication_range: float,
) -> dict[int, list[int]]:
    """Build the candidate graph from the sensed 3-D communication sphere."""

    return build_3d_neighbor_graph(
        {node: state.position for node, state in states.items()},
        communication_range,
    )


def normalized_node_feature(
    node_id: int,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    horizontal_position_scale: float,
    altitude_scale: float,
    velocity_scale: float,
    degree_scale: float,
) -> np.ndarray:
    """Encode one agent's embodied state and local neighborhood density."""

    if min(
        horizontal_position_scale,
        altitude_scale,
        velocity_scale,
        degree_scale,
    ) <= 0.0:
        raise ValueError("feature normalization scales must be positive")
    state = states[node_id]
    return np.asarray(
        [
            state.x / horizontal_position_scale,
            state.y / horizontal_position_scale,
            state.z / altitude_scale,
            state.velocity_x / velocity_scale,
            state.velocity_y / velocity_scale,
            state.velocity_z / velocity_scale,
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
    """Aggregate one-hop agent reports without depending on neighbor order."""

    neighbors = graph.get(node_id, ())
    if not neighbors:
        return np.zeros_like(node_features[node_id])
    return np.mean(
        [node_features[neighbor] for neighbor in neighbors],
        axis=0,
    ).astype(np.float32)


def symmetric_link_feature(
    first: int,
    second: int,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range: float,
    velocity_scale: float,
    degree_scale: float,
) -> np.ndarray:
    """Return six endpoint-symmetric 3-D link sensing features."""

    if min(communication_range, velocity_scale, degree_scale) <= 0.0:
        raise ValueError("link feature scales must be positive")
    first_state = states[first]
    second_state = states[second]
    displacement = tuple(
        second_value - first_value
        for first_value, second_value in zip(
            first_state.position,
            second_state.position,
        )
    )
    relative_velocity = tuple(
        second_value - first_value
        for first_value, second_value in zip(
            first_state.velocity,
            second_state.velocity,
        )
    )
    distance = max(
        math.sqrt(sum(value * value for value in displacement)),
        1e-9,
    )
    relative_speed = math.sqrt(
        sum(value * value for value in relative_velocity)
    )
    separating_speed = sum(
        displacement[index] * relative_velocity[index]
        for index in range(3)
    ) / distance
    first_speed = math.sqrt(
        sum(value * value for value in first_state.velocity)
    )
    second_speed = math.sqrt(
        sum(value * value for value in second_state.velocity)
    )
    if first_speed > 1e-9 and second_speed > 1e-9:
        heading_alignment = sum(
            first_state.velocity[index] * second_state.velocity[index]
            for index in range(3)
        ) / (first_speed * second_speed)
    else:
        heading_alignment = 0.0
    common_neighbors = len(
        set(graph.get(first, ())).intersection(graph.get(second, ()))
    )
    return np.asarray(
        [
            distance / communication_range,
            relative_speed / velocity_scale,
            max(separating_speed, 0.0) / velocity_scale,
            heading_alignment,
            min(common_neighbors / degree_scale, 1.5),
            abs(displacement[2]) / communication_range,
        ],
        dtype=np.float32,
    )


def controlled_link_probability(
    first: int,
    second: int,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range: float,
    velocity_scale: float,
    degree_scale: float,
    environment_shift: float = 0.0,
) -> float:
    """Generate a bounded packet-success probability for controlled studies.

    The target combines 3-D geometry, relative motion, local contention proxy,
    endpoint energy, and an unobserved deterministic link effect.  It is a
    reproducible probability model, not a measured radio channel.
    """

    edge = symmetric_link_feature(
        first,
        second,
        states,
        graph,
        communication_range=communication_range,
        velocity_scale=velocity_scale,
        degree_scale=degree_scale,
    )
    average_degree = (
        len(graph.get(first, ())) + len(graph.get(second, ()))
    ) / (2.0 * degree_scale)
    minimum_energy = min(states[first].energy, states[second].energy)
    endpoint_contexts = []
    for endpoint in (first, second):
        neighbors = graph.get(endpoint, ())
        if not neighbors:
            endpoint_contexts.append((0.0, states[endpoint].energy))
            continue
        mean_neighbor_velocity = tuple(
            sum(states[neighbor].velocity[index] for neighbor in neighbors)
            / len(neighbors)
            for index in range(3)
        )
        velocity_mismatch = math.sqrt(
            sum(
                (
                    mean_neighbor_velocity[index]
                    - states[endpoint].velocity[index]
                )
                ** 2
                for index in range(3)
            )
        ) / velocity_scale
        mean_neighbor_energy = sum(
            states[neighbor].energy for neighbor in neighbors
        ) / len(neighbors)
        endpoint_contexts.append(
            (velocity_mismatch, mean_neighbor_energy)
        )
    neighborhood_motion_mismatch = sum(
        context[0] for context in endpoint_contexts
    ) / 2.0
    neighborhood_energy = min(
        context[1] for context in endpoint_contexts
    )
    logit = (
        2.70
        - 2.75 * float(edge[0]) ** 2
        - 0.50 * float(edge[1])
        - 1.15 * float(edge[2])
        - 0.90 * average_degree
        + 0.30 * float(edge[3])
        + 0.45 * minimum_energy
        - 1.20 * neighborhood_motion_mismatch
        + 0.30 * neighborhood_energy
        - 0.35 * float(edge[5])
        + deterministic_edge_shadow(first, second)
        + environment_shift
    )
    probability = 1.0 / (1.0 + math.exp(-logit))
    return min(0.995, max(0.02, probability))


def endpoint_pair_inputs(dataset: LinkDataset) -> np.ndarray:
    """Flatten only the two embodied endpoint states and link measurements."""

    return np.concatenate(
        [
            dataset.source_self + dataset.target_self,
            np.abs(dataset.source_self - dataset.target_self),
            dataset.edge,
        ],
        axis=1,
    ).astype(np.float32)


def cooperative_perception_inputs(dataset: LinkDataset) -> np.ndarray:
    """Flatten endpoint states plus both one-hop multi-agent summaries."""

    return np.concatenate(
        [
            dataset.source_self + dataset.target_self,
            np.abs(dataset.source_self - dataset.target_self),
            dataset.source_neighbor + dataset.target_neighbor,
            np.abs(dataset.source_neighbor - dataset.target_neighbor),
            dataset.edge,
        ],
        axis=1,
    ).astype(np.float32)


def build_endpoint_pair_mlp(
    input_dimension: int = ENDPOINT_PAIR_INPUT_DIMENSION,
):
    """Build the endpoint-only neural sensing ablation."""

    import tensorflow as tf

    inputs = tf.keras.Input(
        (input_dimension,),
        name="endpoint_pair_features",
    )
    hidden = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="endpoint_dense_1",
    )(inputs)
    hidden = tf.keras.layers.Dense(
        16,
        activation="relu",
        name="endpoint_dense_2",
    )(hidden)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="endpoint_probability",
    )(hidden)
    model = tf.keras.Model(
        inputs,
        output,
        name="endpoint_pair_link_perception",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_cooperative_perception_mlp(
    input_dimension: int = COOPERATIVE_INPUT_DIMENSION,
):
    """Build the shared lightweight multi-agent link perception model."""

    import tensorflow as tf

    inputs = tf.keras.Input(
        (input_dimension,),
        name="cooperative_link_features",
    )
    hidden = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="cooperative_dense_1",
    )(inputs)
    hidden = tf.keras.layers.Dense(
        16,
        activation="relu",
        name="cooperative_dense_2",
    )(hidden)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="cooperative_probability",
    )(hidden)
    model = tf.keras.Model(
        inputs,
        output,
        name="multi_agent_cooperative_link_perception",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_multi_agent_link_dataset(
    routes: Sequence[Sequence[Mapping[str, float]]],
    time_indices: Sequence[int],
    node_count: int,
    communication_range: float,
    horizontal_position_scale: float,
    altitude_scale: float,
    velocity_scale: float,
    degree_scale: float,
    probe_count: int,
    seed: int,
    environment_shift: float = 0.0,
) -> LinkDataset:
    """Construct supervised samples from local multi-agent observations."""

    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
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
        states = snapshot_embodied_states(routes, time_index, node_count)
        graph = build_embodied_neighbor_graph(states, communication_range)
        node_rows = {
            node: normalized_node_feature(
                node,
                states,
                graph,
                horizontal_position_scale=horizontal_position_scale,
                altitude_scale=altitude_scale,
                velocity_scale=velocity_scale,
                degree_scale=degree_scale,
            )
            for node in states
        }
        neighbor_rows = {
            node: neighbor_mean_feature(node, node_rows, graph)
            for node in states
        }
        for first in sorted(graph):
            for second in graph[first]:
                if first >= second:
                    continue
                probability = controlled_link_probability(
                    first,
                    second,
                    states,
                    graph,
                    communication_range=communication_range,
                    velocity_scale=velocity_scale,
                    degree_scale=degree_scale,
                    environment_shift=environment_shift,
                )
                source_self.append(node_rows[first])
                source_neighbor.append(neighbor_rows[first])
                target_self.append(node_rows[second])
                target_neighbor.append(neighbor_rows[second])
                edge_rows.append(
                    symmetric_link_feature(
                        first,
                        second,
                        states,
                        graph,
                        communication_range=communication_range,
                        velocity_scale=velocity_scale,
                        degree_scale=degree_scale,
                    )
                )
                labels.append(
                    float(
                        generator.binomial(probe_count, probability)
                        / probe_count
                    )
                )
                probabilities.append(probability)
                times.append(time_index)
                first_nodes.append(first)
                second_nodes.append(second)

    if not labels:
        return LinkDataset(
            *(np.empty((0, NODE_DIMENSION), dtype=np.float32) for _ in range(4)),
            np.empty((0, EDGE_DIMENSION), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
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


def build_multi_agent_default_bundle(
    routes: Sequence[Sequence[Mapping[str, float]]],
    seed: int,
) -> DatasetBundle:
    """Build the fixed temporal and graph-size generalization split."""

    common = {
        "routes": routes,
        "communication_range": 200.0,
        "horizontal_position_scale": 2000.0,
        "altitude_scale": 120.0,
        "velocity_scale": 25.0,
        "degree_scale": 20.0,
        "probe_count": 20,
    }
    return DatasetBundle(
        train=build_multi_agent_link_dataset(
            time_indices=range(0, 181, 2),
            node_count=60,
            seed=seed,
            environment_shift=0.0,
            **common,
        ),
        validation=build_multi_agent_link_dataset(
            time_indices=range(181, 241, 2),
            node_count=80,
            seed=seed + 1,
            environment_shift=-0.10,
            **common,
        ),
        test=build_multi_agent_link_dataset(
            time_indices=range(241, 301),
            node_count=100,
            seed=seed + 2,
            environment_shift=-0.25,
            **common,
        ),
    )


def lifetime_penalty_map(
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range: float,
    traffic_horizon_s: float,
    lifetime_weight: float,
    minimum_quality: float = 1e-3,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return symmetric RLT penalties and raw predicted lifetimes."""

    if lifetime_weight < 0.0:
        raise ValueError("lifetime_weight cannot be negative")
    penalties: dict[tuple[int, int], float] = {}
    lifetimes: dict[tuple[int, int], float] = {}
    for first in sorted(graph):
        for second in graph[first]:
            if first >= second:
                continue
            lifetime = predicted_link_lifetime_s(
                states[first].position,
                states[first].velocity,
                states[second].position,
                states[second].velocity,
                communication_range,
            )
            quality = link_hold_quality(
                lifetime,
                traffic_horizon_s,
                minimum_quality=minimum_quality,
            )
            edge = (first, second)
            lifetimes[edge] = lifetime
            penalties[edge] = -lifetime_weight * math.log(quality)
    return penalties, lifetimes

