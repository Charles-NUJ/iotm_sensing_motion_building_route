"""Small endpoint-symmetric predictor of remaining mobile-link hold quality.

The predictor consumes only relative motion.  It estimates the fraction of a
short traffic horizon for which a currently connected link remains connected.
The analytical constant-velocity estimate is one input and remains available
as a deterministic fallback; the learned model can correct for turns and
accelerations observed in recorded trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .mobility_lifetime import predicted_link_lifetime_s
from .multi_agent_perception import (
    EmbodiedNodeState,
    build_embodied_neighbor_graph,
    neighbor_mean_feature,
    normalized_node_feature,
    snapshot_embodied_states,
)


LINK_HOLD_INPUT_DIMENSION = 5
RICH_LINK_HOLD_INPUT_DIMENSION = 37
LINK_HOLD_TARGET_HORIZON_S = 10
LINK_HOLD_TRAIN_TIMES = range(0, 171, 2)
LINK_HOLD_VALIDATION_TIMES = range(181, 231, 2)
LINK_HOLD_TEST_TIMES = range(241, 291)


@dataclass(frozen=True)
class LinkHoldDataset:
    inputs: np.ndarray
    rich_inputs: np.ndarray
    target: np.ndarray
    time: np.ndarray
    first_node: np.ndarray
    second_node: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


@dataclass(frozen=True)
class LinkHoldBundle:
    train: LinkHoldDataset
    validation: LinkHoldDataset
    test: LinkHoldDataset


def link_hold_features(
    first_state: EmbodiedNodeState,
    second_state: EmbodiedNodeState,
    *,
    communication_range_m: float,
    velocity_scale_mps: float,
    traffic_horizon_s: float,
) -> np.ndarray:
    """Return five endpoint-symmetric relative-motion features."""

    if min(
        communication_range_m,
        velocity_scale_mps,
        traffic_horizon_s,
    ) <= 0.0:
        raise ValueError("all scales must be positive")
    displacement = np.asarray(second_state.position, dtype=float) - np.asarray(
        first_state.position,
        dtype=float,
    )
    relative_velocity = np.asarray(
        second_state.velocity,
        dtype=float,
    ) - np.asarray(first_state.velocity, dtype=float)
    distance = float(np.linalg.norm(displacement))
    relative_speed = float(np.linalg.norm(relative_velocity))
    radial_speed = (
        float(np.dot(displacement, relative_velocity)) / distance
        if distance > 1e-12
        else 0.0
    )
    transverse_speed = math.sqrt(
        max(0.0, relative_speed**2 - radial_speed**2)
    )
    analytic_lifetime = predicted_link_lifetime_s(
        first_state.position,
        first_state.velocity,
        second_state.position,
        second_state.velocity,
        communication_range_m,
    )
    analytic_quality = (
        1.0
        if math.isinf(analytic_lifetime)
        else min(1.0, analytic_lifetime / traffic_horizon_s)
    )
    return np.asarray(
        [
            min(1.5, distance / communication_range_m),
            min(2.0, relative_speed / velocity_scale_mps),
            float(np.clip(radial_speed / velocity_scale_mps, -2.0, 2.0)),
            min(2.0, transverse_speed / velocity_scale_mps),
            analytic_quality,
        ],
        dtype=np.float32,
    )


def observed_link_hold_quality(
    first_route: Sequence[Mapping[str, float]],
    second_route: Sequence[Mapping[str, float]],
    *,
    time_index: int,
    communication_range_m: float,
    traffic_horizon_s: int,
) -> float:
    """Measure first future range exit in the recorded trajectory."""

    if communication_range_m <= 0.0:
        raise ValueError("communication_range_m must be positive")
    if traffic_horizon_s <= 0:
        raise ValueError("traffic_horizon_s must be positive")
    final_time = time_index + traffic_horizon_s
    if time_index < 0 or final_time >= len(first_route):
        raise ValueError("the observation horizon lies outside the route")
    if len(first_route) != len(second_route):
        raise ValueError("route lengths must match")

    def distance_at(index: int) -> float:
        return math.sqrt(
            sum(
                (
                    float(second_route[index].get(field, 0.0))
                    - float(first_route[index].get(field, 0.0))
                )
                ** 2
                for field in ("x", "y", "z")
            )
        )

    previous_distance = distance_at(time_index)
    if previous_distance > communication_range_m:
        return 0.0
    for offset in range(1, traffic_horizon_s + 1):
        current_distance = distance_at(time_index + offset)
        if current_distance > communication_range_m:
            denominator = current_distance - previous_distance
            fraction = (
                0.0
                if abs(denominator) <= 1e-12
                else (communication_range_m - previous_distance) / denominator
            )
            exit_time = offset - 1 + float(np.clip(fraction, 0.0, 1.0))
            return float(np.clip(exit_time / traffic_horizon_s, 0.0, 1.0))
        previous_distance = current_distance
    return 1.0


def rich_link_hold_features(
    first: int,
    second: int,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range_m: float,
    velocity_scale_mps: float,
    traffic_horizon_s: float,
    horizontal_position_scale_m: float,
    altitude_scale_m: float,
    degree_scale: float,
) -> np.ndarray:
    """Add endpoint and one-hop neighbor summaries to the five motion inputs."""

    node_rows = {
        node: normalized_node_feature(
            node,
            states,
            graph,
            horizontal_position_scale=horizontal_position_scale_m,
            altitude_scale=altitude_scale_m,
            velocity_scale=velocity_scale_mps,
            degree_scale=degree_scale,
        )
        for node in (first, second)
    }
    neighbor_rows = {
        node: neighbor_mean_feature(
            node,
            {
                item: normalized_node_feature(
                    item,
                    states,
                    graph,
                    horizontal_position_scale=horizontal_position_scale_m,
                    altitude_scale=altitude_scale_m,
                    velocity_scale=velocity_scale_mps,
                    degree_scale=degree_scale,
                )
                for item in states
            },
            graph,
        )
        for node in (first, second)
    }
    simple = link_hold_features(
        states[first],
        states[second],
        communication_range_m=communication_range_m,
        velocity_scale_mps=velocity_scale_mps,
        traffic_horizon_s=traffic_horizon_s,
    )
    return np.concatenate(
        (
            simple,
            node_rows[first] + node_rows[second],
            np.abs(node_rows[first] - node_rows[second]),
            neighbor_rows[first] + neighbor_rows[second],
            np.abs(neighbor_rows[first] - neighbor_rows[second]),
        )
    ).astype(np.float32)


def build_link_hold_dataset(
    routes: Sequence[Sequence[Mapping[str, float]]],
    *,
    time_indices: Sequence[int],
    node_count: int,
    communication_range_m: float,
    velocity_scale_mps: float,
    traffic_horizon_s: int,
    horizontal_position_scale_m: float = 2000.0,
    altitude_scale_m: float = 120.0,
    degree_scale: float = 20.0,
) -> LinkHoldDataset:
    """Create current-link samples with future observed hold targets."""

    features = []
    rich_features = []
    targets = []
    times = []
    first_nodes = []
    second_nodes = []
    for time_index in time_indices:
        states = snapshot_embodied_states(routes, time_index, node_count)
        graph = build_embodied_neighbor_graph(states, communication_range_m)
        for first in sorted(graph):
            for second in graph[first]:
                if first >= second:
                    continue
                features.append(
                    link_hold_features(
                        states[first],
                        states[second],
                        communication_range_m=communication_range_m,
                        velocity_scale_mps=velocity_scale_mps,
                        traffic_horizon_s=traffic_horizon_s,
                    )
                )
                rich_features.append(
                    rich_link_hold_features(
                        first,
                        second,
                        states,
                        graph,
                        communication_range_m=communication_range_m,
                        velocity_scale_mps=velocity_scale_mps,
                        traffic_horizon_s=traffic_horizon_s,
                        horizontal_position_scale_m=(
                            horizontal_position_scale_m
                        ),
                        altitude_scale_m=altitude_scale_m,
                        degree_scale=degree_scale,
                    )
                )
                targets.append(
                    observed_link_hold_quality(
                        routes[first],
                        routes[second],
                        time_index=time_index,
                        communication_range_m=communication_range_m,
                        traffic_horizon_s=traffic_horizon_s,
                    )
                )
                times.append(time_index)
                first_nodes.append(first)
                second_nodes.append(second)
    return LinkHoldDataset(
        inputs=np.asarray(features, dtype=np.float32).reshape(
            -1,
            LINK_HOLD_INPUT_DIMENSION,
        ),
        rich_inputs=np.asarray(rich_features, dtype=np.float32).reshape(
            -1,
            RICH_LINK_HOLD_INPUT_DIMENSION,
        ),
        target=np.asarray(targets, dtype=np.float32),
        time=np.asarray(times, dtype=np.int32),
        first_node=np.asarray(first_nodes, dtype=np.int32),
        second_node=np.asarray(second_nodes, dtype=np.int32),
    )


def build_link_hold_default_bundle(
    routes: Sequence[Sequence[Mapping[str, float]]],
) -> LinkHoldBundle:
    """Build temporal splits whose complete future labels never overlap.

    A feature vector at time ``t`` uses a target observed through ``t + 10``.
    The unused intervals 171--180 and 231--240 are therefore embargoes.
    """

    common = {
        "routes": routes,
        "communication_range_m": 200.0,
        "velocity_scale_mps": 25.0,
        "traffic_horizon_s": LINK_HOLD_TARGET_HORIZON_S,
    }
    return LinkHoldBundle(
        train=build_link_hold_dataset(
            time_indices=LINK_HOLD_TRAIN_TIMES,
            node_count=60,
            **common,
        ),
        validation=build_link_hold_dataset(
            time_indices=LINK_HOLD_VALIDATION_TIMES,
            node_count=80,
            **common,
        ),
        test=build_link_hold_dataset(
            time_indices=LINK_HOLD_TEST_TIMES,
            node_count=100,
            **common,
        ),
    )


def build_link_hold_mlp():
    """Build a 241-parameter neural correction of analytical link lifetime."""

    import tensorflow as tf

    inputs = tf.keras.Input(
        shape=(LINK_HOLD_INPUT_DIMENSION,),
        name="link_hold_features",
    )
    hidden = tf.keras.layers.Dense(16, activation="relu")(inputs)
    hidden = tf.keras.layers.Dense(8, activation="relu")(hidden)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="hold_quality",
    )(hidden)
    model = tf.keras.Model(inputs, output, name="relative_motion_link_hold")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_rich_link_hold_mlp():
    """Build a 1,761-parameter neighbor-context link-hold comparator."""

    import tensorflow as tf

    inputs = tf.keras.Input(
        shape=(RICH_LINK_HOLD_INPUT_DIMENSION,),
        name="rich_link_hold_features",
    )
    hidden = tf.keras.layers.Dense(32, activation="relu")(inputs)
    hidden = tf.keras.layers.Dense(16, activation="relu")(hidden)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="rich_hold_quality",
    )(hidden)
    model = tf.keras.Model(inputs, output, name="neighbor_context_link_hold")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def predict_ridge_link_hold_map(
    model,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range_m: float,
    velocity_scale_mps: float,
    traffic_horizon_s: float,
    minimum_quality: float = 1e-3,
) -> dict[tuple[int, int], float]:
    """Apply a fitted five-input Ridge model to every current link."""

    edges = [
        (first, second)
        for first in sorted(graph)
        for second in graph[first]
        if first < second
    ]
    inputs = np.stack(
        [
            link_hold_features(
                states[first],
                states[second],
                communication_range_m=communication_range_m,
                velocity_scale_mps=velocity_scale_mps,
                traffic_horizon_s=traffic_horizon_s,
            )
            for first, second in edges
        ]
    )
    predictions = np.asarray(model.predict(inputs), dtype=float).reshape(-1)
    output = {}
    for edge, features, prediction in zip(edges, inputs, predictions):
        value = 1.0 if features[1] <= 1e-9 else float(prediction)
        output[edge] = float(np.clip(value, minimum_quality, 1.0))
    return output


def predict_link_hold_map(
    model,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range_m: float,
    velocity_scale_mps: float,
    traffic_horizon_s: float,
    minimum_quality: float = 1e-3,
) -> dict[tuple[int, int], float]:
    """Predict one hold-quality value for every current undirected edge."""

    edges = [
        (first, second)
        for first in sorted(graph)
        for second in graph[first]
        if first < second
    ]
    inputs = np.stack(
        [
            link_hold_features(
                states[first],
                states[second],
                communication_range_m=communication_range_m,
                velocity_scale_mps=velocity_scale_mps,
                traffic_horizon_s=traffic_horizon_s,
            )
            for first, second in edges
        ]
    )
    predictions = np.asarray(model(inputs, training=False)).reshape(-1)
    output = {}
    for edge, features, prediction in zip(edges, inputs, predictions):
        # A zero-relative-speed static link has an exact safe fallback.
        value = 1.0 if features[1] <= 1e-9 else float(prediction)
        output[edge] = float(np.clip(value, minimum_quality, 1.0))
    return output


def predict_rich_link_hold_map(
    model,
    states: Mapping[int, EmbodiedNodeState],
    graph: Mapping[int, Sequence[int]],
    *,
    communication_range_m: float,
    velocity_scale_mps: float,
    traffic_horizon_s: float,
    horizontal_position_scale_m: float,
    altitude_scale_m: float,
    degree_scale: float,
    minimum_quality: float = 1e-3,
) -> dict[tuple[int, int], float]:
    edges = [
        (first, second)
        for first in sorted(graph)
        for second in graph[first]
        if first < second
    ]
    inputs = np.stack(
        [
            rich_link_hold_features(
                first,
                second,
                states,
                graph,
                communication_range_m=communication_range_m,
                velocity_scale_mps=velocity_scale_mps,
                traffic_horizon_s=traffic_horizon_s,
                horizontal_position_scale_m=horizontal_position_scale_m,
                altitude_scale_m=altitude_scale_m,
                degree_scale=degree_scale,
            )
            for first, second in edges
        ]
    )
    predictions = np.asarray(model(inputs, training=False)).reshape(-1)
    output = {}
    for edge, prediction in zip(edges, predictions):
        first, second = edge
        simple = link_hold_features(
            states[first],
            states[second],
            communication_range_m=communication_range_m,
            velocity_scale_mps=velocity_scale_mps,
            traffic_horizon_s=traffic_horizon_s,
        )
        value = 1.0 if simple[1] <= 1e-9 else float(prediction)
        output[edge] = float(np.clip(value, minimum_quality, 1.0))
    return output

