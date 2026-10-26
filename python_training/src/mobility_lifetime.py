"""Endpoint-symmetric relative link lifetime for embodied low-altitude routing.

Each node senses local position and velocity.  A link expires when the relative
trajectory of its endpoints exits the communication sphere.  The implementation
solves the three-dimensional quadratic directly, which also covers approaching
nodes and zero relative velocity without direction-specific special cases.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence


Vector = Sequence[float]
TimedLinkProbability = Callable[[int, int, int], float]


def build_3d_neighbor_graph(
    positions: Mapping[int, Vector],
    communication_range_m: float,
) -> dict[int, list[int]]:
    """Build an undirected graph using the same 3-D communication sphere."""

    if communication_range_m <= 0.0:
        raise ValueError("communication_range_m must be positive")
    normalized = {
        int(node): _as_three_vector(position, f"position[{node}]")
        for node, position in positions.items()
    }
    graph = {node: [] for node in normalized}
    nodes = sorted(graph)
    for offset, first in enumerate(nodes):
        for second in nodes[offset + 1:]:
            distance = math.sqrt(
                sum(
                    (
                        normalized[first][dimension]
                        - normalized[second][dimension]
                    )
                    ** 2
                    for dimension in range(3)
                )
            )
            if distance <= communication_range_m:
                graph[first].append(second)
                graph[second].append(first)
    return graph


def predicted_link_lifetime_s(
    first_position: Vector,
    first_velocity: Vector,
    second_position: Vector,
    second_velocity: Vector,
    communication_range_m: float,
) -> float:
    """Return the constant-velocity time until the endpoints leave radio range.

    The result is zero when the endpoints are already outside the range and
    infinity when their relative position is constant inside the range.
    Swapping the two endpoints changes both relative vectors' signs and
    therefore leaves the result unchanged.
    """

    if communication_range_m <= 0.0:
        raise ValueError("communication_range_m must be positive")
    first_p = _as_three_vector(first_position, "first_position")
    second_p = _as_three_vector(second_position, "second_position")
    first_v = _as_three_vector(first_velocity, "first_velocity")
    second_v = _as_three_vector(second_velocity, "second_velocity")
    displacement = tuple(
        second_p[index] - first_p[index] for index in range(3)
    )
    relative_velocity = tuple(
        second_v[index] - first_v[index] for index in range(3)
    )
    quadratic = sum(value * value for value in relative_velocity)
    linear = 2.0 * sum(
        displacement[index] * relative_velocity[index]
        for index in range(3)
    )
    constant = (
        sum(value * value for value in displacement)
        - communication_range_m * communication_range_m
    )
    tolerance = 1e-12 * max(1.0, communication_range_m**2)
    if constant > tolerance:
        return 0.0
    if quadratic <= 1e-18:
        return math.inf
    discriminant = linear * linear - 4.0 * quadratic * constant
    if discriminant < 0.0 and discriminant > -tolerance:
        discriminant = 0.0
    if discriminant < 0.0:
        return 0.0
    exit_time = (
        -linear + math.sqrt(discriminant)
    ) / (2.0 * quadratic)
    return max(0.0, exit_time)


def link_hold_quality(
    lifetime_s: float,
    traffic_horizon_s: float,
    *,
    minimum_quality: float = 1e-6,
) -> float:
    """Map remaining lifetime to a bounded link-hold quality in ``(0, 1]``."""

    if traffic_horizon_s <= 0.0:
        raise ValueError("traffic_horizon_s must be positive")
    if not 0.0 < minimum_quality <= 1.0:
        raise ValueError("minimum_quality must lie in (0, 1]")
    if math.isnan(lifetime_s) or lifetime_s < 0.0:
        raise ValueError("lifetime_s must be nonnegative")
    if math.isinf(lifetime_s):
        return 1.0
    return min(
        1.0,
        max(minimum_quality, lifetime_s / traffic_horizon_s),
    )


def lifetime_augmented_cost(
    reliability_probability: float,
    lifetime_s: float,
    *,
    traffic_horizon_s: float,
    lifetime_weight: float,
    hop_penalty: float,
    minimum_probability: float = 1e-9,
    minimum_lifetime_quality: float = 1e-6,
) -> float:
    """Return a strictly positive additive reliability-and-lifetime edge cost."""

    if lifetime_weight < 0.0:
        raise ValueError("lifetime_weight cannot be negative")
    if hop_penalty <= 0.0:
        raise ValueError("hop_penalty must be positive")
    if not 0.0 < minimum_probability < 0.5:
        raise ValueError("minimum_probability must lie in (0, 0.5)")
    probability = min(
        1.0 - minimum_probability,
        max(minimum_probability, float(reliability_probability)),
    )
    hold_quality = link_hold_quality(
        lifetime_s,
        traffic_horizon_s,
        minimum_quality=minimum_lifetime_quality,
    )
    return (
        -math.log(probability)
        - lifetime_weight * math.log(hold_quality)
        + hop_penalty
    )


def expected_session_delivery_ratio(
    route: Sequence[int],
    packet_time_indices: Sequence[int],
    link_probability: TimedLinkProbability,
) -> float:
    """Average expected end-to-end delivery over a fixed route and time window.

    ``link_probability`` returns zero for a link that has left the neighbor
    graph.  Otherwise it supplies the one-hop success probability at that
    packet time.  Independence is the same path-reliability assumption used by
    the route-level experiments.
    """

    if len(route) < 2 or not packet_time_indices:
        return 0.0
    deliveries = []
    for time_index in packet_time_indices:
        path_probability = 1.0
        for first, second in zip(route, route[1:]):
            probability = float(
                link_probability(int(time_index), int(first), int(second))
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError("link probability must lie in [0, 1]")
            path_probability *= probability
            if path_probability == 0.0:
                break
        deliveries.append(path_probability)
    return sum(deliveries) / len(deliveries)


def _as_three_vector(values: Vector, name: str) -> tuple[float, float, float]:
    if len(values) not in (2, 3):
        raise ValueError(f"{name} must contain two or three values")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} values must be finite")
    if len(vector) == 2:
        return vector[0], vector[1], 0.0
    return vector

