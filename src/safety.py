import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .obstacles import CircleObstacle, RectObstacle
from .utils import clamp


@dataclass
class SafetyConfig:
    r_stop: float = 0.2
    r_slow: float = 0.5
    v_slow: float = 0.2
    w_slow: float = 0.5
    robot_radius: float = 0.2
    near_miss_dist: float = 0.1


Obstacle = Union[CircleObstacle, RectObstacle]


def _distance_point_to_rect(px: float, py: float, rect: RectObstacle) -> float:
    # Axis-aligned rectangle centered at (rect.x, rect.y)
    dx = abs(px - rect.x) - rect.width / 2.0
    dy = abs(py - rect.y) - rect.height / 2.0
    dx = max(dx, 0.0)
    dy = max(dy, 0.0)
    return math.hypot(dx, dy)


def min_distance_to_obstacles(
    x: float,
    y: float,
    obstacles: List[Obstacle],
    robot_radius: float,
) -> Optional[float]:
    if not obstacles:
        return None

    dists = []
    for obs in obstacles:
        if not getattr(obs, "active", True):
            continue
        if isinstance(obs, CircleObstacle):
            center_dist = math.hypot(x - obs.x, y - obs.y)
            d = center_dist - (obs.radius + robot_radius)
        else:
            d = _distance_point_to_rect(x, y, obs) - robot_radius
        dists.append(d)

    if not dists:
        return None
    return min(dists)


def safety_step(
    v: float,
    w: float,
    x: float,
    y: float,
    obstacles: List[Obstacle],
    config: SafetyConfig,
) -> Tuple[float, float, str, Optional[float], bool, bool]:
    min_dist = min_distance_to_obstacles(x, y, obstacles, config.robot_radius)

    if min_dist is None:
        return v, w, "CLEAR", None, False, False

    collision = min_dist <= 0.0
    near_miss = min_dist <= config.near_miss_dist

    if min_dist <= config.r_stop:
        return 0.0, 0.0, "STOP", min_dist, collision, near_miss

    if min_dist <= config.r_slow:
        v = clamp(v, -config.v_slow, config.v_slow)
        w = clamp(w, -config.w_slow, config.w_slow)
        return v, w, "SLOW", min_dist, collision, near_miss

    return v, w, "CLEAR", min_dist, collision, near_miss
