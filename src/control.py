import math
from typing import Dict, Tuple
from .utils import clamp

State = Tuple[float, float, float]
Target = Tuple[float, float]


def go_to_goal_control(state: State, target: Target, params: Dict) -> Tuple[float, float]:
    """Simple P control with speed scheduling for stable docking."""
    x, y, theta = state
    x_t, y_t = target
    dx = x_t - x
    dy = y_t - y
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    yaw = (bearing - theta + math.pi) % (2.0 * math.pi) - math.pi

    # Speed scheduling: reduce linear speed when yaw error is large
    v_max = params.get("v_max", 0.6)
    w_max = params.get("w_max", 1.5)
    k_v = params.get("k_v", 1.2)
    k_w = params.get("k_w", 2.0)
    yaw_slow = params.get("yaw_slow", math.radians(35))

    v = v_max * math.tanh(k_v * dist)
    if abs(yaw) > yaw_slow:
        v *= 0.2

    w = w_max * math.tanh(k_w * yaw)

    v = clamp(v, -v_max, v_max)
    w = clamp(w, -w_max, w_max)
    return v, w
