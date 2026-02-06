import math
from typing import Dict, List, Tuple
from .dynamics import step_differential_drive
from .utils import wrap_to_pi
from .control import go_to_goal_control

State = Tuple[float, float, float]
Target = Tuple[float, float]


def relative_pose(state: State, target: Target) -> Tuple[float, float]:
    """Return (dist_true, yaw_true) in robot frame."""
    x, y, theta = state
    x_t, y_t = target
    dx = x_t - x
    dy = y_t - y
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    yaw = wrap_to_pi(bearing - theta)
    return dist, yaw


def simulate(
    state0: State,
    target: Target,
    dt: float,
    max_steps: int,
    control_params: Dict,
    dist_tol: float,
    yaw_tol: float,
) -> Dict[str, List[float]]:
    """Run a clean-GT docking simulation. Returns history dict."""
    state = state0

    xs: List[float] = []
    ys: List[float] = []
    thetas: List[float] = []
    dists: List[float] = []
    yaws: List[float] = []
    times: List[float] = []

    success = False

    for k in range(max_steps):
        t = k * dt
        dist, yaw = relative_pose(state, target)

        xs.append(state[0])
        ys.append(state[1])
        thetas.append(state[2])
        dists.append(dist)
        yaws.append(yaw)
        times.append(t)

        if dist <= dist_tol and abs(yaw) <= yaw_tol:
            success = True
            break

        v, w = go_to_goal_control(state, target, control_params)
        state = step_differential_drive(state, (v, w), dt)

    return {
        "x": xs,
        "y": ys,
        "theta": thetas,
        "dist_true": dists,
        "yaw_true": yaws,
        "time": times,
        "success": success,
    }
