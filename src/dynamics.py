import math
from typing import Tuple
from .utils import wrap_to_pi

State = Tuple[float, float, float]  # x, y, theta
Control = Tuple[float, float]       # v, w


def step_differential_drive(state: State, control: Control, dt: float) -> State:
    """Integrate differential-drive kinematics for one timestep."""
    x, y, theta = state
    v, w = control
    x_next = x + v * math.cos(theta) * dt
    y_next = y + v * math.sin(theta) * dt
    theta_next = wrap_to_pi(theta + w * dt)
    return (x_next, y_next, theta_next)
