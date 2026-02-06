import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .utils import clamp


@dataclass
class FSMConfig:
    align_dist: float = 0.6
    finalize_dist: float = 0.25
    yaw_align_thresh: float = math.radians(10)
    finalize_speed_scale: float = 0.35
    align_w_gain: float = 2.0


class DockingFSM:
    """Simple docking FSM: APPROACH -> ALIGN -> FINALIZE."""

    def __init__(self, config: FSMConfig) -> None:
        self.config = config
        self.state = "APPROACH"

    def step(
        self,
        dist: Optional[float],
        yaw: Optional[float],
        base_v: float,
        base_w: float,
        w_max: float,
    ) -> Tuple[float, float, str]:
        if dist is None or yaw is None:
            return 0.0, 0.0, self.state

        if self.state == "APPROACH":
            if dist <= self.config.align_dist and abs(yaw) > self.config.yaw_align_thresh:
                self.state = "ALIGN"
            elif dist <= self.config.finalize_dist:
                self.state = "FINALIZE"
        elif self.state == "ALIGN":
            if abs(yaw) <= self.config.yaw_align_thresh:
                self.state = "FINALIZE"
        elif self.state == "FINALIZE":
            if dist > self.config.finalize_dist * 1.2:
                self.state = "APPROACH"

        if self.state == "APPROACH":
            return base_v, base_w, self.state

        if self.state == "ALIGN":
            w = w_max * math.tanh(self.config.align_w_gain * yaw)
            w = clamp(w, -w_max, w_max)
            return 0.0, w, self.state

        # FINALIZE
        v = base_v * self.config.finalize_speed_scale
        return v, base_w, self.state
