import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .utils import wrap_to_pi


@dataclass
class EMAConfig:
    alpha_dist: float = 0.4
    alpha_yaw: float = 0.4


class EMAFilter:
    """EMA filter for distance and yaw (handles angle wrapping)."""

    def __init__(self, config: EMAConfig) -> None:
        self.config = config
        self.dist_ema: Optional[float] = None
        self.yaw_sin_ema: Optional[float] = None
        self.yaw_cos_ema: Optional[float] = None

    def update(self, meas: Tuple[float, float]) -> Tuple[float, float]:
        dist, yaw = meas
        if self.dist_ema is None:
            self.dist_ema = dist
        else:
            a = self.config.alpha_dist
            self.dist_ema = a * dist + (1.0 - a) * self.dist_ema

        sin_y = math.sin(yaw)
        cos_y = math.cos(yaw)

        if self.yaw_sin_ema is None:
            self.yaw_sin_ema = sin_y
            self.yaw_cos_ema = cos_y
        else:
            a = self.config.alpha_yaw
            self.yaw_sin_ema = a * sin_y + (1.0 - a) * self.yaw_sin_ema
            self.yaw_cos_ema = a * cos_y + (1.0 - a) * self.yaw_cos_ema

        yaw_filt = math.atan2(self.yaw_sin_ema, self.yaw_cos_ema)
        return self.dist_ema, wrap_to_pi(yaw_filt)
