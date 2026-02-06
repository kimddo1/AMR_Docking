import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .filtering import EMAConfig, EMAFilter
from .utils import wrap_to_pi


@dataclass
class GatingConfig:
    max_jump_dist: float = 0.4
    max_jump_yaw: float = math.radians(30)
    hold_last: bool = True
    stale_jump_scale: float = 0.0


@dataclass
class RobustConfig:
    enable_filter: bool = True
    enable_gating: bool = True
    ema: EMAConfig = field(default_factory=EMAConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)


class RobustPoseEstimator:
    """Apply gating + filtering to relative pose measurements."""

    def __init__(self, config: RobustConfig) -> None:
        self.config = config
        self.filter = EMAFilter(config.ema) if config.enable_filter else None
        self.last_good: Optional[Tuple[float, float]] = None
        self.stale_count: int = 0

    def _is_outlier(self, meas: Tuple[float, float]) -> bool:
        if self.last_good is None:
            return False
        dist, yaw = meas
        last_dist, last_yaw = self.last_good
        scale = 1.0 + self.stale_count * self.config.gating.stale_jump_scale
        max_jump_dist = self.config.gating.max_jump_dist * scale
        max_jump_yaw = self.config.gating.max_jump_yaw * scale
        if abs(dist - last_dist) > max_jump_dist:
            return True
        yaw_jump = wrap_to_pi(yaw - last_yaw)
        if abs(yaw_jump) > max_jump_yaw:
            return True
        return False

    def step(self, meas: Optional[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], dict]:
        """Return (estimate, info). info includes flags for gating/dropout."""
        info = {
            "input_missing": meas is None,
            "rejected": False,
            "used_hold": False,
            "used_filter": False,
            "stale_count": 0,
        }

        if meas is None:
            self.stale_count += 1
            info["stale_count"] = self.stale_count
            if self.config.gating.hold_last and self.last_good is not None:
                info["used_hold"] = True
                return self.last_good, info
            return None, info

        if self.config.enable_gating and self._is_outlier(meas):
            info["rejected"] = True
            self.stale_count += 1
            info["stale_count"] = self.stale_count
            if self.config.gating.hold_last and self.last_good is not None:
                info["used_hold"] = True
                return self.last_good, info
            return None, info

        self.last_good = meas
        self.stale_count = 0
        info["stale_count"] = 0

        if self.filter is not None:
            info["used_filter"] = True
            return self.filter.update(meas), info

        return meas, info
