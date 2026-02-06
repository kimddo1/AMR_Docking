import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from .utils import wrap_to_pi


@dataclass
class SensorConfig:
    sigma_d: float = 0.0
    sigma_yaw: float = 0.0
    p_drop: float = 0.0
    latency_steps: int = 0
    p_outlier: float = 0.0
    outlier_dist: float = 0.5
    outlier_yaw: float = math.radians(20)
    seed: Optional[int] = None


class RelativePoseSensor:
    """Corrupt GT relative pose with noise, dropout, latency, and outliers."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        maxlen = max(1, config.latency_steps + 1)
        self.buffer: Deque[Optional[Tuple[float, float]]] = deque(maxlen=maxlen)

    def _apply_noise(self, dist: float, yaw: float) -> Tuple[float, float]:
        if self.config.sigma_d > 0.0:
            dist += self.rng.gauss(0.0, self.config.sigma_d)
        if self.config.sigma_yaw > 0.0:
            yaw += self.rng.gauss(0.0, self.config.sigma_yaw)
        return dist, wrap_to_pi(yaw)

    def _apply_outlier(self, dist: float, yaw: float) -> Tuple[float, float]:
        if self.config.p_outlier > 0.0 and self.rng.random() < self.config.p_outlier:
            dist += self.config.outlier_dist * (1.0 if self.rng.random() < 0.5 else -1.0)
            yaw += self.config.outlier_yaw * (1.0 if self.rng.random() < 0.5 else -1.0)
        return dist, wrap_to_pi(yaw)

    def measure(self, dist_true: float, yaw_true: float) -> Optional[Tuple[float, float]]:
        """Return (dist, yaw) or None (dropout). Latency modeled by buffer."""
        # Dropout
        if self.config.p_drop > 0.0 and self.rng.random() < self.config.p_drop:
            raw_meas = None
        else:
            dist, yaw = self._apply_noise(dist_true, yaw_true)
            dist, yaw = self._apply_outlier(dist, yaw)
            raw_meas = (max(0.0, dist), wrap_to_pi(yaw))

        self.buffer.append(raw_meas)

        # Latency: return measurement from L steps ago
        L = self.config.latency_steps
        if len(self.buffer) <= L:
            return None
        return self.buffer[-(L + 1)]
