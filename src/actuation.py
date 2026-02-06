from dataclasses import dataclass
from typing import Optional, Tuple
import random


@dataclass
class ActuationNoiseConfig:
    sigma_v: float = 0.03
    sigma_w: float = 0.05
    seed: Optional[int] = None


class ActuationNoise:
    def __init__(self, config: ActuationNoiseConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def apply(self, v: float, w: float) -> Tuple[float, float]:
        if self.config.sigma_v > 0.0:
            v = v * (1.0 + self.rng.gauss(0.0, self.config.sigma_v))
        if self.config.sigma_w > 0.0:
            w = w * (1.0 + self.rng.gauss(0.0, self.config.sigma_w))
        return v, w
