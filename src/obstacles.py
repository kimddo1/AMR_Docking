from dataclasses import dataclass
from typing import List, Optional
import random


@dataclass
class CircleObstacle:
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    spawn_time: float = 0.0
    despawn_time: Optional[float] = None
    active: bool = True

    def update(self, t: float, dt: float) -> None:
        self.active = t >= self.spawn_time and (self.despawn_time is None or t <= self.despawn_time)
        if self.active:
            self.x += self.vx * dt
            self.y += self.vy * dt


@dataclass
class RectObstacle:
    x: float
    y: float
    width: float
    height: float
    vx: float = 0.0
    vy: float = 0.0
    spawn_time: float = 0.0
    despawn_time: Optional[float] = None
    active: bool = True

    def update(self, t: float, dt: float) -> None:
        self.active = t >= self.spawn_time and (self.despawn_time is None or t <= self.despawn_time)
        if self.active:
            self.x += self.vx * dt
            self.y += self.vy * dt


def make_crossing_scenario() -> List[CircleObstacle]:
    """Obstacle crosses the nominal path from start to target."""
    return [
        CircleObstacle(x=-1.5, y=0.4, radius=0.3, vx=0.7, vy=0.0, spawn_time=0.0),
    ]


def make_cutin_scenario() -> List[CircleObstacle]:
    """Obstacle appears near final approach and stays (cut-in)."""
    return [
        CircleObstacle(x=0.2, y=0.2, radius=0.35, vx=0.0, vy=0.0, spawn_time=6.0),
    ]


def make_cutin_short() -> List[CircleObstacle]:
    """Obstacle appears near final approach and disappears after a while."""
    return [
        CircleObstacle(x=0.2, y=0.2, radius=0.35, vx=0.0, vy=0.0, spawn_time=6.0, despawn_time=10.0),
    ]


def make_mixed_scenario(case_index: int) -> List[CircleObstacle]:
    """Deterministic mix of obstacle patterns (brief/long/persistent)."""
    rng = random.Random(case_index)
    mode = rng.choice(["crossing", "cutin_short", "cutin_persist"])

    if mode == "crossing":
        return make_crossing_scenario()
    if mode == "cutin_short":
        return make_cutin_short()
    return make_cutin_scenario()
