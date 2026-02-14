from dataclasses import dataclass
from typing import Deque, Optional, Tuple
from collections import deque


@dataclass
class RecoveryConfig:
    stop_persist_steps: int = 20
    progress_window: int = 30
    min_progress: float = 0.05
    wait_steps: int = 10
    rotate_steps: int = 20
    backoff_steps: int = 15
    backoff_speed: float = 0.15
    rotate_speed: float = 0.8
    reapproach_steps: int = 15
    max_retries: int = 3


class RecoveryManager:
    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.mode = "IDLE"
        self.mode_step = 0
        self.stop_streak = 0
        self.dist_hist: Deque[float] = deque(maxlen=config.progress_window)
        self.retries = 0
        self.deadlock = False
        self.recovery_count = 0
        # Debug/analysis hooks (last update)
        self.last_trigger = False
        self.last_stalled = False
        self.last_stop_streak = 0

    def _progress_stalled(self) -> bool:
        if len(self.dist_hist) < self.config.progress_window:
            return False
        return (self.dist_hist[0] - self.dist_hist[-1]) < self.config.min_progress

    def _enter_mode(self, mode: str) -> None:
        self.mode = mode
        self.mode_step = 0

    def update_and_override(
        self,
        dist: float,
        safety_mode: str,
    ) -> Tuple[Optional[float], Optional[float], str, bool]:
        """Return (v_override, w_override, recovery_mode, deadlock)."""
        if self.deadlock:
            return 0.0, 0.0, "DEADLOCK", True

        self.dist_hist.append(dist)

        if safety_mode == "STOP":
            self.stop_streak += 1
        else:
            self.stop_streak = 0

        stalled = self._progress_stalled()
        trigger = (self.stop_streak >= self.config.stop_persist_steps) or stalled
        # expose debug state
        self.last_trigger = trigger
        self.last_stalled = stalled
        self.last_stop_streak = self.stop_streak

        if self.mode == "IDLE" and trigger:
            if self.retries >= self.config.max_retries:
                self.deadlock = True
                return 0.0, 0.0, "DEADLOCK", True
            self.retries += 1
            self.recovery_count += 1
            self._enter_mode("WAIT")

        if self.mode == "WAIT":
            self.mode_step += 1
            if self.mode_step >= self.config.wait_steps:
                self._enter_mode("ROTATE")
            return 0.0, 0.0, "WAIT", False

        if self.mode == "ROTATE":
            self.mode_step += 1
            if self.mode_step >= self.config.rotate_steps:
                self._enter_mode("BACKOFF")
            # alternate rotation direction each retry
            direction = -1.0 if (self.retries % 2 == 0) else 1.0
            return 0.0, direction * self.config.rotate_speed, "ROTATE", False

        if self.mode == "BACKOFF":
            self.mode_step += 1
            if self.mode_step >= self.config.backoff_steps:
                self._enter_mode("REAPPROACH")
            return -self.config.backoff_speed, 0.0, "BACKOFF", False

        if self.mode == "REAPPROACH":
            self.mode_step += 1
            if self.mode_step >= self.config.reapproach_steps:
                self._enter_mode("IDLE")
            return None, None, "REAPPROACH", False

        return None, None, "IDLE", False
