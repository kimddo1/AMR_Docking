import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RecoveryConfig:
    stop_persist_steps: int = 20
    clear_confirm_steps: int = 5
    wait_steps: int = 10
    turn_angle_deg: float = 90.0
    turn_speed: float = 0.9
    shift_distance: float = 0.4
    shift_speed: float = 0.15
    turnback_k: float = 2.0
    turnback_yaw_tol_deg: float = 6.0
    turnback_hold_steps: int = 4
    turnback_max_steps: int = 120
    max_retries: int = 3
    cooldown_steps: int = 20
    max_blocked_steps: int = 700


class RecoveryManager:
    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.mode = "IDLE"
        self.mode_step = 0
        self.stop_streak = 0
        self.retries = 0
        self.deadlock = False
        self.recovery_count = 0
        self.cooldown = 0
        self.recovery_steps_total = 0
        self.blocked_steps = 0
        self.clear_streak = 0
        self.no_improve_cycles = 0
        self.maneuver_dir = 1.0
        self.turn_out_progress = 0.0
        self.shift_progress = 0.0
        self.turn_back_progress = 0.0
        self.turn_back_aligned_hold = 0
        # Debug/analysis hooks (last update)
        self.last_trigger = False
        self.last_stalled = False
        self.last_stop_streak = 0
        self.last_trigger_reason = "NONE"
        self.last_cycle_improved = False

    def _enter_mode(self, mode: str) -> None:
        self.mode = mode
        self.mode_step = 0

    def _start_maneuver(self) -> None:
        self.retries += 1
        self.recovery_count += 1
        # Alternate side to avoid repeated same-side traps.
        self.maneuver_dir = -1.0 if (self.recovery_count % 2 == 0) else 1.0
        self.turn_out_progress = 0.0
        self.shift_progress = 0.0
        self.turn_back_progress = 0.0
        self.turn_back_aligned_hold = 0
        self._enter_mode("WAIT")

    def _finish_cycle(self, improved: bool = True) -> None:
        self._enter_mode("IDLE")
        self.cooldown = self.config.cooldown_steps
        self.last_cycle_improved = improved
        if improved:
            self.retries = 0

    def update_and_override(
        self,
        dist: float,
        safety_mode: str,
        yaw_to_target: float,
        dt: float,
    ) -> Tuple[Optional[float], Optional[float], str, bool]:
        """Return (v_override, w_override, recovery_mode, deadlock)."""
        if self.deadlock:
            return 0.0, 0.0, "DEADLOCK", True

        self.last_cycle_improved = False
        self.last_stalled = False

        if safety_mode == "STOP":
            self.stop_streak += 1
            self.blocked_steps += 1
            self.clear_streak = 0
        else:
            self.stop_streak = 0
            self.blocked_steps = 0
            self.clear_streak += 1

        # Trigger on persistent STOP (global), not just near target.
        trigger_stop = self.stop_streak >= self.config.stop_persist_steps
        trigger = trigger_stop and self.cooldown == 0
        # expose debug state
        self.last_trigger = trigger
        self.last_stop_streak = self.stop_streak
        if trigger_stop:
            self.last_trigger_reason = "STOP_PERSIST"
        else:
            self.last_trigger_reason = "NONE"

        if self.mode == "IDLE" and self.cooldown > 0:
            self.cooldown -= 1

        if self.mode == "IDLE" and trigger:
            if self.retries >= self.config.max_retries:
                # Keep behavior safe and predictable if blocked for too long.
                if self.blocked_steps >= self.config.max_blocked_steps:
                    self.deadlock = True
                    return 0.0, 0.0, "DEADLOCK", True
                return 0.0, 0.0, "BLOCKED_WAIT", False
            self._start_maneuver()

        if self.mode != "IDLE":
            self.recovery_steps_total += 1

        if self.mode == "WAIT":
            # If obstacle clearance is confirmed, return control quickly.
            if self.clear_streak >= self.config.clear_confirm_steps:
                self._finish_cycle(improved=True)
                return None, None, "RECOVERED_CLEAR", False
            self.mode_step += 1
            if self.mode_step >= self.config.wait_steps:
                self._enter_mode("TURN_OUT")
            return 0.0, 0.0, "WAIT", False

        turn_goal = math.radians(max(1.0, self.config.turn_angle_deg))

        if self.mode == "TURN_OUT":
            w_cmd = self.maneuver_dir * abs(self.config.turn_speed)
            self.turn_out_progress += abs(w_cmd) * dt
            if self.turn_out_progress >= turn_goal:
                self._enter_mode("SHIFT")
            return 0.0, w_cmd, "TURN_OUT", False

        if self.mode == "SHIFT":
            if self.clear_streak >= self.config.clear_confirm_steps:
                self._finish_cycle(improved=True)
                return None, None, "RECOVERED_CLEAR", False
            v_cmd = abs(self.config.shift_speed)
            self.shift_progress += v_cmd * dt
            if self.shift_progress >= max(0.05, self.config.shift_distance):
                self._enter_mode("TURN_BACK")
            return v_cmd, 0.0, "SHIFT", False

        if self.mode == "TURN_BACK":
            # Align based on current relative yaw after side-shift (recomputed each step).
            w_cmd = self.config.turnback_k * yaw_to_target
            w_cmd = max(-abs(self.config.turn_speed), min(abs(self.config.turn_speed), w_cmd))
            self.turn_back_progress += 1

            yaw_tol = math.radians(max(0.5, self.config.turnback_yaw_tol_deg))
            if abs(yaw_to_target) <= yaw_tol:
                self.turn_back_aligned_hold += 1
            else:
                self.turn_back_aligned_hold = 0

            if self.turn_back_aligned_hold >= max(1, self.config.turnback_hold_steps):
                self._finish_cycle(improved=True)
                return None, None, "TURN_BACK_ALIGNED", False
            if self.turn_back_progress >= max(1, self.config.turnback_max_steps):
                # Timeout guard to avoid getting stuck in this mode forever.
                self._finish_cycle(improved=False)
                return None, None, "TURN_BACK_TIMEOUT", False
            return 0.0, w_cmd, "TURN_BACK", False

        return None, None, "IDLE", False
