import math
from typing import Dict, List, Optional, Tuple
from .dynamics import step_differential_drive
from .utils import wrap_to_pi
from .control import go_to_goal_control, relative_pose_control
from .sensor import RelativePoseSensor
from .robust import RobustPoseEstimator
from .fsm import DockingFSM
from .safety import SafetyConfig, safety_step, Obstacle
from .actuation import ActuationNoise

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
    sensor: Optional[RelativePoseSensor] = None,
    estimator: Optional[RobustPoseEstimator] = None,
    use_relative_control: bool = False,
    fsm: Optional[DockingFSM] = None,
    safety_config: Optional[SafetyConfig] = None,
    obstacles: Optional[List[Obstacle]] = None,
    actuation: Optional[ActuationNoise] = None,
) -> Dict[str, List[float]]:
    """Run docking simulation with optional sensor, FSM, and safety supervisor."""
    state = state0

    xs: List[float] = []
    ys: List[float] = []
    thetas: List[float] = []
    dists: List[float] = []
    yaws: List[float] = []
    times: List[float] = []
    meas_dists: List[Optional[float]] = []
    meas_yaws: List[Optional[float]] = []
    est_dists: List[Optional[float]] = []
    est_yaws: List[Optional[float]] = []
    gate_rejected: List[bool] = []
    meas_missing: List[bool] = []
    fsm_states: List[str] = []
    safety_modes: List[str] = []
    min_dists: List[Optional[float]] = []
    collisions: List[bool] = []
    near_misses: List[bool] = []

    obs_trajs: List[Dict[str, List[Optional[float]]]] = []
    if obstacles is None:
        obstacles = []
    if obstacles:
        obs_trajs = [{"x": [], "y": [], "active": []} for _ in obstacles]

    success = False
    stop_count = 0
    slow_count = 0

    for k in range(max_steps):
        t = k * dt
        if obstacles:
            for obs in obstacles:
                obs.update(t, dt)

            for i, obs in enumerate(obstacles):
                if getattr(obs, "active", True):
                    obs_trajs[i]["x"].append(getattr(obs, "x", None))
                    obs_trajs[i]["y"].append(getattr(obs, "y", None))
                    obs_trajs[i]["active"].append(True)
                else:
                    obs_trajs[i]["x"].append(None)
                    obs_trajs[i]["y"].append(None)
                    obs_trajs[i]["active"].append(False)
        dist, yaw = relative_pose(state, target)
        meas = (dist, yaw)
        if sensor is not None:
            meas = sensor.measure(dist, yaw)

        if estimator is not None:
            est, info = estimator.step(meas)
        else:
            est, info = meas, {
                "input_missing": meas is None,
                "rejected": False,
                "used_hold": False,
                "used_filter": False,
                "stale_count": 0,
            }

        xs.append(state[0])
        ys.append(state[1])
        thetas.append(state[2])
        dists.append(dist)
        yaws.append(yaw)
        times.append(t)
        if meas is None:
            meas_dists.append(None)
            meas_yaws.append(None)
        else:
            meas_dists.append(meas[0])
            meas_yaws.append(meas[1])
        if est is None:
            est_dists.append(None)
            est_yaws.append(None)
        else:
            est_dists.append(est[0])
            est_yaws.append(est[1])
        gate_rejected.append(bool(info.get("rejected", False)))
        meas_missing.append(bool(info.get("input_missing", False)))

        if dist <= dist_tol and abs(yaw) <= yaw_tol:
            fsm_states.append("DONE")
            safety_modes.append("DONE")
            min_dists.append(None)
            collisions.append(False)
            near_misses.append(False)
            success = True
            break

        if use_relative_control:
            if est is None:
                v, w = 0.0, 0.0
            else:
                v, w = relative_pose_control(est[0], est[1], control_params)
                stale_count = int(info.get("stale_count", 0))
                if stale_count > 0:
                    stop_steps = int(control_params.get("stale_stop_steps", 6))
                    if stale_count >= stop_steps:
                        v, w = 0.0, 0.0
                    else:
                        decay = float(control_params.get("stale_decay", 0.15))
                        min_scale = float(control_params.get("stale_min_scale", 0.2))
                        scale = max(min_scale, 1.0 - decay * stale_count)
                        v *= scale
                        w *= scale
        else:
            v, w = go_to_goal_control(state, target, control_params)

        if fsm is not None:
            w_max = float(control_params.get("w_max", 1.5))
            dist_ref = est[0] if est is not None else None
            yaw_ref = est[1] if est is not None else None
            v, w, fsm_state = fsm.step(dist_ref, yaw_ref, v, w, w_max)
        else:
            fsm_state = "NONE"
        fsm_states.append(fsm_state)

        if safety_config is not None:
            v, w, mode, min_dist, collision, near_miss = safety_step(
                v, w, state[0], state[1], obstacles, safety_config
            )
            safety_modes.append(mode)
            min_dists.append(min_dist)
            collisions.append(collision)
            near_misses.append(near_miss)
            if mode == "STOP":
                stop_count += 1
            elif mode == "SLOW":
                slow_count += 1
        else:
            safety_modes.append("NONE")
            min_dists.append(None)
            collisions.append(False)
            near_misses.append(False)
        state = step_differential_drive(state, (v, w), dt, actuation=actuation)

    return {
        "x": xs,
        "y": ys,
        "theta": thetas,
        "dist_true": dists,
        "yaw_true": yaws,
        "time": times,
        "dist_meas": meas_dists,
        "yaw_meas": meas_yaws,
        "dist_est": est_dists,
        "yaw_est": est_yaws,
        "gate_rejected": gate_rejected,
        "meas_missing": meas_missing,
        "fsm_state": fsm_states,
        "safety_mode": safety_modes,
        "min_dist_obs": min_dists,
        "collision": collisions,
        "near_miss": near_misses,
        "stop_count": stop_count,
        "slow_count": slow_count,
        "obstacles": obs_trajs,
        "success": success,
    }
