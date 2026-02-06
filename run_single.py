import argparse
import math
from typing import Tuple

import matplotlib.pyplot as plt

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.fsm import DockingFSM, FSMConfig
from src.safety import SafetyConfig
from src.obstacles import make_crossing_scenario, make_cutin_scenario
from src.actuation import ActuationNoise, ActuationNoiseConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docking demo (clean GT or Phase 2 sensor pipeline)")
    parser.add_argument("--x0", type=float, default=-2.0)
    parser.add_argument("--y0", type=float, default=-1.0)
    parser.add_argument("--theta0", type=float, default=math.radians(45))
    parser.add_argument("--xt", type=float, default=0.0)
    parser.add_argument("--yt", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max_time", type=float, default=20.0)
    parser.add_argument("--dist_tol", type=float, default=0.05)
    parser.add_argument("--yaw_tol", type=float, default=math.radians(5))
    parser.add_argument("--v_max", type=float, default=0.6)
    parser.add_argument("--w_max", type=float, default=1.5)
    parser.add_argument("--k_v", type=float, default=1.2)
    parser.add_argument("--k_w", type=float, default=2.0)
    # Sensor corruption (Phase 2)
    parser.add_argument("--use_sensor", action="store_true")
    parser.add_argument("--sigma_d", type=float, default=0.0)
    parser.add_argument("--sigma_yaw", type=float, default=0.0)
    parser.add_argument("--p_drop", type=float, default=0.0)
    parser.add_argument("--latency", type=int, default=0)
    parser.add_argument("--p_out", type=float, default=0.0)
    parser.add_argument("--outlier_dist", type=float, default=0.5)
    parser.add_argument("--outlier_yaw_deg", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    # Robust stack
    parser.add_argument("--use_filter", action="store_true")
    parser.add_argument("--ema_dist", type=float, default=0.4)
    parser.add_argument("--ema_yaw", type=float, default=0.4)
    parser.add_argument("--use_gating", action="store_true")
    parser.add_argument("--hold_last", action="store_true")
    parser.add_argument("--max_jump_dist", type=float, default=0.4)
    parser.add_argument("--max_jump_yaw_deg", type=float, default=30.0)
    parser.add_argument("--stale_jump_scale", type=float, default=0.0)
    parser.add_argument("--stale_decay", type=float, default=0.15)
    parser.add_argument("--stale_min_scale", type=float, default=0.2)
    parser.add_argument("--stale_stop_steps", type=int, default=6)
    parser.add_argument("--show_meas", action="store_true")
    # FSM
    parser.add_argument("--use_fsm", action="store_true")
    parser.add_argument("--align_dist", type=float, default=0.6)
    parser.add_argument("--finalize_dist", type=float, default=0.25)
    parser.add_argument("--yaw_align_deg", type=float, default=10.0)
    parser.add_argument("--finalize_speed_scale", type=float, default=0.35)
    parser.add_argument("--align_w_gain", type=float, default=2.0)
    # Safety + obstacles
    parser.add_argument("--use_safety", action="store_true")
    parser.add_argument("--r_stop", type=float, default=0.2)
    parser.add_argument("--r_slow", type=float, default=0.5)
    parser.add_argument("--v_slow", type=float, default=0.2)
    parser.add_argument("--w_slow", type=float, default=0.5)
    parser.add_argument("--robot_radius", type=float, default=0.2)
    parser.add_argument("--near_miss_dist", type=float, default=0.1)
    parser.add_argument("--scenario", type=str, default="none", choices=["none", "crossing", "cutin"])
    # Actuation noise (slip-like)
    parser.add_argument("--use_actuation_noise", action="store_true")
    parser.add_argument("--sigma_v", type=float, default=0.03)
    parser.add_argument("--sigma_w", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state0: Tuple[float, float, float] = (args.x0, args.y0, args.theta0)
    target = (args.xt, args.yt)

    max_steps = int(args.max_time / args.dt)
    control_params = {
        "v_max": args.v_max,
        "w_max": args.w_max,
        "k_v": args.k_v,
        "k_w": args.k_w,
        "stale_decay": args.stale_decay,
        "stale_min_scale": args.stale_min_scale,
        "stale_stop_steps": args.stale_stop_steps,
    }

    sensor = None
    estimator = None
    use_relative_control = False
    fsm = None
    safety_cfg = None
    obstacles = []
    actuation = None

    if args.use_sensor:
        sensor_cfg = SensorConfig(
            sigma_d=args.sigma_d,
            sigma_yaw=args.sigma_yaw,
            p_drop=args.p_drop,
            latency_steps=args.latency,
            p_outlier=args.p_out,
            outlier_dist=args.outlier_dist,
            outlier_yaw=math.radians(args.outlier_yaw_deg),
            seed=args.seed,
        )
        sensor = RelativePoseSensor(sensor_cfg)
        use_relative_control = True

        robust_cfg = RobustConfig(
            enable_filter=args.use_filter,
            enable_gating=args.use_gating,
            ema=EMAConfig(alpha_dist=args.ema_dist, alpha_yaw=args.ema_yaw),
            gating=GatingConfig(
                max_jump_dist=args.max_jump_dist,
                max_jump_yaw=math.radians(args.max_jump_yaw_deg),
                hold_last=args.hold_last,
                stale_jump_scale=args.stale_jump_scale,
            ),
        )
        estimator = RobustPoseEstimator(robust_cfg)

    if args.use_fsm:
        fsm_cfg = FSMConfig(
            align_dist=args.align_dist,
            finalize_dist=args.finalize_dist,
            yaw_align_thresh=math.radians(args.yaw_align_deg),
            finalize_speed_scale=args.finalize_speed_scale,
            align_w_gain=args.align_w_gain,
        )
        fsm = DockingFSM(fsm_cfg)

    if args.use_safety:
        safety_cfg = SafetyConfig(
            r_stop=args.r_stop,
            r_slow=args.r_slow,
            v_slow=args.v_slow,
            w_slow=args.w_slow,
            robot_radius=args.robot_radius,
            near_miss_dist=args.near_miss_dist,
        )
        if args.scenario == "crossing":
            obstacles = make_crossing_scenario()
        elif args.scenario == "cutin":
            obstacles = make_cutin_scenario()

    if args.use_actuation_noise:
        actuation = ActuationNoise(ActuationNoiseConfig(
            sigma_v=args.sigma_v,
            sigma_w=args.sigma_w,
            seed=args.seed,
        ))

    hist = simulate(
        state0=state0,
        target=target,
        dt=args.dt,
        max_steps=max_steps,
        control_params=control_params,
        dist_tol=args.dist_tol,
        yaw_tol=args.yaw_tol,
        sensor=sensor,
        estimator=estimator,
        use_relative_control=use_relative_control,
        fsm=fsm,
        safety_config=safety_cfg,
        obstacles=obstacles,
        actuation=actuation,
    )

    xs = hist["x"]
    ys = hist["y"]
    ts = hist["time"]
    dists = hist["dist_true"]
    yaws = hist["yaw_true"]
    meas_dists = hist["dist_meas"]
    meas_yaws = hist["yaw_meas"]
    est_dists = hist["dist_est"]
    est_yaws = hist["yaw_est"]

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(xs, ys, label="trajectory", linewidth=2)
    ax0.scatter([xs[0]], [ys[0]], c="tab:green", label="start")
    ax0.scatter([xs[-1]], [ys[-1]], c="tab:blue", label="end")
    ax0.scatter([target[0]], [target[1]], c="tab:red", marker="*", s=120, label="target")
    if hist.get("obstacles"):
        for obs in hist["obstacles"]:
            ax0.plot(obs["x"], obs["y"], linestyle="--", color="tab:gray", alpha=0.6)
    ax0.set_title("2D Trajectory")
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(ts, dists, color="tab:orange", label="dist_true")
    if args.show_meas and args.use_sensor:
        ax1.plot(ts, meas_dists, color="tab:gray", alpha=0.6, label="dist_meas")
    if args.use_sensor:
        ax1.plot(ts, est_dists, color="tab:blue", alpha=0.8, label="dist_est")
    ax1.set_title("dist_true vs time")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("distance (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(ts, yaws, color="tab:purple", label="yaw_true")
    if args.show_meas and args.use_sensor:
        ax2.plot(ts, meas_yaws, color="tab:gray", alpha=0.6, label="yaw_meas")
    if args.use_sensor:
        ax2.plot(ts, est_yaws, color="tab:blue", alpha=0.8, label="yaw_est")
    ax2.set_title("yaw_true vs time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("yaw (rad)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    success = hist["success"]
    status = "SUCCESS" if success else "TIMEOUT"
    fig.suptitle(f"Docking (clean GT) - {status}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
