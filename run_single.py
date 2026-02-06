import argparse
import math
from typing import Tuple

import matplotlib.pyplot as plt

from src.sim import simulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 clean-GT docking demo")
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
    }

    hist = simulate(
        state0=state0,
        target=target,
        dt=args.dt,
        max_steps=max_steps,
        control_params=control_params,
        dist_tol=args.dist_tol,
        yaw_tol=args.yaw_tol,
    )

    xs = hist["x"]
    ys = hist["y"]
    ts = hist["time"]
    dists = hist["dist_true"]
    yaws = hist["yaw_true"]

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(xs, ys, label="trajectory", linewidth=2)
    ax0.scatter([xs[0]], [ys[0]], c="tab:green", label="start")
    ax0.scatter([xs[-1]], [ys[-1]], c="tab:blue", label="end")
    ax0.scatter([target[0]], [target[1]], c="tab:red", marker="*", s=120, label="target")
    ax0.set_title("2D Trajectory")
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(ts, dists, color="tab:orange")
    ax1.set_title("dist_true vs time")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("distance (m)")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(ts, yaws, color="tab:purple")
    ax2.set_title("yaw_true vs time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("yaw (rad)")
    ax2.grid(True, alpha=0.3)

    success = hist["success"]
    status = "SUCCESS" if success else "TIMEOUT"
    fig.suptitle(f"Docking (clean GT) - {status}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
