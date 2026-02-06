import argparse
import csv
import math
import os
import random
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig

State = Tuple[float, float, float]


def parse_float_list(value: str) -> List[float]:
    parts = [v.strip() for v in value.split(',') if v.strip()]
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep noise/dropout for baseline vs improved policy")
    p.add_argument("--num_trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_time", type=float, default=25.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--dist_tol", type=float, default=0.05)
    p.add_argument("--yaw_tol_deg", type=float, default=5.0)

    # Sweep ranges
    p.add_argument("--sigma_d", type=str, default="0.0,0.02,0.05,0.08")
    p.add_argument("--sigma_yaw", type=str, default="0.0,0.05,0.1,0.2")
    p.add_argument("--p_drop", type=str, default="0.0,0.1,0.2,0.3,0.4")

    # Fixed outlier settings
    p.add_argument("--p_out", type=float, default=0.05)
    p.add_argument("--outlier_dist", type=float, default=0.6)
    p.add_argument("--outlier_yaw_deg", type=float, default=25.0)

    # Controller
    p.add_argument("--v_max", type=float, default=0.6)
    p.add_argument("--w_max", type=float, default=1.5)
    p.add_argument("--k_v", type=float, default=1.2)
    p.add_argument("--k_w", type=float, default=2.0)
    p.add_argument("--yaw_slow_deg", type=float, default=35.0)

    # Robust stack
    p.add_argument("--ema_dist", type=float, default=0.3)
    p.add_argument("--ema_yaw", type=float, default=0.3)
    p.add_argument("--max_jump_dist", type=float, default=0.3)
    p.add_argument("--max_jump_yaw_deg", type=float, default=20.0)
    p.add_argument("--hold_last", action="store_true", default=True)

    # Improved policy params
    p.add_argument("--stale_jump_scale", type=float, default=0.15)
    p.add_argument("--stale_decay", type=float, default=0.2)
    p.add_argument("--stale_min_scale", type=float, default=0.2)
    p.add_argument("--stale_stop_steps", type=int, default=6)

    p.add_argument("--out_csv", type=str, default="outputs/sweep_noise_dropout.csv")
    p.add_argument("--out_plot", type=str, default="outputs/sweep_noise_dropout.png")
    return p.parse_args()


def sample_initial_states(num_trials: int, seed: int) -> List[State]:
    rng = random.Random(seed)
    states = []
    while len(states) < num_trials:
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        if math.hypot(x, y) < 0.6:
            continue
        theta = rng.uniform(-math.pi, math.pi)
        states.append((x, y, theta))
    return states


def run_policy(states: List[State], sensor_cfg: SensorConfig, robust_cfg: RobustConfig, control_params: Dict,
               dt: float, max_steps: int, dist_tol: float, yaw_tol_deg: float) -> Dict:
    successes = 0
    steps = []
    for i, state0 in enumerate(states):
        sensor = RelativePoseSensor(sensor_cfg)
        estimator = RobustPoseEstimator(robust_cfg)
        hist = simulate(
            state0=state0,
            target=(0.0, 0.0),
            dt=dt,
            max_steps=max_steps,
            control_params=control_params,
            dist_tol=dist_tol,
            yaw_tol=math.radians(yaw_tol_deg),
            sensor=sensor,
            estimator=estimator,
            use_relative_control=True,
        )
        if hist["success"]:
            successes += 1
        steps.append(len(hist["time"]))
    return {
        "success_rate": successes / len(states),
        "steps_mean": sum(steps) / len(steps),
    }


def main() -> None:
    args = parse_args()
    states = sample_initial_states(args.num_trials, args.seed)

    sigma_d_list = parse_float_list(args.sigma_d)
    sigma_yaw_list = parse_float_list(args.sigma_yaw)
    p_drop_list = parse_float_list(args.p_drop)

    control_params = {
        "v_max": args.v_max,
        "w_max": args.w_max,
        "k_v": args.k_v,
        "k_w": args.k_w,
        "yaw_slow": math.radians(args.yaw_slow_deg),
        "stale_decay": args.stale_decay,
        "stale_min_scale": args.stale_min_scale,
        "stale_stop_steps": args.stale_stop_steps,
    }

    baseline_cfg = RobustConfig(
        enable_filter=True,
        enable_gating=True,
        ema=EMAConfig(alpha_dist=args.ema_dist, alpha_yaw=args.ema_yaw),
        gating=GatingConfig(
            max_jump_dist=args.max_jump_dist,
            max_jump_yaw=math.radians(args.max_jump_yaw_deg),
            hold_last=args.hold_last,
            stale_jump_scale=0.0,
        ),
    )

    improved_cfg = RobustConfig(
        enable_filter=True,
        enable_gating=True,
        ema=EMAConfig(alpha_dist=args.ema_dist, alpha_yaw=args.ema_yaw),
        gating=GatingConfig(
            max_jump_dist=args.max_jump_dist,
            max_jump_yaw=math.radians(args.max_jump_yaw_deg),
            hold_last=args.hold_last,
            stale_jump_scale=args.stale_jump_scale,
        ),
    )

    rows = []
    for sigma_d in sigma_d_list:
        for sigma_yaw in sigma_yaw_list:
            for p_drop in p_drop_list:
                sensor_cfg = SensorConfig(
                    sigma_d=sigma_d,
                    sigma_yaw=sigma_yaw,
                    p_drop=p_drop,
                    p_outlier=args.p_out,
                    outlier_dist=args.outlier_dist,
                    outlier_yaw=math.radians(args.outlier_yaw_deg),
                    seed=0,
                )

                base = run_policy(states, sensor_cfg, baseline_cfg, control_params, args.dt,
                                  int(args.max_time / args.dt), args.dist_tol, args.yaw_tol_deg)
                imp = run_policy(states, sensor_cfg, improved_cfg, control_params, args.dt,
                                 int(args.max_time / args.dt), args.dist_tol, args.yaw_tol_deg)

                rows.append({
                    "sigma_d": sigma_d,
                    "sigma_yaw": sigma_yaw,
                    "p_drop": p_drop,
                    "success_rate_baseline": base["success_rate"],
                    "success_rate_improved": imp["success_rate"],
                    "success_gain": imp["success_rate"] - base["success_rate"],
                    "steps_mean_baseline": base["steps_mean"],
                    "steps_mean_improved": imp["steps_mean"],
                    "steps_gain": base["steps_mean"] - imp["steps_mean"],
                })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Plot: success gain heatmap for each p_drop
    fig, axes = plt.subplots(1, len(p_drop_list), figsize=(4 * len(p_drop_list), 4), sharey=True)
    if len(p_drop_list) == 1:
        axes = [axes]

    for ax, p_drop in zip(axes, p_drop_list):
        # build grid
        grid = [[0.0 for _ in sigma_yaw_list] for _ in sigma_d_list]
        for r in rows:
            if r["p_drop"] != p_drop:
                continue
            i = sigma_d_list.index(r["sigma_d"])
            j = sigma_yaw_list.index(r["sigma_yaw"])
            grid[i][j] = r["success_gain"]

        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm",
                       extent=[min(sigma_yaw_list), max(sigma_yaw_list),
                               min(sigma_d_list), max(sigma_d_list)])
        ax.set_title(f"p_drop={p_drop}")
        ax.set_xlabel("sigma_yaw")
        ax.set_ylabel("sigma_d")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Success Gain (Improved - Baseline)")
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=150)

    print("saved", args.out_csv)
    print("saved", args.out_plot)


if __name__ == "__main__":
    main()
