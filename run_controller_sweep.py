import argparse
import csv
import itertools
import math
import os
import random
from typing import Dict, Iterable, List, Tuple

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
    p = argparse.ArgumentParser(description="Controller parameter sweep over random initial states")
    p.add_argument("--num_trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_time", type=float, default=25.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--dist_tol", type=float, default=0.05)
    p.add_argument("--yaw_tol_deg", type=float, default=5.0)

    # Controller grid
    p.add_argument("--v_max", type=str, default="0.4,0.6,0.8")
    p.add_argument("--w_max", type=str, default="1.0,1.5,2.0")
    p.add_argument("--k_v", type=str, default="0.8,1.2,1.6")
    p.add_argument("--k_w", type=str, default="1.5,2.0,2.5")
    p.add_argument("--yaw_slow_deg", type=str, default="25,35,45")

    # Sensor corruption
    p.add_argument("--sigma_d", type=float, default=0.05)
    p.add_argument("--sigma_yaw", type=float, default=0.1)
    p.add_argument("--p_drop", type=float, default=0.2)
    p.add_argument("--latency", type=int, default=0)
    p.add_argument("--p_out", type=float, default=0.05)
    p.add_argument("--outlier_dist", type=float, default=0.6)
    p.add_argument("--outlier_yaw_deg", type=float, default=25.0)

    # Robust stack
    p.add_argument("--ema_dist", type=float, default=0.3)
    p.add_argument("--ema_yaw", type=float, default=0.3)
    p.add_argument("--max_jump_dist", type=float, default=0.3)
    p.add_argument("--max_jump_yaw_deg", type=float, default=20.0)
    p.add_argument("--hold_last", action="store_true", default=True)
    p.add_argument("--stale_jump_scale", type=float, default=0.15)
    p.add_argument("--stale_decay", type=float, default=0.2)
    p.add_argument("--stale_min_scale", type=float, default=0.2)
    p.add_argument("--stale_stop_steps", type=int, default=6)

    # Scoring weights
    p.add_argument("--w_success", type=float, default=1000.0)
    p.add_argument("--w_steps", type=float, default=-1.0)
    p.add_argument("--w_dist", type=float, default=-50.0)
    p.add_argument("--w_yaw", type=float, default=-50.0)

    p.add_argument("--out_csv", type=str, default="outputs/controller_sweep.csv")
    p.add_argument("--out_plot", type=str, default="outputs/controller_sweep.png")
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


def score_row(row: Dict, w_success: float, w_steps: float, w_dist: float, w_yaw: float) -> float:
    return (
        w_success * row["success_rate"]
        + w_steps * row["steps_mean"]
        + w_dist * row["final_dist_mean"]
        + w_yaw * row["final_yaw_mean"]
    )


def run_sweep(args: argparse.Namespace) -> List[Dict]:
    states = sample_initial_states(args.num_trials, args.seed)

    v_max_list = parse_float_list(args.v_max)
    w_max_list = parse_float_list(args.w_max)
    k_v_list = parse_float_list(args.k_v)
    k_w_list = parse_float_list(args.k_w)
    yaw_slow_list = [math.radians(v) for v in parse_float_list(args.yaw_slow_deg)]

    results: List[Dict] = []

    for v_max, w_max, k_v, k_w, yaw_slow in itertools.product(
        v_max_list, w_max_list, k_v_list, k_w_list, yaw_slow_list
    ):
        successes = 0
        final_dist = []
        final_yaw = []
        steps = []
        reject_frac = []
        missing_frac = []

        for i, state0 in enumerate(states):
            sensor_cfg = SensorConfig(
                sigma_d=args.sigma_d,
                sigma_yaw=args.sigma_yaw,
                p_drop=args.p_drop,
                latency_steps=args.latency,
                p_outlier=args.p_out,
                outlier_dist=args.outlier_dist,
                outlier_yaw=math.radians(args.outlier_yaw_deg),
                seed=i,
            )
            sensor = RelativePoseSensor(sensor_cfg)
            robust_cfg = RobustConfig(
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
            estimator = RobustPoseEstimator(robust_cfg)

            control_params = {
                "v_max": v_max,
                "w_max": w_max,
                "k_v": k_v,
                "k_w": k_w,
                "yaw_slow": yaw_slow,
                "stale_decay": args.stale_decay,
                "stale_min_scale": args.stale_min_scale,
                "stale_stop_steps": args.stale_stop_steps,
            }

            hist = simulate(
                state0=state0,
                target=(0.0, 0.0),
                dt=args.dt,
                max_steps=int(args.max_time / args.dt),
                control_params=control_params,
                dist_tol=args.dist_tol,
                yaw_tol=math.radians(args.yaw_tol_deg),
                sensor=sensor,
                estimator=estimator,
                use_relative_control=True,
            )

            if hist["success"]:
                successes += 1
                final_dist.append(hist["dist_true"][-1])
                final_yaw.append(abs(hist["yaw_true"][-1]))
            steps.append(len(hist["time"]))

            if len(hist["gate_rejected"]) > 0:
                reject_frac.append(sum(hist["gate_rejected"]) / len(hist["gate_rejected"]))
            if len(hist["meas_missing"]) > 0:
                missing_frac.append(sum(hist["meas_missing"]) / len(hist["meas_missing"]))

        row = {
            "v_max": v_max,
            "w_max": w_max,
            "k_v": k_v,
            "k_w": k_w,
            "yaw_slow_deg": math.degrees(yaw_slow),
            "success_rate": successes / args.num_trials,
            "final_dist_mean": sum(final_dist) / len(final_dist) if final_dist else float("nan"),
            "final_yaw_mean": sum(final_yaw) / len(final_yaw) if final_yaw else float("nan"),
            "steps_mean": sum(steps) / len(steps) if steps else float("nan"),
            "reject_frac_mean": sum(reject_frac) / len(reject_frac) if reject_frac else float("nan"),
            "missing_frac_mean": sum(missing_frac) / len(missing_frac) if missing_frac else float("nan"),
        }
        row["score"] = score_row(row, args.w_success, args.w_steps, args.w_dist, args.w_yaw)
        results.append(row)

    return results


def write_csv(rows: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_plot(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    top = rows_sorted[:20]

    labels = [
        f"v{r['v_max']}_w{r['w_max']}_kv{r['k_v']}_kw{r['k_w']}_ys{int(r['yaw_slow_deg'])}"
        for r in top
    ]
    success = [r["success_rate"] for r in top]
    steps = [r["steps_mean"] for r in top]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(labels, success, color="tab:green", alpha=0.7)
    ax1.set_ylabel("success_rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(labels, steps, color="tab:blue", marker="o")
    ax2.set_ylabel("steps_mean")

    fig.suptitle("Top 20 Controller Settings by Score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    write_csv(rows, args.out_csv)
    make_plot(rows, args.out_plot)

    best = max(rows, key=lambda r: r["score"]) if rows else None
    if best:
        print("BEST", best)
        print("saved", args.out_csv)
        print("saved", args.out_plot)


if __name__ == "__main__":
    main()
