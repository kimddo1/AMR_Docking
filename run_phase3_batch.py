import argparse
import csv
import math
import os
import random
from typing import Dict, List, Tuple

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.fsm import DockingFSM, FSMConfig
from src.safety import SafetyConfig
from src.obstacles import make_crossing_scenario, make_cutin_scenario

State = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 batch evaluation with dynamic obstacles")
    p.add_argument("--num_trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario", type=str, default="crossing", choices=["crossing", "cutin"])
    p.add_argument("--safety", type=str, default="on", choices=["on", "off"])
    p.add_argument("--out_csv", type=str, default="outputs/phase3_batch.csv")
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


def main() -> None:
    args = parse_args()
    states = sample_initial_states(args.num_trials, args.seed)

    sensor_cfg = SensorConfig(
        sigma_d=0.05,
        sigma_yaw=0.1,
        p_drop=0.2,
        p_outlier=0.05,
        outlier_dist=0.6,
        outlier_yaw=math.radians(25),
        seed=0,
    )

    robust_cfg = RobustConfig(
        enable_filter=True,
        enable_gating=True,
        ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
        gating=GatingConfig(
            max_jump_dist=0.3,
            max_jump_yaw=math.radians(20),
            hold_last=True,
            stale_jump_scale=0.15,
        ),
    )

    # Safety config: if safety=off, keep thresholds negative so no override but still measure distances
    if args.safety == "on":
        safety_cfg = SafetyConfig(r_stop=0.2, r_slow=0.5, v_slow=0.2, w_slow=0.5)
    else:
        safety_cfg = SafetyConfig(r_stop=-1.0, r_slow=-1.0, v_slow=10.0, w_slow=10.0)

    rows: List[Dict] = []

    for i, state0 in enumerate(states):
        sensor_cfg.seed = i
        sensor = RelativePoseSensor(sensor_cfg)
        estimator = RobustPoseEstimator(robust_cfg)
        fsm = DockingFSM(FSMConfig())

        if args.scenario == "crossing":
            obstacles = make_crossing_scenario()
        else:
            obstacles = make_cutin_scenario()

        hist = simulate(
            state0=state0,
            target=(0.0, 0.0),
            dt=0.05,
            max_steps=int(25.0 / 0.05),
            control_params={
                "v_max": 0.6,
                "w_max": 1.5,
                "k_v": 1.2,
                "k_w": 2.0,
                "yaw_slow": math.radians(35),
                "stale_decay": 0.2,
                "stale_min_scale": 0.2,
                "stale_stop_steps": 6,
            },
            dist_tol=0.05,
            yaw_tol=math.radians(5),
            sensor=sensor,
            estimator=estimator,
            use_relative_control=True,
            fsm=fsm,
            safety_config=safety_cfg,
            obstacles=obstacles,
        )

        row = {
            "trial": i,
            "success": hist["success"],
            "steps": len(hist["time"]),
            "collision": any(hist["collision"]),
            "near_miss": any(hist["near_miss"]),
            "stop_count": hist["stop_count"],
            "slow_count": hist["slow_count"],
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("saved", args.out_csv)


if __name__ == "__main__":
    main()
