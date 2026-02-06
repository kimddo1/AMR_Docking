import argparse
import json
import math
import os
from typing import Dict, List, Tuple

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.test_cases import generate_test_cases, load_test_cases
from src.actuation import ActuationNoise, ActuationNoiseConfig

State = Tuple[float, float, float]


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate noise strength to target baseline success rate")
    p.add_argument("--cases", type=str, default="outputs/test_cases.csv")
    p.add_argument("--num_trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_min", type=float, default=0.6)
    p.add_argument("--target_max", type=float, default=0.9)
    p.add_argument("--max_time", type=float, default=35.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--dist_tol", type=float, default=0.05)
    p.add_argument("--yaw_tol_deg", type=float, default=5.0)

    # Base corruption values
    p.add_argument("--sigma_d", type=float, default=0.05)
    p.add_argument("--sigma_yaw", type=float, default=0.1)
    p.add_argument("--p_drop", type=float, default=0.2)
    p.add_argument("--p_out", type=float, default=0.05)
    p.add_argument("--outlier_dist", type=float, default=0.6)
    p.add_argument("--outlier_yaw_deg", type=float, default=25.0)

    # Multipliers to search
    p.add_argument("--scales", type=str, default="0.6,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0,3.5,4.0")
    p.add_argument("--p_drop_cap", type=float, default=0.6)

    # Actuation noise (enabled during calibration to reflect reality)
    p.add_argument("--sigma_v", type=float, default=0.03)
    p.add_argument("--sigma_w", type=float, default=0.05)
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def ensure_cases(path: str, num: int, seed: int) -> List[State]:
    if os.path.exists(path):
        return load_test_cases(path)
    return generate_test_cases(path, num, seed)


def run_success_rate(states: List[State], sensor_cfg: SensorConfig, actuation: ActuationNoise, args) -> float:
    successes = 0

    for i, s in enumerate(states):
        sensor_cfg.seed = i
        sensor = RelativePoseSensor(sensor_cfg)

        hist = simulate(
            state0=s,
            target=(0.0, 0.0),
            dt=args.dt,
            max_steps=int(args.max_time / args.dt),
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
            dist_tol=args.dist_tol,
            yaw_tol=math.radians(args.yaw_tol_deg),
            sensor=sensor,
            estimator=None,
            use_relative_control=True,
            actuation=actuation,
        )
        if hist["success"]:
            successes += 1

    return successes / len(states)


def main():
    args = parse_args()
    states = ensure_cases(args.cases, args.num_trials, args.seed)

    actuation = ActuationNoise(ActuationNoiseConfig(
        sigma_v=args.sigma_v,
        sigma_w=args.sigma_w,
        seed=args.seed,
    ))

    scales = parse_float_list(args.scales)
    best = None

    for scale in scales:
        sensor_cfg = SensorConfig(
            sigma_d=args.sigma_d * scale,
            sigma_yaw=args.sigma_yaw * scale,
            p_drop=min(args.p_drop_cap, args.p_drop * scale),
            p_outlier=min(0.5, args.p_out * scale),
            outlier_dist=args.outlier_dist * scale,
            outlier_yaw=math.radians(args.outlier_yaw_deg) * scale,
        )
        rate = run_success_rate(states, sensor_cfg, actuation, args)
        record = {
            "scale": scale,
            "success_rate": rate,
        }
        if args.target_min <= rate <= args.target_max:
            best = record
            best["sensor_cfg"] = sensor_cfg
            break

    if best is None:
        # pick closest to target range
        target = (args.target_min + args.target_max) / 2.0
        scored = []
        for scale in scales:
            sensor_cfg = SensorConfig(
                sigma_d=args.sigma_d * scale,
                sigma_yaw=args.sigma_yaw * scale,
                p_drop=min(args.p_drop_cap, args.p_drop * scale),
                p_outlier=min(0.5, args.p_out * scale),
                outlier_dist=args.outlier_dist * scale,
                outlier_yaw=math.radians(args.outlier_yaw_deg) * scale,
            )
            rate = run_success_rate(states, sensor_cfg, actuation, args)
            scored.append((abs(rate - target), scale, rate, sensor_cfg))
        scored.sort(key=lambda x: x[0])
        _, scale, rate, sensor_cfg = scored[0]
        best = {"scale": scale, "success_rate": rate, "sensor_cfg": sensor_cfg}

    out = {
        "scale": best["scale"],
        "success_rate": best["success_rate"],
        "sensor": {
            "sigma_d": best["sensor_cfg"].sigma_d,
            "sigma_yaw": best["sensor_cfg"].sigma_yaw,
            "p_drop": best["sensor_cfg"].p_drop,
            "p_outlier": best["sensor_cfg"].p_outlier,
            "outlier_dist": best["sensor_cfg"].outlier_dist,
            "outlier_yaw": best["sensor_cfg"].outlier_yaw,
        },
        "actuation": {
            "sigma_v": args.sigma_v,
            "sigma_w": args.sigma_w,
        },
    }

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/noise_calibration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("saved", out_path)
    print(out)


if __name__ == "__main__":
    main()
