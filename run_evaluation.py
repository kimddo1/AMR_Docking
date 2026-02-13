import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.fsm import DockingFSM, FSMConfig
from src.safety import SafetyConfig
from src.obstacles import make_crossing_scenario, make_cutin_scenario, make_mixed_scenario
from src.test_cases import generate_test_cases, load_test_cases
from src.actuation import ActuationNoise, ActuationNoiseConfig
from src.recovery import RecoveryConfig

State = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 evaluation runner")
    p.add_argument("--num_trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cases", type=str, default="outputs/test_cases.csv")
    p.add_argument("--calibration", type=str, default="outputs/noise_calibration.json")
    p.add_argument("--max_time", type=float, default=35.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--dist_tol", type=float, default=0.05)
    p.add_argument("--yaw_tol_deg", type=float, default=5.0)

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
    p.add_argument("--stale_jump_scale", type=float, default=0.15)
    p.add_argument("--stale_decay", type=float, default=0.1)
    p.add_argument("--stale_min_scale", type=float, default=0.3)
    p.add_argument("--stale_stop_steps", type=int, default=12)

    # Actuation noise
    p.add_argument("--sigma_v", type=float, default=0.03)
    p.add_argument("--sigma_w", type=float, default=0.05)

    # Output
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def ensure_cases(path: str, num: int, seed: int) -> List[State]:
    if os.path.exists(path):
        return load_test_cases(path)
    return generate_test_cases(path, num, seed)


def base_control_params(args: argparse.Namespace) -> Dict:
    return {
        "v_max": args.v_max,
        "w_max": args.w_max,
        "k_v": args.k_v,
        "k_w": args.k_w,
        "yaw_slow": math.radians(args.yaw_slow_deg),
        "stale_decay": args.stale_decay,
        "stale_min_scale": args.stale_min_scale,
        "stale_stop_steps": args.stale_stop_steps,
    }


def run_trials(
    states: List[State],
    sensor_cfg: SensorConfig,
    robust_cfg: RobustConfig,
    use_relative_control: bool,
    use_fsm: bool,
    safety_cfg: SafetyConfig | None,
    scenario: str | None,
    args: argparse.Namespace,
    actuation: ActuationNoise,
    recovery_cfg: RecoveryConfig | None,
) -> Dict:
    successes = 0
    final_dist = []
    final_yaw = []
    steps = []
    safe_successes = 0
    safe_steps = []
    collision = 0
    near_miss = 0
    stop_count = 0
    slow_count = 0
    deadlocks = 0
    recovery_total = 0

    for i, state0 in enumerate(states):
        sensor_cfg.seed = i
        sensor = RelativePoseSensor(sensor_cfg) if use_relative_control else None
        estimator = RobustPoseEstimator(robust_cfg) if (use_relative_control and (robust_cfg.enable_filter or robust_cfg.enable_gating)) else None
        fsm = DockingFSM(FSMConfig()) if use_fsm else None

        obstacles = []
        if scenario == "crossing":
            obstacles = make_crossing_scenario()
        elif scenario == "cutin":
            obstacles = make_cutin_scenario()
        elif scenario == "mixed":
            obstacles = make_mixed_scenario(i)

        hist = simulate(
            state0=state0,
            target=(0.0, 0.0),
            dt=args.dt,
            max_steps=int(args.max_time / args.dt),
            control_params=base_control_params(args),
            dist_tol=args.dist_tol,
            yaw_tol=math.radians(args.yaw_tol_deg),
            sensor=sensor,
            estimator=estimator,
            use_relative_control=use_relative_control,
            fsm=fsm,
            safety_config=safety_cfg,
            obstacles=obstacles,
            actuation=actuation,
            recovery_config=recovery_cfg,
        )

        if hist["success"]:
            successes += 1
            final_dist.append(hist["dist_true"][-1])
            final_yaw.append(abs(hist["yaw_true"][-1]))
            if not any(hist.get("collision", [])):
                safe_successes += 1
                safe_steps.append(len(hist["time"]))
        steps.append(len(hist["time"]))

        if hist.get("collision"):
            collision += 1 if any(hist["collision"]) else 0
        if hist.get("near_miss"):
            near_miss += 1 if any(hist["near_miss"]) else 0
        stop_count += int(hist.get("stop_count", 0))
        slow_count += int(hist.get("slow_count", 0))
        recovery_total += int(hist.get("recovery_count", 0))
        deadlocks += 1 if hist.get("deadlock", False) else 0

    n = len(states)
    return {
        "success_rate": successes / n,
        "final_dist_mean": sum(final_dist) / len(final_dist) if final_dist else float("nan"),
        "final_yaw_mean": sum(final_yaw) / len(final_yaw) if final_yaw else float("nan"),
        "steps_mean": sum(steps) / len(steps),
        "collision_rate": collision / n,
        "near_miss_rate": near_miss / n,
        "stop_mean": stop_count / n,
        "slow_mean": slow_count / n,
        "safe_success_rate": safe_successes / n,
        "safe_steps_mean": sum(safe_steps) / len(safe_steps) if safe_steps else float("nan"),
        "recovery_mean": recovery_total / n,
        "deadlock_rate": deadlocks / n,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    states = ensure_cases(args.cases, args.num_trials, args.seed)

    # Load calibrated sensor settings
    with open(args.calibration, "r") as f:
        calib = json.load(f)

    noisy_sensor = SensorConfig(
        sigma_d=calib["sensor"]["sigma_d"],
        sigma_yaw=calib["sensor"]["sigma_yaw"],
        p_drop=calib["sensor"]["p_drop"],
        p_outlier=calib["sensor"]["p_outlier"],
        outlier_dist=calib["sensor"]["outlier_dist"],
        outlier_yaw=calib["sensor"]["outlier_yaw"],
    )

    clean_sensor = SensorConfig(sigma_d=0.0, sigma_yaw=0.0, p_drop=0.0, latency_steps=0, p_outlier=0.0)

    ema_cfg = EMAConfig(alpha_dist=args.ema_dist, alpha_yaw=args.ema_yaw)
    # Scale gating thresholds with calibrated noise to avoid rejecting normal changes
    gating_cfg = GatingConfig(
        max_jump_dist=max(args.max_jump_dist, 3.0 * noisy_sensor.sigma_d),
        max_jump_yaw=max(math.radians(args.max_jump_yaw_deg), 3.0 * noisy_sensor.sigma_yaw),
        hold_last=args.hold_last,
        stale_jump_scale=args.stale_jump_scale,
    )

    robust_off = RobustConfig(enable_filter=False, enable_gating=False)
    robust_filter = RobustConfig(enable_filter=True, enable_gating=False, ema=ema_cfg, gating=gating_cfg)
    robust_filter_gating = RobustConfig(enable_filter=True, enable_gating=True, ema=ema_cfg, gating=gating_cfg)

    actuation = ActuationNoise(ActuationNoiseConfig(
        sigma_v=args.sigma_v,
        sigma_w=args.sigma_w,
        seed=args.seed,
    ))

    rows = []

    # Ablation: clean
    rows.append({
        "name": "baseline_clean",
        **run_trials(states, clean_sensor, robust_off, use_relative_control=False, use_fsm=False,
                    safety_cfg=None, scenario=None, args=args, actuation=actuation, recovery_cfg=None)
    })

    # + noise (true baseline: no estimator, no hold-last)
    rows.append({
        "name": "noise_only",
        **run_trials(states, noisy_sensor, robust_off, use_relative_control=True, use_fsm=False,
                    safety_cfg=None, scenario=None, args=args, actuation=actuation, recovery_cfg=None)
    })

    # + filter
    rows.append({
        "name": "noise_filter",
        **run_trials(states, noisy_sensor, robust_filter, use_relative_control=True, use_fsm=False,
                    safety_cfg=None, scenario=None, args=args, actuation=actuation, recovery_cfg=None)
    })

    # + gating
    rows.append({
        "name": "noise_filter_gating",
        **run_trials(states, noisy_sensor, robust_filter_gating, use_relative_control=True, use_fsm=False,
                    safety_cfg=None, scenario=None, args=args, actuation=actuation, recovery_cfg=None)
    })

    # + FSM
    rows.append({
        "name": "noise_filter_gating_fsm",
        **run_trials(states, noisy_sensor, robust_filter_gating, use_relative_control=True, use_fsm=True,
                    safety_cfg=None, scenario=None, args=args, actuation=actuation, recovery_cfg=None)
    })

    out_csv = os.path.join(args.out_dir, "phase4_ablation.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Safety on/off with dynamic obstacles
    safety_cfg = SafetyConfig(r_stop=0.2, r_slow=0.5, v_slow=0.2, w_slow=0.5)
    safety_off_cfg = SafetyConfig(r_stop=-1.0, r_slow=-1.0, v_slow=10.0, w_slow=10.0)
    recovery_cfg = RecoveryConfig()

    safety_rows = []
    safety_rows.append({
        "name": "safety_on",
        **run_trials(states, noisy_sensor, robust_filter_gating, use_relative_control=True, use_fsm=True,
                    safety_cfg=safety_cfg, scenario="mixed", args=args, actuation=actuation, recovery_cfg=recovery_cfg)
    })
    safety_rows.append({
        "name": "safety_off",
        **run_trials(states, noisy_sensor, robust_filter_gating, use_relative_control=True, use_fsm=True,
                    safety_cfg=safety_off_cfg, scenario="mixed", args=args, actuation=actuation, recovery_cfg=None)
    })

    safety_csv = os.path.join(args.out_dir, "phase4_safety.csv")
    with open(safety_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(safety_rows[0].keys()))
        writer.writeheader()
        for r in safety_rows:
            writer.writerow(r)

    print("saved", out_csv)
    print("saved", safety_csv)


if __name__ == "__main__":
    main()
