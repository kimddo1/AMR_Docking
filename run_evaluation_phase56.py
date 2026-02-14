import argparse
import json
import math
import os
import csv
from typing import Dict, List, Tuple

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.fsm import DockingFSM, FSMConfig
from src.safety import SafetyConfig
from src.obstacles import make_mixed_scenario
from src.test_cases import load_test_cases
from src.actuation import ActuationNoise, ActuationNoiseConfig
from src.recovery import RecoveryConfig

State = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5-6 evaluation (extended ablation + recovery)")
    p.add_argument("--cases", type=str, default="outputs/test_cases.csv")
    p.add_argument("--noise", type=str, default="outputs/noise_calibration.json")
    p.add_argument("--best", type=str, default="outputs/best_config.json")
    p.add_argument("--recovery", type=str, default="outputs/recovery_best_config.json")
    p.add_argument("--max_cases", type=int, default=200)
    p.add_argument("--latency_steps", type=int, default=5)
    p.add_argument("--max_time", type=float, default=35.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def make_control_params(cfg: Dict) -> Dict:
    return {
        "v_max": cfg["v_max"],
        "w_max": cfg["w_max"],
        "k_v": cfg["kv"],
        "k_w": cfg["kw"],
        "yaw_slow": math.radians(cfg["yaw_slow_deg"]),
        "stale_decay": cfg["stale_decay"],
        "stale_min_scale": cfg["stale_min_scale"],
        "stale_stop_steps": cfg["stale_stop_steps"],
    }


def eval_cases(
    cases: List[State],
    sensor_cfg: SensorConfig,
    robust_cfg: RobustConfig | None,
    fsm_cfg: FSMConfig | None,
    safety_cfg: SafetyConfig | None,
    recovery_cfg: RecoveryConfig | None,
    control_params: Dict,
    dist_tol: float,
    yaw_tol_deg: float,
    actuation: ActuationNoise,
    use_relative_control: bool,
    scenario: str | None,
    dt: float,
    max_time: float,
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
    recovery_total = 0
    deadlocks = 0

    for i, state0 in enumerate(cases):
        sensor_cfg.seed = i
        sensor = RelativePoseSensor(sensor_cfg) if use_relative_control else None
        estimator = None
        if robust_cfg is not None and (robust_cfg.enable_filter or robust_cfg.enable_gating):
            estimator = RobustPoseEstimator(robust_cfg)
        fsm = DockingFSM(fsm_cfg) if fsm_cfg is not None else None

        obstacles = []
        if scenario == "mixed":
            obstacles = make_mixed_scenario(i)

        hist = simulate(
            state0=state0,
            target=(0.0, 0.0),
            dt=dt,
            max_steps=int(max_time / dt),
            control_params=control_params,
            dist_tol=dist_tol,
            yaw_tol=math.radians(yaw_tol_deg),
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
            steps.append(len(hist["time"]))
            if not any(hist.get("collision", [])):
                safe_successes += 1
                safe_steps.append(len(hist["time"]))
        else:
            steps.append(len(hist["time"]))

        if hist.get("collision"):
            collision += 1 if any(hist["collision"]) else 0
        if hist.get("near_miss"):
            near_miss += 1 if any(hist["near_miss"]) else 0
        stop_count += int(hist.get("stop_count", 0))
        slow_count += int(hist.get("slow_count", 0))
        recovery_total += int(hist.get("recovery_count", 0))
        deadlocks += 1 if hist.get("deadlock", False) else 0

    n = len(cases)
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

    cases = load_test_cases(args.cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    noise = load_json(args.noise)
    sensor_cfg = SensorConfig(
        sigma_d=noise["sensor"]["sigma_d"],
        sigma_yaw=noise["sensor"]["sigma_yaw"],
        p_drop=noise["sensor"]["p_drop"],
        p_outlier=noise["sensor"]["p_outlier"],
        outlier_dist=noise["sensor"]["outlier_dist"],
        outlier_yaw=noise["sensor"]["outlier_yaw"],
        latency_steps=args.latency_steps,
    )

    actuation = ActuationNoise(ActuationNoiseConfig(
        sigma_v=noise["actuation"]["sigma_v"],
        sigma_w=noise["actuation"]["sigma_w"],
    ))

    # Base configs
    ema_cfg = EMAConfig(alpha_dist=0.3, alpha_yaw=0.3)
    gating_cfg = GatingConfig(
        max_jump_dist=max(0.3, 3.0 * sensor_cfg.sigma_d),
        max_jump_yaw=max(math.radians(20.0), 3.0 * sensor_cfg.sigma_yaw),
        hold_last=True,
        stale_jump_scale=0.15,
    )

    robust_off = RobustConfig(enable_filter=False, enable_gating=False)
    robust_filter = RobustConfig(enable_filter=True, enable_gating=False, ema=ema_cfg, gating=gating_cfg)
    robust_filter_gating = RobustConfig(enable_filter=True, enable_gating=True, ema=ema_cfg, gating=gating_cfg)

    fsm_default = FSMConfig()
    safety_on = SafetyConfig(r_stop=0.2, r_slow=0.5, v_slow=0.2, w_slow=0.5)
    safety_off = SafetyConfig(r_stop=-1.0, r_slow=-1.0, v_slow=10.0, w_slow=10.0)

    # Best tuned controller (Phase 5)
    best = load_json(args.best)["best_config"]
    tuned_control = make_control_params(best)
    tuned_fsm = FSMConfig(
        align_dist=best["align_dist"],
        finalize_dist=best["finalize_dist"],
        yaw_align_thresh=math.radians(best["yaw_align_deg"]),
        finalize_speed_scale=0.35,
        align_w_gain=2.0,
    )

    tuned_ema = EMAConfig(alpha_dist=best["alpha_dist"], alpha_yaw=best["alpha_yaw"])
    tuned_gating = GatingConfig(
        max_jump_dist=max(0.3, best["k_gate_dist"] * sensor_cfg.sigma_d),
        max_jump_yaw=max(math.radians(20.0), best["k_gate_yaw"] * sensor_cfg.sigma_yaw),
        hold_last=True,
        stale_jump_scale=0.15,
    )
    tuned_robust = RobustConfig(enable_filter=True, enable_gating=True, ema=tuned_ema, gating=tuned_gating)

    recovery_best = load_json(args.recovery)
    recovery_cfg = RecoveryConfig(**{k: recovery_best[k] for k in recovery_best if k in RecoveryConfig().__dict__})

    # Ablation extended
    rows = []

    rows.append({
        "name": "baseline_clean",
        **eval_cases(cases, SensorConfig(sigma_d=0, sigma_yaw=0, p_drop=0, p_outlier=0),
                     robust_off, None, None, None,
                     control_params={"v_max":0.6,"w_max":1.5,"k_v":1.2,"k_w":2.0,"yaw_slow":math.radians(35),
                                     "stale_decay":0.1,"stale_min_scale":0.3,"stale_stop_steps":12},
                     dist_tol=0.05, yaw_tol_deg=5.0, actuation=actuation,
                     use_relative_control=False, scenario=None, dt=args.dt, max_time=args.max_time)
    })

    rows.append({
        "name": "noise_only",
        **eval_cases(cases, sensor_cfg, robust_off, None, None, None,
                     control_params={"v_max":0.6,"w_max":1.5,"k_v":1.2,"k_w":2.0,"yaw_slow":math.radians(35),
                                     "stale_decay":0.1,"stale_min_scale":0.3,"stale_stop_steps":12},
                     dist_tol=0.05, yaw_tol_deg=5.0, actuation=actuation,
                     use_relative_control=True, scenario=None, dt=args.dt, max_time=args.max_time)
    })

    rows.append({
        "name": "noise_filter",
        **eval_cases(cases, sensor_cfg, robust_filter, None, None, None,
                     control_params={"v_max":0.6,"w_max":1.5,"k_v":1.2,"k_w":2.0,"yaw_slow":math.radians(35),
                                     "stale_decay":0.1,"stale_min_scale":0.3,"stale_stop_steps":12},
                     dist_tol=0.05, yaw_tol_deg=5.0, actuation=actuation,
                     use_relative_control=True, scenario=None, dt=args.dt, max_time=args.max_time)
    })

    rows.append({
        "name": "noise_filter_gating",
        **eval_cases(cases, sensor_cfg, robust_filter_gating, None, None, None,
                     control_params={"v_max":0.6,"w_max":1.5,"k_v":1.2,"k_w":2.0,"yaw_slow":math.radians(35),
                                     "stale_decay":0.1,"stale_min_scale":0.3,"stale_stop_steps":12},
                     dist_tol=0.05, yaw_tol_deg=5.0, actuation=actuation,
                     use_relative_control=True, scenario=None, dt=args.dt, max_time=args.max_time)
    })

    rows.append({
        "name": "noise_filter_gating_fsm",
        **eval_cases(cases, sensor_cfg, robust_filter_gating, fsm_default, None, None,
                     control_params={"v_max":0.6,"w_max":1.5,"k_v":1.2,"k_w":2.0,"yaw_slow":math.radians(35),
                                     "stale_decay":0.1,"stale_min_scale":0.3,"stale_stop_steps":12},
                     dist_tol=0.05, yaw_tol_deg=5.0, actuation=actuation,
                     use_relative_control=True, scenario=None, dt=args.dt, max_time=args.max_time)
    })

    rows.append({
        "name": "tuned_params",
        **eval_cases(cases, sensor_cfg, tuned_robust, tuned_fsm, None, None,
                     control_params=tuned_control,
                     dist_tol=best["dist_tol"], yaw_tol_deg=best["yaw_tol_deg"], actuation=actuation,
                     use_relative_control=True, scenario=None, dt=args.dt, max_time=args.max_time)
    })

    rows.append({
        "name": "tuned_params_recovery",
        **eval_cases(cases, sensor_cfg, tuned_robust, tuned_fsm, safety_on, recovery_cfg,
                     control_params=tuned_control,
                     dist_tol=best["dist_tol"], yaw_tol_deg=best["yaw_tol_deg"], actuation=actuation,
                     use_relative_control=True, scenario="mixed", dt=args.dt, max_time=args.max_time)
    })

    ablation_csv = os.path.join(args.out_dir, "phase56_ablation.csv")
    with open(ablation_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Safety comparison with recovery
    safety_rows = []
    safety_rows.append({
        "name": "safety_on",
        **eval_cases(cases, sensor_cfg, tuned_robust, tuned_fsm, safety_on, None,
                     control_params=tuned_control,
                     dist_tol=best["dist_tol"], yaw_tol_deg=best["yaw_tol_deg"], actuation=actuation,
                     use_relative_control=True, scenario="mixed", dt=args.dt, max_time=args.max_time)
    })
    safety_rows.append({
        "name": "safety_on_recovery",
        **eval_cases(cases, sensor_cfg, tuned_robust, tuned_fsm, safety_on, recovery_cfg,
                     control_params=tuned_control,
                     dist_tol=best["dist_tol"], yaw_tol_deg=best["yaw_tol_deg"], actuation=actuation,
                     use_relative_control=True, scenario="mixed", dt=args.dt, max_time=args.max_time)
    })
    safety_rows.append({
        "name": "safety_off",
        **eval_cases(cases, sensor_cfg, tuned_robust, tuned_fsm, safety_off, None,
                     control_params=tuned_control,
                     dist_tol=best["dist_tol"], yaw_tol_deg=best["yaw_tol_deg"], actuation=actuation,
                     use_relative_control=True, scenario="mixed", dt=args.dt, max_time=args.max_time)
    })

    safety_csv = os.path.join(args.out_dir, "phase56_safety.csv")
    with open(safety_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(safety_rows[0].keys()))
        writer.writeheader()
        for r in safety_rows:
            writer.writerow(r)

    print("saved", ablation_csv)
    print("saved", safety_csv)


if __name__ == "__main__":
    main()
