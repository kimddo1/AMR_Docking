import argparse
import csv
import json
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
from src.obstacles import make_mixed_scenario
from src.test_cases import load_test_cases
from src.actuation import ActuationNoise, ActuationNoiseConfig
from src.recovery import RecoveryConfig

State = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 recovery tuning (random search)")
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_cases", type=int, default=200)
    p.add_argument("--load_cases", type=str, default="outputs/test_cases.csv")
    p.add_argument("--load_noise", type=str, default="outputs/noise_calibration.json")
    p.add_argument("--best_config", type=str, default="outputs/best_config.json")
    p.add_argument("--match_phase56", action="store_true")
    p.add_argument("--latency_steps", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def sample_recovery(rng: random.Random) -> Dict:
    return {
        "stop_persist_steps": rng.randint(40, 140),
        "clear_confirm_steps": rng.randint(2, 8),
        "wait_steps": rng.randint(10, 60),
        "turn_angle_deg": rng.uniform(70.0, 110.0),
        "turn_speed": rng.uniform(0.6, 1.8),
        "shift_distance": rng.uniform(0.20, 0.90),
        "shift_speed": rng.uniform(0.08, 0.35),
        "turnback_k": rng.uniform(1.2, 4.5),
        "turnback_yaw_tol_deg": rng.uniform(2.0, 12.0),
        "turnback_hold_steps": rng.randint(2, 8),
        "turnback_max_steps": rng.randint(40, 220),
        "max_retries": rng.randint(1, 3),
        "cooldown_steps": rng.randint(20, 100),
        "max_blocked_steps": rng.randint(500, 1200),
    }


def score(metrics: Dict) -> float:
    # Prioritize collision-free success while discouraging deadlock/timeout loops.
    if metrics["success_rate"] <= 0.0:
        return -1e6
    steps = metrics.get("safe_steps_mean")
    if steps is None or math.isnan(steps):
        steps = metrics.get("steps_mean", 0.0)
    timeout_rate = metrics.get("timeout_rate", 0.0)
    recovery_steps = metrics.get("recovery_steps_mean", 0.0)
    crossing_success = metrics.get("success_crossing", 0.0)
    cutin_short_success = metrics.get("success_cutin_short", 0.0)
    return (
        160.0 * metrics["success_rate"]
        + 120.0 * metrics["safe_success_rate"]
        + 120.0 * crossing_success
        + 120.0 * cutin_short_success
        - 120.0 * metrics["deadlock_rate"]
        - 60.0 * timeout_rate
        - 30.0 * metrics["collision_rate"]
        - 0.05 * steps
        - 0.02 * recovery_steps
    )


def eval_recovery(
    cases: List[State],
    sensor_cfg: SensorConfig,
    robust_cfg: RobustConfig,
    fsm_cfg: FSMConfig,
    safety_cfg: SafetyConfig,
    recovery_cfg: RecoveryConfig,
    actuation: ActuationNoise,
    control_params: Dict,
    dist_tol: float,
    yaw_tol_deg: float,
) -> Dict:
    successes = 0
    safe_successes = 0
    failures = 0
    timeout_failures = 0
    steps_sum = 0
    safe_steps_sum = 0
    collision_count = 0
    deadlocks = 0
    recovery_steps_total = 0
    recovery_count_total = 0
    mode_counts = {"crossing": 0, "cutin_short": 0, "cutin_persist": 0}
    mode_success = {"crossing": 0, "cutin_short": 0, "cutin_persist": 0}

    for i, state0 in enumerate(cases):
        sensor_cfg.seed = i
        sensor = RelativePoseSensor(sensor_cfg)
        estimator = RobustPoseEstimator(robust_cfg)
        fsm = DockingFSM(fsm_cfg)
        obstacles = make_mixed_scenario(i)
        mode = random.Random(i).choice(["crossing", "cutin_short", "cutin_persist"])
        mode_counts[mode] += 1

        hist = simulate(
            state0=state0,
            target=(0.0, 0.0),
            dt=0.05,
            max_steps=int(35.0 / 0.05),
            control_params=control_params,
            dist_tol=dist_tol,
            yaw_tol=math.radians(yaw_tol_deg),
            sensor=sensor,
            estimator=estimator,
            use_relative_control=True,
            fsm=fsm,
            safety_config=safety_cfg,
            obstacles=obstacles,
            actuation=actuation,
            recovery_config=recovery_cfg,
        )

        success = hist["success"]
        collided = any(hist.get("collision", []))
        if success:
            successes += 1
            mode_success[mode] += 1
            steps_sum += len(hist["time"])
            if not collided:
                safe_successes += 1
                safe_steps_sum += len(hist["time"])
        else:
            failures += 1
            if not hist.get("deadlock", False):
                timeout_failures += 1
        if collided:
            collision_count += 1
        if hist.get("deadlock", False):
            deadlocks += 1
        recovery_steps_total += int(hist.get("recovery_steps_total", 0))
        recovery_count_total += int(hist.get("recovery_count", 0))

    n = len(cases)
    def _safe_div(a: int, b: int) -> float:
        return (a / b) if b > 0 else 0.0
    return {
        "success_rate": successes / n,
        "safe_success_rate": safe_successes / n,
        "failure_rate": failures / n,
        "timeout_rate": timeout_failures / n,
        "collision_rate": collision_count / n,
        "deadlock_rate": deadlocks / n,
        "steps_mean": steps_sum / successes if successes > 0 else float("nan"),
        "safe_steps_mean": safe_steps_sum / safe_successes if safe_successes > 0 else float("nan"),
        "recovery_steps_mean": recovery_steps_total / n,
        "recovery_count_mean": recovery_count_total / n,
        "success_crossing": _safe_div(mode_success["crossing"], mode_counts["crossing"]),
        "success_cutin_short": _safe_div(mode_success["cutin_short"], mode_counts["cutin_short"]),
        "success_cutin_persist": _safe_div(mode_success["cutin_persist"], mode_counts["cutin_persist"]),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cases = load_test_cases(args.load_cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    with open(args.load_noise, "r") as f:
        noise = json.load(f)
    best = None
    if args.match_phase56:
        with open(args.best_config, "r") as f:
            best = json.load(f)["best_config"]
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
        seed=args.seed,
    ))

    if args.match_phase56 and best is not None:
        control_params = {
            "v_max": best["v_max"],
            "w_max": best["w_max"],
            "k_v": best["kv"],
            "k_w": best["kw"],
            "yaw_slow": math.radians(best["yaw_slow_deg"]),
            "stale_decay": best["stale_decay"],
            "stale_min_scale": best["stale_min_scale"],
            "stale_stop_steps": best["stale_stop_steps"],
        }
        dist_tol = best["dist_tol"]
        yaw_tol_deg = best["yaw_tol_deg"]
        ema_cfg = EMAConfig(alpha_dist=best["alpha_dist"], alpha_yaw=best["alpha_yaw"])
        gating_cfg = GatingConfig(
            max_jump_dist=max(0.3, best["k_gate_dist"] * sensor_cfg.sigma_d),
            max_jump_yaw=max(math.radians(20.0), best["k_gate_yaw"] * sensor_cfg.sigma_yaw),
            hold_last=True,
            stale_jump_scale=0.15,
        )
        fsm_cfg = FSMConfig(
            align_dist=best["align_dist"],
            finalize_dist=best["finalize_dist"],
            yaw_align_thresh=math.radians(best["yaw_align_deg"]),
            finalize_speed_scale=0.35,
            align_w_gain=2.0,
        )
    else:
        control_params = {
            "v_max": 0.6,
            "w_max": 1.5,
            "k_v": 1.2,
            "k_w": 2.0,
            "yaw_slow": math.radians(35),
            "stale_decay": 0.1,
            "stale_min_scale": 0.3,
            "stale_stop_steps": 12,
        }
        dist_tol = 0.05
        yaw_tol_deg = 5.0
        ema_cfg = EMAConfig(alpha_dist=0.3, alpha_yaw=0.3)
        gating_cfg = GatingConfig(
            max_jump_dist=max(0.3, 3.0 * sensor_cfg.sigma_d),
            max_jump_yaw=max(math.radians(20.0), 3.0 * sensor_cfg.sigma_yaw),
            hold_last=True,
            stale_jump_scale=0.15,
        )
        fsm_cfg = FSMConfig()
    robust_cfg = RobustConfig(enable_filter=True, enable_gating=True, ema=ema_cfg, gating=gating_cfg)
    safety_cfg = SafetyConfig(r_stop=0.2, r_slow=0.5, v_slow=0.2, w_slow=0.5)

    results = []
    for t in range(args.trials):
        rec = sample_recovery(rng)
        recovery_cfg = RecoveryConfig(**rec)

        metrics = eval_recovery(
            cases,
            sensor_cfg,
            robust_cfg,
            fsm_cfg,
            safety_cfg,
            recovery_cfg,
            actuation,
            control_params,
            dist_tol,
            yaw_tol_deg,
        )

        row = {
            "trial": t,
            "profile": "phase56" if args.match_phase56 else "baseline",
            **rec,
            **metrics,
            "score": score(metrics),
        }
        results.append(row)

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0] if results else None

    out_csv = os.path.join(args.out_dir, "recovery_tuning_results.csv")
    out_best = os.path.join(args.out_dir, "recovery_best_config.json")
    out_summary = os.path.join(args.out_dir, "recovery_tuning_summary.md")

    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        if best:
            with open(out_best, "w") as f:
                json.dump(best, f, indent=2)

            with open(out_summary, "w") as f:
                f.write("# Recovery Tuning Summary\n\n")
                f.write(f"Best score: {best['score']:.3f}\n\n")
                f.write("## Best Config\n")
                for k, v in best.items():
                    if k in {"trial", "score"}:
                        continue
                    f.write(f"- {k}: {v}\n")

    print("saved", out_csv)
    print("saved", out_best)
    print("saved", out_summary)


if __name__ == "__main__":
    main()
