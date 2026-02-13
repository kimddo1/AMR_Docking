import argparse
import csv
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.fsm import DockingFSM, FSMConfig
from src.safety import SafetyConfig
from src.obstacles import make_mixed_scenario
from src.test_cases import load_test_cases
from src.actuation import ActuationNoise, ActuationNoiseConfig

State = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 auto-tuning (random search)")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_cases", type=int, default=200)

    p.add_argument("--load_cases", type=str, default="outputs/test_cases.csv")
    p.add_argument("--val_cases", type=str, default="")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--save_splits", action="store_true")

    p.add_argument("--load_noise", type=str, default="outputs/noise_calibration.json")
    p.add_argument("--latency_steps", type=int, default=5)

    # Feature toggles
    p.add_argument("--use_sensor", action="store_true", default=True)
    p.add_argument("--use_filter", action="store_true", default=True)
    p.add_argument("--use_gating", action="store_true", default=True)
    p.add_argument("--hold_last", action="store_true", default=True)
    p.add_argument("--use_fsm", action="store_true", default=True)
    p.add_argument("--use_safety", action="store_true", default=True)

    p.add_argument("--scenario", type=str, default="mixed")

    # Optional safety tuning
    p.add_argument("--tune_safety", action="store_true", default=False)

    # Output
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def split_cases(cases: List[State], val_fraction: float, seed: int) -> Tuple[List[State], List[State]]:
    idx = list(range(len(cases)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(len(cases) * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train = [cases[i] for i in train_idx]
    val = [cases[i] for i in val_idx]
    return train, val


def load_noise(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def sample_config(rng: random.Random, tune_safety: bool) -> Dict:
    # Controller
    kv = rng.uniform(0.3, 2.5)
    kw = rng.uniform(0.8, 6.0)
    v_max = rng.uniform(0.2, 1.0)
    w_max = rng.uniform(1.0, 6.0)
    yaw_slow_deg = rng.uniform(10.0, 60.0)

    # FSM
    finalize_dist = rng.uniform(0.05, 0.5)
    align_dist = rng.uniform(max(0.2, finalize_dist + 0.05), 1.0)
    yaw_align_deg = rng.uniform(2.0, 20.0)

    # Tolerances
    dist_tol = rng.uniform(0.01, 0.05)
    yaw_tol_deg = rng.uniform(0.5, 5.0)

    # EMA
    alpha_dist = rng.uniform(0.05, 0.5)
    alpha_yaw = rng.uniform(0.05, 0.5)

    # Gating (scaled)
    k_gate_dist = rng.uniform(1.5, 5.0)
    k_gate_yaw = rng.uniform(1.5, 5.0)

    # Stale handling
    stale_stop_steps = rng.randint(6, 20)
    stale_decay = rng.uniform(0.05, 0.25)
    stale_min_scale = rng.uniform(0.1, 0.5)

    config = {
        "kv": kv,
        "kw": kw,
        "v_max": v_max,
        "w_max": w_max,
        "yaw_slow_deg": yaw_slow_deg,
        "align_dist": align_dist,
        "finalize_dist": finalize_dist,
        "yaw_align_deg": yaw_align_deg,
        "dist_tol": dist_tol,
        "yaw_tol_deg": yaw_tol_deg,
        "alpha_dist": alpha_dist,
        "alpha_yaw": alpha_yaw,
        "k_gate_dist": k_gate_dist,
        "k_gate_yaw": k_gate_yaw,
        "stale_stop_steps": stale_stop_steps,
        "stale_decay": stale_decay,
        "stale_min_scale": stale_min_scale,
    }

    if tune_safety:
        r_stop = rng.uniform(0.1, 0.3)
        r_slow = rng.uniform(max(0.3, r_stop + 0.1), 0.8)
        config["r_stop"] = r_stop
        config["r_slow"] = r_slow
    return config


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
    robust_cfg: Optional[RobustConfig],
    fsm_cfg: Optional[FSMConfig],
    safety_cfg: Optional[SafetyConfig],
    control_params: Dict,
    dist_tol: float,
    yaw_tol_deg: float,
    actuation: ActuationNoise,
    use_relative_control: bool,
    scenario: str,
) -> Dict:
    successes = 0
    safe_successes = 0
    dist_sum = 0.0
    yaw_sum = 0.0
    steps_sum = 0
    safe_steps_sum = 0
    collision_count = 0

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
            dt=0.05,
            max_steps=int(35.0 / 0.05),
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
        )

        success = hist["success"]
        collided = any(hist.get("collision", [])) if safety_cfg is not None else False

        if success:
            successes += 1
            dist_sum += hist["dist_true"][-1]
            yaw_sum += abs(hist["yaw_true"][-1])
            steps_sum += len(hist["time"])
            if not collided:
                safe_successes += 1
                safe_steps_sum += len(hist["time"])

        if collided:
            collision_count += 1

    n = len(cases)
    if successes == 0:
        return {
            "success_rate": 0.0,
            "safe_success_rate": 0.0,
            "collision_rate": collision_count / n if n else 0.0,
            "dist_mean": float("nan"),
            "yaw_mean": float("nan"),
            "steps_mean": float("nan"),
            "safe_steps_mean": float("nan"),
        }

    return {
        "success_rate": successes / n,
        "safe_success_rate": safe_successes / n,
        "collision_rate": collision_count / n if n else 0.0,
        "dist_mean": dist_sum / successes,
        "yaw_mean": yaw_sum / successes,
        "steps_mean": steps_sum / successes,
        "safe_steps_mean": safe_steps_sum / safe_successes if safe_successes > 0 else float("nan"),
    }


def score_metrics(metrics: Dict, use_safety: bool) -> float:
    if metrics["success_rate"] <= 0.0:
        return -1e6

    dist_cm = metrics["dist_mean"] * 100.0
    yaw_deg = math.degrees(metrics["yaw_mean"])

    if use_safety:
        steps = metrics["safe_steps_mean"]
        if math.isnan(steps):
            steps = metrics["steps_mean"]
        return (
            100.0 * metrics["safe_success_rate"]
            - 30.0 * metrics["collision_rate"]
            - 3.0 * dist_cm
            - 1.0 * yaw_deg
            - 0.05 * steps
        )

    return (
        100.0 * metrics["success_rate"]
        - 3.0 * dist_cm
        - 1.0 * yaw_deg
        - 0.05 * metrics["steps_mean"]
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    cases = load_test_cases(args.load_cases)
    if args.val_cases:
        train_cases = cases
        val_cases = load_test_cases(args.val_cases)
    else:
        train_cases, val_cases = split_cases(cases, args.val_fraction, args.seed)

    if args.max_cases > 0:
        train_cases = train_cases[: args.max_cases]
        if val_cases:
            val_cases = val_cases[: max(1, int(args.max_cases * args.val_fraction))]

    if args.save_splits and not args.val_cases:
        train_path = os.path.join(args.out_dir, "train_cases.csv")
        val_path = os.path.join(args.out_dir, "val_cases.csv")
        with open(train_path, "w", newline="") as f:
            f.write("x,y,theta\n")
            for s in train_cases:
                f.write(f"{s[0]},{s[1]},{s[2]}\n")
        with open(val_path, "w", newline="") as f:
            f.write("x,y,theta\n")
            for s in val_cases:
                f.write(f"{s[0]},{s[1]},{s[2]}\n")

    noise = load_noise(args.load_noise)
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

    results = []

    for t in range(args.trials):
        cfg = sample_config(rng, args.tune_safety)

        # Robust config
        if args.use_filter or args.use_gating:
            ema_cfg = EMAConfig(alpha_dist=cfg["alpha_dist"], alpha_yaw=cfg["alpha_yaw"])

            base_jump_dist = 0.3
            base_jump_yaw = math.radians(20.0)
            max_jump_dist = max(base_jump_dist, cfg["k_gate_dist"] * sensor_cfg.sigma_d)
            max_jump_yaw = max(base_jump_yaw, cfg["k_gate_yaw"] * sensor_cfg.sigma_yaw)

            gating_cfg = GatingConfig(
                max_jump_dist=max_jump_dist,
                max_jump_yaw=max_jump_yaw,
                hold_last=args.hold_last,
                stale_jump_scale=0.15,
            )
            robust_cfg = RobustConfig(
                enable_filter=args.use_filter,
                enable_gating=args.use_gating,
                ema=ema_cfg,
                gating=gating_cfg,
            )
        else:
            robust_cfg = None

        # FSM config
        fsm_cfg = None
        if args.use_fsm:
            fsm_cfg = FSMConfig(
                align_dist=cfg["align_dist"],
                finalize_dist=cfg["finalize_dist"],
                yaw_align_thresh=math.radians(cfg["yaw_align_deg"]),
                finalize_speed_scale=0.35,
                align_w_gain=2.0,
            )

        # Safety config
        safety_cfg = None
        if args.use_safety:
            if args.tune_safety:
                r_stop = cfg["r_stop"]
                r_slow = cfg["r_slow"]
            else:
                r_stop = 0.2
                r_slow = 0.5
            safety_cfg = SafetyConfig(r_stop=r_stop, r_slow=r_slow, v_slow=0.2, w_slow=0.5)

        control_params = make_control_params(cfg)
        use_relative = args.use_sensor

        train_metrics = eval_cases(
            train_cases,
            sensor_cfg,
            robust_cfg,
            fsm_cfg,
            safety_cfg,
            control_params,
            cfg["dist_tol"],
            cfg["yaw_tol_deg"],
            actuation,
            use_relative,
            args.scenario,
        )
        train_score = score_metrics(train_metrics, args.use_safety)

        val_score = None
        val_metrics = None
        if val_cases:
            val_metrics = eval_cases(
                val_cases,
                sensor_cfg,
                robust_cfg,
                fsm_cfg,
                safety_cfg,
                control_params,
                cfg["dist_tol"],
                cfg["yaw_tol_deg"],
                actuation,
                use_relative,
                args.scenario,
            )
            val_score = score_metrics(val_metrics, args.use_safety)
        else:
            val_score = train_score
            val_metrics = train_metrics

        row = {
            "trial": t,
            "train_score": train_score,
            "val_score": val_score,
            **cfg,
            "train_success_rate": train_metrics["success_rate"],
            "train_safe_success_rate": train_metrics["safe_success_rate"],
            "train_collision_rate": train_metrics["collision_rate"],
            "train_dist_mean": train_metrics["dist_mean"],
            "train_yaw_mean": train_metrics["yaw_mean"],
            "train_steps_mean": train_metrics["steps_mean"],
            "val_success_rate": val_metrics["success_rate"],
            "val_safe_success_rate": val_metrics["safe_success_rate"],
            "val_collision_rate": val_metrics["collision_rate"],
            "val_dist_mean": val_metrics["dist_mean"],
            "val_yaw_mean": val_metrics["yaw_mean"],
            "val_steps_mean": val_metrics["steps_mean"],
        }
        results.append(row)

    results_sorted = sorted(results, key=lambda r: r["val_score"], reverse=True)
    topk = results_sorted[:10]
    best = results_sorted[0] if results_sorted else None

    out_csv = os.path.join(args.out_dir, "tuning_results.csv")
    out_topk = os.path.join(args.out_dir, "tuning_topk.csv")
    out_best = os.path.join(args.out_dir, "best_config.json")
    out_summary = os.path.join(args.out_dir, "tuning_summary.md")

    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        with open(out_topk, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(topk[0].keys()))
            writer.writeheader()
            for r in topk:
                writer.writerow(r)

        if best:
            best_out = {
                "best_config": best,
                "meta": {
                    "trials": args.trials,
                    "seed": args.seed,
                    "max_cases": args.max_cases,
                    "latency_steps": args.latency_steps,
                    "noise": args.load_noise,
                    "cases": args.load_cases,
                    "val_cases": args.val_cases,
                },
            }
            with open(out_best, "w") as f:
                json.dump(best_out, f, indent=2)

            with open(out_summary, "w") as f:
                f.write("# Tuning Summary\n\n")
                f.write(f"Best val score: {best['val_score']:.3f}\n\n")
                f.write("## Best Parameters\n")
                for k, v in best.items():
                    if k in {"trial", "train_score", "val_score"}:
                        continue
                    f.write(f"- {k}: {v}\n")

    print("saved", out_csv)
    print("saved", out_topk)
    print("saved", out_best)
    print("saved", out_summary)


if __name__ == "__main__":
    main()
