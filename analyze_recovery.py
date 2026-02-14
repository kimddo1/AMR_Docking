import argparse
import json
import math
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze recovery triggers on a specific case")
    p.add_argument("--cases", type=str, default="outputs/test_cases_holdout.csv")
    p.add_argument("--case_index", type=int, default=0)
    p.add_argument("--noise", type=str, default="outputs/noise_calibration.json")
    p.add_argument("--best", type=str, default="outputs/best_config.json")
    p.add_argument("--recovery", type=str, default="outputs/recovery_best_config.json")
    p.add_argument("--latency_steps", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_time", type=float, default=35.0)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cases = load_test_cases(args.cases)
    if not cases:
        raise RuntimeError("No cases loaded.")
    idx = max(0, min(args.case_index, len(cases) - 1))
    state0 = cases[idx]

    noise = load_json(args.noise)
    sensor_cfg = SensorConfig(
        sigma_d=noise["sensor"]["sigma_d"],
        sigma_yaw=noise["sensor"]["sigma_yaw"],
        p_drop=noise["sensor"]["p_drop"],
        p_outlier=noise["sensor"]["p_outlier"],
        outlier_dist=noise["sensor"]["outlier_dist"],
        outlier_yaw=noise["sensor"]["outlier_yaw"],
        latency_steps=args.latency_steps,
        seed=idx,
    )

    actuation = ActuationNoise(ActuationNoiseConfig(
        sigma_v=noise["actuation"]["sigma_v"],
        sigma_w=noise["actuation"]["sigma_w"],
        seed=42,
    ))

    # Tuned control + robust + fsm
    best = load_json(args.best)["best_config"]
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

    ema_cfg = EMAConfig(alpha_dist=best["alpha_dist"], alpha_yaw=best["alpha_yaw"])
    gating_cfg = GatingConfig(
        max_jump_dist=max(0.3, best["k_gate_dist"] * sensor_cfg.sigma_d),
        max_jump_yaw=max(math.radians(20.0), best["k_gate_yaw"] * sensor_cfg.sigma_yaw),
        hold_last=True,
        stale_jump_scale=0.15,
    )
    robust_cfg = RobustConfig(enable_filter=True, enable_gating=True, ema=ema_cfg, gating=gating_cfg)
    fsm_cfg = FSMConfig(
        align_dist=best["align_dist"],
        finalize_dist=best["finalize_dist"],
        yaw_align_thresh=math.radians(best["yaw_align_deg"]),
        finalize_speed_scale=0.35,
        align_w_gain=2.0,
    )

    recovery_best = load_json(args.recovery)
    recovery_cfg = RecoveryConfig(**{k: recovery_best[k] for k in recovery_best if k in RecoveryConfig().__dict__})

    sensor = RelativePoseSensor(sensor_cfg)
    estimator = RobustPoseEstimator(robust_cfg)
    fsm = DockingFSM(fsm_cfg)
    safety_cfg = SafetyConfig(r_stop=0.2, r_slow=0.5, v_slow=0.2, w_slow=0.5)
    obstacles = make_mixed_scenario(idx)

    hist = simulate(
        state0=state0,
        target=(0.0, 0.0),
        dt=args.dt,
        max_steps=int(args.max_time / args.dt),
        control_params=control_params,
        dist_tol=best["dist_tol"],
        yaw_tol=math.radians(best["yaw_tol_deg"]),
        sensor=sensor,
        estimator=estimator,
        use_relative_control=True,
        fsm=fsm,
        safety_config=safety_cfg,
        obstacles=obstacles,
        actuation=actuation,
        recovery_config=recovery_cfg,
    )

    t = hist["time"]
    dist = hist["dist_true"]
    yaw = [abs(y) for y in hist["yaw_true"]]
    safety = hist["safety_mode"]
    recovery_mode = hist["recovery_mode"]
    trigger = hist["recovery_trigger"]
    stalled = hist["recovery_stalled"]
    stop_streak = hist["recovery_stop_streak"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(t, dist, label="dist")
    axes[0].set_ylabel("dist (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, yaw, label="|yaw|")
    axes[1].set_ylabel("|yaw| (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, stop_streak, label="STOP streak")
    axes[2].plot(t, [int(x) for x in trigger], label="trigger", alpha=0.6)
    axes[2].plot(t, [int(x) for x in stalled], label="stalled", alpha=0.6)
    axes[2].set_ylabel("recovery flags")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Encode modes as text annotations
    axes[3].plot(t, [0]*len(t), alpha=0)  # placeholder
    axes[3].set_ylabel("modes")
    axes[3].set_yticks([])
    for i, tt in enumerate(t):
        if i % 10 == 0:
            axes[3].text(tt, 0.0, f"S:{safety[i]}\nR:{recovery_mode[i]}", fontsize=7)
    axes[3].set_xlabel("time (s)")

    fig.suptitle(f"Recovery Analysis (case {idx})")
    fig.tight_layout()

    out_path = os.path.join(args.out_dir, f"recovery_case_{idx}.png")
    fig.savefig(out_path, dpi=150)
    print("saved", out_path)


if __name__ == "__main__":
    main()
