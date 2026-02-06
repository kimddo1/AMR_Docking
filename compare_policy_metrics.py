import math
import statistics as stats

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig

N = 30

state0 = (-2.0, -1.0, math.radians(45))
target = (0.0, 0.0)

dt = 0.05
max_steps = int(20.0 / dt)
control_params = {
    "v_max": 0.6,
    "w_max": 1.5,
    "k_v": 1.2,
    "k_w": 2.0,
    "stale_decay": 0.2,
    "stale_min_scale": 0.2,
    "stale_stop_steps": 6,
}

dist_tol = 0.05
yaw_tol = math.radians(5)


policies = {
    "Baseline gating": RobustConfig(
        enable_filter=True,
        enable_gating=True,
        ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
        gating=GatingConfig(
            max_jump_dist=0.3,
            max_jump_yaw=math.radians(20),
            hold_last=True,
            stale_jump_scale=0.0,
        ),
    ),
    "Improved gating + stale slowdown": RobustConfig(
        enable_filter=True,
        enable_gating=True,
        ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
        gating=GatingConfig(
            max_jump_dist=0.3,
            max_jump_yaw=math.radians(20),
            hold_last=True,
            stale_jump_scale=0.15,
        ),
    ),
}


def run_policy(label, robust_cfg):
    successes = 0
    final_dist = []
    final_yaw = []
    steps = []
    reject_frac = []
    missing_frac = []

    for seed in range(N):
        sensor_cfg = SensorConfig(
            sigma_d=0.05,
            sigma_yaw=0.1,
            p_drop=0.2,
            p_outlier=0.05,
            outlier_dist=0.6,
            outlier_yaw=math.radians(25),
            seed=seed,
        )
        sensor = RelativePoseSensor(sensor_cfg)
        estimator = RobustPoseEstimator(robust_cfg)

        hist = simulate(
            state0=state0,
            target=target,
            dt=dt,
            max_steps=max_steps,
            control_params=control_params,
            dist_tol=dist_tol,
            yaw_tol=yaw_tol,
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

    return {
        "label": label,
        "success_rate": successes / N,
        "final_dist_mean": stats.mean(final_dist) if final_dist else float("nan"),
        "final_yaw_mean": stats.mean(final_yaw) if final_yaw else float("nan"),
        "steps_mean": stats.mean(steps) if steps else float("nan"),
        "reject_frac_mean": stats.mean(reject_frac) if reject_frac else float("nan"),
        "missing_frac_mean": stats.mean(missing_frac) if missing_frac else float("nan"),
        "N": N,
    }


results = [run_policy(label, cfg) for label, cfg in policies.items()]

print("N", N)
for r in results:
    print(r)
