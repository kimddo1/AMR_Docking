import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig


def run_case(label, robust_cfg):
    sensor_cfg = SensorConfig(
        sigma_d=0.05,
        sigma_yaw=0.1,
        p_drop=0.2,
        p_outlier=0.05,
        outlier_dist=0.6,
        outlier_yaw=math.radians(25),
        seed=0,
    )
    sensor = RelativePoseSensor(sensor_cfg)
    estimator = RobustPoseEstimator(robust_cfg)

    hist = simulate(
        state0=(-2.0, -1.0, math.radians(45)),
        target=(0.0, 0.0),
        dt=0.05,
        max_steps=int(20.0 / 0.05),
        control_params={
            "v_max": 0.6,
            "w_max": 1.5,
            "k_v": 1.2,
            "k_w": 2.0,
            # improved stale handling params (only used if stale_count > 0)
            "stale_decay": 0.2,
            "stale_min_scale": 0.2,
            "stale_stop_steps": 6,
        },
        dist_tol=0.05,
        yaw_tol=math.radians(5),
        sensor=sensor,
        estimator=estimator,
        use_relative_control=True,
    )
    return label, hist


# Baseline policy (original gating/hold_last, no stale scaling)
robust_baseline = RobustConfig(
    enable_filter=True,
    enable_gating=True,
    ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
    gating=GatingConfig(
        max_jump_dist=0.3,
        max_jump_yaw=math.radians(20),
        hold_last=True,
        stale_jump_scale=0.0,
    ),
)

# Improved policy (stale gate scaling)
robust_improved = RobustConfig(
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

cases = [
    run_case("Baseline gating", robust_baseline),
    run_case("Improved gating + stale slowdown", robust_improved),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='row')

for idx, (label, hist) in enumerate(cases):
    ts = hist["time"]

    ax_dist = axes[0, idx]
    ax_dist.plot(ts, hist["dist_true"], color="tab:orange", label="dist_true")
    ax_dist.plot(ts, hist["dist_meas"], color="tab:gray", alpha=0.5, label="dist_meas")
    ax_dist.plot(ts, hist["dist_est"], color="tab:blue", alpha=0.8, label="dist_est")
    ax_dist.set_title(label)
    ax_dist.set_ylabel("distance (m)")
    ax_dist.grid(True, alpha=0.3)

    ax_yaw = axes[1, idx]
    ax_yaw.plot(ts, hist["yaw_true"], color="tab:purple", label="yaw_true")
    ax_yaw.plot(ts, hist["yaw_meas"], color="tab:gray", alpha=0.5, label="yaw_meas")
    ax_yaw.plot(ts, hist["yaw_est"], color="tab:blue", alpha=0.8, label="yaw_est")
    ax_yaw.set_xlabel("time (s)")
    ax_yaw.set_ylabel("yaw (rad)")
    ax_yaw.grid(True, alpha=0.3)

axes[0, 0].legend(loc="best")
axes[1, 0].legend(loc="best")

fig.suptitle("Policy Comparison: Baseline vs Improved")
fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
out_path = os.path.join("outputs", "phase2_policy_compare.png")
fig.savefig(out_path, dpi=150)
print("saved", out_path)
