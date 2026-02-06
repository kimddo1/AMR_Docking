import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig


def run_case(label, sensor_cfg, robust_cfg=None, use_relative_control=True):
    sensor = RelativePoseSensor(sensor_cfg)
    estimator = None
    if robust_cfg is not None:
        estimator = RobustPoseEstimator(robust_cfg)

    hist = simulate(
        state0=(-2.0, -1.0, math.radians(45)),
        target=(0.0, 0.0),
        dt=0.05,
        max_steps=int(20.0 / 0.05),
        control_params={"v_max": 0.6, "w_max": 1.5, "k_v": 1.2, "k_w": 2.0},
        dist_tol=0.05,
        yaw_tol=math.radians(5),
        sensor=sensor,
        estimator=estimator,
        use_relative_control=use_relative_control,
    )
    return label, hist


cases = []

# 1) Noise only
cases.append(
    run_case(
        "Noise",
        SensorConfig(sigma_d=0.05, sigma_yaw=0.1, seed=0),
        robust_cfg=None,
    )
)

# 2) Noise + filter
cases.append(
    run_case(
        "Noise + EMA",
        SensorConfig(sigma_d=0.05, sigma_yaw=0.1, seed=0),
        robust_cfg=RobustConfig(
            enable_filter=True,
            enable_gating=False,
            ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
            gating=GatingConfig(hold_last=True),
        ),
    )
)

# 3) Noise + filter + gating + dropout + outliers
cases.append(
    run_case(
        "Noise + EMA + Gating + Drop/Out",
        SensorConfig(
            sigma_d=0.05,
            sigma_yaw=0.1,
            p_drop=0.2,
            p_outlier=0.05,
            outlier_dist=0.6,
            outlier_yaw=math.radians(25),
            seed=0,
        ),
        robust_cfg=RobustConfig(
            enable_filter=True,
            enable_gating=True,
            ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
            gating=GatingConfig(
                max_jump_dist=0.3,
                max_jump_yaw=math.radians(20),
                hold_last=True,
            ),
        ),
    )
)

# 4) Noise + filter + gating + latency
cases.append(
    run_case(
        "Noise + EMA + Gating + Latency",
        SensorConfig(
            sigma_d=0.05,
            sigma_yaw=0.1,
            latency_steps=5,
            seed=0,
        ),
        robust_cfg=RobustConfig(
            enable_filter=True,
            enable_gating=True,
            ema=EMAConfig(alpha_dist=0.3, alpha_yaw=0.3),
            gating=GatingConfig(
                max_jump_dist=0.3,
                max_jump_yaw=math.radians(20),
                hold_last=True,
            ),
        ),
    )
)

fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex='row')

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

# Add legends only once
axes[0, 0].legend(loc="best")
axes[1, 0].legend(loc="best")

fig.suptitle("Phase 2 Comparison (same base settings)")
fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
out_path = os.path.join("outputs", "phase2_compare.png")
fig.savefig(out_path, dpi=150)
print("saved", out_path)
