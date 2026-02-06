import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.sim import simulate
from src.sensor import RelativePoseSensor, SensorConfig
from src.robust import RobustPoseEstimator, RobustConfig, GatingConfig
from src.filtering import EMAConfig
from src.fsm import DockingFSM, FSMConfig
from src.safety import SafetyConfig
from src.obstacles import make_crossing_scenario


def main() -> None:
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
    estimator = RobustPoseEstimator(robust_cfg)

    fsm = DockingFSM(FSMConfig())
    safety_cfg = SafetyConfig(r_stop=0.2, r_slow=0.5, v_slow=0.2, w_slow=0.5)
    obstacles = make_crossing_scenario()

    hist = simulate(
        state0=(-2.0, -1.0, math.radians(45)),
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

    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(hist["x"], hist["y"], label="trajectory", linewidth=2)
    ax.scatter([hist["x"][0]], [hist["y"][0]], c="tab:green", label="start")
    ax.scatter([hist["x"][-1]], [hist["y"][-1]], c="tab:blue", label="end")
    ax.scatter([0.0], [0.0], c="tab:red", marker="*", s=120, label="target")
    if hist.get("obstacles"):
        for obs in hist["obstacles"]:
            ax.plot(obs["x"], obs["y"], linestyle="--", color="tab:gray", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Trajectory + Obstacles")
    ax.legend(loc="best")

    ax2 = axes[1]
    ax2.plot(hist["time"], hist["min_dist_obs"], color="tab:orange")
    ax2.axhline(safety_cfg.r_slow, color="tab:blue", linestyle="--", label="r_slow")
    ax2.axhline(safety_cfg.r_stop, color="tab:red", linestyle="--", label="r_stop")
    ax2.set_title("Min Distance to Obstacles")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("distance (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    status = "SUCCESS" if hist["success"] else "TIMEOUT"
    fig.suptitle(f"Phase 3 Demo ({status})")
    fig.tight_layout()

    out_path = os.path.join("outputs", "phase3_demo.png")
    fig.savefig(out_path, dpi=150)

    print("saved", out_path)
    print("success", hist["success"], "stop_count", hist["stop_count"], "slow_count", hist["slow_count"])


if __name__ == "__main__":
    main()
