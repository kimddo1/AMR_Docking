import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Make Phase 5-6 plots (custom titles + filters)")
    p.add_argument("--ablation_csv", type=str, default="outputs/phase56_ablation.csv")
    p.add_argument("--safety_csv", type=str, default="outputs/phase56_safety.csv")
    p.add_argument("--out_ablation", type=str, default="outputs/phase56_ablation_plots.png")
    p.add_argument("--out_safety", type=str, default="outputs/phase56_safety_plots.png")
    return p.parse_args()


def load_rows(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ["success_rate","final_dist_mean","final_yaw_mean","steps_mean",
                      "collision_rate","near_miss_rate","stop_mean","slow_mean",
                      "safe_success_rate","safe_steps_mean","recovery_mean","deadlock_rate"]:
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def plot_ablation(rows, out_path):
    labels = [r["name"] for r in rows]
    success = [r["success_rate"] for r in rows]
    steps = [r["steps_mean"] for r in rows]
    dist = [r["final_dist_mean"] for r in rows]
    yaw = [r["final_yaw_mean"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(labels, success, color="tab:green")
    ax.set_title("Success rate")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[0, 1]
    ax.bar(labels, steps, color="tab:blue")
    ax.set_title("Mean steps")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 0]
    ax.bar(labels, dist, color="tab:orange")
    ax.set_title("Final distance mean")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 1]
    ax.bar(labels, yaw, color="tab:purple")
    ax.set_title("Final yaw mean")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle("Phase 5–6 Ablation Summary")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)


def plot_safety(rows, out_path):
    labels = [r["name"] for r in rows]
    coll = [r["collision_rate"] for r in rows]
    near = [r["near_miss_rate"] for r in rows]
    safe_success = [r.get("safe_success_rate", 0.0) for r in rows]
    safe_steps = [r.get("safe_steps_mean", 0.0) for r in rows]
    recovery_mean = [r.get("recovery_mean", 0.0) for r in rows]
    deadlock_rate = [r.get("deadlock_rate", 0.0) for r in rows]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.bar(labels, coll, color="tab:red")
    ax.set_title("Collision rate")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[0, 1]
    ax.bar(labels, near, color="tab:orange")
    ax.set_title("Near-miss rate")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 0]
    ax.bar(labels, safe_success, color="tab:green")
    ax.set_title("Collision-free success rate")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 1]
    ax.bar(labels, safe_steps, color="tab:blue")
    ax.set_title("Collision-free mean steps")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[2, 0]
    ax.bar(labels, recovery_mean, color="tab:purple")
    ax.set_title("Recovery count (mean)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[2, 1]
    ax.bar(labels, deadlock_rate, color="tab:gray")
    ax.set_title("Deadlock rate")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle("Phase 5–6 Safety Comparison")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main():
    args = parse_args()
    ablation_rows = load_rows(args.ablation_csv)
    # remove tuned_params_recovery from ablation
    ablation_rows = [r for r in ablation_rows if r.get("name") != "tuned_params_recovery"]
    safety_rows = load_rows(args.safety_csv)

    plot_ablation(ablation_rows, args.out_ablation)
    plot_safety(safety_rows, args.out_safety)

    print("saved", args.out_ablation)
    print("saved", args.out_safety)


if __name__ == "__main__":
    main()
