import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot tuning results (Top-K)")
    p.add_argument("--csv", type=str, default="outputs/tuning_results.csv")
    p.add_argument("--out", type=str, default="outputs/tuning_results_topk.png")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.csv, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ["val_score","val_success_rate","val_safe_success_rate",
                      "val_collision_rate","val_steps_mean"]:
                row[k] = float(row[k])
            rows.append(row)

    rows.sort(key=lambda r: r["val_score"], reverse=True)
    rows = rows[: min(args.topk, len(rows))]

    labels = [f"t{r['trial']}" for r in rows]
    scores = [r["val_score"] for r in rows]
    success = [r["val_success_rate"] for r in rows]
    safe_success = [r["val_safe_success_rate"] for r in rows]
    collision = [r["val_collision_rate"] for r in rows]
    steps = [r["val_steps_mean"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(labels, scores, color="tab:blue")
    ax.set_title("Validation Score (Top-K)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(labels, success, color="tab:green", label="success")
    ax.bar(labels, safe_success, color="tab:orange", alpha=0.7, label="safe success")
    ax.set_title("Success vs Safe Success")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(labels, steps, color="tab:purple")
    ax.set_title("Mean Steps (Validation)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(labels, collision, color="tab:red")
    ax.set_title("Collision Rate (Validation)")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Phase 5 Tuning Results (Top-K)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
