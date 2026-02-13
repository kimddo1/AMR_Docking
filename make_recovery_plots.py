import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot recovery tuning results")
    p.add_argument("--csv", type=str, default="outputs/recovery_tuning_results.csv")
    p.add_argument("--out", type=str, default="outputs/recovery_tuning_results.png")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.csv, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ["score","success_rate","safe_success_rate","collision_rate","deadlock_rate","steps_mean"]:
                row[k] = float(row[k])
            rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[: min(args.topk, len(rows))]

    labels = [f"t{r['trial']}" for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    ax = axes[0, 0]
    ax.bar(labels, [r["score"] for r in rows], color="tab:blue")
    ax.set_title("Score (Top-K)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(labels, [r["success_rate"] for r in rows], color="tab:green", label="success")
    ax.bar(labels, [r["safe_success_rate"] for r in rows], color="tab:orange", alpha=0.7, label="safe success")
    ax.set_title("Success vs Safe Success")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(loc="best")

    ax = axes[0, 2]
    ax.bar(labels, [r["collision_rate"] for r in rows], color="tab:red")
    ax.set_title("Collision Rate")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 0]
    ax.bar(labels, [r["deadlock_rate"] for r in rows], color="tab:gray")
    ax.set_title("Deadlock Rate")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 1]
    ax.bar(labels, [r["steps_mean"] for r in rows], color="tab:purple")
    ax.set_title("Mean Steps")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1, 2]
    ax.axis('off')

    fig.suptitle("Phase 6 Recovery Tuning Results (Top-K)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
