import argparse
import csv
import os
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Analyze accuracy vs speed trade-offs for tuning results")
    p.add_argument("--csv", type=str, default="outputs/tuning_results.csv")
    p.add_argument("--out_csv", type=str, default="outputs/tuning_tradeoff.csv")
    p.add_argument("--out_plot", type=str, default="outputs/tuning_tradeoff.png")
    p.add_argument("--out_summary", type=str, default="outputs/tuning_tradeoff_summary.md")
    p.add_argument("--w_dist", type=float, default=1.0, help="Weight for dist error (cm)")
    p.add_argument("--w_yaw", type=float, default=1.0, help="Weight for yaw error (deg)")
    p.add_argument("--min_success", type=float, default=0.0)
    p.add_argument("--min_safe_success", type=float, default=0.0)
    return p.parse_args()


def pareto_front(rows: List[Dict], key_x: str, key_y: str) -> List[Dict]:
    # Minimize both x and y
    rows_sorted = sorted(rows, key=lambda r: (r[key_x], r[key_y]))
    front = []
    best_y = float('inf')
    for r in rows_sorted:
        if r[key_y] < best_y:
            front.append(r)
            best_y = r[key_y]
    return front


def main():
    args = parse_args()

    rows = []
    with open(args.csv, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # parse numeric fields used in tradeoff
            for k in ["val_score","val_success_rate","val_safe_success_rate","val_collision_rate",
                      "val_dist_mean","val_yaw_mean","val_steps_mean"]:
                row[k] = float(row[k])
            rows.append(row)

    # filter by success thresholds
    rows = [r for r in rows if r["val_success_rate"] >= args.min_success and r["val_safe_success_rate"] >= args.min_safe_success]

    # compute accuracy metric: weighted dist(cm) + yaw(deg)
    for r in rows:
        dist_cm = r["val_dist_mean"] * 100.0
        yaw_deg = r["val_yaw_mean"] * 57.29577951308232
        r["acc_score"] = args.w_dist * dist_cm + args.w_yaw * yaw_deg
        r["dist_cm"] = dist_cm
        r["yaw_deg"] = yaw_deg

    # Pareto front on (acc_score, steps_mean)
    front = pareto_front(rows, "acc_score", "val_steps_mean")

    # Save tradeoff CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = [
            "trial","acc_score","dist_cm","yaw_deg","val_steps_mean",
            "val_success_rate","val_safe_success_rate","val_collision_rate","val_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # Plot scatter + Pareto front
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([r["val_steps_mean"] for r in rows], [r["acc_score"] for r in rows],
               alpha=0.4, label="All trials")

    ax.scatter([r["val_steps_mean"] for r in front], [r["acc_score"] for r in front],
               color="tab:red", label="Pareto front")

    ax.set_xlabel("Mean steps (validation)")
    ax.set_ylabel("Accuracy score (lower is better)")
    ax.set_title("Accuracy vs Speed Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=150)

    # Summary
    fastest = sorted(rows, key=lambda r: r["val_steps_mean"])[:5]
    most_accurate = sorted(rows, key=lambda r: r["acc_score"])[:5]

    with open(args.out_summary, "w") as f:
        f.write("# Tuning Trade-off Summary\n\n")
        f.write(f"Accuracy metric: {args.w_dist}*dist_cm + {args.w_yaw}*yaw_deg\n\n")
        f.write(f"Total trials (filtered): {len(rows)}\n")
        f.write(f"Pareto front size: {len(front)}\n\n")

        f.write("## Fastest (Top 5 by steps)\n")
        for r in fastest:
            f.write(f"- trial {r['trial']}: steps={r['val_steps_mean']:.2f}, acc={r['acc_score']:.3f}, success={r['val_success_rate']:.3f}\n")
        f.write("\n## Most Accurate (Top 5 by accuracy)\n")
        for r in most_accurate:
            f.write(f"- trial {r['trial']}: acc={r['acc_score']:.3f}, steps={r['val_steps_mean']:.2f}, success={r['val_success_rate']:.3f}\n")

    print("saved", args.out_csv)
    print("saved", args.out_plot)
    print("saved", args.out_summary)


if __name__ == "__main__":
    main()
