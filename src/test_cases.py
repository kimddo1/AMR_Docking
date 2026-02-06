import csv
import math
import os
import random
from typing import List, Tuple

State = Tuple[float, float, float]


def generate_test_cases(path: str, num: int, seed: int) -> List[State]:
    rng = random.Random(seed)
    states: List[State] = []
    while len(states) < num:
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        if math.hypot(x, y) < 0.6:
            continue
        theta = rng.uniform(-math.pi, math.pi)
        states.append((x, y, theta))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "theta"])
        for s in states:
            writer.writerow(s)
    return states


def load_test_cases(path: str) -> List[State]:
    states: List[State] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            states.append((float(row["x"]), float(row["y"]), float(row["theta"])))
    return states
