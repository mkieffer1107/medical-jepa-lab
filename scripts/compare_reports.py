from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "linear_probe/test_accuracy",
    "linear_probe/test_balanced_accuracy",
    "knn/test_accuracy",
    "knn/test_balanced_accuracy",
    "clustering/test_silhouette",
    "pca/explained_variance_2d",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare JEPA evaluator metrics.json files")
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    rows = []
    for report in args.reports:
        path = Path(report)
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append((path.parent.name, metrics))

    header = ["run", *KEYS]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for name, metrics in rows:
        values = [name]
        for key in KEYS:
            value = metrics.get(key)
            values.append("—" if value is None else f"{float(value):.4f}")
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
