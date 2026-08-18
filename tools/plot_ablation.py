#!/usr/bin/env python3
"""Create wall-clock/quality scatter plots for ViT ablation run directories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_record(run_dir: Path, clock: str) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {metrics_path}")
    manifest = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label = manifest.get("config", {}).get("experiment_name", run_dir.name)
    train_key = "train_full_hard_ce" if "train_full_hard_ce" in rows[-1] else "train_full_ce"
    return {
        "run": str(run_dir),
        "label": label,
        "best_train_full_ce": min(_float(row, train_key) for row in rows),
        "best_val_accuracy": max(_float(row, "val_accuracy") for row in rows),
        "clock_minutes": _float(rows[-1], clock, _float(rows[-1], "wall_clock_seconds")) / 60.0,
        "evd_rate": _float(rows[-1], "evd_rate"),
    }


def plot(records: list[dict[str, Any]], metric: str, ylabel: str, destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for record in records:
        axis.scatter(record["clock_minutes"], record[metric])
        axis.annotate(
            record["label"],
            (record["clock_minutes"], record[metric]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Cumulative time (min.)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reconstructed FOAM ablation scatter figures.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", default="plots/ablation")
    parser.add_argument(
        "--clock",
        choices=("train_compute_seconds_cumulative", "end_to_end_wall_clock_seconds"),
        default="train_compute_seconds_cumulative",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install requirements-dev.txt") from exc

    records = [load_record(run, args.clock) for run in args.runs]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot(records, "best_train_full_ce", "Best deterministic full-train CE", output / "train_ce_scatter.png")
    plot(records, "best_val_accuracy", "Best validation accuracy (%)", output / "validation_scatter.png")
    print(output)


if __name__ == "__main__":
    main()
