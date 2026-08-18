#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_run(path: Path):
    with (path / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest_path = path / "run_manifest.json"
    label = path.name
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = manifest.get("config", {}).get("experiment_name", label)
    return label, rows


def plot_metric(
    runs,
    metric: str,
    ylabel: str,
    destination: Path,
    clock: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for label, rows in runs:
        x = [
            float(row.get(clock, row.get("wall_clock_seconds", 0.0))) / 60.0
            for row in rows
        ]
        y = [float(row[metric]) for row in rows]
        axis.plot(x, y, marker="o", label=label)
    axis.set_xlabel(
        "Cumulative training compute time (min.)"
        if clock == "train_compute_seconds_cumulative"
        else "End-to-end wall-clock time (min.)"
    )
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot wall-clock FOAM/ViT metrics.")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument(
        "--clock",
        choices=(
            "train_compute_seconds_cumulative",
            "end_to_end_wall_clock_seconds",
        ),
        default="train_compute_seconds_cumulative",
    )
    args = parser.parse_args()
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required; install requirements-dev.txt"
        ) from exc

    runs = [load_run(Path(value)) for value in args.runs]
    output = Path(args.output_dir)
    train_metric = (
        "train_full_hard_ce"
        if all("train_full_hard_ce" in rows[-1] for _, rows in runs)
        else "train_full_ce"
    )
    plot_metric(
        runs,
        train_metric,
        "Deterministic full-train hard-label CE",
        output / "train_full_ce_vs_time.png",
        args.clock,
    )
    plot_metric(
        runs,
        "val_accuracy",
        "Validation top-1 accuracy (%)",
        output / "val_accuracy_vs_time.png",
        args.clock,
    )
    if all("evd_rate" in rows[-1] for _, rows in runs):
        plot_metric(
            runs,
            "evd_rate",
            "Cumulative EVD operation rate",
            output / "evd_rate_vs_time.png",
            args.clock,
        )
    print(output)


if __name__ == "__main__":
    main()
