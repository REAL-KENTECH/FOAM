#!/usr/bin/env python3
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


def summarize(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows in {metrics_path}")
    manifest = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final = rows[-1]
    best_train = min((_float(row, "train_full_ce") for row in rows), default=float("nan"))
    best_val = max((_float(row, "val_accuracy") for row in rows), default=float("nan"))
    return {
        "run": run_dir.name,
        "optimizer": manifest.get("config", {}).get("optimizer", "unknown"),
        "best_train_full_ce": best_train,
        "best_val_accuracy": best_val,
        "final_train_compute_seconds": _float(
            final,
            "train_compute_seconds_cumulative",
            _float(final, "wall_clock_seconds"),
        ),
        "final_end_to_end_wall_clock_seconds": float(
            summary.get(
                "end_to_end_wall_clock_seconds",
                summary.get(
                    "wall_clock_seconds",
                    _float(
                        final,
                        "end_to_end_wall_clock_seconds",
                        _float(final, "wall_clock_seconds"),
                    ),
                ),
            )
        ),
        "final_evd_rate": _float(final, "evd_rate"),
        "global_step": int(float(final.get("global_step", 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reconstructed FOAM run directories.")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", default="run_summary.csv")
    args = parser.parse_args()
    records = [summarize(Path(value)) for value in args.runs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(output)


if __name__ == "__main__":
    main()
