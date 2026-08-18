#!/usr/bin/env python3
"""Plot FOAM damping/proxy/EVD dynamics from ``factor_diagnostics.csv``."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _load_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "factor_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No factor diagnostic rows in {path}")
    return rows


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def aggregate(rows: Iterable[dict[str, str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        epoch = int(float(row.get("epoch", 0)))
        axis = row.get("axis", "unknown") or "unknown"
        groups[(epoch, axis)].append(row)

    records: list[dict[str, float | str]] = []
    for (epoch, axis), values in sorted(groups.items()):
        count = len(values)
        checks = sum(_as_float(row, "check_calls", _as_float(row, "checks")) for row in values)
        evd_calls = sum(_as_float(row, "evd_calls") for row in values)
        proxy_calls = sum(_as_float(row, "proxy_calls") for row in values)
        records.append(
            {
                "epoch": epoch,
                "axis": axis,
                "factor_count": float(count),
                "epsilon_mean": sum(_as_float(row, "epsilon") for row in values) / count,
                "epsilon_max": max(_as_float(row, "epsilon") for row in values),
                "proxy_mean": sum(_as_float(row, "last_proxy", _as_float(row, "proxy")) for row in values) / count,
                "proxy_max": max(_as_float(row, "last_proxy", _as_float(row, "proxy")) for row in values),
                "evd_rate": evd_calls / checks if checks > 0 else 0.0,
                "proxy_rate": proxy_calls / checks if checks > 0 else 0.0,
            }
        )
    return records


def _plot(
    records: list[dict[str, float | str]],
    metric: str,
    ylabel: str,
    destination: Path,
    *,
    log_y: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis_plot = plt.subplots(figsize=(7.2, 4.8))
    axes = sorted({str(record["axis"]) for record in records})
    for axis_name in axes:
        selected = [record for record in records if record["axis"] == axis_name]
        axis_plot.plot(
            [int(record["epoch"]) for record in selected],
            [float(record[metric]) for record in selected],
            marker="o",
            label=axis_name,
        )
    axis_plot.set_xlabel("Epoch")
    axis_plot.set_ylabel(ylabel)
    if log_y:
        axis_plot.set_yscale("log")
    axis_plot.grid(True, alpha=0.3)
    axis_plot.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot epsilon, proxy and cumulative EVD dynamics by factor axis."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", default="plots/diagnostics")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install requirements-dev.txt") from exc

    records = aggregate(_load_rows(args.run))
    output = Path(args.output_dir)
    _plot(records, "epsilon_mean", "Mean adaptive damping", output / "epsilon_mean.png", log_y=True)
    _plot(records, "epsilon_max", "Maximum adaptive damping", output / "epsilon_max.png", log_y=True)
    _plot(records, "proxy_mean", "Mean last FOAM proxy", output / "proxy_mean.png")
    _plot(records, "proxy_max", "Maximum last FOAM proxy", output / "proxy_max.png")
    _plot(records, "evd_rate", "Cumulative EVD operation rate", output / "evd_rate.png")
    print(output)


if __name__ == "__main__":
    main()
