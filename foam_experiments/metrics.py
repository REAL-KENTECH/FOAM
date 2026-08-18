from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .distributed import DistributedContext, gather_objects
from .optim import local_preconditioner_diagnostics, local_preconditioner_profile


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def append_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def read_all(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temp.replace(destination)


def _axis_summary(rows: Sequence[Dict[str, Any]], axis: int, prefix: str) -> Dict[str, Any]:
    # Figure 3 in the paper concerns left/right factors of matrix-valued
    # blocks. Exclude vector/scalar blocks from the L/R summaries.
    selected = [
        row
        for row in rows
        if int(row.get("block_order", 0)) == 2 and int(row["axis"]) == axis
    ]
    if not selected:
        return {
            f"{prefix}_factor_count": 0,
            f"{prefix}_checks": 0,
            f"{prefix}_evd_calls": 0,
            f"{prefix}_reuse_calls": 0,
            f"{prefix}_evd_rate": 0.0,
            f"{prefix}_epsilon_mean": 0.0,
            f"{prefix}_epsilon_max": 0.0,
            f"{prefix}_proxy_mean": 0.0,
            f"{prefix}_proxy_max": 0.0,
        }
    checks = sum(int(row["checks"]) for row in selected)
    evd_calls = sum(int(row["evd_calls"]) for row in selected)
    reuse_calls = sum(int(row["reuse_calls"]) for row in selected)
    epsilons = [float(row["epsilon"]) for row in selected]
    proxies = [
        float(row["last_proxy"])
        for row in selected
        if math.isfinite(float(row["last_proxy"]))
    ]
    return {
        f"{prefix}_factor_count": len(selected),
        f"{prefix}_checks": checks,
        f"{prefix}_evd_calls": evd_calls,
        f"{prefix}_reuse_calls": reuse_calls,
        f"{prefix}_evd_rate": evd_calls / max(checks, 1),
        f"{prefix}_epsilon_mean": mean(epsilons),
        f"{prefix}_epsilon_max": max(epsilons),
        f"{prefix}_proxy_mean": mean(proxies) if proxies else float("nan"),
        f"{prefix}_proxy_max": max(proxies) if proxies else float("nan"),
    }


def summarize_factor_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "factor_count": 0,
            "optimizer_checks": 0,
            "evd_calls": 0,
            "proxy_calls": 0,
            "residual_calls": 0,
            "reuse_calls": 0,
            "evd_rate": 0.0,
            **_axis_summary([], 0, "left"),
            **_axis_summary([], 1, "right"),
        }
    checks = sum(int(row["checks"]) for row in rows)
    evd_calls = sum(int(row["evd_calls"]) for row in rows)
    proxy_calls = sum(int(row.get("proxy_calls", 0)) for row in rows)
    residual_calls = sum(int(row.get("residual_calls", 0)) for row in rows)
    reuse_calls = sum(int(row["reuse_calls"]) for row in rows)
    cap_refreshes = sum(int(row["cap_refreshes"]) for row in rows)
    damping_updates = sum(int(row["damping_updates"]) for row in rows)
    dimensions = sorted({int(row["dimension"]) for row in rows})
    return {
        "factor_count": len(rows),
        "optimizer_checks": checks,
        "evd_calls": evd_calls,
        "proxy_calls": proxy_calls,
        "residual_calls": residual_calls,
        "reuse_calls": reuse_calls,
        "evd_rate": evd_calls / max(checks, 1),
        "cap_refreshes": cap_refreshes,
        "damping_updates": damping_updates,
        "factor_dimensions": ";".join(str(value) for value in dimensions),
        **_axis_summary(rows, 0, "left"),
        **_axis_summary(rows, 1, "right"),
    }


def collect_optimizer_metrics(
    optimizer,
    context: DistributedContext,
    epoch: int,
    global_step: int,
    wall_clock_seconds: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    local_rows = local_preconditioner_diagnostics(optimizer)
    local_profile = local_preconditioner_profile(optimizer, reset=False)
    gathered_rows = gather_objects(local_rows, dst=0)
    gathered_profiles = gather_objects(local_profile, dst=0)

    if not context.is_main:
        return {}, []

    rows = [row for rank_rows in (gathered_rows or []) for row in rank_rows]
    rows.sort(key=lambda row: (int(row.get("group", 0)), str(row["factor_id"])))
    enriched = [
        {
            "epoch": epoch,
            "global_step": global_step,
            "wall_clock_seconds": wall_clock_seconds,
            **row,
        }
        for row in rows
    ]
    summary = summarize_factor_rows(rows)
    profiles = gathered_profiles or []
    for key in ("proxy_seconds", "evd_seconds", "reuse_seconds"):
        # Work is distributed across ranks. Summing reports total GPU-seconds;
        # max approximates the critical path. Both are useful.
        values = [float(profile.get(key, 0.0)) for profile in profiles]
        summary[f"{key}_sum"] = sum(values)
        summary[f"{key}_max_rank"] = max(values, default=0.0)
    return summary, enriched
