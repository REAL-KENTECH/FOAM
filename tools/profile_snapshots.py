#!/usr/bin/env python3
"""Profile the exact FOAM proxy and a direct inverse-root EVD on saved factors."""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    function: Callable[[], torch.Tensor],
    repeats: int,
    warmup: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def foam_proxy(
    factor: torch.Tensor,
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    epsilon: float,
    root: int,
) -> torch.Tensor:
    rotated = eigenvectors.T @ factor @ eigenvectors
    shifted = (eigenvalues.clamp_min(0) + epsilon).clamp_min(
        torch.finfo(eigenvalues.dtype).tiny
    )
    relative_change = torch.linalg.norm(
        (rotated - torch.diag(eigenvalues))
        / torch.outer(shifted.sqrt(), shifted.sqrt()),
        ord="fro",
    )
    inverse_root_eigenvalues = shifted.pow(-1.0 / root)
    alpha = inverse_root_eigenvalues.abs().max() / (
        torch.linalg.vector_norm(inverse_root_eigenvalues) + 1e-25
    )
    return relative_change * alpha / root


def direct_inverse_root(
    factor: torch.Tensor, epsilon: float, root: int
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(factor)
    shifted = eigenvalues.clamp_min(0).add(epsilon)
    return (
        eigenvectors * shifted.pow(-1.0 / root).unsqueeze(0)
    ) @ eigenvectors.T


def iter_snapshots(paths: Iterable[Path]):
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for snapshot in payload.get("snapshots", []):
            if snapshot.get("has_eigendecomposition", False):
                yield path, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile FOAM h(epsilon) and direct inverse-root EVD on saved factors."
    )
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="snapshot_profile.csv")
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative.")

    device = torch.device(args.device)
    rows = []
    for source, item in iter_snapshots(args.snapshots):
        factor = item["factor_matrix"].to(device=device, dtype=torch.float32)
        eigenvectors = item["eigenvectors"].to(device=device, dtype=torch.float32)
        eigenvalues = item["eigenvalues"].to(device=device, dtype=torch.float32)
        epsilon = float(item["epsilon"])
        root = int(item["root"])

        proxy_seconds = benchmark(
            lambda: foam_proxy(
                factor, eigenvectors, eigenvalues, epsilon, root
            ),
            args.repeats,
            args.warmup,
            device,
        )
        evd_seconds = benchmark(
            lambda: direct_inverse_root(factor, epsilon, root),
            args.repeats,
            args.warmup,
            device,
        )
        rows.append(
            {
                "source": str(source),
                "factor_index": item.get("factor_index", ""),
                "dimension": int(factor.shape[0]),
                "root": root,
                "epsilon": epsilon,
                "proxy_ms": proxy_seconds * 1000.0,
                "inverse_root_evd_ms": evd_seconds * 1000.0,
                "evd_over_proxy": evd_seconds / max(proxy_seconds, 1e-30),
            }
        )

    if not rows:
        raise SystemExit("No stored factor with a valid eigendecomposition was found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dimension"]].append(row)
    print("dimension,median_proxy_ms,median_inverse_root_evd_ms,median_ratio")
    for dimension in sorted(grouped):
        group = grouped[dimension]
        print(
            f"{dimension},"
            f"{statistics.median(row['proxy_ms'] for row in group):.6f},"
            f"{statistics.median(row['inverse_root_evd_ms'] for row in group):.6f},"
            f"{statistics.median(row['evd_over_proxy'] for row in group):.3f}"
        )
    print(output)


if __name__ == "__main__":
    main()
