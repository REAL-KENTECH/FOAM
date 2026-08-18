#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _as_override(key: str, value: Any) -> str:
    # JSON is a strict subset of YAML, so ExperimentConfig.apply_overrides() can
    # parse it with yaml.safe_load without the document terminator (``...``)
    # emitted by yaml.safe_dump for scalar values.
    rendered = json.dumps(value, separators=(",", ":"))
    return f"{key}={rendered}"


def build_command(spec: dict[str, Any], run: dict[str, Any]) -> list[str]:
    launcher = str(spec.get("launcher", "torchrun"))
    nproc = int(spec.get("nproc_per_node", 1))
    base_config = str(spec["base_config"])
    command = (
        [launcher, "--standalone", f"--nproc-per-node={nproc}"]
        if nproc > 1
        else ["python"]
    )
    command += ["-m", "foam_experiments.train_vit", "--config", base_config]
    overrides = dict(run.get("overrides", {}))
    name = str(run["name"])
    overrides.setdefault("experiment_name", name)
    overrides.setdefault("output_dir", f"runs/sweeps/{name}")
    overrides.setdefault("expected_world_size", nproc)
    for key, value in overrides.items():
        command += ["--set", _as_override(key, value)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a YAML-defined FOAM sweep sequentially.")
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="Run name to include; repeatable.")
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    with sweep_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    selected = set(args.only)
    for run in spec.get("runs", []):
        if selected and run["name"] not in selected:
            continue
        command = build_command(spec, run)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
