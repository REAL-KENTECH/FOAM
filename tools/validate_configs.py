#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foam_experiments.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all experiment YAML files.")
    parser.add_argument("root", nargs="?", default="configs/vit")
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    paths = sorted(Path(args.root).rglob("*.yaml"))
    if not paths:
        raise SystemExit(f"No YAML files found under {args.root}")
    for path in paths:
        config = ExperimentConfig.from_yaml(path)
        config.validate(args.world_size)
        print(path)
    print(f"validated={len(paths)}")


if __name__ == "__main__":
    main()
