#!/usr/bin/env python3
"""Verify epoch-boundary checkpoint/resume against an uninterrupted CPU run.

The test uses the execution-only ``stop_after_epoch`` hook, so both paths keep
exactly the same planned number of optimization steps and learning-rate
schedule. It compares the final model and optimizer tensor states.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foam_experiments.config import ExperimentConfig
from foam_experiments.train_vit import run


def _base_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="resume_equivalence",
        output_dir=str(output_dir),
        seed=123,
        deterministic=True,
        data_backend="synthetic",
        augmentation_backend="torchvision",
        synthetic_train_samples=16,
        synthetic_eval_samples=8,
        image_size=16,
        num_classes=4,
        patch_size=4,
        embedding_dim=16,
        depth=1,
        num_heads=4,
        mlp_dim=32,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        embedding_dropout=0.0,
        per_device_batch_size=4,
        eval_batch_size=4,
        workers=0,
        epochs=2,
        max_steps=-1,
        base_lr=1.0e-3,
        warmup_ratio=0.25,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        mixup=0.0,
        cutmix=0.0,
        label_smoothing=0.0,
        optimizer="foam",
        epsilon=1.0e-4,
        matrix_root_inv_threshold=0.5,
        max_epsilon=1.0e-1,
        precondition_frequency=1,
        start_preconditioning_step=1,
        max_preconditioner_dim=32,
        inv_root_override=2,
        adam_grafting_beta2=0.99,
        grafting_epsilon=1.0e-6,
        full_train_eval_interval=1,
        validation_interval=1,
        save_interval=1,
        save_best_model=False,
        log_interval=0,
        factor_diagnostics_interval=1,
    )


def _compare(left: Any, right: Any, path: str = "root") -> tuple[int, float]:
    """Return ``(tensor_count, maximum_absolute_difference)`` or raise."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"Type mismatch at {path}: Tensor vs {type(right)}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(
                f"Tensor metadata mismatch at {path}: "
                f"{left.shape}/{left.dtype} vs {right.shape}/{right.dtype}"
            )
        if left.is_floating_point() or left.is_complex():
            difference = float((left - right).abs().max().item()) if left.numel() else 0.0
        else:
            difference = 0.0 if torch.equal(left, right) else float("inf")
        if not torch.equal(left, right):
            raise AssertionError(f"Tensor mismatch at {path}; max_abs_diff={difference}")
        return 1, difference
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"Dictionary-key mismatch at {path}")
        count = 0
        maximum = 0.0
        for key in sorted(left, key=str):
            child_count, child_maximum = _compare(left[key], right[key], f"{path}.{key}")
            count += child_count
            maximum = max(maximum, child_maximum)
        return count, maximum
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"Sequence mismatch at {path}")
        count = 0
        maximum = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child_count, child_maximum = _compare(
                left_item, right_item, f"{path}[{index}]"
            )
            count += child_count
            maximum = max(maximum, child_maximum)
        return count, maximum
    if left != right:
        raise AssertionError(f"Value mismatch at {path}: {left!r} vs {right!r}")
    return 0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check exact epoch-boundary checkpoint/resume equivalence."
    )
    parser.add_argument("--output-dir", default="runs/verification_resume_equivalence")
    parser.add_argument("--keep", action="store_true", help="Keep generated runs.")
    args = parser.parse_args()

    root = Path(args.output_dir)
    if root.exists():
        shutil.rmtree(root)
    full_dir = root / "uninterrupted"
    stage_dir = root / "stage1"
    resumed_dir = root / "resumed"

    full_config = _base_config(full_dir)
    full_summary = run(full_config, force_cpu=True)

    stage_config = _base_config(stage_dir)
    stage_config.stop_after_epoch = 1
    stage_summary = run(stage_config, force_cpu=True)
    if stage_summary["epochs_completed"] != 1:
        raise AssertionError("The staged run did not stop after epoch 1.")

    resumed_config = _base_config(resumed_dir)
    resumed_config.resume = str(stage_dir / "checkpoints" / "last")
    resumed_summary = run(resumed_config, force_cpu=True)

    full_model = torch.load(
        full_dir / "checkpoints" / "last" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_model = torch.load(
        resumed_dir / "checkpoints" / "last" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_tensors, model_max_diff = _compare(full_model, resumed_model, "model")

    full_rank = torch.load(
        full_dir / "checkpoints" / "last" / "rank_0000.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_rank = torch.load(
        resumed_dir / "checkpoints" / "last" / "rank_0000.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer_tensors, optimizer_max_diff = _compare(
        full_rank["optimizer"], resumed_rank["optimizer"], "optimizer"
    )

    result = {
        "status": "PASS",
        "global_step": resumed_summary["global_step"],
        "full_epochs": full_summary["epochs_completed"],
        "resumed_epochs": resumed_summary["epochs_completed"],
        "model_tensor_count": model_tensors,
        "optimizer_tensor_count": optimizer_tensors,
        "model_max_abs_diff": model_max_diff,
        "optimizer_max_abs_diff": optimizer_max_diff,
        "output_dir": str(root),
    }
    report = root / "resume_equivalence.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if not args.keep:
        # Keep the compact report, remove the much larger checkpoints.
        for directory in (full_dir, stage_dir, resumed_dir):
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
