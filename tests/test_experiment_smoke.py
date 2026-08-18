from __future__ import annotations

import csv
import json
from pathlib import Path

from foam_experiments.config import ExperimentConfig
from foam_experiments.train_vit import run


def test_cpu_synthetic_training_writes_reproducibility_artifacts(tmp_path: Path):
    output = tmp_path / "run"
    config = ExperimentConfig(
        experiment_name="pytest_smoke",
        output_dir=str(output),
        data_backend="synthetic",
        augmentation_backend="torchvision",
        image_size=16,
        patch_size=4,
        embedding_dim=16,
        depth=1,
        num_heads=4,
        mlp_dim=32,
        num_classes=4,
        per_device_batch_size=4,
        eval_batch_size=4,
        workers=0,
        synthetic_train_samples=16,
        synthetic_eval_samples=8,
        epochs=1,
        max_steps=2,
        base_lr=1.0e-3,
        warmup_ratio=0.5,
        mixup=0.0,
        label_smoothing=0.0,
        optimizer="foam",
        epsilon=1.0e-4,
        matrix_root_inv_threshold=0.5,
        max_epsilon=0.1,
        precondition_frequency=1,
        start_preconditioning_step=1,
        max_preconditioner_dim=32,
        inv_root_override=2,
        full_train_eval_interval=1,
        validation_interval=1,
        save_interval=1,
        log_interval=0,
        profile_preconditioner=True,
    )
    summary = run(config, force_cpu=True)
    assert summary["global_step"] == 2
    for relative in (
        "resolved_config.yaml",
        "run_manifest.json",
        "metrics.csv",
        "factor_diagnostics.csv",
        "summary.json",
        "checkpoints/last/model.pt",
        "checkpoints/last/rank_0000.pt",
    ):
        assert (output / relative).exists(), relative

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["warmup_steps"] == 1
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert "train_full_ce" in rows[0]
    assert "wall_clock_seconds" in rows[0]
    assert "evd_rate" in rows[0]
