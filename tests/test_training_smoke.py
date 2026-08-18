from __future__ import annotations

import csv
import json
from pathlib import Path

from foam_experiments.config import ExperimentConfig
from foam_experiments.train_vit import run


def test_cpu_training_smoke_writes_reproducibility_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    config = ExperimentConfig(
        experiment_name="pytest_smoke",
        output_dir=str(output),
        data_backend="synthetic",
        augmentation_backend="torchvision",
        synthetic_train_samples=16,
        synthetic_eval_samples=8,
        image_size=16,
        num_classes=4,
        patch_size=8,
        embedding_dim=16,
        depth=1,
        num_heads=2,
        mlp_dim=32,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        embedding_dropout=0.0,
        per_device_batch_size=4,
        eval_batch_size=4,
        workers=0,
        epochs=1,
        base_lr=1e-3,
        warmup_ratio=0.0,
        optimizer="foam",
        epsilon=1e-5,
        matrix_root_inv_threshold=0.5,
        max_epsilon=1e-2,
        precondition_frequency=1,
        start_preconditioning_step=1,
        max_preconditioner_dim=16,
        inv_root_override=2,
        mixup=0.0,
        cutmix=0.0,
        label_smoothing=0.0,
        full_train_eval_interval=1,
        validation_interval=1,
        save_interval=1,
        factor_diagnostics_interval=1,
    )

    summary = run(config, force_cpu=True)
    assert summary["global_step"] == 4
    assert summary["epochs_completed"] == 1

    required = [
        "resolved_config.yaml",
        "run_manifest.json",
        "metrics.csv",
        "factor_diagnostics.csv",
        "summary.json",
        "checkpoints/last/model.pt",
        "checkpoints/last/rank_0000.pt",
        "checkpoints/last/metadata.json",
    ]
    for relative in required:
        assert (output / relative).exists(), relative

    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["train_full_hard_ce"]
    assert rows[0]["end_to_end_wall_clock_seconds"]
    assert float(rows[0]["proxy_calls"]) > 0

    on_disk_summary = json.loads((output / "summary.json").read_text())
    assert on_disk_summary["global_step"] == summary["global_step"]
