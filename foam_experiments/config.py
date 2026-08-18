from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Flat, serializable configuration for the ViT/ImageNet experiments."""

    # Run identity and reproducibility.
    experiment_name: str = "vit_foam"
    output_dir: str = "runs/vit_foam"
    seed: int = 42
    deterministic: bool = False
    resume: str = ""
    expected_world_size: int = 0  # 0 disables the check; paper ViT configs use 4.

    # Data.
    data_backend: str = "imagefolder"  # imagefolder | huggingface | synthetic
    data_path: str = "./data/imagenet"
    hf_cache_dir: str = "./data/hf_cache"
    hf_dataset_name: str = "imagenet-1k"
    hf_train_split: str = "train"
    hf_val_split: str = "validation"
    hf_image_key: str = "image"
    hf_label_key: str = "label"
    image_size: int = 224
    num_classes: int = 1000
    per_device_batch_size: int = 256
    eval_batch_size: int = 256
    workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = True
    synthetic_train_samples: int = 4096
    synthetic_eval_samples: int = 1024
    augmentation_backend: str = "timm"  # timm | torchvision
    auto_augment: str = "rand-m15-n2-mstd0.5"
    interpolation: str = "bicubic"
    mixup: float = 0.2
    cutmix: float = 0.0
    label_smoothing: float = 0.1
    random_erasing: float = 0.0

    # Model. Defaults preserve the uploaded ViT-S/16 implementation.
    model_impl: str = "paper_custom"  # paper_custom | timm_vit_small_patch16_224
    patch_size: int = 16
    embedding_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_dim: int = 1536
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.1
    embedding_dropout: float = 0.1
    init_scheme: str = "source"  # source | vit
    compile_model: bool = False

    # Training.
    epochs: int = 90
    max_steps: int = -1
    # Execution-only preemption hook. It does not change the planned LR schedule.
    # A positive value stops this invocation after that absolute epoch and writes
    # a resumable checkpoint.
    stop_after_epoch: int = -1
    base_lr: float = 2.0e-3
    warmup_ratio: float = 0.05
    warmup_steps: int = -1
    weight_decay: float = 4.2e-4
    beta1: float = 0.95
    beta2: float = 0.995
    precision: str = "fp32"  # fp32 | bf16 | fp16
    grad_clip_norm: float = 0.0
    log_interval: int = 50
    validation_interval: int = 1
    full_train_eval_interval: int = 1
    save_interval: int = 45
    save_best_model: bool = True

    # Optimizer family.
    optimizer: str = "foam"  # foam | stale_shampoo | ablations | dr_shampoo | adamw | soap
    epsilon: float = 1.0e-9
    epsilon_left: Optional[float] = None
    epsilon_right: Optional[float] = None
    matrix_root_inv_threshold: float = 0.5
    max_epsilon: float = 3.0e-7
    diagonal_residual_threshold: float = 0.75
    precondition_frequency: int = 20
    start_preconditioning_step: int = 20
    max_preconditioner_dim: int = 1024
    inv_root_override: int = 2
    exponent_multiplier: float = 1.0
    adam_grafting_beta2: float = 0.995
    grafting_epsilon: float = 1.0e-9
    use_bias_correction: bool = True
    use_merge_dims: bool = True
    use_decoupled_weight_decay: bool = True
    use_normalized_grafting: bool = False
    profile_preconditioner: bool = False
    communication_dtype: str = "fp32"
    trainers_per_group: int = -1
    communicate_params: bool = False

    # Optional external SOAP baseline. The source is not bundled.
    soap_module_path: str = ""
    soap_shampoo_beta: float = -1.0
    soap_precondition_1d: bool = False
    soap_normalize_grads: bool = False

    # Logging.
    use_wandb: bool = False
    wandb_project: str = "ViT_FOAM"
    wandb_entity: str = ""
    factor_diagnostics_interval: int = 1
    factor_snapshot_interval: int = 0
    factor_snapshot_max_per_rank: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise KeyError(f"Unknown configuration keys: {unknown}")
        return cls(**dict(values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise TypeError(f"Expected a mapping in {config_path}, got {type(values).__name__}.")
        return cls.from_mapping(values)

    def apply_overrides(self, overrides: Iterable[str]) -> None:
        valid = {field.name for field in fields(self)}
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Override must be KEY=VALUE, got {override!r}.")
            key, raw_value = override.split("=", 1)
            key = key.strip()
            if key not in valid:
                raise KeyError(f"Unknown override key: {key}")
            value = yaml.safe_load(raw_value)
            setattr(self, key, value)

    def validate(self, world_size: int = 1) -> None:
        if self.data_backend not in {"imagefolder", "huggingface", "synthetic"}:
            raise ValueError(f"Unsupported data_backend={self.data_backend!r}.")
        if self.augmentation_backend not in {"timm", "torchvision"}:
            raise ValueError(f"Unsupported augmentation_backend={self.augmentation_backend!r}.")
        if self.model_impl not in {"paper_custom", "timm_vit_small_patch16_224"}:
            raise ValueError(f"Unsupported model_impl={self.model_impl!r}.")
        valid_optimizers = {
            "foam",
            "stale_shampoo",
            "foam_no_adaptive_epsilon",
            "foam_no_evd_refresh",
            "dr_shampoo",
            "adamw",
            "soap",
        }
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Unsupported optimizer={self.optimizer!r}; expected {sorted(valid_optimizers)}.")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported precision={self.precision!r}.")
        if self.per_device_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if self.precondition_frequency < 1:
            raise ValueError("precondition_frequency must be at least 1.")
        if self.start_preconditioning_step < self.precondition_frequency:
            raise ValueError("start_preconditioning_step must be >= precondition_frequency.")
        if self.optimizer != "adamw" and self.epsilon <= 0:
            raise ValueError("Shampoo damping epsilon must be positive.")
        if self.optimizer in {
            "foam",
            "foam_no_adaptive_epsilon",
            "foam_no_evd_refresh",
        } and self.matrix_root_inv_threshold <= 0:
            raise ValueError("FOAM modes require matrix_root_inv_threshold > 0.")
        if self.max_epsilon < self.epsilon:
            raise ValueError("max_epsilon must be at least epsilon.")
        if self.diagonal_residual_threshold < 0:
            raise ValueError("diagonal_residual_threshold must be non-negative.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if self.expected_world_size < 0:
            raise ValueError("expected_world_size must be non-negative.")
        if self.expected_world_size and world_size != self.expected_world_size:
            raise ValueError(
                "This configuration requires a fixed world size to preserve the "
                "reported global batch size: "
                f"expected={self.expected_world_size}, actual={world_size}."
            )

    @property
    def global_batch_size(self) -> int:
        # Filled with the actual world size in the run manifest; this property is
        # intentionally per-process independent.
        return self.per_device_batch_size

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
