from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
    PreconditionerUpdateMode,
)

from .config import ExperimentConfig
from .distributed import DistributedContext
from .soap_adapter import build_soap_optimizer


_OPTIMIZER_TO_MODE = {
    "foam": PreconditionerUpdateMode.FOAM,
    "stale_shampoo": PreconditionerUpdateMode.STALE,
    "foam_no_adaptive_epsilon": PreconditionerUpdateMode.FOAM_NO_ADAPTIVE_EPSILON,
    "foam_no_evd_refresh": PreconditionerUpdateMode.FOAM_NO_EVD_REFRESH,
    "dr_shampoo": PreconditionerUpdateMode.DR_SHAMPOO,
}


def build_optimizer(
    config: ExperimentConfig,
    model: nn.Module,
    context: DistributedContext,
) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            betas=(config.beta1, config.beta2),
            eps=config.grafting_epsilon,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "soap":
        return build_soap_optimizer(config, model.parameters())

    communication_dtype = {
        "fp32": CommunicationDType.FP32,
        "fp16": CommunicationDType.FP16,
        "bf16": CommunicationDType.BF16,
        "default": CommunicationDType.DEFAULT,
    }.get(config.communication_dtype.lower())
    if communication_dtype is None:
        raise ValueError(f"Unknown communication_dtype={config.communication_dtype!r}.")

    distributed_config = None
    if context.distributed:
        trainers_per_group = (
            context.world_size
            if config.trainers_per_group == -1
            else config.trainers_per_group
        )
        distributed_config = DDPShampooConfig(
            communication_dtype=communication_dtype,
            num_trainers_per_group=trainers_per_group,
            communicate_params=config.communicate_params,
        )

    mode = _OPTIMIZER_TO_MODE[config.optimizer]
    return DistributedShampoo(
        params=model.parameters(),
        lr=config.base_lr,
        betas=(config.beta1, config.beta2),
        epsilon=config.epsilon,
        epsilon_left=config.epsilon_left,
        epsilon_right=config.epsilon_right,
        momentum=0.0,
        weight_decay=config.weight_decay,
        max_preconditioner_dim=config.max_preconditioner_dim,
        precondition_frequency=config.precondition_frequency,
        start_preconditioning_step=config.start_preconditioning_step,
        inv_root_override=config.inv_root_override,
        exponent_multiplier=config.exponent_multiplier,
        use_bias_correction=config.use_bias_correction,
        use_decoupled_weight_decay=config.use_decoupled_weight_decay,
        grafting_config=AdamGraftingConfig(
            beta2=config.adam_grafting_beta2,
            epsilon=config.grafting_epsilon,
        ),
        use_normalized_grafting=config.use_normalized_grafting,
        use_merge_dims=config.use_merge_dims,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        use_protected_eigh=True,
        track_root_inv_residuals=False,
        matrix_root_inv_threshold=(
            0.0 if mode is PreconditionerUpdateMode.STALE else config.matrix_root_inv_threshold
        ),
        max_epsilon=config.max_epsilon,
        preconditioner_update_mode=mode,
        diagonal_residual_threshold=config.diagonal_residual_threshold,
        profile_preconditioner=config.profile_preconditioner,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def optimizer_name(optimizer: torch.optim.Optimizer) -> str:
    return optimizer.__class__.__name__


def has_foam_diagnostics(optimizer: torch.optim.Optimizer) -> bool:
    return hasattr(optimizer, "get_preconditioner_diagnostics")


def local_preconditioner_diagnostics(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    if hasattr(optimizer, "get_factor_diagnostics"):
        return list(optimizer.get_factor_diagnostics())
    if not has_foam_diagnostics(optimizer):
        return []
    payload = optimizer.get_preconditioner_diagnostics(include_factors=True)
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(payload.get("groups", [])):
        for row in group.get("factors", []):
            rows.append({"group": group_index, **row})
    return rows


def local_preconditioner_profile(
    optimizer: torch.optim.Optimizer, reset: bool = False
) -> dict[str, float]:
    if not hasattr(optimizer, "get_preconditioner_profile"):
        return {"proxy_seconds": 0.0, "evd_seconds": 0.0, "reuse_seconds": 0.0}
    return dict(optimizer.get_preconditioner_profile(reset=reset))
