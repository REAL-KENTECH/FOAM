from __future__ import annotations

import torch

from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    PreconditionerUpdateMode,
)


def make(model: torch.nn.Module) -> DistributedShampoo:
    return DistributedShampoo(
        model.parameters(),
        lr=1e-2,
        betas=(0.0, 0.9),
        epsilon=1e-4,
        max_preconditioner_dim=8,
        precondition_frequency=1,
        start_preconditioning_step=1,
        inv_root_override=2,
        grafting_config=AdamGraftingConfig(beta2=0.9, epsilon=1e-8),
        preconditioner_update_mode=PreconditionerUpdateMode.FOAM,
        matrix_root_inv_threshold=0.5,
        max_epsilon=1e-2,
    )


def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scale: float) -> None:
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = torch.arange(
            parameter.numel(), dtype=parameter.dtype
        ).reshape_as(parameter).add(1).mul(scale)
    optimizer.step()


def compact(rows):
    keys = (
        "factor_id",
        "epsilon",
        "last_proxy",
        "evd_calls",
        "proxy_calls",
        "reuse_calls",
        "damping_updates",
        "cap_refreshes",
        "last_refresh_step",
    )
    return [{key: row[key] for key in keys} for row in rows]


def test_distributed_state_dict_preserves_foam_controller_state() -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = make(model)
    step(model, optimizer, 1.0)
    step(model, optimizer, 0.25)

    state = optimizer.distributed_state_dict(
        key_to_param=model.named_parameters(), save_param_groups=True
    )
    before = compact(optimizer.get_factor_diagnostics())

    restored_model = torch.nn.Linear(3, 2, bias=False)
    restored_optimizer = make(restored_model)
    restored_optimizer.load_distributed_state_dict(
        state,
        key_to_param=restored_model.named_parameters(),
        save_param_groups=True,
    )
    after = compact(restored_optimizer.get_factor_diagnostics())

    assert after == before
