from __future__ import annotations

from typing import Callable

import pytest
import torch

from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    PreconditionerUpdateMode,
)
from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import (
    ShampooPreconditionerList,
)


def build_optimizer(
    model: torch.nn.Module,
    mode: PreconditionerUpdateMode,
    *,
    epsilon: float = 1.0e-3,
    tau: float = 0.5,
    max_epsilon: float = 1.0e-2,
) -> DistributedShampoo:
    return DistributedShampoo(
        model.parameters(),
        lr=1.0e-2,
        betas=(0.0, 1.0),
        epsilon=epsilon,
        momentum=0.0,
        weight_decay=0.0,
        max_preconditioner_dim=8,
        precondition_frequency=1,
        start_preconditioning_step=1,
        inv_root_override=2,
        exponent_multiplier=1.0,
        use_bias_correction=True,
        grafting_config=AdamGraftingConfig(beta2=0.9, epsilon=1.0e-8),
        preconditioner_update_mode=mode,
        matrix_root_inv_threshold=0.0 if mode is PreconditionerUpdateMode.STALE else tau,
        max_epsilon=max_epsilon,
    )


def take_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, gradient_value: float) -> None:
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, gradient_value)
    optimizer.step()


def preconditioner(optimizer: DistributedShampoo) -> ShampooPreconditionerList:
    return optimizer._per_group_state_lists[0]["shampoo_preconditioner_list"]


def test_proxy_matches_paper_formula_and_includes_one_over_p() -> None:
    controller = object.__new__(ShampooPreconditionerList)
    controller._exponent_multiplier = 1.0

    stale = torch.tensor([[3.0, 0.4], [0.4, 1.5]], dtype=torch.float64)
    current = torch.tensor([[3.2, 0.2], [0.2, 1.7]], dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(stale)
    epsilon = 0.1
    root = 2

    rc, alpha, proxy = controller._compute_foam_proxy(
        current, eigenvectors, eigenvalues, epsilon, root
    )

    rotated_drift = eigenvectors.T @ current @ eigenvectors - torch.diag(eigenvalues)
    shifted = eigenvalues + epsilon
    expected_rc = torch.linalg.norm(
        rotated_drift / torch.outer(shifted.sqrt(), shifted.sqrt()), ord="fro"
    )
    inverse_root_eigenvalues = shifted.pow(-1.0 / root)
    expected_alpha = inverse_root_eigenvalues.max() / torch.linalg.vector_norm(
        inverse_root_eigenvalues
    )
    expected_proxy = expected_rc * expected_alpha / root

    torch.testing.assert_close(rc, expected_rc)
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(proxy, expected_proxy)
    assert not torch.isclose(proxy, expected_rc * expected_alpha)


def test_stale_mode_does_not_compute_foam_proxy() -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = build_optimizer(model, PreconditionerUpdateMode.STALE)
    take_step(model, optimizer, 1.0)

    controller = preconditioner(optimizer)

    def fail(*args, **kwargs):
        raise AssertionError("The stale baseline must not pay FOAM proxy cost.")

    controller._compute_foam_proxy = fail  # type: ignore[method-assign]
    take_step(model, optimizer, 0.5)

    diagnostics = optimizer.get_preconditioner_diagnostics()["groups"][0]
    assert diagnostics["proxy_calls"] == 0
    assert diagnostics["evd_calls"] == diagnostics["check_calls"]


def test_adaptive_epsilon_never_drops_below_base_floor() -> None:
    base_epsilon = 1.0e-3
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = build_optimizer(
        model, PreconditionerUpdateMode.FOAM, epsilon=base_epsilon
    )
    take_step(model, optimizer, 1.0)
    controller = preconditioner(optimizer)

    for factors in controller._local_kronecker_factors_list:
        for epsilon_state in factors.adaptive_epsilons:
            epsilon_state.fill_(1.0e-2)

    def zero_proxy(*args, **kwargs):
        scalar = torch.tensor(0.0)
        return scalar, scalar, scalar

    controller._compute_foam_proxy = zero_proxy  # type: ignore[method-assign]
    take_step(model, optimizer, 0.0)

    rows = optimizer.get_factor_diagnostics()
    assert rows
    assert all(row["epsilon"] == pytest.approx(base_epsilon) for row in rows)
    assert all(row["evd_calls"] == 1 for row in rows)


def test_epsilon_cap_triggers_fresh_evd_and_resets_damping() -> None:
    base_epsilon = 1.0e-3
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = build_optimizer(
        model,
        PreconditionerUpdateMode.FOAM,
        epsilon=base_epsilon,
        tau=0.5,
        max_epsilon=2.0e-3,
    )
    take_step(model, optimizer, 1.0)
    controller = preconditioner(optimizer)

    def large_proxy(*args, **kwargs):
        return torch.tensor(1.0), torch.tensor(1.0), torch.tensor(10.0)

    controller._compute_foam_proxy = large_proxy  # type: ignore[method-assign]
    take_step(model, optimizer, 0.5)

    rows = optimizer.get_factor_diagnostics()
    assert all(row["epsilon"] == pytest.approx(base_epsilon) for row in rows)
    assert all(row["evd_calls"] == 2 for row in rows)
    assert all(row["cap_refreshes"] == 1 for row in rows)
