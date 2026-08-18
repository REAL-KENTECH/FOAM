import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd import profiler

from .shampoo_block_info import BlockInfo
from .shampoo_utils import compress_list, get_dtype_size
from ...matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
)
from ...optimizer_modules import OptimizerModule


logger: logging.Logger = logging.getLogger(__name__)


def _dist_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


RWS_ADAGRAD = "rws_adagrad"
ADAGRAD = "adagrad"
SHAMPOO = "shampoo"


class PreconditionerList(ABC):
    """Preconditioner base class."""
    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._numel_list: Tuple[int, ...] = (0,) * len(block_list)
        self._dims_list: Tuple[torch.Size, ...] = tuple(
            block.size() for block in block_list
        )
        self._num_bytes_list: Tuple[int, ...] = (0,) * len(block_list)

    @abstractmethod
    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        ...

    @abstractmethod
    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        ...

    @abstractmethod
    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        ...

    @property
    def numel_list(self) -> Tuple[int, ...]:
        return self._numel_list

    @property
    def dims_list(self) -> Tuple[torch.Size, ...]:
        return self._dims_list

    @property
    def num_bytes_list(self) -> Tuple[int, ...]:
        return self._num_bytes_list

    def numel(self) -> int:
        return sum(self._numel_list)

    def num_bytes(self) -> int:
        return sum(self._num_bytes_list)


class SGDPreconditionerList(PreconditionerList):
    """SGD (identity) preconditioners for a list of parameters."""
    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__(block_list)

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        return

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return masked_grad_list

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        return


class RWSAdagradPreconditionerList(PreconditionerList):
    """Row-Wise Adagrad / Adam / RMSProp preconditioners."""
    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)
        self._beta2 = beta2
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        preconditioner_list = []
        for block, block_info in zip(block_list, block_info_list, ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[RWS_ADAGRAD] = block_info.allocate_zeros_tensor(
                block.shape[0], block.dtype, block.device
            )
            preconditioner_list.append(block_info.get_tensor(block_state[RWS_ADAGRAD]))

            logger.info(
                f"Instantiated RWS Adagrad Preconditioner {preconditioner_index} ({block_state[RWS_ADAGRAD].shape}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        self._local_preconditioner_list: Tuple[Tensor, ...] = compress_list(
            preconditioner_list, distributor_selector
        )
        self._masked_preconditioner_list: Tuple[
            Tensor, ...
        ] = self._local_preconditioner_list

        self._numel_list: Tuple[int, ...] = tuple(
            preconditioner.numel() for preconditioner in preconditioner_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            preconditioner.numel() * preconditioner.element_size()
            for preconditioner in preconditioner_list
        )

        logger.info(
            f"Rank {_dist_rank()}: RWSAdaGradPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {_dist_rank()}: RWSAdaGradPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {_dist_rank()}: RWSAdaGradPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {_dist_rank()}: RWSAdaGradPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
        )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            masked_avg_rws_grad_norm_sq_list = tuple(
                torch.mean(grad * grad, axis=tuple(torch.arange(1, grad.dim())))
                for grad in masked_grad_list
            )
            if self._beta2 == 1.0:
                torch._foreach_add_(
                    self._masked_preconditioner_list,
                    masked_avg_rws_grad_norm_sq_list,
                    value=1.0,
                )
            else:
                torch._foreach_mul_(self._masked_preconditioner_list, self._beta2)
                torch._foreach_add_(
                    self._masked_preconditioner_list,
                    masked_avg_rws_grad_norm_sq_list,
                    alpha=1.0 - self._beta2,
                )

            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = 1.0 - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            masked_bias_corrected_preconditioner_list = torch._foreach_div(
                self._masked_preconditioner_list, self._bias_correction2
            )
            torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
            torch._foreach_add_(
                masked_bias_corrected_preconditioner_list, self._epsilon
            )
            return tuple(
                grad / bias_corrected_preconditioner[(...,) + (None,) * (grad.dim() - 1)]
                for grad, bias_corrected_preconditioner in zip(
                    masked_grad_list, masked_bias_corrected_preconditioner_list
                )
            )

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_preconditioner_list = compress_list(
                self._local_preconditioner_list, local_grad_selector
            )


class AdagradPreconditionerList(PreconditionerList):
    """Adagrad / Adam / RMSProp preconditioners."""
    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)
        self._beta2 = beta2
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        preconditioner_list = []
        for block, block_info in zip(block_list, block_info_list, ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[ADAGRAD] = block_info.allocate_zeros_tensor(
                block.size(), block.dtype, block.device
            )
            preconditioner_list.append(block_info.get_tensor(block_state[ADAGRAD]))

            logger.info(
                f"Instantiated Adagrad Preconditioner {preconditioner_index} ({block_state[ADAGRAD].shape}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        self._local_preconditioner_list: Tuple[Tensor, ...] = compress_list(
            preconditioner_list, distributor_selector
        )
        self._masked_preconditioner_list: Tuple[
            Tensor, ...
        ] = self._local_preconditioner_list

        self._numel_list: Tuple[int, ...] = tuple(
            preconditioner.numel() for preconditioner in preconditioner_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            preconditioner.numel() * preconditioner.element_size()
            for preconditioner in preconditioner_list
        )

        logger.info(
            f"Rank {_dist_rank()}: AdaGradPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {_dist_rank()}: AdaGradPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {_dist_rank()}: AdaGradPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {_dist_rank()}: AdaGradPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
        )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            if self._beta2 == 1.0:
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list,
                    masked_grad_list,
                    masked_grad_list,
                    value=1.0,
                )
            else:
                torch._foreach_mul_(self._masked_preconditioner_list, self._beta2)
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list,
                    masked_grad_list,
                    masked_grad_list,
                    value=1 - self._beta2,
                )

            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = 1.0 - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            masked_bias_corrected_preconditioner_list = torch._foreach_div(
                self._masked_preconditioner_list, self._bias_correction2
            )
            torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
            torch._foreach_add_(
                masked_bias_corrected_preconditioner_list, self._epsilon
            )
            return torch._foreach_div(
                masked_grad_list, masked_bias_corrected_preconditioner_list
            )

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_preconditioner_list = compress_list(
                self._local_preconditioner_list, local_grad_selector
            )



@dataclass
class ShampooKroneckerFactors(OptimizerModule):
    """Checkpointable Shampoo/FOAM state for one parameter block.

    All algorithmically relevant FOAM state is stored as tensors so that
    ``distributed_state_dict`` preserves the stale eigenbasis, eigenvalues,
    adaptive damping, and controller counters across resume.
    """

# Modified in the FOAM experiment reconstruction (2026); see MODIFICATIONS.md.

    factor_matrices: Tuple[Tensor, ...]
    inv_factor_matrices: Tuple[Tensor, ...]
    factor_matrix_indices: Tuple[str, ...]
    is_factor_matrices_diagonal: Tuple[Tensor, ...] = field(default_factory=tuple)
    eigenvalues: Tuple[Tensor, ...] = field(default_factory=tuple)
    eigenvectors: Tuple[Tensor, ...] = field(default_factory=tuple)
    adaptive_epsilons: Tuple[Tensor, ...] = field(default_factory=tuple)
    has_eigendecomposition: Tuple[Tensor, ...] = field(default_factory=tuple)
    last_proxy: Tuple[Tensor, ...] = field(default_factory=tuple)
    last_relative_condition: Tuple[Tensor, ...] = field(default_factory=tuple)
    last_alpha: Tuple[Tensor, ...] = field(default_factory=tuple)
    check_calls: Tuple[Tensor, ...] = field(default_factory=tuple)
    evd_calls: Tuple[Tensor, ...] = field(default_factory=tuple)
    proxy_calls: Tuple[Tensor, ...] = field(default_factory=tuple)
    reuse_calls: Tuple[Tensor, ...] = field(default_factory=tuple)
    damping_updates: Tuple[Tensor, ...] = field(default_factory=tuple)
    cap_refreshes: Tuple[Tensor, ...] = field(default_factory=tuple)
    residual_calls: Tuple[Tensor, ...] = field(default_factory=tuple)
    last_refresh_step: Tuple[Tensor, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__init__()
        count = len(self.factor_matrices)
        if not (
            count == len(self.inv_factor_matrices) == len(self.factor_matrix_indices)
        ):
            raise ValueError("Inconsistent number of Shampoo factor states.")

        def _scalar_tuple(value: float, dtype: torch.dtype) -> Tuple[Tensor, ...]:
            return tuple(
                torch.tensor(value, dtype=dtype, device=factor.device)
                for factor in self.factor_matrices
            )

        if not self.is_factor_matrices_diagonal:
            self.is_factor_matrices_diagonal = tuple(
                torch.tensor(True, dtype=torch.bool, device=factor.device)
                for factor in self.factor_matrices
            )
        if not self.eigenvalues:
            self.eigenvalues = tuple(
                torch.zeros(
                    factor.shape[0] if factor.ndim == 2 else 0,
                    dtype=factor.dtype,
                    device=factor.device,
                )
                for factor in self.factor_matrices
            )
        if not self.eigenvectors:
            self.eigenvectors = tuple(torch.zeros_like(factor) for factor in self.factor_matrices)
        if not self.adaptive_epsilons:
            self.adaptive_epsilons = _scalar_tuple(0.0, torch.float64)
        if not self.has_eigendecomposition:
            self.has_eigendecomposition = tuple(
                torch.tensor(False, dtype=torch.bool, device=factor.device)
                for factor in self.factor_matrices
            )
        if not self.last_proxy:
            self.last_proxy = _scalar_tuple(float("nan"), torch.float64)
        if not self.last_relative_condition:
            self.last_relative_condition = _scalar_tuple(float("nan"), torch.float64)
        if not self.last_alpha:
            self.last_alpha = _scalar_tuple(float("nan"), torch.float64)
        for name in (
            "check_calls",
            "evd_calls",
            "proxy_calls",
            "reuse_calls",
            "damping_updates",
            "cap_refreshes",
            "residual_calls",
        ):
            if not getattr(self, name):
                setattr(
                    self,
                    name,
                    tuple(
                        torch.tensor(0, dtype=torch.int64, device=factor.device)
                        for factor in self.factor_matrices
                    ),
                )
        if not self.last_refresh_step:
            self.last_refresh_step = tuple(
                torch.tensor(-1, dtype=torch.int64, device=factor.device)
                for factor in self.factor_matrices
            )


class ShampooPreconditionerList(PreconditionerList):
    """Shampoo preconditioners with canonical stale/FOAM refresh policies."""

    _VALID_MODES = {
        "stale",
        "foam",
        "foam_no_adaptive_epsilon",
        "foam_no_evd_refresh",
        "dr_shampoo",
    }

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        epsilon_left: Optional[float] = None,
        epsilon_right: Optional[float] = None,
        matrix_root_inv_threshold: float = 0.0,
        max_epsilon: float = 1.0,
        preconditioner_update_mode: str = "stale",
        diagonal_residual_threshold: float = 0.1,
        profile_preconditioner: bool = False,
        inv_root_override: Union[int, Tuple[int, ...]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
        use_protected_eigh: bool = True,
        # Backward-compatible ignored arguments.
        use_adaptive_epsilon: bool = False,
        condition_thresholds: Optional[Dict[float, float]] = None,
        is_default_config: bool = False,
        use_trace_correction: bool = False,
    ) -> None:
        del use_adaptive_epsilon, condition_thresholds, is_default_config, use_trace_correction
        super().__init__(block_list)

        mode = str(getattr(preconditioner_update_mode, "value", preconditioner_update_mode)).lower()
        if mode == "diagonal_residual":
            mode = "dr_shampoo"
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown preconditioner_update_mode={preconditioner_update_mode!r}. "
                f"Valid modes: {sorted(self._VALID_MODES)}."
            )
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if max_epsilon < epsilon:
            raise ValueError("max_epsilon must be greater than or equal to epsilon.")
        if mode.startswith("foam") and matrix_root_inv_threshold <= 0.0:
            raise ValueError(
                "FOAM modes require matrix_root_inv_threshold (tau) > 0. "
                "Use mode='stale' for fixed-cadence Shampoo."
            )
        if diagonal_residual_threshold < 0.0:
            raise ValueError("diagonal_residual_threshold must be non-negative.")

        self._beta2 = beta2
        self._epsilon = float(epsilon)
        self._epsilon_left = float(epsilon_left if epsilon_left is not None else epsilon)
        self._epsilon_right = float(epsilon_right if epsilon_right is not None else epsilon)
        self._use_per_dim_epsilon = epsilon_left is not None or epsilon_right is not None
        self._matrix_root_inv_threshold = float(matrix_root_inv_threshold)
        self._max_epsilon = float(max_epsilon)
        self._preconditioner_update_mode = mode
        self._diagonal_residual_threshold = float(diagonal_residual_threshold)
        self._profile_preconditioner = bool(profile_preconditioner)
        self._inv_root_override = inv_root_override
        self._exponent_multiplier = float(exponent_multiplier)
        self._factor_matrix_dtype = factor_matrix_dtype
        self._use_bias_correction = use_bias_correction
        self._use_protected_eigh = use_protected_eigh
        self._bias_correction2: Tensor = torch.tensor(1.0)
        self._runtime_profile: Dict[str, float] = {
            "proxy_seconds": 0.0,
            "evd_seconds": 0.0,
            "reuse_seconds": 0.0,
        }

        kronecker_factors_list = []
        for block, block_info, dims in zip(block_list, block_info_list, self._dims_list):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            factor_matrices = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim), self._factor_matrix_dtype, block_info.param.device
                )
                for dim in dims
            )
            inv_factor_matrices = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim), block.dtype, block_info.param.device
                )
                for dim in dims
            )
            eigenvalues = tuple(
                block_info.allocate_zeros_tensor(
                    (dim,), self._factor_matrix_dtype, block_info.param.device
                )
                for dim in dims
            )
            eigenvectors = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim), self._factor_matrix_dtype, block_info.param.device
                )
                for dim in dims
            )

            def allocate_scalars(dtype: torch.dtype) -> Tuple[Tensor, ...]:
                return tuple(
                    block_info.allocate_zeros_tensor((), dtype, block_info.param.device)
                    for _ in dims
                )

            is_diagonal = allocate_scalars(torch.bool)
            adaptive_epsilons = allocate_scalars(torch.float64)
            has_eigendecomposition = allocate_scalars(torch.bool)
            last_proxy = allocate_scalars(torch.float64)
            last_relative_condition = allocate_scalars(torch.float64)
            last_alpha = allocate_scalars(torch.float64)
            check_calls = allocate_scalars(torch.int64)
            evd_calls = allocate_scalars(torch.int64)
            proxy_calls = allocate_scalars(torch.int64)
            reuse_calls = allocate_scalars(torch.int64)
            damping_updates = allocate_scalars(torch.int64)
            cap_refreshes = allocate_scalars(torch.int64)
            residual_calls = allocate_scalars(torch.int64)
            last_refresh_step = allocate_scalars(torch.int64)

            preconditioner_index = f"{param_index}.{block_index}"
            factor_matrix_indices = tuple(
                f"{preconditioner_index}.{k}" for k in range(len(dims))
            )
            state_kf = ShampooKroneckerFactors(
                factor_matrices=factor_matrices,
                inv_factor_matrices=inv_factor_matrices,
                factor_matrix_indices=factor_matrix_indices,
                is_factor_matrices_diagonal=is_diagonal,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                adaptive_epsilons=adaptive_epsilons,
                has_eigendecomposition=has_eigendecomposition,
                last_proxy=last_proxy,
                last_relative_condition=last_relative_condition,
                last_alpha=last_alpha,
                check_calls=check_calls,
                evd_calls=evd_calls,
                proxy_calls=proxy_calls,
                reuse_calls=reuse_calls,
                damping_updates=damping_updates,
                cap_refreshes=cap_refreshes,
                residual_calls=residual_calls,
                last_refresh_step=last_refresh_step,
            )
            block_state[SHAMPOO] = state_kf

            local_kf = ShampooKroneckerFactors(
                factor_matrices=tuple(block_info.get_tensor(t) for t in factor_matrices),
                inv_factor_matrices=tuple(block_info.get_tensor(t) for t in inv_factor_matrices),
                factor_matrix_indices=factor_matrix_indices,
                is_factor_matrices_diagonal=tuple(block_info.get_tensor(t) for t in is_diagonal),
                eigenvalues=tuple(block_info.get_tensor(t) for t in eigenvalues),
                eigenvectors=tuple(block_info.get_tensor(t) for t in eigenvectors),
                adaptive_epsilons=tuple(block_info.get_tensor(t) for t in adaptive_epsilons),
                has_eigendecomposition=tuple(block_info.get_tensor(t) for t in has_eigendecomposition),
                last_proxy=tuple(block_info.get_tensor(t) for t in last_proxy),
                last_relative_condition=tuple(block_info.get_tensor(t) for t in last_relative_condition),
                last_alpha=tuple(block_info.get_tensor(t) for t in last_alpha),
                check_calls=tuple(block_info.get_tensor(t) for t in check_calls),
                evd_calls=tuple(block_info.get_tensor(t) for t in evd_calls),
                proxy_calls=tuple(block_info.get_tensor(t) for t in proxy_calls),
                reuse_calls=tuple(block_info.get_tensor(t) for t in reuse_calls),
                damping_updates=tuple(block_info.get_tensor(t) for t in damping_updates),
                cap_refreshes=tuple(block_info.get_tensor(t) for t in cap_refreshes),
                residual_calls=tuple(block_info.get_tensor(t) for t in residual_calls),
                last_refresh_step=tuple(block_info.get_tensor(t) for t in last_refresh_step),
            )

            for factor_idx, _ in enumerate(dims):
                base_epsilon = self._base_epsilon_for_factor(len(dims), factor_idx)
                if local_kf.adaptive_epsilons[factor_idx].numel() > 0:
                    local_kf.adaptive_epsilons[factor_idx].fill_(base_epsilon)
                    local_kf.is_factor_matrices_diagonal[factor_idx].fill_(True)
                    local_kf.last_proxy[factor_idx].fill_(float("nan"))
                    local_kf.last_relative_condition[factor_idx].fill_(float("nan"))
                    local_kf.last_alpha[factor_idx].fill_(float("nan"))
                    local_kf.last_refresh_step[factor_idx].fill_(-1)

            kronecker_factors_list.append(local_kf)
            logger.info(
                "Instantiated Shampoo preconditioner %s for parameter %s, block %s, "
                "mode=%s, base_epsilon=%s.",
                preconditioner_index,
                tuple(block_info.param.shape),
                tuple(block.shape),
                self._preconditioner_update_mode,
                [self._base_epsilon_for_factor(len(dims), i) for i in range(len(dims))],
            )

        local_block_list = compress_list(block_list, distributor_selector)
        self._local_kronecker_factors_list: Tuple[ShampooKroneckerFactors, ...] = compress_list(
            kronecker_factors_list, distributor_selector
        )
        self._local_order_list = tuple(block.dim() for block in local_block_list)
        self._local_root_list = self._get_inverse_roots_from_override(
            self._inv_root_override, self._local_order_list
        )
        self._masked_order_list = self._local_order_list
        self._masked_root_list = self._local_root_list
        self._masked_kronecker_factors_list = self._local_kronecker_factors_list

        # Preserve the upstream accounting convention: factor and inverse-factor
        # matrices only. Controller state is intentionally excluded from these
        # legacy diagnostics.
        self._numel_list = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list = tuple(
            numel
            * (get_dtype_size(self._factor_matrix_dtype) + get_dtype_size(block.dtype))
            // 2
            for numel, block in zip(self._numel_list, local_block_list)
        )

    def _base_epsilon_for_factor(self, factor_count: int, factor_idx: int) -> float:
        if not self._use_per_dim_epsilon or factor_count == 1:
            return self._epsilon
        return self._epsilon_left if factor_idx == 0 else self._epsilon_right

    @staticmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]], order_list: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        if isinstance(inv_root_override, Sequence):
            return tuple(
                2 * order if order >= len(inv_root_override) else inv_root_override[order]
                for order in order_list
            )
        return (
            tuple(2 * order for order in order_list)
            if inv_root_override == 0
            else (inv_root_override,) * len(order_list)
        )

    @staticmethod
    def _scalar(tensor: Tensor) -> float:
        return float(tensor.detach().item())

    @staticmethod
    def _increment(tensor: Tensor, value: int = 1) -> None:
        tensor.add_(value)

    @staticmethod
    def _ensure_finite(tensor: Tensor, name: str) -> None:
        if torch.isinf(tensor).any():
            raise ValueError(f"Encountered inf values in {name}")
        if torch.isnan(tensor).any():
            raise ValueError(f"Encountered nan values in {name}")

    def _profile_cuda_device(self) -> Optional[torch.device]:
        """Return the local CUDA device used by this preconditioner, if any.

        Profiling must not synchronize an unrelated/default CUDA device when the
        optimizer itself is executing on CPU (for example on a CUDA-capable
        workstation running a CPU smoke test).
        """
        for factors in self._masked_kronecker_factors_list:
            for factor_matrix in factors.factor_matrices:
                if factor_matrix.is_cuda:
                    return factor_matrix.device
        return None

    def _timed(self, key: str, function):
        if not self._profile_preconditioner:
            return function()
        cuda_device = self._profile_cuda_device()
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        start = time.perf_counter()
        result = function()
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        self._runtime_profile[key] += time.perf_counter() - start
        return result

    def update_preconditioners(
        self, masked_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            for grad, order, kronecker_factors in zip(
                masked_grad_list,
                self._masked_order_list,
                self._masked_kronecker_factors_list,
            ):
                if self._beta2 != 1.0:
                    torch._foreach_mul_(kronecker_factors.factor_matrices, self._beta2)
                outer_product_list = tuple(
                    torch.tensordot(
                        grad,
                        grad,
                        dims=[[*chain(range(k), range(k + 1, order))]] * 2,
                    )
                    for k in range(order)
                )
                torch._foreach_add_(
                    kronecker_factors.factor_matrices,
                    outer_product_list,
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = 1.0 - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            def precondition_masked_grad(
                masked_grad: Tensor,
                inv_factor_matrices: Tuple[Tensor, ...],
            ) -> Tensor:
                for inv_factor_matrix in inv_factor_matrices:
                    masked_grad = torch.tensordot(
                        masked_grad, inv_factor_matrix, [[0], [0]]
                    )
                return masked_grad

            return tuple(
                precondition_masked_grad(masked_grad, factors.inv_factor_matrices)
                for masked_grad, factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list
                )
            )

    def _compute_relative_condition_number(
        self,
        factor_matrix: Tensor,
        prev_eigenvectors: Tensor,
        prev_eigenvalues: Tensor,
        epsilon: float,
    ) -> Tensor:
        """Compute ``||(D+eps I)^-1/2 (Q^T L Q-D) (D+eps I)^-1/2||_F``."""
        rotated = torch.linalg.multi_dot(
            [prev_eigenvectors.T, factor_matrix, prev_eigenvectors]
        )
        shifted = (prev_eigenvalues.clamp_min(0) + epsilon).clamp_min(
            torch.finfo(prev_eigenvalues.dtype).tiny
        )
        denominator = torch.outer(shifted.sqrt(), shifted.sqrt())
        return torch.linalg.norm(
            (rotated - torch.diag(prev_eigenvalues)) / denominator,
            ord="fro",
        )

    def _compute_foam_proxy(
        self,
        factor_matrix: Tensor,
        eigenvectors: Tensor,
        eigenvalues: Tensor,
        epsilon: float,
        root: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        rc = self._compute_relative_condition_number(
            factor_matrix, eigenvectors, eigenvalues, epsilon
        )
        inverse_root_eigenvalues = (
            eigenvalues.clamp_min(0) + epsilon
        ).pow(-self._exponent_multiplier / root)
        alpha = inverse_root_eigenvalues.abs().max() / (
            torch.linalg.vector_norm(inverse_root_eigenvalues) + 1e-25
        )
        # Paper Eq. (3): h = RC * alpha / p.
        proxy = rc * (alpha / root)
        return rc, alpha, proxy

    @staticmethod
    def _compute_diagonalization_residual(
        factor_matrix: Tensor, eigenvectors: Tensor
    ) -> Tuple[Tensor, Tensor]:
        rotated = torch.linalg.multi_dot([eigenvectors.T, factor_matrix, eigenvectors])
        diagonal = torch.diagonal(rotated).clamp_min(0)
        off_diagonal = rotated - torch.diag(diagonal)
        residual = torch.linalg.norm(off_diagonal, ord="fro") / (
            torch.linalg.norm(rotated, ord="fro") + 1e-25
        )
        return residual, diagonal

    def _reuse_basis(
        self,
        inv_factor_matrix: Tensor,
        eigenvectors: Tensor,
        eigenvalues: Tensor,
        epsilon: float,
        root: int,
    ) -> None:
        def compute() -> Tensor:
            shifted = (eigenvalues.clamp_min(0) + epsilon).clamp_min(
                torch.finfo(eigenvalues.dtype).tiny
            )
            eigen_term = shifted.pow(-self._exponent_multiplier / root)
            return (eigenvectors * eigen_term.unsqueeze(0)) @ eigenvectors.T

        computed = self._timed("reuse_seconds", compute)
        self._ensure_finite(computed, "inverse factor matrix")
        inv_factor_matrix.copy_(computed.to(dtype=inv_factor_matrix.dtype))

    def _fresh_eigendecomposition(
        self,
        factor_matrix: Tensor,
        inv_factor_matrix: Tensor,
        is_factor_matrix_diagonal: Tensor,
        factor_matrix_index: str,
        root: int,
        epsilon: float,
        kronecker_factors: ShampooKroneckerFactors,
        factor_idx: int,
        step: int,
    ) -> None:
        if bool(is_factor_matrix_diagonal.item()) and not check_diagonal(factor_matrix):
            is_factor_matrix_diagonal.fill_(False)
            logger.debug("Factor matrix %s is not diagonal.", factor_matrix_index)

        def compute():
            return matrix_inverse_root(
                A=factor_matrix,
                root=root,
                epsilon=epsilon,
                exponent_multiplier=self._exponent_multiplier,
                is_diagonal=is_factor_matrix_diagonal,
                retry_double_precision=self._use_protected_eigh,
            )

        result = self._timed("evd_seconds", compute)
        if isinstance(result, Tensor):
            computed_inv_factor_matrix = result
            used_epsilon: Union[float, Tensor] = epsilon
            damped_eigenvalues = None
            eigenvectors = None
        else:
            computed_inv_factor_matrix, used_epsilon, damped_eigenvalues, eigenvectors = result

        self._ensure_finite(computed_inv_factor_matrix, "inverse factor matrix")
        inv_factor_matrix.copy_(
            computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
        )

        used_epsilon_float = (
            float(used_epsilon.item())
            if isinstance(used_epsilon, Tensor)
            else float(used_epsilon)
        )
        if damped_eigenvalues is None or eigenvectors is None:
            if factor_matrix.numel() == 1:
                raw_eigenvalues = factor_matrix.reshape(-1).clamp_min(0)
                eigenvectors = torch.ones_like(factor_matrix)
            elif bool(is_factor_matrix_diagonal.item()):
                raw_eigenvalues = torch.diagonal(factor_matrix).clamp_min(0)
                eigenvectors = torch.eye(
                    factor_matrix.shape[0],
                    dtype=factor_matrix.dtype,
                    device=factor_matrix.device,
                )
            else:
                # This branch is only expected for a non-EVD root method. The
                # inverse factor remains usable, but no stale basis can be reused.
                kronecker_factors.has_eigendecomposition[factor_idx].fill_(False)
                kronecker_factors.adaptive_epsilons[factor_idx].fill_(
                    used_epsilon_float
                )
                self._increment(kronecker_factors.evd_calls[factor_idx])
                kronecker_factors.last_refresh_step[factor_idx].fill_(step)
                return
        else:
            raw_eigenvalues = (
                damped_eigenvalues - used_epsilon_float
            ).clamp_min(0)

        kronecker_factors.eigenvalues[factor_idx].copy_(
            raw_eigenvalues.to(
                dtype=kronecker_factors.eigenvalues[factor_idx].dtype
            )
        )
        kronecker_factors.eigenvectors[factor_idx].copy_(
            eigenvectors.to(
                dtype=kronecker_factors.eigenvectors[factor_idx].dtype
            )
        )
        kronecker_factors.adaptive_epsilons[factor_idx].fill_(used_epsilon_float)
        kronecker_factors.has_eigendecomposition[factor_idx].fill_(True)
        kronecker_factors.last_refresh_step[factor_idx].fill_(step)
        self._increment(kronecker_factors.evd_calls[factor_idx])

    def _compute_single_root_inverse(
        self,
        factor_matrix: Tensor,
        inv_factor_matrix: Tensor,
        is_factor_matrix_diagonal: Tensor,
        factor_matrix_index: str,
        root: int,
        epsilon_value: float,
        kronecker_factors: ShampooKroneckerFactors,
        factor_idx: int,
        step: int = -1,
    ) -> None:
        bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
        self._ensure_finite(
            bias_corrected_factor_matrix, "bias-corrected factor matrix"
        )
        self._increment(kronecker_factors.check_calls[factor_idx])

        has_basis = bool(
            kronecker_factors.has_eigendecomposition[factor_idx].item()
        )
        if self._preconditioner_update_mode == "stale" or not has_basis:
            self._fresh_eigendecomposition(
                bias_corrected_factor_matrix,
                inv_factor_matrix,
                is_factor_matrix_diagonal,
                factor_matrix_index,
                root,
                epsilon_value,
                kronecker_factors,
                factor_idx,
                step,
            )
            return

        eigenvectors = kronecker_factors.eigenvectors[factor_idx]
        eigenvalues = kronecker_factors.eigenvalues[factor_idx]
        current_epsilon = max(
            epsilon_value,
            self._scalar(kronecker_factors.adaptive_epsilons[factor_idx]),
        )

        try:
            if self._preconditioner_update_mode == "dr_shampoo":
                residual, current_diagonal = self._timed(
                    "proxy_seconds",
                    lambda: self._compute_diagonalization_residual(
                        bias_corrected_factor_matrix, eigenvectors
                    ),
                )
                self._increment(kronecker_factors.residual_calls[factor_idx])
                kronecker_factors.last_proxy[factor_idx].fill_(
                    self._scalar(residual)
                )
                if self._scalar(residual) > self._diagonal_residual_threshold:
                    self._fresh_eigendecomposition(
                        bias_corrected_factor_matrix,
                        inv_factor_matrix,
                        is_factor_matrix_diagonal,
                        factor_matrix_index,
                        root,
                        epsilon_value,
                        kronecker_factors,
                        factor_idx,
                        step,
                    )
                    return
                eigenvalues.copy_(current_diagonal.to(dtype=eigenvalues.dtype))
                kronecker_factors.adaptive_epsilons[factor_idx].fill_(epsilon_value)
                self._reuse_basis(
                    inv_factor_matrix,
                    eigenvectors,
                    eigenvalues,
                    epsilon_value,
                    root,
                )
                self._increment(kronecker_factors.reuse_calls[factor_idx])
                return

            rc, alpha, proxy = self._timed(
                "proxy_seconds",
                lambda: self._compute_foam_proxy(
                    bias_corrected_factor_matrix,
                    eigenvectors,
                    eigenvalues,
                    current_epsilon,
                    root,
                ),
            )
            self._increment(kronecker_factors.proxy_calls[factor_idx])
            proxy_value = self._scalar(proxy)
            kronecker_factors.last_relative_condition[factor_idx].fill_(
                self._scalar(rc)
            )
            kronecker_factors.last_alpha[factor_idx].fill_(self._scalar(alpha))
            kronecker_factors.last_proxy[factor_idx].fill_(proxy_value)

            if self._preconditioner_update_mode == "foam_no_adaptive_epsilon":
                if proxy_value > self._matrix_root_inv_threshold:
                    self._fresh_eigendecomposition(
                        bias_corrected_factor_matrix,
                        inv_factor_matrix,
                        is_factor_matrix_diagonal,
                        factor_matrix_index,
                        root,
                        epsilon_value,
                        kronecker_factors,
                        factor_idx,
                        step,
                    )
                    return
                next_epsilon = epsilon_value
            else:
                next_epsilon = max(
                    epsilon_value,
                    current_epsilon
                    * proxy_value
                    / self._matrix_root_inv_threshold,
                )
                if abs(next_epsilon - current_epsilon) > 0.0:
                    self._increment(
                        kronecker_factors.damping_updates[factor_idx]
                    )

                if self._preconditioner_update_mode == "foam":
                    if next_epsilon > self._max_epsilon:
                        self._increment(
                            kronecker_factors.cap_refreshes[factor_idx]
                        )
                        self._fresh_eigendecomposition(
                            bias_corrected_factor_matrix,
                            inv_factor_matrix,
                            is_factor_matrix_diagonal,
                            factor_matrix_index,
                            root,
                            epsilon_value,
                            kronecker_factors,
                            factor_idx,
                            step,
                        )
                        return
                elif self._preconditioner_update_mode == "foam_no_evd_refresh":
                    next_epsilon = min(next_epsilon, self._max_epsilon)

            kronecker_factors.adaptive_epsilons[factor_idx].fill_(next_epsilon)
            self._reuse_basis(
                inv_factor_matrix,
                eigenvectors,
                eigenvalues,
                next_epsilon,
                root,
            )
            self._increment(kronecker_factors.reuse_calls[factor_idx])
        except Exception as exception:
            if not self._use_protected_eigh:
                raise
            logger.warning(
                "Controller update failed for factor matrix %s with exception %s. "
                "Falling back to a fresh eigendecomposition.",
                factor_matrix_index,
                exception,
            )
            self._fresh_eigendecomposition(
                bias_corrected_factor_matrix,
                inv_factor_matrix,
                is_factor_matrix_diagonal,
                factor_matrix_index,
                root,
                epsilon_value,
                kronecker_factors,
                factor_idx,
                step,
            )

    def compute_root_inverse(self, step: Optional[Tensor] = None) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compute_root_inverse.__name__} ##"
        ):
            step_value = int(step.item()) if isinstance(step, Tensor) else -1
            for kronecker_factors, root in zip(
                self._local_kronecker_factors_list,
                self._local_root_list,
            ):
                factor_count = len(kronecker_factors.factor_matrices)
                for idx, (
                    factor_matrix,
                    inv_factor_matrix,
                    is_factor_matrix_diagonal,
                    factor_matrix_index,
                ) in enumerate(
                    zip(
                        kronecker_factors.factor_matrices,
                        kronecker_factors.inv_factor_matrices,
                        kronecker_factors.is_factor_matrices_diagonal,
                        kronecker_factors.factor_matrix_indices,
                    )
                ):
                    self._compute_single_root_inverse(
                        factor_matrix=factor_matrix,
                        inv_factor_matrix=inv_factor_matrix,
                        is_factor_matrix_diagonal=is_factor_matrix_diagonal,
                        factor_matrix_index=factor_matrix_index,
                        root=root,
                        epsilon_value=self._base_epsilon_for_factor(
                            factor_count, idx
                        ),
                        kronecker_factors=kronecker_factors,
                        factor_idx=idx,
                        step=step_value,
                    )

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_order_list = compress_list(
                self._local_order_list, local_grad_selector
            )
            self._masked_root_list = compress_list(
                self._local_root_list, local_grad_selector
            )
            self._masked_kronecker_factors_list = compress_list(
                self._local_kronecker_factors_list, local_grad_selector
            )

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        relative_errors = []
        relative_residuals = []
        for kronecker_factors, root in zip(
            self._masked_kronecker_factors_list,
            self._masked_root_list,
        ):
            for factor_idx, (factor_matrix, inv_factor_matrix) in enumerate(
                zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.inv_factor_matrices,
                )
            ):
                epsilon = self._scalar(
                    kronecker_factors.adaptive_epsilons[factor_idx]
                )
                relative_error, relative_residual = (
                    compute_matrix_root_inverse_residuals(
                        factor_matrix / self._bias_correction2,
                        inv_factor_matrix,
                        root,
                        epsilon,
                        self._exponent_multiplier,
                    )
                )
                relative_errors.append(relative_error)
                relative_residuals.append(relative_residual)
        return tuple(relative_errors), tuple(relative_residuals)

    def get_diagnostics(self, include_factors: bool = False) -> Dict[str, Any]:
        records = []
        for factors, root in zip(
            self._local_kronecker_factors_list, self._local_root_list
        ):
            for idx, factor in enumerate(factors.factor_matrices):
                if factor.numel() == 0:
                    continue
                records.append(
                    {
                        "factor_index": factors.factor_matrix_indices[idx],
                        "factor_id": factors.factor_matrix_indices[idx],
                        "axis": int(idx),
                        "block_order": int(len(factors.factor_matrices)),
                        "dimension": int(factor.shape[0]),
                        "root": int(root),
                        "epsilon": self._scalar(factors.adaptive_epsilons[idx]),
                        "proxy": self._scalar(factors.last_proxy[idx]),
                        "last_proxy": self._scalar(factors.last_proxy[idx]),
                        "relative_condition": self._scalar(
                            factors.last_relative_condition[idx]
                        ),
                        "alpha": self._scalar(factors.last_alpha[idx]),
                        "check_calls": int(factors.check_calls[idx].item()),
                        "checks": int(factors.check_calls[idx].item()),
                        "evd_calls": int(factors.evd_calls[idx].item()),
                        "proxy_calls": int(factors.proxy_calls[idx].item()),
                        "reuse_calls": int(factors.reuse_calls[idx].item()),
                        "damping_updates": int(
                            factors.damping_updates[idx].item()
                        ),
                        "cap_refreshes": int(
                            factors.cap_refreshes[idx].item()
                        ),
                        "residual_calls": int(
                            factors.residual_calls[idx].item()
                        ),
                        "last_refresh_step": int(
                            factors.last_refresh_step[idx].item()
                        ),
                    }
                )

        def total(key: str) -> int:
            return sum(int(record[key]) for record in records)

        epsilons = [record["epsilon"] for record in records]
        proxies = [
            record["proxy"]
            for record in records
            if math.isfinite(record["proxy"])
        ]
        dimensions: Dict[int, int] = {}
        for record in records:
            dim = int(record["dimension"])
            dimensions[dim] = dimensions.get(dim, 0) + 1

        def summarize(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    "min": float("nan"),
                    "median": float("nan"),
                    "mean": float("nan"),
                    "max": float("nan"),
                }
            ordered = sorted(values)
            middle = len(ordered) // 2
            median = (
                ordered[middle]
                if len(ordered) % 2
                else 0.5 * (ordered[middle - 1] + ordered[middle])
            )
            return {
                "min": min(values),
                "median": median,
                "mean": sum(values) / len(values),
                "max": max(values),
            }

        check_calls = total("check_calls")
        diagnostics: Dict[str, Any] = {
            "mode": self._preconditioner_update_mode,
            "factor_count": len(records),
            "factor_dimensions": {str(k): v for k, v in sorted(dimensions.items())},
            "check_calls": check_calls,
            "evd_calls": total("evd_calls"),
            "proxy_calls": total("proxy_calls"),
            "reuse_calls": total("reuse_calls"),
            "damping_updates": total("damping_updates"),
            "cap_refreshes": total("cap_refreshes"),
            "residual_calls": total("residual_calls"),
            "evd_operation_rate": (
                total("evd_calls") / check_calls if check_calls else float("nan")
            ),
            "epsilon": summarize(epsilons),
            "proxy": summarize(proxies),
            "runtime_profile_seconds": dict(self._runtime_profile),
        }
        if include_factors:
            diagnostics["factors"] = records
        return diagnostics

    def get_runtime_profile(self, reset: bool = False) -> Dict[str, float]:
        profile = dict(self._runtime_profile)
        if reset:
            for key in self._runtime_profile:
                self._runtime_profile[key] = 0.0
        return profile

    def export_factor_snapshots(self) -> List[Dict[str, Any]]:
        snapshots = []
        for factors, root in zip(
            self._local_kronecker_factors_list, self._local_root_list
        ):
            for idx, factor in enumerate(factors.factor_matrices):
                if factor.numel() == 0:
                    continue
                snapshots.append(
                    {
                        "factor_index": factors.factor_matrix_indices[idx],
                        "factor_matrix": (
                            factor / self._bias_correction2
                        ).detach().cpu(),
                        "eigenvalues": factors.eigenvalues[idx].detach().cpu(),
                        "eigenvectors": factors.eigenvectors[idx].detach().cpu(),
                        "epsilon": self._scalar(
                            factors.adaptive_epsilons[idx]
                        ),
                        "has_eigendecomposition": bool(
                            factors.has_eigendecomposition[idx].item()
                        ),
                        "root": int(root),
                    }
                )
        return snapshots

