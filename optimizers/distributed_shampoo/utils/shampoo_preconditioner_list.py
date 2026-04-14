import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from itertools import chain
from typing import Any, DefaultDict, Optional, Sequence, List, Tuple, Union, Dict

import torch
import torch.distributed as dist
from .shampoo_block_info import BlockInfo
from .shampoo_utils import (
    compress_list,
    get_dtype_size,
)

from ...matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
)
from ...optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

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
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: RWSAdaGradPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
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
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

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
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Numel Breakdown: {self._numel_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Bytes Breakdown: {self._num_bytes_list}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Total Elements: {sum(self._numel_list)}"
        )
        logger.info(
            f"Rank {dist.get_rank()}: AdaGradPreconditionerList Total Bytes: {sum(self._num_bytes_list)}"
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
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

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
    """Shampoo Kronecker Factors."""

    factor_matrices: Tuple[Tensor, ...]
    inv_factor_matrices: Tuple[Tensor, ...]
    factor_matrix_indices: Tuple[str, ...]
    is_factor_matrices_diagonal: Tuple[Tensor, ...] = field(init=False)
    eigenvalues: List[Optional[Tensor]] = field(default_factory=list)
    eigenvectors: List[Optional[Tensor]] = field(default_factory=list)
    # DryShampoo: Persist per-block epsilon state
    adaptive_epsilons: List[Optional[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()
        assert (
            len(self.factor_matrices)
            == len(self.inv_factor_matrices)
            == len(self.factor_matrix_indices)
        )
        self.is_factor_matrices_diagonal = tuple(
            torch.tensor(True) for _ in self.factor_matrices
        )
        self.eigenvalues = [None] * len(self.factor_matrices)
        self.eigenvectors = [None] * len(self.factor_matrices)
        self.adaptive_epsilons = [None] * len(self.factor_matrices)


class ShampooPreconditionerList(PreconditionerList):
    """Shampoo preconditioners for list of parameters."""

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        epsilon_left : Optional[float] = None,
        epsilon_right : Optional[float] = None,
        # DryShampoo Hyperparameters
        matrix_root_inv_threshold: float = 0.0, # tau
        max_epsilon: float = 1.0, # epsilon_max
        
        inv_root_override: Union[int, Tuple[int, ...]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
        use_protected_eigh: bool = True,
        # Ignored legacy args
        use_adaptive_epsilon: bool = False,
        condition_thresholds: Optional[Dict[float, float]] = None,
        is_default_config: bool = False,
        use_trace_correction: bool = False,
    ) -> None:
        super().__init__(block_list)

        self._beta2 = beta2
        self._epsilon = epsilon
        
        # DryShampoo Configuration
        self._matrix_root_inv_threshold = matrix_root_inv_threshold
        self._max_epsilon = max_epsilon
        
        # Asymmetric Configuration Support (for initialization)
        self._epsilon_left = epsilon_left if epsilon_left is not None else epsilon
        self._epsilon_right = epsilon_right if epsilon_right is not None else epsilon
        self._use_per_dim_epsilon = (epsilon_left is not None or epsilon_right is not None)
        
        self._inv_root_override = inv_root_override
        self._exponent_multiplier = exponent_multiplier
        self._factor_matrix_dtype = factor_matrix_dtype
        self._use_bias_correction = use_bias_correction
        self._use_protected_eigh = use_protected_eigh
        self._bias_correction2: Tensor = torch.tensor(1.0)
        
        kronecker_factors_list = []
        epsilon_per_dim_list = [] if self._use_per_dim_epsilon else None
    
        for block, block_info, dims in zip(
            block_list, block_info_list, self._dims_list, 
        ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Determine initial epsilon for each dimension (Asymmetric setup)
            current_block_epsilons = []
            if self._use_per_dim_epsilon:
                block_epsilon_per_dim = []
                for dim_idx, dim in enumerate(dims):
                    if len(dims) == 1:
                        eps = self._epsilon
                    elif dim_idx == 0:
                        eps = self._epsilon_left
                    else:
                        eps = self._epsilon_right
                    block_epsilon_per_dim.append(eps)
                    current_block_epsilons.append(eps)
                epsilon_per_dim_list.append(tuple(block_epsilon_per_dim))
            else:
                current_block_epsilons = [self._epsilon] * len(dims)

            factor_matrices = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim),
                    self._factor_matrix_dtype,
                    block_info.param.device,
                )
                for dim in dims
            )
            inv_factor_matrices = tuple(
                block_info.allocate_zeros_tensor(
                    (dim, dim),
                    block.dtype,
                    block_info.param.device,
                )
                for dim in dims
            )

            preconditioner_index = str(param_index) + "." + str(block_index)
            factor_matrix_indices = tuple(
                preconditioner_index + "." + str(k) for k in range(len(dims))
            )
            
            # Instantiate ShampooKroneckerFactors
            kf_instance = ShampooKroneckerFactors(
                factor_matrices=tuple(
                    block_info.get_tensor(t) for t in factor_matrices
                ),
                inv_factor_matrices=tuple(
                    block_info.get_tensor(t) for t in inv_factor_matrices
                ),
                factor_matrix_indices=factor_matrix_indices,
            )
            # Initialize adaptive epsilons in the state
            kf_instance.adaptive_epsilons = current_block_epsilons
            
            block_state[SHAMPOO] = ShampooKroneckerFactors(
                factor_matrices=factor_matrices,
                inv_factor_matrices=inv_factor_matrices,
                factor_matrix_indices=factor_matrix_indices,
            )
            kronecker_factors_list.append(kf_instance)

            logger.info(
                f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                f"with epsilon: {current_block_epsilons} "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        local_block_list = compress_list(block_list, distributor_selector)
        self._local_kronecker_factors_list: Tuple[
            ShampooKroneckerFactors, ...
        ] = compress_list(kronecker_factors_list, distributor_selector)
        
        self._local_order_list: Tuple[int, ...] = tuple(
            block.dim() for block in local_block_list
        )
        self._local_root_list: Tuple[int, ...] = self._get_inverse_roots_from_override(
            self._inv_root_override, self._local_order_list
        )

        self._masked_order_list: Tuple[int, ...] = self._local_order_list
        self._masked_root_list: Tuple[int, ...] = self._local_root_list
        self._masked_kronecker_factors_list: Tuple[
            ShampooKroneckerFactors, ...
        ] = self._local_kronecker_factors_list
        
        self._numel_list: Tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            numel
            * (get_dtype_size(self._factor_matrix_dtype) + get_dtype_size(block.dtype))
            // 2
            for numel, block in zip(self._numel_list, local_block_list)
        )

    def _compute_relative_condition_number(
            self,
            factor_matrix : Tensor,
            prev_eigenvectors : Tensor,
            prev_eigenvalues : Tensor,
            epsilon : float
        ) -> Tensor:
            """
            Equation (2):
            RC(eps_t) = || (L_tilde_ij - delta_ij * d_i) / (sqrt(d_i + eps) * sqrt(d_j + eps)) ||_F
            where L_tilde = Q^T * L_t * Q
            """
            # L_tilde = Q^T * L_t * Q (Whitened Perturbation Matrix)
            # Note: factor_matrix here is usually bias_corrected_factor_matrix (L_t)
            L_tilde = torch.linalg.multi_dot([prev_eigenvectors.T, factor_matrix, prev_eigenvectors])
            
            # d_term = sqrt(d + epsilon)
            d_term = torch.sqrt(prev_eigenvalues + epsilon)
            
            # Denominator matrix D_ij = sqrt(d_i + eps) * sqrt(d_j + eps)
            # Using outer product for efficient computation broadcasting
            denominator = torch.outer(d_term, d_term)
            
            # Numerator E_ij = L_tilde_ij - delta_ij * d_i (Diagonal elements subtracted)
            # This is equivalent to L_tilde - diag(prev_eigenvalues)
            numerator = L_tilde - torch.diag(prev_eigenvalues)
            
            # Element-wise division
            scaled_diff = numerator / denominator
            
            # RC_t = ||E||_F
            rc_t = torch.linalg.norm(scaled_diff, ord = 'fro')
            return rc_t

    @staticmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]], order_list: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        if isinstance(inv_root_override, Sequence):
            return tuple(
                2 * order
                if order >= len(inv_root_override)
                else inv_root_override[order]
                for order in order_list
            )
        else:
            return (
                tuple(2 * order for order in order_list)
                if inv_root_override == 0
                else (inv_root_override,) * len(order_list)
            )

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
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

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
                precondition_masked_grad(
                    masked_grad=masked_grad,
                    inv_factor_matrices=kronecker_factors.inv_factor_matrices,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, 
                )
            )

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
    ) -> None:
        """Compute root inverse for a single factor matrix using DryShampoo adaptive logic."""
        
        # Current factor matrix (bias corrected L_t)
        bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
        
        # Current State
        prev_Q = kronecker_factors.eigenvectors[factor_idx]
        prev_D = kronecker_factors.eigenvalues[factor_idx]
        
        # Use stored adaptive epsilon if available, else base epsilon
        current_epsilon = kronecker_factors.adaptive_epsilons[factor_idx]
        if current_epsilon is None:
            current_epsilon = epsilon_value

        should_recompute_eigen = True
        
        # Algorithm 3: Check conditions if we have stale eigenbases
        if prev_Q is not None and prev_D is not None and self._matrix_root_inv_threshold > 0.0:
            try:
                # Line 6: Calculate RC(epsilon_{t-1})
                rc_t = self._compute_relative_condition_number(
                    bias_corrected_factor_matrix,
                    prev_Q,
                    prev_D,
                    current_epsilon
                )

                inv_root_exponent = -self._exponent_multiplier / root
                h_eigenvalues = (prev_D + current_epsilon).pow(inv_root_exponent)

                spectral_norm = h_eigenvalues.abs().max()
                frobenius_norm = torch.norm(h_eigenvalues, p = 2)

                alpha = spectral_norm / (frobenius_norm + 1e-25)
                
                # Line 7: Update Damping factor
                # epsilon_t = epsilon_{t-1} * (RC / tau) * alpha
                # [FIX] vit.py와 동일하게 alpha를 포함하여 계산
                new_epsilon = current_epsilon * ((rc_t * alpha) / self._matrix_root_inv_threshold)
                
                # Line 8: if RC * alpha >= tau
                if (rc_t * alpha) >= self._matrix_root_inv_threshold:
                    # Line 9: if epsilon_t < epsilon_max
                    if new_epsilon < self._max_epsilon:
                        # Line 10: Apply updated damping to previous eigenfactors (Fast Update)
                        # Keep old Q, D. Just update epsilon.
                        current_epsilon = float(new_epsilon)
                        should_recompute_eigen = False
                        
                        # Construct H = Q * (D + eps*I)^(-1/p) * Q^T
                        alpha_pow = -self._exponent_multiplier / root
                        # prev_D는 이제 Raw Eigenvalues이므로 epsilon을 여기서 더해줍니다.
                        eig_term = (prev_D + current_epsilon).pow(alpha_pow)
                        computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                        
                        # Update state
                        computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                        inv_factor_matrix.copy_(computed_inv_factor_matrix)
                        kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"{factor_matrix_index}: Fast update. New eps={current_epsilon:.2e}, RC={rc_t:.4f}")
                    else:
                        # Line 11-13: Recompute Eigendecomposition (Slow Update)
                        # Reset epsilon to BASE epsilon
                        current_epsilon = epsilon_value
                        should_recompute_eigen = True 
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"{factor_matrix_index}: Max epsilon reached. Resetting and recomputing.")
                
                else:
                    # Line 15: else (RC < tau)
                    # Line 16: Apply updated damping factor (Fast Update)
                    current_epsilon = float(new_epsilon)
                    should_recompute_eigen = False
                    
                    alpha_pow = -self._exponent_multiplier / root
                    # prev_D는 Raw Eigenvalues이므로 epsilon을 여기서 더해줍니다.
                    eig_term = (prev_D + current_epsilon).pow(alpha_pow)
                    computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                    
                    computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                    inv_factor_matrix.copy_(computed_inv_factor_matrix)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                    
            except Exception as e:
                logger.warning(f"Failed to compute RC for {factor_matrix_index}: {e}. Forcing recompute.")
                should_recompute_eigen = True

        if should_recompute_eigen:
            # Perform standard Eigendecomposition Update
            if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                is_factor_matrix_diagonal.copy_(torch.tensor(False))

            try:
                # Use matrix_inverse_root from matrix_functions
                # [FIX] used_epsilon을 반환받습니다.
                result = matrix_inverse_root(
                    A=bias_corrected_factor_matrix,
                    root=root,
                    epsilon=current_epsilon,
                    exponent_multiplier=self._exponent_multiplier,
                    is_diagonal=is_factor_matrix_diagonal,
                    retry_double_precision=self._use_protected_eigh,
                )
                
                computed_inv_factor_matrix, used_epsilon, L, Q = result
                
                if L is not None and Q is not None:
                    # [FIX] L은 epsilon이 더해진 상태이므로, 저장할 때는 epsilon을 뺍니다.
                    # 이를 통해 다음 step에서 (prev_D + current_epsilon) 계산 시 중복 더해짐을 방지합니다.
                    raw_eigenvalues = L - used_epsilon
                    kronecker_factors.eigenvalues[factor_idx] = raw_eigenvalues.to(dtype=factor_matrix.dtype)
                    kronecker_factors.eigenvectors[factor_idx] = Q.to(dtype=factor_matrix.dtype)
                    # Store the epsilon used for this decomposition
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                
                computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                inv_factor_matrix.copy_(computed_inv_factor_matrix)

            except Exception as exception:
                if (
                    not self._use_protected_eigh
                    or "Encountered nan or inf values in inverse factor matrix"
                    in str(exception)
                ):
                    raise exception
                else:
                    logger.warning(
                        f"Matrix inverse root computation failed for factor matrix {factor_matrix_index} "
                        f"with exception {exception}. Using previous inv_factor_matrix and continuing..."
                    )

    def compute_root_inverse(self) -> None:
        """
        Call _compute_single_root_inverse for all blocks.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compute_root_inverse.__name__} ##"
        ):
            for kronecker_factors, root in zip(
                self._local_kronecker_factors_list,
                self._local_root_list,
            ):
                for idx, (
                    factor_matrix,
                    inv_factor_matrix,
                    is_factor_matrix_diagonal,
                    factor_matrix_index,
                ) in enumerate(zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.inv_factor_matrices,
                    kronecker_factors.is_factor_matrices_diagonal,
                    kronecker_factors.factor_matrix_indices,
                )):
                    # Determine base epsilon for this block/dimension
                    base_epsilon = self._epsilon
                    if self._use_per_dim_epsilon and len(kronecker_factors.factor_matrices) > 1:
                        base_epsilon = self._epsilon_left if idx == 0 else self._epsilon_right
                    
                    self._compute_single_root_inverse(
                        factor_matrix=factor_matrix,
                        inv_factor_matrix=inv_factor_matrix,
                        is_factor_matrix_diagonal=is_factor_matrix_diagonal,
                        factor_matrix_index=factor_matrix_index,
                        root=root,
                        epsilon_value=base_epsilon,
                        kronecker_factors=kronecker_factors,
                        factor_idx=idx
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
            self._masked_kronecker_factors_list: Tuple[
                ShampooKroneckerFactors, ...
            ] = compress_list(self._local_kronecker_factors_list, local_grad_selector)

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        relative_errors = []
        relative_residuals = []

        for kronecker_factors, root in zip(
            self._masked_kronecker_factors_list,
            self._masked_root_list,
        ):
            for factor_matrix, inv_factor_matrix in zip(
                kronecker_factors.factor_matrices,
                kronecker_factors.inv_factor_matrices,
            ):
                bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
                (
                    relative_error,
                    relative_residual,
                ) = compute_matrix_root_inverse_residuals(
                    bias_corrected_factor_matrix,
                    inv_factor_matrix,
                    root,
                    self._epsilon,
                    self._exponent_multiplier,
                )
                relative_errors.append(relative_error)
                relative_residuals.append(relative_residual)

        return (
            tuple(relative_errors),
            tuple(relative_residuals),
        )
