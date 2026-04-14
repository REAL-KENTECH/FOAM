"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from .shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    BETAS,
    DDPShampooConfig,
    DistributedConfig,  # <--- [FIX] 누락된 DistributedConfig 추가
    DISTRIBUTOR,
    EPSILON,
    EPSILON_LEFT,
    EPSILON_RIGHT,
    EXPONENT_MULTIPLIER,
    FILTERED_GRAD,
    FILTERED_GRAD_LIST,
    FSDPShampooConfig,
    GRAFTING_CONFIG,
    GRAFTING_PRECONDITIONER_LIST,
    GraftingConfig,
    INV_ROOT_OVERRIDE,
    LR,
    MASKED_BLOCKED_GRADS,
    MASKED_BLOCKED_PARAMS,
    MASKED_FILTERED_GRAD_LIST,
    MASKED_MOMENTUM_LIST,
    MAX_PRECONDITIONER_DIM,
    MATRIX_ROOT_INV_THRESHOLD,
    MAX_EPSILON,
    MOMENTUM,
    MOMENTUM_LIST,
    PARAMS,
    PRECONDITION_FREQUENCY,
    PRECONDITIONER_DTYPE,
    PREVIOUS_GRAD_SELECTOR,
    RMSpropGraftingConfig,
    RWSAdaGradGraftingConfig,
    SGDGraftingConfig,
    SHAMPOO_PRECONDITIONER_LIST,
    START_PRECONDITIONING_STEP,
    STEP,
    USE_BIAS_CORRECTION,
    USE_DECOUPLED_WEIGHT_DECAY,
    USE_EMA_MOMENTUM,
    USE_MERGE_DIMS,
    USE_NADAM,
    USE_NESTEROV,
    USE_NORMALIZED_GRAFTING,
    WEIGHT_DECAY,
)

from .utils.shampoo_checkpoint_utils import (
    extract_state_dict_content,
    flatten,
    unflatten,
    update_param_state_dict_object,
)
from .utils.shampoo_ddp_distributor import (
    DDPDistributor,
)
from .utils.shampoo_distributor import Distributor
from .utils.shampoo_fsdp_distributor import (
    FSDPDistributor,
)

from .utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    RWSAdagradPreconditionerList,
    SGDPreconditionerList,
    ShampooPreconditionerList,
)
from .utils.shampoo_utils import compress_list

logger: logging.Logger = logging.getLogger(__name__)

EPSILON_DEFAULT = 1e-16


class DistributedShampoo(torch.optim.Optimizer):
    """Implements distributed Shampoo algorithm (DryShampoo)."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        epsilon: float = 1e-12,
        epsilon_left: Optional[float] = None,
        epsilon_right: Optional[float] = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        inv_root_override: Union[int, Sequence[int]] = 0,
        exponent_multiplier: float = 1.0,
        use_nadam: bool = False,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        grafting_config: Optional[GraftingConfig] = None,
        use_normalized_grafting: bool = False,
        use_ema_momentum: bool = True,
        use_merge_dims: bool = True,
        use_pytorch_compile: bool = False,
        distributed_config: Optional[DistributedConfig] = None,
        preconditioner_dtype: torch.dtype = torch.float32,
        use_protected_eigh: bool = True,
        track_root_inv_residuals: bool = False,
        # DryShampoo Specific Arguments
        matrix_root_inv_threshold: float = 0.0,  # tau
        max_epsilon: float = 1.0, # epsilon_max
        use_adaptive_epsilon: bool = False, # Compatibility argument
        condition_thresholds: Optional[Dict[float, float]] = None, # Compatibility argument
    ) -> None:
        # Hyperparameter checks.
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be >= 0.0.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}. Must be in [0.0, 1.0)."
            )
        if not 0.0 < betas[1] <= 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}. Must be in (0.0, 1.0]."
            )
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Must be > 0.0.")
        
        actual_epsilon_left = epsilon_left if epsilon_left is not None else epsilon
        actual_epsilon_right = epsilon_right if epsilon_right is not None else epsilon
        
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Invalid momentum parameter: {momentum}. Must be [0.0, 1.0)."
            )
        if not weight_decay >= 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}. Must be >= 0.0."
            )
        if not max_preconditioner_dim >= 1:
            raise ValueError(
                f"Invalid max preconditioner dimension: {max_preconditioner_dim}. Must be >= 1."
            )
        if not precondition_frequency >= 1:
            raise ValueError(
                f"Invalid precondition frequency: {precondition_frequency}. Must be >= 1."
            )
        if not start_preconditioning_step >= -1:
            raise ValueError(
                f"Invalid start_preconditioning_step: {start_preconditioning_step}. Must be >= -1."
            )
        if isinstance(inv_root_override, Sequence):
            if not all(e >= 0 for e in inv_root_override):
                raise ValueError(
                    f"Invalid exponent override list: {inv_root_override}. All values must be >= 0."
                )
        else:
            if not inv_root_override >= 0:
                raise ValueError(
                    f"Invalid exponent override: {inv_root_override}. Must be >= 0."
                )
        if not 0.0 <= matrix_root_inv_threshold < 1.0:
            raise ValueError(
                f"Invalid matrix_root_inv_threshold : {matrix_root_inv_threshold}."
                "Must be in [0.0, 1.0)."
            )
        
        if track_root_inv_residuals:
            logger.setLevel(logging.DEBUG)

        # Provide warning/error for start_preconditioning_step.
        if start_preconditioning_step == -1:
            start_preconditioning_step = precondition_frequency
            logger.warning(
                "start_preconditioning_step set to -1. Setting start_preconditioning_step equal to "
                f"precondition frequency {precondition_frequency} by default."
            )
        if start_preconditioning_step < precondition_frequency:
            raise ValueError(
                f"Invalid start_preconditioning_step value: {start_preconditioning_step}. Must be >= {precondition_frequency=}."
            )

        if use_nadam and betas[0] == 0.0:
            logger.warning(
                "NAdam flag is enabled but beta1 parameter is zero! "
                "Continuing without using NAdam..."
            )

        if use_nesterov and momentum == 0.0:
            logger.warning(
                "Nesterov flag is enabled but momentum parameter is zero! "
                "Continuing without using momentum or Nesterov acceleration..."
            )

        if use_pytorch_compile and not torch.cuda.is_available():
            raise ValueError(
                "Backend does NOT support Pytorch 2.0 compile. Switch to use_pytorch_compile=False."
            )

        super().__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                EPSILON: epsilon,
                EPSILON_LEFT: actual_epsilon_left,
                EPSILON_RIGHT: actual_epsilon_right,
                MATRIX_ROOT_INV_THRESHOLD: matrix_root_inv_threshold,
                MAX_EPSILON: max_epsilon,
                MOMENTUM: momentum,
                WEIGHT_DECAY: weight_decay,
                MAX_PRECONDITIONER_DIM: max_preconditioner_dim,
                PRECONDITION_FREQUENCY: precondition_frequency,
                START_PRECONDITIONING_STEP: start_preconditioning_step,
                INV_ROOT_OVERRIDE: inv_root_override,
                EXPONENT_MULTIPLIER: exponent_multiplier,
                USE_NADAM: use_nadam,
                USE_NESTEROV: use_nesterov,
                USE_BIAS_CORRECTION: use_bias_correction,
                USE_DECOUPLED_WEIGHT_DECAY: use_decoupled_weight_decay,
                GRAFTING_CONFIG: grafting_config,
                USE_NORMALIZED_GRAFTING: use_normalized_grafting,
                USE_EMA_MOMENTUM: use_ema_momentum,
                USE_MERGE_DIMS: use_merge_dims,
                PRECONDITIONER_DTYPE: preconditioner_dtype,
            },
        )

        # Initialize non-group-related fields.
        self._distributed_config = distributed_config
        self._use_protected_eigh = use_protected_eigh
        self._track_root_inv_residuals = track_root_inv_residuals
        self._use_pytorch_compile = use_pytorch_compile

        # Initialize dictionary containing lists of .
        self._per_group_state_lists: List[Dict[str, Any]] = [
            {} for _ in self.param_groups
        ]

        # Block parameters and instantiate optimizer states.
        self._instantiate_distributor()
        
        if start_preconditioning_step < torch.inf:
            self._instantiate_shampoo_preconditioner_list()
        self._instantiate_grafting()
        self._instantiate_steps()
        self._instantiate_momentum()
        self._instantiate_filtered_grads()
        self._instantiate_device()

        # Use PT2 to compile the step function for each parameter group
        self._per_group_step: Callable = (
            torch.compile(self._per_group_step_impl, backend="inductor")
            if self._use_pytorch_compile
            else self._per_group_step_impl
        )

    @torch.no_grad()
    def _instantiate_distributor(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups,
        ):
            # Instantiate distributors for each group.
            if self._distributed_config is None:
                state_lists[DISTRIBUTOR] = Distributor(
                    param_group=group,
                )
            elif isinstance(self._distributed_config, DDPShampooConfig):
                state_lists[DISTRIBUTOR] = DDPDistributor(
                    param_group=group,
                    distributed_config=self._distributed_config,
                )
            elif isinstance(self._distributed_config, FSDPShampooConfig):
                state_lists[DISTRIBUTOR] = FSDPDistributor(
                    param_group=group,
                    distributed_config=self._distributed_config,
                )
            else:
                raise NotImplementedError(f"{self._distributed_config=} not supported!")

            # Compile blocked parameters and block-to-parameter metadata into group lists.
            state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
                DISTRIBUTOR
            ].local_blocked_params
            # First PREVIOUS_GRAD_SELECTOR is set to None.
            state_lists[PREVIOUS_GRAD_SELECTOR] = None

    @torch.no_grad()
    def _instantiate_shampoo_preconditioner_list(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups,
        ):
            state_lists[SHAMPOO_PRECONDITIONER_LIST] = ShampooPreconditionerList(
                block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                state=self.state,
                block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                beta2=group[BETAS][1],
                epsilon=group[EPSILON],
                epsilon_left=group[EPSILON_LEFT],
                epsilon_right=group[EPSILON_RIGHT],
                # DryShampoo params
                matrix_root_inv_threshold=group[MATRIX_ROOT_INV_THRESHOLD],
                max_epsilon=group[MAX_EPSILON],
                
                inv_root_override=group[INV_ROOT_OVERRIDE],
                exponent_multiplier=group[EXPONENT_MULTIPLIER],
                use_bias_correction=group[USE_BIAS_CORRECTION],
                factor_matrix_dtype=group[PRECONDITIONER_DTYPE],
                use_protected_eigh=self._use_protected_eigh,
            )

    @torch.no_grad()
    def _instantiate_grafting(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups,
        ):
            if group[GRAFTING_CONFIG] is None:
                state_lists[GRAFTING_PRECONDITIONER_LIST] = None
            elif isinstance(group[GRAFTING_CONFIG], SGDGraftingConfig):
                state_lists[GRAFTING_PRECONDITIONER_LIST] = SGDPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                )
            elif isinstance(
                group[GRAFTING_CONFIG],
                (AdaGradGraftingConfig, RMSpropGraftingConfig, AdamGraftingConfig),
            ):
                state_lists[GRAFTING_PRECONDITIONER_LIST] = AdagradPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                    distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                    beta2=1.0
                    if isinstance(group[GRAFTING_CONFIG], AdaGradGraftingConfig)
                    else group[GRAFTING_CONFIG].beta2,
                    epsilon=group[GRAFTING_CONFIG].epsilon,
                    use_bias_correction=isinstance(
                        group[GRAFTING_CONFIG], AdamGraftingConfig
                    ),
                )
            elif isinstance(group[GRAFTING_CONFIG], RWSAdaGradGraftingConfig):
                state_lists[GRAFTING_PRECONDITIONER_LIST] = RWSAdagradPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                    distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                    beta2=group[GRAFTING_CONFIG].beta2,
                    epsilon=group[GRAFTING_CONFIG].epsilon,
                    use_bias_correction=group[GRAFTING_CONFIG].use_bias_correction,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported grafting config: {group[GRAFTING_CONFIG]=}."
                )

    @torch.no_grad()
    def _instantiate_steps(self) -> None:
        for state_lists in self._per_group_state_lists:
            assert (
                len(state_lists[DISTRIBUTOR].global_block_info_list) > 0
            ), "There is no params in your param_group."
            # NOTE: We instantiate a single step tensor on CPU for each group in order
            #       to track the number of steps taken by all parameters within the group.
            #       Instantiating on CPU avoids GPU synchronization.
            state_lists[STEP] = torch.tensor(0, dtype=torch.int64, device="cpu")

            # In order to ensure that the step counter is checkpointed correctly, we store it
            # as a tensor (which is replicated across all devices) under the first parameter's state.
            block_info = state_lists[DISTRIBUTOR].global_block_info_list[0]
            self.state[block_info.param][STEP] = state_lists[STEP]

    @torch.no_grad()
    def _instantiate_momentum(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups,
        ):
            if group[MOMENTUM] == 0.0:
                continue

            # Construct global momentum list.
            global_momentum_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].global_blocked_params,
                state_lists[DISTRIBUTOR].global_block_info_list,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[
                    block_info.param
                ], f"{block_index=} not found in {self.state[block_info.param]=}."
                block_state = self.state[block_info.param][block_index]

                block_state[MOMENTUM] = block_info.allocate_zeros_tensor(
                    shape=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                global_momentum_list.append(
                    block_info.get_tensor(block_state[MOMENTUM])
                )

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[MOMENTUM_LIST] = compress_list(
                global_momentum_list,
                state_lists[DISTRIBUTOR].distributor_selector,
            )
            # Here, we set masked momentum list to momentum list because we assume
            # all parameters are active.
            state_lists[MASKED_MOMENTUM_LIST] = state_lists[MOMENTUM_LIST]

    @torch.no_grad()
    def _instantiate_filtered_grads(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups,
        ):
            if group[BETAS][0] == 0.0:
                continue

            # Construct global filtered gradient list.
            global_filtered_grad_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].global_blocked_params,
                state_lists[DISTRIBUTOR].global_block_info_list,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[
                    block_info.param
                ], f"{block_index=} not found in {self.state[block_info.param]=}."
                block_state = self.state[block_info.param][block_index]

                block_state[FILTERED_GRAD] = block_info.allocate_zeros_tensor(
                    shape=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                global_filtered_grad_list.append(
                    block_info.get_tensor(block_state[FILTERED_GRAD])
                )

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[FILTERED_GRAD_LIST] = compress_list(
                global_filtered_grad_list,
                state_lists[DISTRIBUTOR].distributor_selector,
            )
            # Here, we set masked filtered grad list to filtered grad list because we assume
            # all parameters are active.
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[FILTERED_GRAD_LIST]

    @torch.no_grad()
    def _instantiate_device(self) -> None:
        # NOTE: Assume all parameter groups consistently exist on the same rank
        self._device = self._per_group_state_lists[0][MASKED_BLOCKED_PARAMS][0].device

    @staticmethod
    @torch.no_grad()
    def _mask_state_lists(state_lists: Dict[str, Any], group: Dict[str, Any]) -> None:
        if (
            state_lists[DISTRIBUTOR].local_grad_selector
            == state_lists[PREVIOUS_GRAD_SELECTOR]
        ):
            return

        # Updates masked state lists if previous block selector disagrees with current selector.
        # State list compression is necessary in order to avoid handling gradients with None.
        state_lists[PREVIOUS_GRAD_SELECTOR] = state_lists[
            DISTRIBUTOR
        ].local_grad_selector
        state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
            DISTRIBUTOR
        ].local_masked_blocked_params
        if SHAMPOO_PRECONDITIONER_LIST in state_lists:
            state_lists[SHAMPOO_PRECONDITIONER_LIST].compress_preconditioner_list(
                local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[GRAFTING_CONFIG] is not None:
            state_lists[GRAFTING_PRECONDITIONER_LIST].compress_preconditioner_list(
                local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[BETAS][0] != 0.0:
            state_lists[MASKED_FILTERED_GRAD_LIST] = compress_list(
                state_lists[FILTERED_GRAD_LIST],
                state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[MOMENTUM] != 0.0:
            state_lists[MASKED_MOMENTUM_LIST] = compress_list(
                state_lists[MOMENTUM_LIST],
                state_lists[DISTRIBUTOR].local_grad_selector,
            )

    @torch.no_grad()
    @torch.compiler.disable
    def _compute_root_inverse(
        self, state_lists: Dict[str, Any], compute_root_inverse: bool
    ) -> None:
        if compute_root_inverse:
            state_lists[SHAMPOO_PRECONDITIONER_LIST].compute_root_inverse()

    @torch.no_grad()
    def _per_group_step_impl(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        weight_decay: float,
        momentum_param: float,
        grafting_config_not_none: bool,
        compute_root_inverse: bool,
        use_decoupled_weight_decay: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        use_nadam: bool,
        use_nesterov: bool,
        use_normalized_grafting: bool,
        use_ema_momentum: bool,
    ) -> None:
        # Incorporate L2-regularization or decoupled weight decay.
        if weight_decay != 0.0 and not use_decoupled_weight_decay:
            torch._foreach_add_(
                state_lists[MASKED_BLOCKED_GRADS],
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

        # Update Shampoo and grafting preconditioners.
        if SHAMPOO_PRECONDITIONER_LIST in state_lists:
            state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
                masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )
        if grafting_config_not_none:
            state_lists[GRAFTING_PRECONDITIONER_LIST].update_preconditioners(
                masked_grad_list=tuple(
                    blocked_grad / (grad_norm + EPSILON_DEFAULT) for blocked_grad, grad_norm in zip(
                        state_lists[MASKED_BLOCKED_GRADS],
                        torch._foreach_norm(state_lists[MASKED_BLOCKED_GRADS]),
                    )
                ) if use_normalized_grafting else state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )

        # Compute matrix root inverse.
        self._compute_root_inverse(state_lists, compute_root_inverse)

        # Compute filtered gradient or EMA of the gradients.
        if beta1 != 0.0:
            torch._foreach_mul_(state_lists[MASKED_FILTERED_GRAD_LIST], beta1)
            torch._foreach_add_(
                state_lists[MASKED_FILTERED_GRAD_LIST],
                state_lists[MASKED_BLOCKED_GRADS],
                alpha=1 - beta1,
            )
            masked_filtered_grad_list = state_lists[MASKED_FILTERED_GRAD_LIST]

            if use_nadam:
                masked_filtered_grad_list = torch._foreach_mul(masked_filtered_grad_list, beta1)
                torch._foreach_add_(
                    masked_filtered_grad_list,
                    state_lists[MASKED_BLOCKED_GRADS],
                    alpha=1 - beta1,
                )

            if use_bias_correction:
                bias_correction1 = 1.0 - beta1**step
                masked_filtered_grad_list = torch._foreach_div(
                    masked_filtered_grad_list, bias_correction1
                )

        else:
            masked_filtered_grad_list = state_lists[MASKED_BLOCKED_GRADS]

        # Precondition gradients.
        # If the step count is less than start_preconditioning_step, then we use the grafting method.
        # Assumes that the step state is consistent across all parameters.
        if use_grafting_method:
            masked_blocked_search_directions = state_lists[
                GRAFTING_PRECONDITIONER_LIST
            ].precondition(
                masked_grad_list=masked_filtered_grad_list,
            )

        # Otherwise, we use Shampoo.
        else:
            masked_blocked_search_directions = state_lists[
                SHAMPOO_PRECONDITIONER_LIST
            ].precondition(
                masked_grad_list=masked_filtered_grad_list,
            )

            # Apply grafting.
            if grafting_config_not_none:
                # We apply normalized grafting by normalizing the per-block gradient list.
                # Note that this is not equal to the per-block filtered gradient list.
                if use_normalized_grafting:
                    masked_filtered_grad_list = tuple(
                        blocked_grad / (grad_norm + EPSILON_DEFAULT) for blocked_grad, grad_norm in zip(
                            state_lists[MASKED_BLOCKED_GRADS],
                            torch._foreach_norm(state_lists[MASKED_BLOCKED_GRADS]),
                        )
                    )

                grafting_norm_list = torch._foreach_norm(
                    state_lists[GRAFTING_PRECONDITIONER_LIST].precondition(
                        masked_grad_list=masked_filtered_grad_list,
                    )
                )
                shampoo_norm_list = torch._foreach_norm(
                    masked_blocked_search_directions
                )
                torch._foreach_add_(shampoo_norm_list, EPSILON_DEFAULT)
                torch._foreach_div_(grafting_norm_list, shampoo_norm_list)
                torch._foreach_mul_(
                    masked_blocked_search_directions, grafting_norm_list
                )

        # Incorporate decoupled weight decay.
        if weight_decay != 0.0 and use_decoupled_weight_decay:
            torch._foreach_add_(
                masked_blocked_search_directions,
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

        # Update momentum.
        if momentum_param != 0.0:
            torch._foreach_mul_(state_lists[MASKED_MOMENTUM_LIST], momentum_param)
            torch._foreach_add_(
                state_lists[MASKED_MOMENTUM_LIST],
                masked_blocked_search_directions,
                alpha=1.0 - momentum_param if use_ema_momentum else 1.0,
            )

            # Incorporates Nesterov momentum.
            if use_nesterov:
                if use_ema_momentum:
                    torch._foreach_mul_(masked_blocked_search_directions, 1.0 - momentum_param)

                torch._foreach_add_(
                    masked_blocked_search_directions,
                    state_lists[MASKED_MOMENTUM_LIST],
                    alpha=momentum_param,
                )

            else:
                torch._foreach_copy_(
                    masked_blocked_search_directions,
                    state_lists[MASKED_MOMENTUM_LIST],
                )

        # Updates parameters in distributed fashion.
        # If DDP, executes AllGather communication to ensure all parameters are updated after local updates.
        torch._foreach_mul_(masked_blocked_search_directions, -lr)
        state_lists[DISTRIBUTOR].update_params(
            masked_blocked_search_directions=masked_blocked_search_directions
        )


    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups,
        ):
            # Construct blocked gradient list.
            state_lists[MASKED_BLOCKED_GRADS] = state_lists[
                DISTRIBUTOR
            ].merge_and_block_gradients()

            # Based on the current block selector, mask lists of parameters and optimizer states.
            DistributedShampoo._mask_state_lists(state_lists, group)

            # Check if gradient list is empty. If so, continue.
            if not state_lists[MASKED_BLOCKED_GRADS]:
                continue

            # Iterate group step counter and define Python scalar step.
            step = state_lists[STEP].add_(1)
            # NOTE: Wrap scalar of group[LR] into a 0D tensor to avoid PT2 recompilation;
            # Send 0D tensor to GPU in `non_blocking` to avoid QPS regression. Remove the gpu
            # tensor impl once PT2 supports cpu 0D tensor properly.
            lr = torch.tensor(group[LR], dtype=torch.float).to(
                self._device, non_blocking=True
            )
            beta1 = group[BETAS][0]
            weight_decay = group[WEIGHT_DECAY]
            momentum_param = group[MOMENTUM]
            grafting_config_not_none = group[GRAFTING_CONFIG] is not None
            
            # Check compute root inverse or not for preconditioner
            # For DryShampoo, we *always* call compute_root_inverse, but inside it decides whether to actually update or not
            # based on the threshold. So we pass True here if we are past the start step.
            compute_root_inverse = (
                step % group[PRECONDITION_FREQUENCY] == 0
                and step > group[START_PRECONDITIONING_STEP]
                or step == group[START_PRECONDITIONING_STEP]
            )
            
            use_decoupled_weight_decay = group[USE_DECOUPLED_WEIGHT_DECAY]
            use_bias_correction = group[USE_BIAS_CORRECTION]
            # Check applying grafting method or not
            use_grafting_method = (
                step < group[START_PRECONDITIONING_STEP] and grafting_config_not_none
            )
            use_nadam = group[USE_NADAM]
            use_nesterov = group[USE_NESTEROV]
            use_normalized_grafting = group[USE_NORMALIZED_GRAFTING]
            use_ema_momentum = group[USE_EMA_MOMENTUM]

            self._per_group_step(
                state_lists,
                step,
                lr,
                beta1,
                weight_decay,
                momentum_param,
                grafting_config_not_none,
                compute_root_inverse,
                use_decoupled_weight_decay,
                use_bias_correction,
                use_grafting_method,
                use_nadam,
                use_nesterov,
                use_normalized_grafting,
                use_ema_momentum,
            )

        return loss


    @staticmethod
    def _construct_param_group_key(
        group: Dict[str, Any], param_to_key: Dict[torch.Tensor, str]
    ) -> str:
        return "/".join(sorted(param_to_key[param] for param in group[PARAMS]))

    def distributed_state_dict(
        self,
        key_to_param: Iterator[Tuple[str, torch.Tensor]],
        save_param_groups: bool = True,
    ) -> Dict[str, Any]:
        """Distributed state dict simplified from TorchRec's KeyedOptimizer.
        Compatible with torch.distributed.checkpoint with DTensor.

        Returned state and param_groups will contain parameter keys
        instead of parameter indices in torch.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.

        Can also handle classes and supported data structures that follow the PyTorch stateful
        protocol.

        Args:
            key_to_param (Iterator[Tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
                maps a FQN to the parameters in the model.
            save_param_groups (bool): Flag for saving parameter groups. (Default: True)

        Returns:
            state_dict (Dict[str, Any]): Dictionary containing the optimizer state and potentially parameter
                groups.

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        param_to_key = {param: key for key, param in key_to_param}
        ret: Dict[str, Any] = {
            "state": {
                param_to_key[param]: flatten(extract_state_dict_content(param_state))
                for param, param_state in self.state.items()
            }
        }
        if not save_param_groups:
            return ret

        # Store parameter groups with unique parameter group identifier.
        # NOTE: The parameters are ignored since they are assumed to be checkpointed separately.
        ret["param_groups"] = {
            self._construct_param_group_key(group, param_to_key): {
                k: deepcopy(v) for k, v in group.items() if k != PARAMS
            }
            for group in self.param_groups
        }

        return ret

    def load_distributed_state_dict(
        self,
        state_dict: Mapping[str, Any],
        key_to_param: Iterator[Tuple[str, torch.Tensor]],
        save_param_groups: bool = True,
        enable_missing_key_check: bool = True,
    ) -> None:
        """Load state dict simplified from TorchRec's KeyedOptimizer.
        Compatible with torch.distributed.checkpoint.

        This implementation is much stricter than the one in torch.Optimizer:
        it requires implementations to fully initialize their state during first optimization iteration,
        and it prohibits loading an empty state into already initialized KeyedOptimizer and vise versa.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves usability
            * avoid state duplication by directly copying into state tensors, e.g.
              optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard if needed
              optimizer.load_state_dict(sd)  # replace param_groups

        Args:
            state_dict (Dict[str, Any]): State dictionary to load containing the optimizer state and
                parameter groups.
            key_to_param (Iterator[Tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
                maps a FQN to the parameters in the model.
            save_param_groups (bool): Flag for saving parameter groups. (Default: True)
            enable_missing_key_check (bool): Flag for enabling missing key check. (Default: True)

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        state_to_load = state_dict["state"]
        key_to_param_mapping = dict(key_to_param)

        # Load state
        for param_key, param_state in state_to_load.items():
            # Check if parameter exists in current parameter state dict.
            if param_key not in key_to_param_mapping:
                if enable_missing_key_check:
                    raise KeyError(
                        f"Parameter key {param_key} not found in key_to_param mapping!"
                    )
                else:
                    logger.warning(
                        f"Parameter key {param_key} not found in key_to_param mapping!"
                    )
                    continue

            param = key_to_param_mapping[param_key]

            if param not in self.state:
                if enable_missing_key_check:
                    raise KeyError(f"Parameter {param} not found in state!")
                else:
                    logger.warning(f"Parameter {param} not found in state!")
                    continue

            # Update parameter state.
            update_param_state_dict_object(
                self.state[param],
                unflatten(param_state),
            )

        # Load param_groups.
        if save_param_groups:
            param_groups_to_load = state_dict["param_groups"]
            param_groups = self.param_groups

            if len(param_groups) != len(param_groups_to_load):
                raise ValueError(
                    f"Different param_groups count: {len(param_groups)} vs {len(param_groups_to_load)}"
                )
            param_to_key = {param: key for key, param in key_to_param_mapping.items()}

            # Loading the parameter group based on the unique parameter group key.
            for group in param_groups:
                param_group_key = self._construct_param_group_key(group, param_to_key)
                if param_group_key not in param_groups_to_load:
                    raise ValueError(
                        f"Param group {param_group_key} not found in param_groups_to_load!"
                    )
                param_group_to_load = param_groups_to_load[param_group_key]
                for key, value in param_group_to_load.items():
                    group[key] = deepcopy(value)
