from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Type

import torch

from .config import ExperimentConfig


def _resolve_soap_class(module_path: str = "") -> Type[torch.optim.Optimizer]:
    """Resolve the external reference ``SOAP`` class without vendoring it.

    ``module_path`` may point either to a directory containing ``soap.py`` or
    directly to the file. The module is intentionally optional because the
    uploaded FOAM artifact did not include SOAP's source.
    """
    if module_path:
        path = Path(module_path).expanduser().resolve()
        search_dir = path.parent if path.is_file() else path
        sys.path.insert(0, str(search_dir))
    try:
        module = importlib.import_module("soap")
    except ImportError as exc:
        raise RuntimeError(
            "optimizer=soap requires the external reference implementation as "
            "soap.py. Set soap_module_path in the YAML configuration."
        ) from exc
    if not hasattr(module, "SOAP"):
        raise RuntimeError("The resolved soap module does not define a SOAP class.")
    return module.SOAP


def build_soap_optimizer(
    config: ExperimentConfig, params
) -> torch.optim.Optimizer:
    soap_class = _resolve_soap_class(config.soap_module_path)
    return soap_class(
        params,
        lr=config.base_lr,
        betas=(config.beta1, config.beta2),
        shampoo_beta=config.soap_shampoo_beta,
        eps=config.grafting_epsilon,
        weight_decay=config.weight_decay,
        precondition_frequency=config.precondition_frequency,
        max_precond_dim=config.max_preconditioner_dim,
        merge_dims=config.use_merge_dims,
        precondition_1d=config.soap_precondition_1d,
        normalize_grads=config.soap_normalize_grads,
        data_format="channels_first",
        correct_bias=config.use_bias_correction,
    )
