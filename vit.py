#!/usr/bin/env python3
"""
Official public training script for Vision Transformer (ViT) with FOAM.

Features
--------
- ViT training on ImageNet-1K or synthetic data
- FOAM-enabled DistributedShampoo optimizer only
- Single-process or DistributedDataParallel execution
- Epoch-level reporting of:
    * train loss
    * validation loss
    * validation accuracy
- CSV metric logging and checkpoint saving

This script intentionally removes:
- stale / residual / SOAP baselines
- wall-clock profiling / timing instrumentation
- QR benchmarks
- dimension-scaling microbenchmarks
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from timm.data import Mixup, create_transform
except Exception:
    Mixup = None
    create_transform = None

try:
    import wandb
except Exception:
    wandb = None

try:
    from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
    from optimizers.distributed_shampoo.shampoo_types import (
        AdamGraftingConfig,
        CommunicationDType,
        DDPShampooConfig,
    )
    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import (
        ShampooPreconditionerList,
    )
    from optimizers.matrix_functions import check_diagonal, matrix_inverse_root

    _HAVE_SHAMPOO = True
except Exception:
    DistributedShampoo = None
    AdamGraftingConfig = None
    CommunicationDType = None
    DDPShampooConfig = None
    ShampooPreconditionerList = None
    check_diagonal = None
    matrix_inverse_root = None
    _HAVE_SHAMPOO = False


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def dist_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def dist_barrier() -> None:
    if is_dist():
        dist.barrier()


def maybe_init_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and not args.cpu

    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return world_size > 1, dist_rank(), dist_world_size(), device


def maybe_cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def all_reduce_sum(value: float, device: torch.device) -> float:
    if not is_dist():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, rank: int = 0) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(full_seed)
        torch.cuda.manual_seed_all(full_seed)

    os.environ["PYTHONHASHSEED"] = str(full_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, attn_dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embedding_dim)
        return self.out_proj(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = CustomMultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_classes: int = 1000,
        embedding_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.classifier_norm = nn.LayerNorm(embedding_dim)
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.classifier_norm(x)
        return self.classifier_head(x[:, 0])


def build_model(args: argparse.Namespace) -> nn.Module:
    return VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
        attn_dropout=args.attn_dropout,
        mlp_dropout=args.mlp_dropout,
        embedding_dropout=args.embedding_dropout,
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class SyntheticImageNet(Dataset):
    def __init__(self, size: int = 4096, num_classes: int = 1000, image_size: int = 224):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        rng = np.random.default_rng(int(idx))
        image = rng.integers(0, 256, size=(3, self.image_size, self.image_size), dtype=np.uint8)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        label = int(rng.integers(0, self.num_classes))
        return {"pixel_values": image, "label": label}


def apply_transforms(examples: Dict[str, List[Image.Image]], transform):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "label": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
    }


def make_dataloaders(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    distributed: bool,
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available() and not args.cpu

    if args.synthetic_data:
        train_dataset = SyntheticImageNet(
            size=args.synthetic_train_samples,
            num_classes=args.num_classes,
            image_size=args.image_size,
        )
        val_dataset = SyntheticImageNet(
            size=args.synthetic_eval_samples,
            num_classes=args.num_classes,
            image_size=args.image_size,
        )

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if distributed
            else None
        )
        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            if distributed
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader

    if load_dataset is None or create_transform is None:
        raise RuntimeError("datasets and timm are required for ImageNet-1K training mode.")

    train_transform = create_transform(
        input_size=args.image_size,
        is_training=True,
        auto_augment=args.auto_augment,
        interpolation=args.interpolation,
    )
    val_transform = create_transform(
        input_size=args.image_size,
        is_training=False,
        interpolation=args.interpolation,
    )

    if rank == 0:
        load_dataset("imagenet-1k", cache_dir=args.data_path)
    dist_barrier()

    dataset = load_dataset("imagenet-1k", cache_dir=args.data_path)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_dataset.set_transform(functools.partial(apply_transforms, transform=train_transform))
    val_dataset.set_transform(functools.partial(apply_transforms, transform=val_transform))

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# FOAM optimizer patch
# ---------------------------------------------------------------------------


def install_foam_patch(args: argparse.Namespace):
    if not _HAVE_SHAMPOO:
        raise RuntimeError("DistributedShampoo is required for FOAM training.")

    original_single = ShampooPreconditionerList._compute_single_root_inverse

    def patched_compute_single_root_inverse(
        self,
        factor_matrix,
        inv_factor_matrix,
        is_factor_matrix_diagonal,
        factor_matrix_index,
        root,
        epsilon_value,
        kronecker_factors,
        factor_idx,
    ):
        del factor_matrix_index  # not used in the public release

        bias_corrected = factor_matrix / self._bias_correction2
        prev_q = kronecker_factors.eigenvectors[factor_idx]
        prev_d = kronecker_factors.eigenvalues[factor_idx]

        current_epsilon = kronecker_factors.adaptive_epsilons[factor_idx]
        if current_epsilon is None:
            current_epsilon = epsilon_value

        should_recompute = True

        def reuse_stale_basis(new_epsilon: float) -> None:
            exponent = -self._exponent_multiplier / root
            eig_term = (prev_d + new_epsilon).pow(exponent)
            computed_inv = prev_q * eig_term.unsqueeze(0) @ prev_q.T
            inv_factor_matrix.copy_(computed_inv.to(dtype=inv_factor_matrix.dtype))
            kronecker_factors.adaptive_epsilons[factor_idx] = float(new_epsilon)

        if prev_q is not None and prev_d is not None:
            try:
                rc_t = self._compute_relative_condition_number(
                    bias_corrected,
                    prev_q,
                    prev_d,
                    current_epsilon,
                )
                exponent = -self._exponent_multiplier / root
                h_eigs = (prev_d + current_epsilon).pow(exponent)
                alpha = h_eigs.abs().max() / (torch.norm(h_eigs, p=2) + 1e-25)
                proxy = rc_t * (alpha / root)

                new_epsilon = current_epsilon * (
                    float(proxy.item()) / max(args.matrix_root_inv_threshold, 1e-25)
                )

                if float(proxy.item()) >= args.matrix_root_inv_threshold:
                    if new_epsilon < args.max_epsilon:
                        current_epsilon = float(new_epsilon)
                        should_recompute = False
                        reuse_stale_basis(current_epsilon)
                    else:
                        current_epsilon = epsilon_value
                else:
                    current_epsilon = max(float(new_epsilon), float(epsilon_value))
                    should_recompute = False
                    reuse_stale_basis(current_epsilon)
            except Exception:
                should_recompute = True

        if should_recompute:
            if (
                is_factor_matrix_diagonal is not None
                and is_factor_matrix_diagonal is not False
                and check_diagonal is not None
            ):
                try:
                    if bool(is_factor_matrix_diagonal) and not check_diagonal(factor_matrix):
                        is_factor_matrix_diagonal.copy_(torch.tensor(False, device=factor_matrix.device))
                except Exception:
                    pass

            result = matrix_inverse_root(
                A=bias_corrected,
                root=root,
                epsilon=current_epsilon,
                exponent_multiplier=self._exponent_multiplier,
                is_diagonal=is_factor_matrix_diagonal,
                retry_double_precision=self._use_protected_eigh,
            )
            computed_inv, used_epsilon, eigvals, eigvecs = result

            if eigvals is not None and eigvecs is not None:
                raw_eigenvalues = eigvals - used_epsilon
                kronecker_factors.eigenvalues[factor_idx] = raw_eigenvalues.to(dtype=factor_matrix.dtype)
                kronecker_factors.eigenvectors[factor_idx] = eigvecs.to(dtype=factor_matrix.dtype)
                kronecker_factors.adaptive_epsilons[factor_idx] = float(current_epsilon)

            inv_factor_matrix.copy_(computed_inv.to(dtype=inv_factor_matrix.dtype))

    ShampooPreconditionerList._compute_single_root_inverse = patched_compute_single_root_inverse
    return original_single


def restore_foam_patch(original_single) -> None:
    if _HAVE_SHAMPOO:
        ShampooPreconditionerList._compute_single_root_inverse = original_single


def build_optimizer(args: argparse.Namespace, model: nn.Module, world_size: int) -> Any:
    if not _HAVE_SHAMPOO:
        raise RuntimeError("DistributedShampoo is required for FOAM training.")

    distributed_config = None
    if world_size > 1:
        distributed_config = DDPShampooConfig(
            communication_dtype=CommunicationDType.FP32,
            num_trainers_per_group=world_size,
            communicate_params=False,
        )

    return DistributedShampoo(
        params=model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        epsilon=args.epsilon,
        momentum=0.0,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(
            beta2=args.adam_grafting_beta2,
            epsilon=args.grafting_epsilon,
        ),
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        exponent_multiplier=1,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        matrix_root_inv_threshold=args.matrix_root_inv_threshold,
        max_epsilon=args.max_epsilon,
    )


def gather_optimizer_state_from_all_ranks(
    optimizer,
    model,
    rank: int,
    world_size: int,
):
    if not is_dist() or world_size <= 1:
        if hasattr(optimizer, "distributed_state_dict"):
            return optimizer.distributed_state_dict(
                key_to_param=model.module.named_parameters() if isinstance(model, DDP) else model.named_parameters()
            )
        return optimizer.state_dict()

    if not hasattr(optimizer, "distributed_state_dict"):
        return optimizer.state_dict() if rank == 0 else None

    local_state = optimizer.distributed_state_dict(
        key_to_param=model.module.named_parameters() if isinstance(model, DDP) else model.named_parameters()
    )
    all_states = [None] * world_size
    dist.all_gather_object(all_states, local_state)

    if rank != 0:
        return None

    merged_state = {"state": {}, "param_groups": all_states[0].get("param_groups", [])}
    for param_key in all_states[0]["state"].keys():
        merged_state["state"][param_key] = {}
        param_state_keys = set()

        for state in all_states:
            if "state" in state and param_key in state["state"]:
                param_state_keys.update(state["state"][param_key].keys())

        for state_key in param_state_keys:
            merged_value = None
            is_factor_matrix = "factor_matrices" in str(state_key)

            for state in all_states:
                if (
                    "state" in state
                    and param_key in state["state"]
                    and state_key in state["state"][param_key]
                ):
                    value = state["state"][param_key][state_key]
                    if isinstance(value, torch.Tensor):
                        if hasattr(value, "_local_tensor"):
                            value = value._local_tensor
                        if is_factor_matrix:
                            if value.numel() > 0 and (merged_value is None or merged_value.numel() == 0):
                                merged_value = value.clone()
                        else:
                            if merged_value is None or (merged_value.numel() == 0 and value.numel() > 0):
                                merged_value = value.clone()
                    else:
                        if merged_value is None:
                            merged_value = value

            if merged_value is not None:
                merged_state["state"][param_key][state_key] = merged_value

    return merged_state


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def get_warmup_cosine_decay_lr(
    current_step: int,
    base_lr: float,
    num_steps: int,
    warmup_steps: int,
) -> float:
    if current_step < warmup_steps:
        return base_lr * (current_step / max(warmup_steps, 1))

    progress = (current_step - warmup_steps) / max(num_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine_decay


def maybe_build_wandb(args: argparse.Namespace, rank: int):
    if rank != 0 or args.no_wandb or wandb is None:
        return None
    return wandb.init(
        project=args.project,
        entity=args.entity,
        dir=str(args.out_dir),
        config=vars(args),
    )


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    loss_sum = 0.0
    total_examples = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_sum += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == labels).sum().item())

    loss_sum = all_reduce_sum(loss_sum, device)
    total_examples = int(all_reduce_sum(total_examples, device))
    correct = int(all_reduce_sum(correct, device))

    avg_loss = loss_sum / max(total_examples, 1)
    accuracy = 100.0 * correct / max(total_examples, 1)
    return avg_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Any,
    mixup_fn,
    device: torch.device,
    epoch: int,
    total_steps: int,
    args: argparse.Namespace,
    rank: int,
) -> Tuple[float, float]:
    model.train()

    loss_sum = 0.0
    total_examples = 0
    last_lr = optimizer.param_groups[0]["lr"]

    for step, batch in enumerate(loader):
        global_step = epoch * len(loader) + step

        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        lr = get_warmup_cosine_decay_lr(
            current_step=global_step,
            base_lr=args.base_lr,
            num_steps=total_steps,
            warmup_steps=args.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        last_lr = lr

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_sum += float(loss.detach().item()) * batch_size
        total_examples += batch_size

        if rank == 0 and (step + 1) % args.log_interval == 0:
            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"step {step + 1}/{len(loader)} "
                f"loss={float(loss.detach().item()):.4f} "
                f"lr={lr:.3e}"
            )

        if args.max_steps > 0 and global_step + 1 >= args.max_steps:
            break

    loss_sum = all_reduce_sum(loss_sum, device)
    total_examples = int(all_reduce_sum(total_examples, device))
    train_loss = loss_sum / max(total_examples, 1)
    return train_loss, last_lr


def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_command(args: argparse.Namespace) -> None:
    distributed, rank, world_size, device = maybe_init_distributed(args)
    set_seed(args.seed, rank)

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = maybe_build_wandb(args, rank)
    train_loader, val_loader = make_dataloaders(args, rank, world_size, distributed)

    model = build_model(args).to(device)
    if distributed:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids)

    criterion = nn.CrossEntropyLoss().to(device)

    mixup_fn = None
    if not args.synthetic_data and (args.mixup > 0 or args.label_smoothing > 0):
        if Mixup is None:
            raise RuntimeError("timm is required when mixup or label smoothing is enabled.")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=0.0,
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes,
        )

    optimizer = build_optimizer(args, model, world_size)
    original_single = install_foam_patch(args)

    start_epoch = 0
    best_val_accuracy = -float("inf")
    metrics_rows: List[Dict[str, Any]] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        target_model = model.module if isinstance(model, DDP) else model
        target_model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            if hasattr(optimizer, "load_distributed_state_dict"):
                optimizer.load_distributed_state_dict(
                    checkpoint["optimizer_state_dict"],
                    key_to_param=target_model.named_parameters(),
                )
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_accuracy = float(checkpoint.get("best_val_accuracy", best_val_accuracy))

        metrics_path = args.out_dir / "metrics.csv"
        if metrics_path.exists():
            metrics_path.unlink()

    total_steps = len(train_loader) * args.epochs

    try:
        for epoch in range(start_epoch, args.epochs):
            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss, lr = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                mixup_fn=mixup_fn,
                device=device,
                epoch=epoch,
                total_steps=total_steps,
                args=args,
                rank=rank,
            )

            val_loss, val_accuracy = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": lr,
            }
            metrics_rows.append(row)

            if rank == 0:
                print(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_accuracy={val_accuracy:.2f}%"
                )
                write_metrics_csv(args.out_dir / "metrics.csv", metrics_rows)

                if wandb_run is not None:
                    wandb.log(dict(row))

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                optimizer_state = gather_optimizer_state_from_all_ranks(
                    optimizer=optimizer,
                    model=model,
                    rank=rank,
                    world_size=world_size,
                )
                if rank == 0:
                    target_model = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": target_model.state_dict(),
                            "optimizer_state_dict": optimizer_state,
                            "best_val_accuracy": best_val_accuracy,
                            "args": vars(args),
                        },
                        args.out_dir / "best.pt",
                    )

            save_periodic = args.save_interval > 0 and (epoch + 1) % args.save_interval == 0
            if save_periodic:
                optimizer_state = gather_optimizer_state_from_all_ranks(
                    optimizer=optimizer,
                    model=model,
                    rank=rank,
                    world_size=world_size,
                )
                if rank == 0:
                    target_model = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": target_model.state_dict(),
                            "optimizer_state_dict": optimizer_state,
                            "best_val_accuracy": best_val_accuracy,
                            "args": vars(args),
                        },
                        args.out_dir / f"epoch_{epoch + 1}.pt",
                    )

            if args.max_steps > 0 and (epoch + 1) * len(train_loader) >= args.max_steps:
                break

        if rank == 0:
            summary = {
                "best_val_accuracy": best_val_accuracy,
                "epochs_completed": len(metrics_rows),
                "world_size": world_size,
                "device": str(device),
                "output_dir": str(args.out_dir),
            }
            with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    finally:
        restore_foam_patch(original_single)
        if wandb_run is not None:
            wandb.finish()
        maybe_cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vision Transformer training with FOAM optimizer"
    )

    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./runs/vit_foam")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-train-samples", type=int, default=4096)
    parser.add_argument("--synthetic-eval-samples", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.995)
    parser.add_argument("--adam-grafting-beta2", type=float, default=0.995)
    parser.add_argument("--grafting-epsilon", type=float, default=1e-9)
    parser.add_argument("--epsilon", type=float, default=1e-9)

    parser.add_argument("--matrix-root-inv-threshold", type=float, default=0.5, help="FOAM proxy threshold")
    parser.add_argument("--max-epsilon", type=float, default=5e-7, help="FOAM epsilon cap")
    parser.add_argument("--precondition-frequency", type=int, default=20)
    parser.add_argument("--start-preconditioning-step", type=int, default=20)
    parser.add_argument("--max-preconditioner-dim", type=int, default=1024)

    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--auto-augment", type=str, default="rand-m15-n2-mstd0.5")
    parser.add_argument("--interpolation", type=str, default="bicubic")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-dim", type=int, default=1536)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--embedding-dropout", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--save-interval", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--project", type=str, default="ViT_FOAM")
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_command(args)


if __name__ == "__main__":
    main()
