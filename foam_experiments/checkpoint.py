from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import DistributedContext, barrier


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng_state(train_generator: torch.Generator) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_state_dict(optimizer: torch.optim.Optimizer, model: nn.Module) -> Dict[str, Any]:
    target = unwrap_model(model)
    if hasattr(optimizer, "distributed_state_dict"):
        return {
            "format": "distributed",
            "state": optimizer.distributed_state_dict(
                key_to_param=target.named_parameters(), save_param_groups=True
            ),
        }
    return {"format": "torch", "state": optimizer.state_dict()}


def load_optimizer_state_dict(
    optimizer: torch.optim.Optimizer, model: nn.Module, payload: Dict[str, Any]
) -> None:
    target = unwrap_model(model)
    if payload["format"] == "distributed":
        if not hasattr(optimizer, "load_distributed_state_dict"):
            raise TypeError("Checkpoint contains DistributedShampoo state for a different optimizer.")
        optimizer.load_distributed_state_dict(
            payload["state"], key_to_param=target.named_parameters(), save_param_groups=True
        )
    elif payload["format"] == "torch":
        optimizer.load_state_dict(payload["state"])
    else:
        raise ValueError(f"Unknown optimizer checkpoint format {payload['format']!r}.")


class CheckpointManager:
    """Rank-local optimizer checkpoints plus one replicated model checkpoint.

    This avoids lossy rank merging of distributed optimizer tensors. Resume
    requires the same world size, which is recorded in metadata.
    """

    def __init__(self, output_dir: str | Path, context: DistributedContext) -> None:
        self.output_dir = Path(output_dir)
        self.context = context
        self.root = self.output_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    def save_training_state(
        self,
        tag: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        train_generator: torch.Generator,
        epoch: int,
        global_step: int,
        best_val_accuracy: float,
        elapsed_wall_clock_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        checkpoint_dir = self.root / tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rank_payload = {
            "optimizer": optimizer_state_dict(optimizer, model),
            "rng": capture_rng_state(train_generator),
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        _atomic_torch_save(
            rank_payload,
            checkpoint_dir / f"rank_{self.context.rank:04d}.pt",
        )

        if self.context.is_main:
            target = unwrap_model(model)
            _atomic_torch_save(target.state_dict(), checkpoint_dir / "model.pt")
            metadata = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_val_accuracy": float(best_val_accuracy),
                "elapsed_wall_clock_seconds": float(elapsed_wall_clock_seconds),
                "world_size": int(self.context.world_size),
                "extra": extra or {},
            }
            temporary = checkpoint_dir / "metadata.json.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            temporary.replace(checkpoint_dir / "metadata.json")
        barrier()
        return checkpoint_dir

    def save_best_model(
        self,
        model: nn.Module,
        epoch: int,
        global_step: int,
        val_accuracy: float,
    ) -> Path:
        destination = self.root / "best"
        if self.context.is_main:
            destination.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(unwrap_model(model).state_dict(), destination / "model.pt")
            with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "val_accuracy": float(val_accuracy),
                    },
                    handle,
                    indent=2,
                )
        barrier()
        return destination

    def load_training_state(
        self,
        checkpoint: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        train_generator: torch.Generator,
        device: torch.device,
    ) -> Dict[str, Any]:
        checkpoint_dir = Path(checkpoint)
        if checkpoint_dir.name == "auto":
            checkpoint_dir = self.root / "last"
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if int(metadata["world_size"]) != self.context.world_size:
            raise ValueError(
                "Rank-local optimizer checkpoints require the same world size: "
                f"saved={metadata['world_size']}, current={self.context.world_size}."
            )

        model_state = torch.load(
            checkpoint_dir / "model.pt", map_location=device, weights_only=False
        )
        unwrap_model(model).load_state_dict(model_state)

        rank_path = checkpoint_dir / f"rank_{self.context.rank:04d}.pt"
        rank_payload = torch.load(rank_path, map_location=device, weights_only=False)
        load_optimizer_state_dict(optimizer, model, rank_payload["optimizer"])
        if scaler is not None and rank_payload.get("scaler") is not None:
            scaler.load_state_dict(rank_payload["scaler"])
        restore_rng_state(rank_payload["rng"], train_generator)
        barrier()
        return metadata
