from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import CheckpointManager
from .config import ExperimentConfig
from .data import BatchMixup, DataBundle, build_dataloaders
from .distributed import (
    DistributedContext,
    all_reduce_max,
    all_reduce_sum,
    barrier,
    cleanup,
    environment_summary,
    initialize_distributed,
    synchronize,
)
from .metrics import CSVLogger, collect_optimizer_metrics, write_json
from .model import build_model, count_parameters, model_signature
from .optim import build_optimizer, current_learning_rate, set_learning_rate


def set_seed(seed: int, rank: int, deterministic: bool) -> None:
    full_seed = int(seed) + int(rank)
    os.environ["PYTHONHASHSEED"] = str(full_seed)
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def learning_rate_at_step(
    step: int, total_steps: int, base_lr: float, warmup_steps: int
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def precision_context(config: ExperimentConfig, device: torch.device):
    if config.precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return F.cross_entropy(logits, targets)
    return torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1).mean()


def _batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = batch["pixel_values"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    mixup: Optional[BatchMixup],
    config: ExperimentConfig,
    context: DistributedContext,
    epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> Tuple[float, int, float, bool]:
    model.train()
    objective_sum = 0.0
    sample_count = 0
    stop_requested = False

    synchronize(context.device)
    barrier()
    start = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if config.max_steps > 0 and global_step >= config.max_steps:
            stop_requested = True
            break

        images, hard_labels = _batch_to_device(batch, context.device)
        targets: torch.Tensor = hard_labels
        if mixup is not None:
            images, targets = mixup(images, hard_labels)

        learning_rate = learning_rate_at_step(
            global_step, total_steps, config.base_lr, warmup_steps
        )
        set_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)

        with precision_context(config, context.device):
            logits = model(images)
            loss = soft_target_cross_entropy(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        batch_size = images.shape[0]
        objective_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        global_step += 1

        if context.is_main and config.log_interval > 0 and (batch_index + 1) % config.log_interval == 0:
            print(
                f"epoch={epoch + 1}/{config.epochs} "
                f"step={batch_index + 1}/{len(loader)} "
                f"global_step={global_step} "
                f"objective={float(loss.detach().item()):.5f} "
                f"lr={learning_rate:.4e}",
                flush=True,
            )

    synchronize(context.device)
    barrier()
    elapsed = all_reduce_max(time.perf_counter() - start, context.device)
    objective_sum = all_reduce_sum(objective_sum, context.device)
    sample_count = int(all_reduce_sum(sample_count, context.device))
    return objective_sum / max(sample_count, 1), global_step, elapsed, stop_requested


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    context: DistributedContext,
    config: ExperimentConfig,
    compute_accuracy: bool,
) -> Tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    correct = 0

    synchronize(context.device)
    barrier()
    start = time.perf_counter()
    for batch in loader:
        images, labels = _batch_to_device(batch, context.device)
        with precision_context(config, context.device):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        batch_size = images.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        if compute_accuracy:
            correct += int((logits.argmax(dim=-1) == labels).sum().item())

    synchronize(context.device)
    barrier()
    elapsed = all_reduce_max(time.perf_counter() - start, context.device)
    loss_sum = all_reduce_sum(loss_sum, context.device)
    sample_count = int(all_reduce_sum(sample_count, context.device))
    correct = int(all_reduce_sum(correct, context.device)) if compute_accuracy else 0
    loss = loss_sum / max(sample_count, 1)
    accuracy = 100.0 * correct / max(sample_count, 1) if compute_accuracy else float("nan")
    return loss, accuracy, elapsed


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _build_manifest(
    config: ExperimentConfig,
    context: DistributedContext,
    model: nn.Module,
    data: DataBundle,
    total_steps: int,
    warmup_steps: int,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "config": config.to_dict(),
        "environment": environment_summary(context),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
        "git_revision": _git_revision(repo_root),
        "parameter_count": count_parameters(model.module if isinstance(model, DDP) else model),
        "model": model_signature(config),
        "train_samples": data.train_samples,
        "validation_samples": data.val_samples,
        "steps_per_epoch": len(data.train_loader),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "per_device_batch_size": config.per_device_batch_size,
        "global_batch_size": config.per_device_batch_size * context.world_size,
    }


def _make_scaler(config: ExperimentConfig, device: torch.device):
    enabled = config.precision == "fp16" and device.type == "cuda"
    if not enabled:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def _maybe_wandb(config: ExperimentConfig, context: DistributedContext):
    if not config.use_wandb or not context.is_main:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("use_wandb=true requires wandb.") from exc
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=config.experiment_name,
        dir=config.output_dir,
        config=config.to_dict(),
    )


def _save_factor_snapshots(
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    context: DistributedContext,
    epoch: int,
    global_step: int,
    max_per_rank: int,
) -> None:
    if not hasattr(optimizer, "export_preconditioner_snapshots"):
        return
    snapshots = optimizer.export_preconditioner_snapshots()
    if max_per_rank > 0:
        snapshots = snapshots[:max_per_rank]
    destination = output_dir / "factor_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "rank": int(context.rank),
            "snapshots": snapshots,
        },
        destination / f"epoch_{epoch:03d}_rank_{context.rank:04d}.pt",
    )


def run(config: ExperimentConfig, force_cpu: bool = False) -> Dict[str, Any]:
    context = initialize_distributed(force_cpu=force_cpu)
    try:
        config.validate(context.world_size)
        set_seed(config.seed, context.rank, config.deterministic)
        output_dir = Path(config.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            config.save_yaml(output_dir / "resolved_config.yaml")
        barrier()

        data = build_dataloaders(config, context)
        model = build_model(config).to(context.device)
        if context.distributed:
            model = DDP(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        optimizer = build_optimizer(config, model, context)
        scaler = _make_scaler(config, context.device)
        mixup = (
            BatchMixup(
                num_classes=config.num_classes,
                mixup_alpha=config.mixup,
                cutmix_alpha=config.cutmix,
                label_smoothing=config.label_smoothing,
            )
            if config.mixup > 0 or config.cutmix > 0 or config.label_smoothing > 0
            else None
        )

        planned_steps = len(data.train_loader) * config.epochs
        total_steps = min(planned_steps, config.max_steps) if config.max_steps > 0 else planned_steps
        warmup_steps = (
            config.warmup_steps
            if config.warmup_steps >= 0
            else int(round(config.warmup_ratio * total_steps))
        )

        manifest = _build_manifest(
            config, context, model, data, total_steps=total_steps, warmup_steps=warmup_steps
        )
        if context.is_main:
            write_json(output_dir / "run_manifest.json", manifest)

        metrics_logger = CSVLogger(output_dir / "metrics.csv")
        factor_logger = CSVLogger(output_dir / "factor_diagnostics.csv")
        checkpoint_manager = CheckpointManager(output_dir, context)
        wandb_run = _maybe_wandb(config, context)

        start_epoch = 0
        global_step = 0
        best_val_accuracy = -float("inf")
        elapsed_before_resume = 0.0
        train_compute_before_resume = 0.0
        evaluation_before_resume = 0.0
        if config.resume:
            metadata = checkpoint_manager.load_training_state(
                config.resume,
                model,
                optimizer,
                scaler,
                data.train_generator,
                context.device,
            )
            start_epoch = int(metadata["epoch"]) + 1
            global_step = int(metadata["global_step"])
            best_val_accuracy = float(metadata["best_val_accuracy"])
            elapsed_before_resume = float(metadata.get("elapsed_wall_clock_seconds", 0.0))
            timing = metadata.get("extra", {}).get("timing", {})
            train_compute_before_resume = float(timing.get("train_compute_seconds", 0.0))
            evaluation_before_resume = float(timing.get("evaluation_seconds", 0.0))

        barrier()
        synchronize(context.device)
        run_start = time.perf_counter()
        completed_epochs = start_epoch
        stop_requested = False
        cumulative_train_seconds = train_compute_before_resume
        cumulative_evaluation_seconds = evaluation_before_resume

        for epoch in range(start_epoch, config.epochs):
            if isinstance(data.train_loader.sampler, DistributedSampler):
                data.train_loader.sampler.set_epoch(epoch)

            online_objective, global_step, train_seconds, stop_requested = train_one_epoch(
                model=model,
                loader=data.train_loader,
                optimizer=optimizer,
                scaler=scaler,
                mixup=mixup,
                config=config,
                context=context,
                epoch=epoch,
                global_step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

            cumulative_train_seconds += train_seconds

            train_full_loss = float("nan")
            train_full_eval_seconds = 0.0
            if config.full_train_eval_interval > 0 and (
                (epoch + 1) % config.full_train_eval_interval == 0 or stop_requested
            ):
                train_full_loss, _, train_full_eval_seconds = evaluate(
                    model,
                    data.train_eval_loader,
                    context,
                    config,
                    compute_accuracy=False,
                )

            val_loss = float("nan")
            val_accuracy = float("nan")
            val_seconds = 0.0
            if config.validation_interval > 0 and (
                (epoch + 1) % config.validation_interval == 0 or stop_requested
            ):
                val_loss, val_accuracy, val_seconds = evaluate(
                    model, data.val_loader, context, config, compute_accuracy=True
                )

            cumulative_evaluation_seconds += train_full_eval_seconds + val_seconds

            synchronize(context.device)
            barrier()
            wall_clock_seconds = elapsed_before_resume + all_reduce_max(
                time.perf_counter() - run_start, context.device
            )
            optimizer_summary, factor_rows = collect_optimizer_metrics(
                optimizer,
                context,
                epoch=epoch + 1,
                global_step=global_step,
                wall_clock_seconds=wall_clock_seconds,
            )

            metric_row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_online_objective": online_objective,
                "train_full_hard_ce": train_full_loss,
                "train_full_ce": train_full_loss,  # backward-compatible alias
                "val_ce": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_learning_rate(optimizer),
                "train_seconds": train_seconds,
                "train_full_eval_seconds": train_full_eval_seconds,
                "validation_seconds": val_seconds,
                "train_compute_seconds_cumulative": cumulative_train_seconds,
                "evaluation_seconds_cumulative": cumulative_evaluation_seconds,
                "end_to_end_wall_clock_seconds": wall_clock_seconds,
                "wall_clock_seconds": wall_clock_seconds,  # backward-compatible alias
                **optimizer_summary,
            }

            if config.factor_snapshot_interval > 0 and (
                (epoch + 1) % config.factor_snapshot_interval == 0
            ):
                _save_factor_snapshots(
                    optimizer,
                    output_dir,
                    context,
                    epoch + 1,
                    global_step,
                    config.factor_snapshot_max_per_rank,
                )
            barrier()

            if context.is_main:
                metrics_logger.append(metric_row)
                if config.factor_diagnostics_interval > 0 and (
                    (epoch + 1) % config.factor_diagnostics_interval == 0
                ):
                    factor_logger.append_many(factor_rows)
                print(
                    f"epoch={epoch + 1} global_step={global_step} "
                    f"online_objective={online_objective:.5f} "
                    f"train_full_hard_ce={train_full_loss:.5f} "
                    f"val_accuracy={val_accuracy:.3f} "
                    f"wall_clock_min={wall_clock_seconds / 60.0:.2f} "
                    f"evd_rate={float(optimizer_summary.get('evd_rate', 0.0)):.4f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(metric_row, step=global_step)

            if not math.isnan(val_accuracy) and val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if config.save_best_model:
                    checkpoint_manager.save_best_model(
                        model, epoch, global_step, val_accuracy
                    )

            invocation_stop = (
                config.stop_after_epoch > 0
                and (epoch + 1) >= config.stop_after_epoch
            )
            if config.save_interval > 0 and (
                (epoch + 1) % config.save_interval == 0
                or stop_requested
                or invocation_stop
            ):
                checkpoint_manager.save_training_state(
                    "last",
                    model,
                    optimizer,
                    scaler,
                    data.train_generator,
                    epoch,
                    global_step,
                    best_val_accuracy,
                    wall_clock_seconds,
                    extra={
                        "experiment_name": config.experiment_name,
                        "timing": {
                            "train_compute_seconds": cumulative_train_seconds,
                            "evaluation_seconds": cumulative_evaluation_seconds,
                        },
                    },
                )
                checkpoint_manager.save_training_state(
                    f"epoch_{epoch + 1:03d}",
                    model,
                    optimizer,
                    scaler,
                    data.train_generator,
                    epoch,
                    global_step,
                    best_val_accuracy,
                    wall_clock_seconds,
                    extra={
                        "experiment_name": config.experiment_name,
                        "timing": {
                            "train_compute_seconds": cumulative_train_seconds,
                            "evaluation_seconds": cumulative_evaluation_seconds,
                        },
                    },
                )

            completed_epochs = epoch + 1
            if (
                stop_requested
                or invocation_stop
                or (config.max_steps > 0 and global_step >= config.max_steps)
            ):
                break

        synchronize(context.device)
        barrier()
        final_wall_clock = elapsed_before_resume + all_reduce_max(
            time.perf_counter() - run_start, context.device
        )
        summary = {
            "experiment_name": config.experiment_name,
            "optimizer": config.optimizer,
            "epochs_completed": completed_epochs,
            "global_step": global_step,
            "best_val_accuracy": best_val_accuracy,
            "train_compute_seconds": cumulative_train_seconds,
            "evaluation_seconds": cumulative_evaluation_seconds,
            "end_to_end_wall_clock_seconds": final_wall_clock,
            "wall_clock_seconds": final_wall_clock,
            "output_dir": str(output_dir),
            "world_size": context.world_size,
            "global_batch_size": config.per_device_batch_size * context.world_size,
        }
        if context.is_main:
            write_json(output_dir / "summary.json", summary)
        if wandb_run is not None:
            wandb_run.finish()
        return summary
    finally:
        cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible ViT/ImageNet training for FOAM and matched baselines."
    )
    parser.add_argument("--config", required=True, help="YAML experiment configuration.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a configuration value. Can be repeated.",
    )
    parser.add_argument("--resume", default="", help="Checkpoint directory; overrides config.resume.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.apply_overrides(args.overrides)
    if args.resume:
        config.resume = args.resume
    summary = run(config, force_cpu=args.cpu)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
