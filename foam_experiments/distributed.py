from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(force_cpu: bool = False) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and not force_cpu

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistributedContext(
        distributed=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def all_reduce_sum(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_max(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def gather_objects(value: Any, dst: int = 0) -> List[Any] | None:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    if dist.get_rank() == dst:
        output: List[Any] = [None] * dist.get_world_size()
        dist.gather_object(value, object_gather_list=output, dst=dst)
        return output
    dist.gather_object(value, dst=dst)
    return None


def environment_summary(context: DistributedContext) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "world_size": context.world_size,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        summary.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(context.device),
            }
        )
    return summary
