from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig
from .distributed import DistributedContext, barrier

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticImageNet(Dataset):
    """Deterministic synthetic samples for smoke tests and CI."""

    def __init__(self, size: int, num_classes: int, image_size: int) -> None:
        self.size = int(size)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        generator = torch.Generator().manual_seed(int(index))
        image = torch.rand(
            (3, self.image_size, self.image_size), generator=generator, dtype=torch.float32
        )
        label = int(torch.randint(self.num_classes, (1,), generator=generator).item())
        return {"pixel_values": image, "label": label}


class TupleToDictDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, label = self.dataset[index]
        return {"pixel_values": image, "label": int(label)}


class DistributedEvalSampler(Sampler[int]):
    """Non-padding distributed sampler for exact evaluation metrics."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return math.ceil((len(self.dataset) - self.rank) / self.world_size)


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _torchvision_transforms(config: ExperimentConfig, training: bool):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    interpolation = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }.get(config.interpolation.lower(), InterpolationMode.BICUBIC)

    if training:
        operations: list[Any] = [
            transforms.RandomResizedCrop(config.image_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ]
        if config.auto_augment:
            # This is the closest torchvision-native reconstruction of the
            # source's timm policy rand-m15-n2-mstd0.5.
            operations.append(transforms.RandAugment(num_ops=2, magnitude=15))
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        if config.random_erasing > 0:
            operations.append(transforms.RandomErasing(p=config.random_erasing))
        return transforms.Compose(operations)

    resize_size = int(round(config.image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_transforms(config: ExperimentConfig):
    if config.augmentation_backend == "timm":
        try:
            from timm.data import create_transform
        except ImportError as exc:
            raise RuntimeError(
                "augmentation_backend=timm requires timm. Install requirements.txt "
                "or set augmentation_backend=torchvision."
            ) from exc
        train_transform = create_transform(
            input_size=config.image_size,
            is_training=True,
            auto_augment=config.auto_augment or None,
            interpolation=config.interpolation,
            re_prob=config.random_erasing,
        )
        eval_transform = create_transform(
            input_size=config.image_size,
            is_training=False,
            interpolation=config.interpolation,
        )
        return train_transform, eval_transform
    return _torchvision_transforms(config, True), _torchvision_transforms(config, False)


def _hf_transform(
    examples: Dict[str, Sequence[Any]],
    transform,
    image_key: str,
    label_key: str,
) -> Dict[str, Any]:
    images = examples[image_key]
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in images]
    if label_key != "label":
        examples["label"] = examples[label_key]
    return examples


def _collate(batch: Sequence[Any]) -> Dict[str, torch.Tensor]:
    if isinstance(batch[0], dict):
        images = [sample["pixel_values"] for sample in batch]
        labels = [sample["label"] for sample in batch]
    else:
        images = [sample[0] for sample in batch]
        labels = [sample[1] for sample in batch]
    return {
        "pixel_values": torch.stack(images),
        "label": torch.as_tensor(labels, dtype=torch.long),
    }


@dataclass
class DataBundle:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    train_generator: torch.Generator
    train_samples: int
    val_samples: int


def _load_datasets(config: ExperimentConfig, context: DistributedContext):
    if config.data_backend == "synthetic":
        train = SyntheticImageNet(
            config.synthetic_train_samples, config.num_classes, config.image_size
        )
        train_eval = SyntheticImageNet(
            config.synthetic_train_samples, config.num_classes, config.image_size
        )
        val = SyntheticImageNet(
            config.synthetic_eval_samples, config.num_classes, config.image_size
        )
        return train, train_eval, val

    train_transform, eval_transform = build_transforms(config)

    if config.data_backend == "imagefolder":
        from torchvision.datasets import ImageFolder

        root = Path(config.data_path)
        train_root = root / "train"
        val_root = root / "val"
        if not train_root.is_dir() or not val_root.is_dir():
            raise FileNotFoundError(
                f"ImageFolder backend expects {train_root} and {val_root}."
            )
        train = TupleToDictDataset(ImageFolder(train_root, transform=train_transform))
        train_eval = TupleToDictDataset(ImageFolder(train_root, transform=eval_transform))
        val = TupleToDictDataset(ImageFolder(val_root, transform=eval_transform))
        return train, train_eval, val

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("data_backend=huggingface requires the datasets package.") from exc

    if context.is_main:
        load_dataset(config.hf_dataset_name, cache_dir=config.hf_cache_dir)
    barrier()
    dataset = load_dataset(config.hf_dataset_name, cache_dir=config.hf_cache_dir)
    missing_splits = [
        split
        for split in (config.hf_train_split, config.hf_val_split)
        if split not in dataset
    ]
    if missing_splits:
        raise KeyError(
            f"Dataset {config.hf_dataset_name!r} is missing splits {missing_splits}."
        )
    train = dataset[config.hf_train_split].with_transform(
        lambda examples: _hf_transform(
            examples, train_transform, config.hf_image_key, config.hf_label_key
        )
    )
    train_eval = dataset[config.hf_train_split].with_transform(
        lambda examples: _hf_transform(
            examples, eval_transform, config.hf_image_key, config.hf_label_key
        )
    )
    val = dataset[config.hf_val_split].with_transform(
        lambda examples: _hf_transform(
            examples, eval_transform, config.hf_image_key, config.hf_label_key
        )
    )
    return train, train_eval, val


def build_dataloaders(
    config: ExperimentConfig, context: DistributedContext
) -> DataBundle:
    train_dataset, train_eval_dataset, val_dataset = _load_datasets(config, context)
    train_generator = torch.Generator().manual_seed(config.seed)

    if context.distributed:
        train_sampler: Optional[Sampler[int]] = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=config.drop_last,
        )
        train_eval_sampler: Optional[Sampler[int]] = DistributedEvalSampler(
            train_eval_dataset, context.rank, context.world_size
        )
        val_sampler: Optional[Sampler[int]] = DistributedEvalSampler(
            val_dataset, context.rank, context.world_size
        )
    else:
        train_sampler = None
        train_eval_sampler = None
        val_sampler = None

    common = {
        "num_workers": config.workers,
        "pin_memory": config.pin_memory and context.device.type == "cuda",
        "persistent_workers": config.persistent_workers and config.workers > 0,
        "worker_init_fn": _worker_init_fn,
        "collate_fn": _collate,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=config.drop_last,
        generator=train_generator,
        **common,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=config.eval_batch_size,
        sampler=train_eval_sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return DataBundle(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        train_generator=train_generator,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )


class BatchMixup:
    """Self-contained Mixup/CutMix with label smoothing."""

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float,
        cutmix_alpha: float,
        label_smoothing: float,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.label_smoothing = float(label_smoothing)

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing + off_value
        targets = torch.full(
            (labels.shape[0], self.num_classes),
            off_value,
            dtype=torch.float32,
            device=labels.device,
        )
        return targets.scatter_(1, labels.unsqueeze(1), on_value)

    @staticmethod
    def _sample_beta(alpha: float, device: torch.device) -> float:
        if alpha <= 0:
            return 1.0
        distribution = torch.distributions.Beta(alpha, alpha)
        return float(distribution.sample().to(device).item())

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self._targets(labels)
        if self.mixup_alpha <= 0 and self.cutmix_alpha <= 0:
            return images, targets

        permutation = torch.randperm(images.shape[0], device=images.device)
        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or bool(torch.rand((), device=images.device) < 0.5)
        )
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = self._sample_beta(alpha, images.device)

        if use_cutmix:
            height, width = images.shape[-2:]
            cut_ratio = math.sqrt(1.0 - lam)
            cut_height = int(height * cut_ratio)
            cut_width = int(width * cut_ratio)
            center_y = int(torch.randint(height, (), device=images.device).item())
            center_x = int(torch.randint(width, (), device=images.device).item())
            y1 = max(center_y - cut_height // 2, 0)
            y2 = min(center_y + cut_height // 2, height)
            x1 = max(center_x - cut_width // 2, 0)
            x2 = min(center_x + cut_width // 2, width)
            mixed = images.clone()
            mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(height * width))
        else:
            mixed = images.mul(lam).add(images[permutation], alpha=1.0 - lam)

        mixed_targets = targets.mul(lam).add(targets[permutation], alpha=1.0 - lam)
        return mixed, mixed_targets
