from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.act(self.fc1(inputs)))
        return self.dropout(self.fc2(hidden))


class CustomMultiheadAttention(nn.Module):
    """Attention implementation used by the uploaded public ViT script."""

    def __init__(self, embedding_dim: int, num_heads: int, attn_dropout: float) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        queries = self.q_proj(inputs).reshape(shape).permute(0, 2, 1, 3)
        keys = self.k_proj(inputs).reshape(shape).permute(0, 2, 1, 3)
        values = self.v_proj(inputs).reshape(shape).permute(0, 2, 1, 3)

        attention = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.attn_dropout(F.softmax(attention, dim=-1))
        output = torch.matmul(attention, values)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(
            batch_size, sequence_length, self.embedding_dim
        )
        return self.out_proj(output)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attn_dropout: float,
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = CustomMultiheadAttention(embedding_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, mlp_dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(inputs)
        inputs = inputs + self.attn(normalized)
        return inputs + self.mlp(self.norm2(inputs))


class PaperVisionTransformer(nn.Module):
    """ViT-S/16 architecture reconstructed from the uploaded ``vit.py``."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3,
            config.embedding_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, config.embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    attn_dropout=config.attn_dropout,
                    mlp_dropout=config.mlp_dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.classifier_norm = nn.LayerNorm(config.embedding_dim)
        self.classifier_head = nn.Linear(config.embedding_dim, config.num_classes)

        if config.init_scheme == "vit":
            self._initialize_vit_weights()
        elif config.init_scheme != "source":
            raise ValueError(f"Unknown init_scheme={config.init_scheme!r}.")

    def _initialize_vit_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat((cls_token, patches), dim=1)
        hidden = self.embedding_dropout(hidden + self.position_embedding)
        for block in self.encoder_blocks:
            hidden = block(hidden)
        hidden = self.classifier_norm(hidden)
        return self.classifier_head(hidden[:, 0])


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_impl == "paper_custom":
        model: nn.Module = PaperVisionTransformer(config)
    elif config.model_impl == "timm_vit_small_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for model_impl=timm_vit_small_patch16_224.") from exc
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=config.num_classes,
            img_size=config.image_size,
            drop_rate=config.embedding_dropout,
            attn_drop_rate=config.attn_dropout,
            drop_path_rate=0.0,
        )
    else:  # guarded by config validation
        raise ValueError(f"Unsupported model_impl={config.model_impl!r}.")

    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
        model = torch.compile(model)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_signature(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "implementation": config.model_impl,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "embedding_dim": config.embedding_dim,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "mlp_dim": config.mlp_dim,
        "num_classes": config.num_classes,
        "init_scheme": config.init_scheme,
    }
