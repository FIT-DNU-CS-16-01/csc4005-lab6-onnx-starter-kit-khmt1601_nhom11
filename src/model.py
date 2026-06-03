from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def _freeze_all_except(module: nn.Module, trainable_prefixes: tuple[str, ...]) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)


def build_vit_model(
    model_name: Literal["vit_b_16", "vit_b_32"],
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = False,
    train_mode: Literal["head_only", "finetune"] = "head_only",
) -> nn.Module:
    try:
        from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
    except Exception as exc:
        raise ImportError(
            "Vision Transformer cần torchvision cài đúng cặp phiên bản với torch."
        ) from exc

    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
    elif model_name == "vit_b_32":
        weights = ViT_B_32_Weights.DEFAULT if pretrained else None
        model = vit_b_32(weights=weights)
    else:
        raise ValueError(f"Unsupported ViT model: {model_name}")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if train_mode == "head_only":
        _freeze_all_except(model, ("heads",))
    elif train_mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("train_mode must be 'head_only' or 'finetune'.")

    return model


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = False,
    train_mode: str = "head_only",
) -> nn.Module:
    if model_name in {"vit_b_16", "vit_b_32"}:
        return build_vit_model(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            train_mode=train_mode,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_checkpoint_state(checkpoint_path: str, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, checkpoint["model_state_dict"]
    return {"raw_checkpoint": True}, checkpoint


def infer_num_classes_from_checkpoint_metadata(checkpoint_meta: dict, fallback: int = 5) -> int:
    class_to_idx = checkpoint_meta.get("class_to_idx")
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
        return len(class_to_idx)

    args = checkpoint_meta.get("args")
    if isinstance(args, dict):
        classes = args.get("classes")
        if isinstance(classes, list) and len(classes) > 0:
            return len(classes)

    return fallback
