import math
import os
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA residual branch."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be nn.Linear")

        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = max(0, int(rank))
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        ref_weight = base_layer.weight

        if self.rank > 0:
            self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            # Keep LoRA branch on the same device/dtype as the wrapped base layer.
            self.lora_A.to(device=ref_weight.device, dtype=ref_weight.dtype)
            self.lora_B.to(device=ref_weight.device, dtype=ref_weight.dtype)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        base = self.base_layer(x)
        if self.rank <= 0:
            return base
        residual = self.lora_B(self.lora_A(self.dropout(x)))
        return base + residual * self.scaling


def _iter_named_linear_modules(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield module, name, full_name, child
        else:
            yield from _iter_named_linear_modules(child, full_name)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: Sequence[str],
    freeze_base: bool = True,
):
    if rank <= 0:
        return []

    target_keywords = tuple(str(keyword) for keyword in target_keywords if str(keyword))
    replacements = []
    for parent, child_name, full_name, child in list(_iter_named_linear_modules(model)):
        if target_keywords and not any(keyword in full_name for keyword in target_keywords):
            continue
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        replacements.append(full_name)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
        mark_only_lora_trainable(model)

    return replacements


def mark_only_lora_trainable(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None and module.lora_B is not None:
                module.lora_A.weight.requires_grad = True
                module.lora_B.weight.requires_grad = True
            for param in module.base_layer.parameters():
                param.requires_grad = False


def has_lora_layers(model: nn.Module):
    return any(isinstance(module, LoRALinear) for module in model.modules())


def lora_state_dict(model: nn.Module):
    state = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
        state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict, strict: bool = False):
    missing = []
    unexpected = []
    used = set()

    modules = dict(model.named_modules())
    for key, value in state_dict.items():
        if not key.endswith(".weight"):
            unexpected.append(key)
            continue

        if key.endswith(".lora_A.weight"):
            module_name = key[: -len(".lora_A.weight")]
            attr = "lora_A"
        elif key.endswith(".lora_B.weight"):
            module_name = key[: -len(".lora_B.weight")]
            attr = "lora_B"
        else:
            unexpected.append(key)
            continue

        module = modules.get(module_name)
        if not isinstance(module, LoRALinear):
            unexpected.append(key)
            continue

        target = getattr(module, attr).weight
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        used.add(key)

    for name, module in modules.items():
        if not isinstance(module, LoRALinear):
            continue
        key_a = f"{name}.lora_A.weight"
        key_b = f"{name}.lora_B.weight"
        if key_a not in used:
            missing.append(key_a)
        if key_b not in used:
            missing.append(key_b)

    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA load failed, missing={missing}, unexpected={unexpected}")
    return missing, unexpected


def save_lora_adapter(
    model: nn.Module,
    path: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: Sequence[str],
):
    payload = {
        "lora_state_dict": lora_state_dict(model),
        "config": {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "target_keywords": list(target_keywords),
        },
    }
    adapter_dir = os.path.dirname(path)
    if adapter_dir:
        os.makedirs(adapter_dir, exist_ok=True)
    torch.save(payload, path)
    return path


def load_lora_adapter(model: nn.Module, path: str, freeze_base: bool = True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    rank = int(cfg.get("rank", 8))
    alpha = float(cfg.get("alpha", 16.0))
    dropout = float(cfg.get("dropout", 0.0))
    target_keywords = tuple(cfg.get("target_keywords", ()))

    if not has_lora_layers(model):
        inject_lora(
            model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_keywords=target_keywords,
            freeze_base=freeze_base,
        )
    missing, unexpected = load_lora_state_dict(
        model,
        payload.get("lora_state_dict", {}),
        strict=False,
    )
    return {"missing_keys": missing, "unexpected_keys": unexpected, "config": cfg}


def trainable_parameter_summary(model: nn.Module):
    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = int(parameter.numel())
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return {"total": total, "trainable": trainable}
