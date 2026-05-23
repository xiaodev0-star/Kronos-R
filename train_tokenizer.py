"""Tokenizer 训练脚本。"""

import json
import os
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONPATH", None)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import DataConfig, TokenizerConfig
from data_processor import get_dataloaders
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs, export_tokenizer_config
from reproducibility import set_global_seed


def _safe_loader_overrides_for_windows(device):
    """Windows 下禁用多进程 DataLoader，防止 CUDA DLL 重复加载导致崩溃。"""
    if os.name != "nt":
        return None
    return {
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": bool(device.type == "cuda"),
    }


def _history_path():
    """返回 tokenizer 训练历史 JSON 文件的路径。"""
    return os.path.join(os.path.dirname(TokenizerConfig.save_path), "tokenizer_history.json")


def _write_history_json(history):
    """将训练历史写入磁盘 JSON 文件（每 epoch 结束后调用一次）。"""
    history_path = _history_path()
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _aggregate_losses_by_epoch(history):
    """从 batch 级 loss 记录聚合成 epoch 级均值（供绘图和 CSV 导出使用）。"""
    from collections import defaultdict

    epoch_losses = {
        "train_loss": defaultdict(list),
        "val_loss": defaultdict(list),
        "recon_loss": defaultdict(list),
    }

    for key in epoch_losses.keys():
        records = history.get(key, [])
        for item in records:
            epoch = item.get("epoch", 0)
            loss = float(item.get("loss", 0.0))
            epoch_losses[key][epoch].append(loss)

    aggregated = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
    }

    for key in aggregated.keys():
        epoch_dict = epoch_losses[key]
        for epoch in sorted(epoch_dict.keys()):
            losses = epoch_dict[epoch]
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            aggregated[key].append({
                "epoch": epoch,
                "loss": mean_loss,
            })

    return aggregated


def _export_losses_to_csv(history, output_csv):
    """将 epoch 级训练/验证/重建损失导出为 CSV 文件。"""
    import csv

    aggregated = _aggregate_losses_by_epoch(history)

    all_epochs = set()
    for key in aggregated.values():
        for item in key:
            all_epochs.add(item["epoch"])
    all_epochs = sorted(list(all_epochs))

    rows = []
    for epoch in all_epochs:
        row = {"epoch": epoch}
        for key in ["train_loss", "val_loss", "recon_loss"]:
            loss_val = None
            for item in aggregated[key]:
                if item["epoch"] == epoch:
                    loss_val = item["loss"]
                    break
            row[key] = loss_val
        rows.append(row)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "recon_loss"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Losses exported to CSV: {output_csv}")


def _plot_losses_from_json(history_path, output_png):
    """从历史 JSON 文件绘制 loss 曲线并保存为 PNG。"""
    if not os.path.exists(history_path):
        print(f"History json not found, skip plot: {history_path}")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    aggregated = _aggregate_losses_by_epoch(history)

    def _get_epoch_series(key):
        records = aggregated.get(key, [])
        epochs = [item.get("epoch", 0) for item in records]
        losses = [item.get("loss", 0.0) for item in records]
        return epochs, losses

    epochs_train, train_y = _get_epoch_series("train_loss")
    epochs_val, val_y = _get_epoch_series("val_loss")
    epochs_recon, recon_y = _get_epoch_series("recon_loss")

    plt.figure(figsize=(12, 6))
    if train_y:
        plt.plot(epochs_train, train_y, label="train_loss", linewidth=2.0, marker='o')
    if val_y:
        plt.plot(epochs_val, val_y, label="val_loss", linewidth=2.0, marker='s')
    if recon_y:
        plt.plot(epochs_recon, recon_y, label="recon_loss", linewidth=2.0, marker='^')
    plt.title("Tokenizer Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Loss curves saved: {output_png}")


def _choose_amp_dtype(device):
    """选择当前 GPU 支持的最佳 AMP 数据类型（bfloat16 > float16 > None）。"""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(amp_enabled, amp_dtype):
    """构建 AMP autocast 上下文，兼容新旧版 PyTorch API。"""
    if not amp_enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    except Exception:
        return torch.cuda.amp.autocast(dtype=amp_dtype)


class _CUDAPrefetcher:
    """CUDA 数据预取器：使用独立 stream 在 GPU 计算期间异步搬运下一个 batch。"""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._next = None
        self._iter = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch_data = next(self._iter)
        except StopIteration:
            self._next = None
            return
        data = batch_data[0]
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._next = data.to(self.device, non_blocking=True)
        else:
            self._next = data.to(self.device, non_blocking=True)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        # 等预取 stream 完成再交出数据，保证当前 stream 拿到正确结果。
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
        result = self._next
        self._preload()
        return result


def _build_optimizer(model, use_cuda):
    """构建 Adam 优化器，按 fused → foreach → 基础 优先级尝试。"""
    base_kwargs = {"lr": TokenizerConfig.learning_rate}
    use_foreach = bool(getattr(TokenizerConfig, "optimizer_use_foreach", True))
    use_fused = bool(use_cuda and getattr(TokenizerConfig, "optimizer_use_fused", True))

    candidates = []
    if use_fused:
        candidates.append({**base_kwargs, "fused": True})
    if use_foreach:
        candidates.append({**base_kwargs, "foreach": True})
    candidates.append(base_kwargs)

    tried = set()
    last_exc = None
    for kwargs in candidates:
        key = tuple(sorted(kwargs.items()))
        if key in tried:
            continue
        tried.add(key)
        try:
            return optim.Adam(model.parameters(), **kwargs), kwargs
        except (TypeError, RuntimeError, ValueError) as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    return optim.Adam(model.parameters(), **base_kwargs), base_kwargs


def train_tokenizer(train_dataloader, val_dataloader, tokenizer, device):
    """Tokenizer 主训练循环：VQ-VAE 训练 + 每 epoch 验证 + 最佳 checkpoint 保存。"""
    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    use_grad_scaler = bool(use_amp and amp_dtype == torch.float16)
    non_blocking = bool(device.type == "cuda")

    optimizer, optimizer_kwargs = _build_optimizer(tokenizer, use_cuda=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TokenizerConfig.epochs, eta_min=1e-5
    )
    # float16 需要 GradScaler 防止小梯度下溢；bfloat16 动态范围够大，不需要。
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    tokenizer.train()
    history = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
    }

    print("Training Tokenizer...")

    best_loss = float("inf")
    total_samples = len(getattr(train_dataloader, "dataset", []))
    steps_per_epoch = len(train_dataloader)
    batch_size = int(getattr(train_dataloader, "batch_size", 0) or 0)
    print(
        "Data stats - "
        f"samples: {total_samples}, steps/epoch: {steps_per_epoch}, "
        f"batch_size: {batch_size}"
    )

    use_prefetch = device.type == "cuda"

    for epoch in range(TokenizerConfig.epochs):
        tokenizer.train()
        train_epoch_loss = 0.0
        num_batches = 0

        if use_prefetch:
            train_iter = _CUDAPrefetcher(train_dataloader, device)
        else:
            train_iter = (batch[0].to(device, non_blocking=non_blocking) for batch in train_dataloader)

        for batch_idx, data in enumerate(
            tqdm(train_iter, desc=f"Epoch {epoch + 1}/{TokenizerConfig.epochs} [train]")
        ):
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(use_amp, amp_dtype):
                vq_loss, x_recon, perplexities, _ = tokenizer(data, return_all=True)
                recon_loss = F.mse_loss(x_recon, data)
                # 总损失 = 重建损失 + VQ 损失（含 commitment loss + 可能的 codebook loss）
                loss = recon_loss + vq_loss

            if not torch.isfinite(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skip.")
                optimizer.zero_grad(set_to_none=True)
                del data, vq_loss, x_recon, recon_loss, loss
                continue

            scaler.scale(loss).backward()
            del data, x_recon

            scaler.unscale_(optimizer)
            found_inf = torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), TokenizerConfig.grad_clip)
            if use_grad_scaler and not torch.isfinite(found_inf):
                print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skip optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                del vq_loss, recon_loss, loss
                continue
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss += loss.item()
            num_batches += 1

            history["train_loss"].append(
                {
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "loss": float(loss.item()),
                }
            )
            history["recon_loss"].append(
                {
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "loss": float(recon_loss.item()),
                }
            )
            del vq_loss, recon_loss, loss

        # -------- 验证阶段 --------
        tokenizer.eval()
        val_epoch_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch_idx, (data, _, _, _) in enumerate(
                tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{TokenizerConfig.epochs} [val]")
            ):
                data = data.to(device, non_blocking=non_blocking)
                with _autocast_context(use_amp, amp_dtype):
                    vq_loss, x_recon, _, _ = tokenizer(data, return_all=True)
                    recon_loss = F.mse_loss(x_recon, data)
                    val_loss = recon_loss + vq_loss

                if not torch.isfinite(val_loss):
                    del data, vq_loss, x_recon, recon_loss, val_loss
                    continue

                val_epoch_loss += val_loss.item()
                val_batches += 1
                history["val_loss"].append(
                    {
                        "epoch": epoch + 1,
                        "batch": val_batch_idx + 1,
                        "loss": float(val_loss.item()),
                    }
                )
                del data, vq_loss, x_recon, recon_loss, val_loss

        if device.type == "cuda":
            torch.cuda.empty_cache()

        scheduler.step()

        avg_train_loss = train_epoch_loss / max(num_batches, 1)
        avg_val_loss = val_epoch_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch + 1} summary - train_loss: {avg_train_loss:.6f}, "
            f"val_loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_tokenizer(tokenizer, history, epoch, avg_val_loss)

        _write_history_json(history)

    print("-" * 50)
    print(f"Tokenizer training done. best_loss={best_loss:.4f}")
    return tokenizer, history


def save_tokenizer(tokenizer, history, epoch, loss):
    """保存 tokenizer checkpoint 到磁盘。"""
    os.makedirs(os.path.dirname(TokenizerConfig.save_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": tokenizer.state_dict(),
            "config": export_tokenizer_config(),
            "history": history,
            "loss": loss,
        },
        TokenizerConfig.save_path,
    )

    print(f"Tokenizer saved: {TokenizerConfig.save_path}")


def load_tokenizer(device):
    """从磁盘加载预训练的 tokenizer 并设为 eval 模式。"""
    if not os.path.exists(TokenizerConfig.save_path):
        raise FileNotFoundError(f"Tokenizer not found: {TokenizerConfig.save_path}")

    checkpoint = torch.load(TokenizerConfig.save_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(config)).to(device)

    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)
    tokenizer.eval()

    print(f"Tokenizer loaded: {TokenizerConfig.save_path}")
    print(f"Trained epochs: {checkpoint['epoch'] + 1}, loss: {checkpoint['loss']:.4f}")

    return tokenizer


def evaluate_tokenizer(tokenizer, dataloader, device):
    """评估 tokenizer 重建质量（MSE + MAE）。"""
    tokenizer.eval()
    total_mse, total_mae, num_batches = 0.0, 0.0, 0
    amp_dtype = _choose_amp_dtype(device)
    use_amp = device.type == "cuda"
    non_blocking = bool(device.type == "cuda")

    print("\nEvaluating tokenizer reconstruction quality...")

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluate"):
            data = batch_data[0].to(device, non_blocking=non_blocking)
            with _autocast_context(use_amp, amp_dtype):
                vq_loss, x_recon, _, _ = tokenizer(data, return_all=True)

            total_mse += F.mse_loss(x_recon, data).item()
            total_mae += F.l1_loss(x_recon, data).item()
            num_batches += 1
            del data, vq_loss, x_recon

    if device.type == "cuda":
        torch.cuda.empty_cache()

    avg_mse = total_mse / max(num_batches, 1)
    avg_mae = total_mae / max(num_batches, 1)

    print(f"Reconstruction MSE: {avg_mse:.6f}")
    print(f"Reconstruction MAE: {avg_mae:.6f}")

    return avg_mse, avg_mae


def main():
    """Tokenizer 训练入口：加载数据 → 构建模型 → 训练 → 评估 → 导出曲线。"""
    seed = int(getattr(TokenizerConfig, "random_seed", DataConfig.random_seed))
    set_global_seed(seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    loader_overrides = _safe_loader_overrides_for_windows(device)
    tokenizer_batch_size = max(1, int(getattr(TokenizerConfig, "batch_size", 8)))
    print(f"Tokenizer batch_size: {tokenizer_batch_size}")
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=tokenizer_batch_size,
        include_demo=False,
        loader_overrides=loader_overrides,
    )
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)

    print(f"\nTokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}")

    tokenizer, history = train_tokenizer(train_loader, val_loader, tokenizer, device)
    evaluate_tokenizer(tokenizer, val_loader, device)

    history_path = _history_path()
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")

    output_png = os.path.join(os.path.dirname(TokenizerConfig.save_path), "tokenizer_loss_curves.png")
    output_csv = os.path.join(os.path.dirname(TokenizerConfig.save_path), "tokenizer_losses.csv")
    _plot_losses_from_json(history_path, output_png)
    _export_losses_to_csv(history, output_csv)


if __name__ == "__main__":
    main()
