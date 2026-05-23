# -*- coding: utf-8 -*-
"""Kronos-R 主模型训练入口。当前默认训练路径为 full-sequence teacher-forced pretrain。"""

import ctypes
import json
import logging
import os
import socket
import time
from contextlib import nullcontext
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config import DataConfig, EvaluationConfig, ModelConfig, PathConfig, TrainingConfig
from data_processor import collate_fn, collate_fn_v2, get_dataloaders, get_datasets, migrate_cache_to_memmap
from model.kronos_reasoning import KronosReasoningGPT
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from reproducibility import set_global_seed


# ──────────────────────────────────────────────
# 系统 / 环境工具
# ──────────────────────────────────────────────

def _get_available_ram_bytes():
    """跨平台获取当前可用 RAM 字节数。"""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None
    else:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return None
    return None


def _choose_amp_dtype(device):
    """选择当前 GPU 支持的最佳 AMP 数据类型（bfloat16 > float16）。"""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(amp_enabled, amp_dtype):
    """构建 AMP autocast 上下文，兼容新旧 PyTorch API。"""
    if not amp_enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    except Exception:
        return torch.cuda.amp.autocast(dtype=amp_dtype)


def _is_oom_error(exc):
    """判断异常是否为显存不足错误。"""
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _is_compile_error(exc):
    """判断异常是否来自 torch.compile（dynamo / inductor）。"""
    current = exc
    while current is not None:
        module_name = getattr(current.__class__, "__module__", "")
        if module_name.startswith("torch._dynamo") or module_name.startswith("torch._inductor"):
            return True
        text = str(current).lower()
        if (
            "torch._dynamo" in text
            or "torchinductor" in text
            or "backendcompilerfailed" in text
            or "torch.compile" in text
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_dist_ready():
    """分布式环境是否已初始化。"""
    return dist.is_available() and dist.is_initialized()


def _is_main_process(rank):
    """当前进程序列是否为 rank=0 的主进程。"""
    return rank == 0


def _find_free_port():
    """找到一个未被占用的 TCP 端口号。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _distributed_reduce_sum(value, device):
    """分布式环境下对 scalar 做 all_reduce SUM。"""
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if _is_dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


# ──────────────────────────────────────────────
# 模型工具
# ──────────────────────────────────────────────

def _unwrap_model(model):
    """剥掉 DDP / torch.compile 的包装，返回原始模块。"""
    raw = model
    if isinstance(raw, DDP):
        raw = raw.module
    raw = getattr(raw, "_orig_mod", raw)
    return raw


def _export_runtime_model_config(overrides=None):
    """Export runtime model config for DSA+GQA (no sector, no LinearAttention)."""
    keys = (
        "dim", "depth", "heads",
        "num_latent_tokens", "latent_reasoner_depth", "latent_cross_heads",
        "position_encoding", "rope_base", "alibi_decay_base",
        "max_len", "dropout",
        "vocab_size_coarse", "vocab_size_fine",
        "use_revin",
        "num_factor_tokens", "num_kv_heads", "dsa_windows",
    )
    config = {key: getattr(ModelConfig, key) for key in keys if hasattr(ModelConfig, key)}
    config.setdefault("num_kv_heads", 2)
    config.setdefault("dsa_windows", [None, 512, 512, None])
    if overrides:
        config.update(overrides)
    return config


def _clear_model_runtime_caches(model):
    """调用模型各模块的 clear_runtime_caches（如 KV-cache）。"""
    raw_model = _unwrap_model(model)
    for module in raw_model.modules():
        clear_fn = getattr(module, "clear_runtime_caches", None)
        if callable(clear_fn):
            clear_fn()


# ──────────────────────────────────────────────
# 路径解析
# ──────────────────────────────────────────────

def _normalize_path_token(path_value):
    """将路径字符串规范化为统一格式的 token。"""
    if path_value is None:
        return ""
    return str(path_value).strip().replace("\\", "/").rstrip("/")


def _is_default_checkpoint_dir(path_value):
    """路径 token 是否为默认 checkpoint 目录值。"""
    token = _normalize_path_token(path_value)
    return token in {"", ".", "checkpoints", "./checkpoints"}


def _is_default_base_model_path(path_value):
    """路径 token 是否为默认 base_model 路径值。"""
    token = _normalize_path_token(path_value)
    return token in {
        "", "base_model.pt",
        "checkpoints/base_model.pt", "./checkpoints/base_model.pt",
    }


def _abspath(path_value):
    """安全展开 ~ 并转为绝对路径。"""
    return os.path.abspath(os.path.expanduser(str(path_value)))


def _resolve_runtime_paths():
    """运行时路径决议：环境变量优先，其次 config，最后 fallback。"""
    env_checkpoint_dir = os.environ.get("KRONOS_CHECKPOINT_DIR", "").strip()
    env_base_model_path = os.environ.get("KRONOS_BASE_MODEL_PATH", "").strip()
    env_output_dir = os.environ.get("KRONOS_OUTPUT_DIR", "").strip()

    current_checkpoint_dir = str(getattr(PathConfig, "checkpoint_dir", "")).strip()
    current_save_dir = str(getattr(TrainingConfig, "save_dir", "")).strip()
    current_base_model_path = str(getattr(TrainingConfig, "base_model_path", "")).strip()

    if env_checkpoint_dir:
        checkpoint_dir = _abspath(env_checkpoint_dir)
    elif not _is_default_checkpoint_dir(current_checkpoint_dir):
        checkpoint_dir = _abspath(current_checkpoint_dir)
    elif not _is_default_checkpoint_dir(current_save_dir):
        checkpoint_dir = _abspath(current_save_dir)
    elif not _is_default_base_model_path(current_base_model_path):
        base_dir = os.path.dirname(_abspath(current_base_model_path))
        checkpoint_dir = base_dir if base_dir else _abspath("checkpoints")
    elif current_checkpoint_dir:
        checkpoint_dir = _abspath(current_checkpoint_dir)
    elif current_save_dir:
        checkpoint_dir = _abspath(current_save_dir)
    else:
        checkpoint_dir = _abspath("checkpoints")

    if env_base_model_path:
        base_model_path = _abspath(env_base_model_path)
    elif not _is_default_base_model_path(current_base_model_path):
        base_model_path = _abspath(current_base_model_path)
    else:
        base_name = os.path.basename(current_base_model_path) or "base_model.pt"
        base_model_path = os.path.join(checkpoint_dir, base_name)

    PathConfig.checkpoint_dir = checkpoint_dir
    TrainingConfig.save_dir = checkpoint_dir
    TrainingConfig.base_model_path = base_model_path

    if env_output_dir:
        resolved_output_dir = _abspath(env_output_dir)
        PathConfig.output_dir = resolved_output_dir
        EvaluationConfig.output_dir = resolved_output_dir


# ──────────────────────────────────────────────
# 日志工具
# ──────────────────────────────────────────────

def _build_training_logger(enabled=True):
    """创建训练专用 logger，写入 checkpoint_dir/train.log。"""
    if not enabled:
        return None, None

    os.makedirs(PathConfig.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(PathConfig.checkpoint_dir, "train.log")
    logger = logging.getLogger("kronos.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def _log_info(logger, message):
    """安全写 info 日志。"""
    if logger is not None:
        logger.info(message)


def _close_training_logger(logger):
    """关闭训练 logger 所有 handler。"""
    if logger is None:
        return
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


# ──────────────────────────────────────────────
# Checkpoint 保存
# ──────────────────────────────────────────────

def _save_epoch_base_model(model, tokenizer, epoch, loss,
                           model_config_overrides=None, logger=None):
    """每个 epoch 保存一份 basemode-{epoch+1}.pt。"""
    if _is_dist_ready() and (not _is_main_process(dist.get_rank())):
        return None

    os.makedirs(PathConfig.checkpoint_dir, exist_ok=True)
    path = os.path.join(PathConfig.checkpoint_dir, f"basemode-{epoch + 1}.pt")
    raw_model = _unwrap_model(model)
    model_config = _export_runtime_model_config(overrides=model_config_overrides)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "tokenizer_state_dict": tokenizer.state_dict(),
            "model_config": model_config,
            "loss": loss,
        },
        path,
    )
    print(f"Epoch base model saved: {path}")
    _log_info(logger, f"Epoch base model saved: {path}")
    return path


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, loss,
                    model_config_overrides=None, logger=None):
    """验证改善时保存完整 checkpoint + 更新 base_model.pt。"""
    if _is_dist_ready() and (not _is_main_process(dist.get_rank())):
        return

    os.makedirs(PathConfig.checkpoint_dir, exist_ok=True)
    path = os.path.join(
        PathConfig.checkpoint_dir,
        f"checkpoint_epoch{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
    )
    raw_model = _unwrap_model(model)
    model_config = _export_runtime_model_config(overrides=model_config_overrides)
    model_sd = raw_model.state_dict()
    tokenizer_sd = tokenizer.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_sd,
            "tokenizer_state_dict": tokenizer_sd,
            "model_config": model_config,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")
    _log_info(logger, f"Checkpoint saved: {path}")

    base_path = getattr(TrainingConfig, "base_model_path", "")
    if base_path:
        base_dir = os.path.dirname(base_path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_sd,
                "tokenizer_state_dict": tokenizer_sd,
                "model_config": model_config,
                "loss": loss,
            },
            base_path,
        )
        print(f"Base model updated: {base_path}")
        _log_info(logger, f"Base model updated: {base_path}")


# ──────────────────────────────────────────────
# Tokenizer & 模型加载
# ──────────────────────────────────────────────

def load_pretrained_tokenizer(device):
    """加载预训练 tokenizer，置为 eval 模式并冻结所有参数。"""
    tokenizer_path = TrainingConfig.tokenizer_path
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}\n"
            "Please run: python train_tokenizer.py"
        )

    checkpoint = torch.load(tokenizer_path, map_location=device, weights_only=False)
    tokenizer = HierarchicalQuantizer(**build_tokenizer_kwargs(checkpoint["config"])).to(device)
    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    print(f"Loaded pretrained tokenizer: {tokenizer_path}")
    print(f"Tokenizer epoch: {checkpoint['epoch'] + 1}, loss: {checkpoint['loss']:.4f}")
    return tokenizer


def latent_regularization_loss(latent_states):
    """Latent token 正则化：diversity（邻层差异）+ collapse（方差异常）惩罚。"""
    if latent_states is None or latent_states.shape[0] < 2:
        device = latent_states.device if latent_states is not None else "cpu"
        return torch.tensor(0.0, device=device)

    k_steps, batch_size, num_tokens, channels = latent_states.shape
    diff = latent_states[1:] - latent_states[:-1]
    diversity_loss = torch.exp(-diff.pow(2).sum(-1).sqrt().mean())

    latent_flat = latent_states.reshape(k_steps, batch_size * num_tokens, channels)
    collapse_loss = torch.exp(-latent_flat.var(dim=1).mean())

    return (
        TrainingConfig.diversity_weight * diversity_loss
        + TrainingConfig.collapse_weight * collapse_loss
    )


def _resolve_model_vocab_sizes(tokenizer):
    """从已加载的 BSQ tokenizer 推导 coarse / fine 词汇表大小。"""
    vocab_size_coarse = int(tokenizer.bsq_coarse.vocab_size())
    vocab_size_fine = int(tokenizer.bsq_fine.vocab_size()) if tokenizer.num_quantizers > 1 else vocab_size_coarse
    return {
        "vocab_size_coarse": vocab_size_coarse,
        "vocab_size_fine": vocab_size_fine,
    }


def _prepare_batch(features, time_features, tokenizer, device, non_blocking,
                   encoding_coarse=None, encoding_fine=None):
    """Tokenize and prepare batch for causal next-token prediction."""
    if encoding_coarse is not None and encoding_fine is not None:
        idx_coarse = encoding_coarse.to(device, non_blocking=non_blocking)
        idx_fine = encoding_fine.to(device, non_blocking=non_blocking)
    else:
        tokenizer_device = next(tokenizer.parameters()).device
        features_on_device = features.to(tokenizer_device, non_blocking=non_blocking)
        with torch.no_grad():
            idx_coarse, idx_fine = tokenizer.encode(features_on_device)
        del features_on_device
        if tokenizer_device != device:
            idx_coarse = idx_coarse.to(device, non_blocking=non_blocking)
            idx_fine = idx_fine.to(device, non_blocking=non_blocking)

    input_coarse = idx_coarse[:, :-1].long()
    input_fine = idx_fine[:, :-1].long()
    target_coarse = idx_coarse[:, 1:].long()
    target_fine = idx_fine[:, 1:].long()

    t_min = time_features["minute"][:, :-1].to(device, non_blocking=non_blocking).long()
    t_day = time_features["day"][:, :-1].to(device, non_blocking=non_blocking).long()
    t_month = time_features["month"][:, :-1].to(device, non_blocking=non_blocking).long()
    t_year = time_features["year"][:, :-1].to(device, non_blocking=non_blocking).long()
    target_day = time_features["day"][:, 1:].to(device, non_blocking=non_blocking).long()
    target_month = time_features["month"][:, 1:].to(device, non_blocking=non_blocking).long()

    return {
        "input_coarse": input_coarse,
        "input_fine": input_fine,
        "target_coarse": target_coarse,
        "target_fine": target_fine,
        "t_min": t_min,
        "t_day": t_day,
        "t_month": t_month,
        "t_year": t_year,
        "target_day": target_day,
        "target_month": target_month,
    }


def _unpack_batch(batch_data):
    """Unpack DataLoader 4-tuple → features, time_features, encodings (skip sector)."""
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 3:
        return batch_data[0], batch_data[2], batch_data[3]
    return batch_data[0], None, None


# ──────────────────────────────────────────────
# CUDA 预取器
# ──────────────────────────────────────────────

class _CUDAPrefetcher:
    """CUDA 数据预取器：使用独立 stream 在 GPU 计算期间异步搬运下一个 batch。"""

    def __init__(self, loader, prepare_fn, device):
        self.loader = loader
        self.prepare_fn = prepare_fn
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
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._next = self.prepare_fn(batch_data)
        else:
            self._next = self.prepare_fn(batch_data)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
        result = self._next
        self._preload()
        return result


# ──────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────

def _compute_sequence_cross_entropy(criterion, logits_coarse, logits_fine,
                                    target_coarse, target_fine):
    """计算序列最后一步的 coarse + fine 交叉熵（带越界检查）。"""
    coarse_vocab = int(logits_coarse.size(-1))
    fine_vocab = int(logits_fine.size(-1))
    coarse_min = int(target_coarse.min().item())
    coarse_max = int(target_coarse.max().item())
    fine_min = int(target_fine.min().item())
    fine_max = int(target_fine.max().item())

    if coarse_min < 0 or coarse_max >= coarse_vocab:
        raise ValueError(
            f"Coarse CE target out of range: min={coarse_min}, max={coarse_max}, vocab={coarse_vocab}"
        )
    if fine_min < 0 or fine_max >= fine_vocab:
        raise ValueError(
            f"Fine CE target out of range: min={fine_min}, max={fine_max}, vocab={fine_vocab}"
        )

    pred_loss = criterion(
        logits_coarse.reshape(-1, coarse_vocab),
        target_coarse.reshape(-1),
    )
    pred_loss = pred_loss + criterion(
        logits_fine.reshape(-1, fine_vocab),
        target_fine.reshape(-1),
    )
    return pred_loss


def base_one_step_loss(model, criterion,
                       input_coarse, input_fine,
                       target_coarse, target_fine,
                       t_min, t_day, t_month, t_year,
                       with_latent_loss=True):
    """Full-sequence teacher-forced next-token prediction loss (no sector)."""
    logits_coarse, logits_fine, latent_states = model(
        input_coarse, input_fine,
        t_min, t_day, t_month, t_year,
    )

    pred_loss = _compute_sequence_cross_entropy(
        criterion, logits_coarse, logits_fine,
        target_coarse, target_fine,
    )

    if with_latent_loss:
        latent_loss = latent_regularization_loss(latent_states)
    else:
        latent_loss = pred_loss * 0.0

    return pred_loss, latent_loss


# ──────────────────────────────────────────────
# 优化器 & 编译
# ──────────────────────────────────────────────

def _build_optimizer(model, use_cuda):
    """构建 AdamW 优化器，按 fused → foreach → 基础优先级尝试。"""
    base_kwargs = {
        "lr": TrainingConfig.learning_rate,
        "weight_decay": TrainingConfig.weight_decay,
    }
    use_foreach = bool(getattr(TrainingConfig, "optimizer_use_foreach", True))
    use_fused = bool(use_cuda and getattr(TrainingConfig, "optimizer_use_fused", True))

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
            optimizer = optim.AdamW(model.parameters(), **kwargs)
            return optimizer, kwargs
        except (TypeError, RuntimeError, ValueError) as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    optimizer = optim.AdamW(model.parameters(), **base_kwargs)
    return optimizer, base_kwargs


def _get_batch_and_accumulation_config():
    """根据 GPU 显存量自动调整 batch_size 和 accumulation_steps。"""
    base_batch = max(1, int(TrainingConfig.batch_size))
    base_accum = max(1, int(getattr(TrainingConfig, "accumulation_steps", 1)))

    if torch.cuda.is_available():
        try:
            _, total_vram = torch.cuda.mem_get_info()
            vram_gb = total_vram / (1024 ** 3)

            if vram_gb >= 22:
                auto_batch = 96
            elif vram_gb >= 14:
                auto_batch = 4
            elif vram_gb >= 10:
                auto_batch = 24
            elif vram_gb >= 7:
                auto_batch = 16
            else:
                auto_batch = 8

            if base_batch <= 5 and auto_batch > base_batch:
                batch_size = auto_batch
                effective_target = max(base_batch * base_accum, auto_batch)
                accumulation_steps = max(1, (effective_target + batch_size - 1) // batch_size)
            else:
                batch_size = base_batch
                accumulation_steps = base_accum
        except Exception:
            batch_size = base_batch
            accumulation_steps = base_accum
    else:
        batch_size = base_batch
        accumulation_steps = base_accum

    target_effective = batch_size * accumulation_steps
    return batch_size, accumulation_steps, target_effective


def _maybe_compile_model(model, distributed):
    """尝试 torch.compile，失败则回退到 eager 模式。"""
    if not bool(getattr(TrainingConfig, "enable_torch_compile", False)):
        return model, False
    if distributed:
        print("Skip torch.compile in distributed mode for stability.")
        return model, False
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this environment. Skip compile.")
        return model, False

    if bool(getattr(TrainingConfig, "torch_compile_suppress_errors", True)):
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
        except Exception:
            pass

    try:
        backend = str(getattr(TrainingConfig, "torch_compile_backend", "inductor")).strip()
        compile_kwargs = {
            "mode": str(getattr(TrainingConfig, "torch_compile_mode", "max-autotune")),
            "dynamic": bool(getattr(TrainingConfig, "torch_compile_dynamic", True)),
        }
        if backend:
            compile_kwargs["backend"] = backend
        compiled = torch.compile(model, **compile_kwargs)
        return compiled, True
    except Exception as exc:
        print(f"torch.compile failed ({exc}). Continue without compile.")
        return model, False


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_model(local_rank=0, world_size=1, distributed=False):
    """解析运行环境并进入训练主流程。"""
    _resolve_runtime_paths()

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rank = dist.get_rank() if _is_dist_ready() else 0
    is_main = _is_main_process(rank)
    train_logger = None
    log_path = None
    if is_main:
        train_logger, log_path = _build_training_logger(enabled=True)
        print(f"Checkpoint dir: {PathConfig.checkpoint_dir}")
        print(f"Base model path: {TrainingConfig.base_model_path}")
        _log_info(
            train_logger,
            f"Training start | distributed={distributed} | world_size={world_size} | local_rank={local_rank}",
        )
        _log_info(train_logger, f"Checkpoint dir: {PathConfig.checkpoint_dir}")
        _log_info(train_logger, f"Base model path: {TrainingConfig.base_model_path}")

    try:
        return _train_model_inner(
            device=device,
            rank=rank,
            is_main=is_main,
            train_logger=train_logger,
            log_path=log_path,
            local_rank=local_rank,
            world_size=world_size,
            distributed=distributed,
        )
    finally:
        _close_training_logger(train_logger)


def _train_model_inner(device, rank, is_main, train_logger, log_path,
                       local_rank, world_size, distributed):
    """训练核心流程：数据准备 → 建模 → 训练 → 验证 → 保存。"""
    base_seed = int(getattr(TrainingConfig, "random_seed", DataConfig.random_seed))
    seed_value = base_seed + (int(local_rank) if distributed else 0)
    set_global_seed(seed_value, deterministic=bool(getattr(TrainingConfig, "deterministic", True)))

    amp_enabled = device.type == "cuda"
    amp_dtype = _choose_amp_dtype(device)
    use_grad_scaler = bool(amp_enabled and amp_dtype == torch.float16)
    deterministic = bool(getattr(TrainingConfig, "deterministic", True))

    if amp_enabled:
        if bool(getattr(TrainingConfig, "use_tf32", True)) and (not deterministic):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.set_float32_matmul_precision("high")

    if is_main:
        print(f"Device: {device}")
        print(f"Seed (base/rank): {base_seed}/{seed_value}")
        print(f"AMP enabled: {amp_enabled}, dtype: {amp_dtype}")
        print(f"CPU cores: {os.cpu_count() or 1}")
        _log_info(train_logger, f"Device: {device}")
        _log_info(train_logger, f"Seed (base/rank): {base_seed}/{seed_value}")
        _log_info(train_logger, f"AMP enabled: {amp_enabled}, dtype: {amp_dtype}")
    available_ram = _get_available_ram_bytes()
    if is_main and available_ram is not None:
        print(f"Available RAM: {available_ram / (1024**3):.2f} GB")
    if is_main and amp_enabled:
        free_vram, total_vram = torch.cuda.mem_get_info(device=device)
        print(f"GPU: {torch.cuda.get_device_name(device)} | free/total VRAM: {free_vram / (1024**3):.2f}/{total_vram / (1024**3):.2f} GB")

    # ── 加载 tokenizer 并预计算编码 ──
    tokenizer = load_pretrained_tokenizer(device)
    use_memmap = bool(getattr(TrainingConfig, "use_memmap_cache", False))

    if use_memmap and is_main:
        memmap_cache_dir = str(getattr(TrainingConfig, "memmap_cache_dir", "dataset_cache"))
        if not os.path.isabs(memmap_cache_dir):
            memmap_cache_dir = os.path.join(os.path.dirname(PathConfig.checkpoint_dir), memmap_cache_dir)
        print(f"\n{'='*60}")
        print("Memmap cache mode enabled. Migrating .pt caches to memmap...")
        print(f"Target dir: {memmap_cache_dir}")
        print(f"{'='*60}")
        migrate_cache_to_memmap(cache_dir=memmap_cache_dir)

    train_dataset, val_dataset, demo_dataset = get_datasets(
        include_demo=False, use_memmap=use_memmap,
    )

    if is_main:
        print("\n" + "="*60)
        print("Precomputing tokenizer encodings for datasets...")
        print("="*60)

    try:
        if is_main:
            if use_memmap:
                train_dataset.precompute_encodings(tokenizer, device, force=False)
                val_dataset.precompute_encodings(tokenizer, device, force=False)
            else:
                train_dataset.precompute_encodings(tokenizer, device)
                val_dataset.precompute_encodings(tokenizer, device)

        if _is_dist_ready():
            dist.barrier()

        tokenizer = tokenizer.to("cpu")
        if is_main:
            print("Tokenizer offloaded to CPU to free GPU memory.")
    except Exception as exc:
        if is_main:
            print(f"Warning: Tokenizer encoding precompute failed ({exc}). Continue with dynamic encoding.")

    # ── 构建模型 (DSA + GQA, no sector, no RevIN, no factor tokens) ──
    runtime_model_config_overrides = {}
    runtime_model_config_overrides.update(_resolve_model_vocab_sizes(tokenizer))

    model = KronosReasoningGPT(
        dim=ModelConfig.dim, depth=ModelConfig.depth, heads=ModelConfig.heads,
        num_kv_heads=ModelConfig.num_kv_heads,
        dsa_windows=ModelConfig.dsa_windows,
        dropout=ModelConfig.dropout,
        vocab_size_coarse=runtime_model_config_overrides["vocab_size_coarse"],
        vocab_size_fine=runtime_model_config_overrides["vocab_size_fine"],
        num_latent_tokens=ModelConfig.num_latent_tokens,
        latent_reasoner_depth=ModelConfig.latent_reasoner_depth,
        latent_cross_heads=ModelConfig.latent_cross_heads,
        position_encoding=ModelConfig.position_encoding,
        rope_base=ModelConfig.rope_base,
        alibi_decay_base=ModelConfig.alibi_decay_base,
        max_len=ModelConfig.max_len,
        use_revin=ModelConfig.use_revin,
        num_factor_tokens=ModelConfig.num_factor_tokens,
    ).to(device)

    use_gradient_checkpointing = bool(getattr(TrainingConfig, "use_gradient_checkpointing", True))
    if use_gradient_checkpointing:
        model.enable_gradient_checkpointing(True)
        if is_main:
            print("Gradient checkpointing: ENABLED")

    if is_main:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            "Resolved token vocab sizes: "
            f"coarse={runtime_model_config_overrides['vocab_size_coarse']}, "
            f"fine={runtime_model_config_overrides['vocab_size_fine']}"
        )

    # ── 构建 DataLoader / 优化器 / Scheduler ──
    criterion = nn.CrossEntropyLoss()
    tuned_batch_size, accumulation_steps, target_effective = _get_batch_and_accumulation_config()
    effective_batch_size = tuned_batch_size * accumulation_steps

    train_loader, val_loader, _demo_loader, _ = get_dataloaders(
        train_dataset=train_dataset, val_dataset=val_dataset,
        demo_dataset=demo_dataset, batch_size=tuned_batch_size,
        include_demo=False, distributed=distributed,
        rank=rank, world_size=world_size,
        collate_fn_override=collate_fn_v2 if use_memmap else None,
    )

    model, compiled = _maybe_compile_model(model, distributed=distributed)
    if distributed and world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer, optimizer_kwargs = _build_optimizer(model, use_cuda=amp_enabled)
    schedule_by_updates = bool(getattr(TrainingConfig, "scheduler_by_updates", False))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(getattr(TrainingConfig, "scheduler_T_max", TrainingConfig.epochs))),
        eta_min=TrainingConfig.scheduler_eta_min,
    )
    # float16 需要 GradScaler 防止小梯度下溢；bfloat16 动态范围够大，不需要。
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    batch_log_interval = max(1, int(getattr(TrainingConfig, "train_log_every_n_batches", 1)))

    if is_main:
        print("\nStart training Kronos reasoning GPT...")
        print(f"Epochs: {TrainingConfig.epochs}")
        print("Training target: full-sequence teacher-forced base pretrain")
        print(f"Batch size: {tuned_batch_size}, accumulation: {accumulation_steps}, effective: {effective_batch_size}")
        print(f"Optimizer kwargs: {optimizer_kwargs}")
        print(f"torch.compile enabled: {compiled}")
        print(f"CUDA prefetch enabled: {bool(getattr(TrainingConfig, 'use_cuda_prefetch', True))}")
        print(f"Learning rate: {TrainingConfig.learning_rate}")
        print(f"Scheduler by updates: {schedule_by_updates}")
        print("-" * 50)
        _log_info(
            train_logger,
            f"Train setup | epochs={TrainingConfig.epochs} | target=full-sequence-base-pretrain | "
            f"batch_size={tuned_batch_size} | accumulation={accumulation_steps} | "
            f"lr={TrainingConfig.learning_rate} | compile={compiled}",
        )

    # ── 训练循环 ──
    history = {
        "train_loss": [], "val_loss": [], "val_1step_ce": [],
        "train_pred_loss": [], "train_latent_loss": [], "lr": [],
        "batch_size": tuned_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "max_train_updates": max(0, int(getattr(TrainingConfig, "max_train_updates", 0))),
        "optimizer_updates_completed": 0,
        "scheduler_by_updates": schedule_by_updates,
        "scheduler_T_max": int(getattr(TrainingConfig, "scheduler_T_max", TrainingConfig.epochs)),
    }
    best_val_loss = float("inf")
    early_stop_patience = max(1, int(getattr(TrainingConfig, "early_stop_patience", 5)))
    patience_counter = 0
    transfer_non_blocking = bool(getattr(TrainingConfig, "pin_memory", True))
    total_epochs = TrainingConfig.epochs
    max_train_updates = max(0, int(getattr(TrainingConfig, "max_train_updates", 0)))
    optimizer_updates_completed = 0
    stop_after_validation = False

    def _write_training_history_snapshot():
        if not is_main:
            return
        os.makedirs(PathConfig.checkpoint_dir, exist_ok=True)
        history_path = os.path.join(PathConfig.checkpoint_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    def _make_prepare_fn(is_train):
        """构建 batch 预处理闭包（供 CUDAPrefetcher / 常规迭代复用）。"""
        def prepare(batch_data):
            features, time_features, encodings = _unpack_batch(batch_data)
            result = _prepare_batch(
                features, time_features, tokenizer,
                device, transfer_non_blocking,
                encoding_coarse=encodings["idx_coarse"] if encodings else None,
                encoding_fine=encodings["idx_fine"] if encodings else None,
            )
            return result, is_train
        return prepare

    epoch = 0
    while epoch < total_epochs:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if is_main:
            _log_info(train_logger, f"EPOCH {epoch + 1}/{total_epochs} | START")
        train_sampler = getattr(train_loader, "sampler", None)
        if distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        total_pred = 0.0
        total_latent = 0.0
        num_batches = 0
        train_compile_exc = None

        use_prefetch = (
            device.type == "cuda"
            and not distributed
            and bool(getattr(TrainingConfig, "use_cuda_prefetch", True))
        )
        if use_prefetch:
            train_iter = _CUDAPrefetcher(train_loader, _make_prepare_fn(True), device)
        else:
            train_iter = train_loader

        phase_name = "BaseModel"
        pbar = tqdm(
            train_iter,
            desc=f"Epoch {epoch + 1}/{total_epochs} [{phase_name}]",
            disable=not is_main,
        )
        reached_update_budget = False
        for batch_idx, batch_data in enumerate(pbar, start=1):
            try:
                if use_prefetch:
                    batch_tensors, _ = batch_data
                else:
                    features, time_features, encodings = _unpack_batch(batch_data)
                    batch_tensors = _prepare_batch(
                        features, time_features, tokenizer,
                        device, transfer_non_blocking,
                        encoding_coarse=encodings["idx_coarse"] if encodings else None,
                        encoding_fine=encodings["idx_fine"] if encodings else None,
                    )
                    del features, time_features, encodings

                with _autocast_context(amp_enabled, amp_dtype):
                    pred_loss, latent_loss = base_one_step_loss(
                        model=model, criterion=criterion,
                        input_coarse=batch_tensors["input_coarse"],
                        input_fine=batch_tensors["input_fine"],
                        target_coarse=batch_tensors["target_coarse"],
                        target_fine=batch_tensors["target_fine"],
                        t_min=batch_tensors["t_min"],
                        t_day=batch_tensors["t_day"],
                        t_month=batch_tensors["t_month"],
                        t_year=batch_tensors["t_year"],
                        with_latent_loss=True,
                    )
                    step_loss = pred_loss + latent_loss

                del batch_tensors

                if not torch.isfinite(step_loss):
                    print(f"Warning: NaN/Inf loss at step {batch_idx}, skip.")
                    optimizer.zero_grad(set_to_none=True)
                    del pred_loss, latent_loss, step_loss
                    continue

                total_loss += step_loss.item()
                total_pred += pred_loss.item()
                total_latent += latent_loss.item()
                num_batches += 1

                step_loss = step_loss / accumulation_steps
                scaler.scale(step_loss).backward()

                should_step = batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    found_inf = torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                    if use_grad_scaler and not torch.isfinite(found_inf):
                        print(f"Warning: NaN/Inf gradients at step {batch_idx}, skip optimizer step.")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        del pred_loss, latent_loss, step_loss
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_updates_completed += 1
                    history["optimizer_updates_completed"] = int(optimizer_updates_completed)
                    if schedule_by_updates:
                        scheduler.step()
                    if max_train_updates > 0 and optimizer_updates_completed >= max_train_updates:
                        reached_update_budget = True

                if is_main:
                    avg_train_loss = total_loss / max(num_batches, 1)
                    pbar.set_postfix({"TRAIN-LOSS": avg_train_loss})
                    if batch_idx % batch_log_interval == 0 or batch_idx == len(train_loader):
                        _log_info(
                            train_logger,
                            f"EPOCH {epoch + 1}/{total_epochs} | BATCH {batch_idx}/{len(train_loader)} | "
                            f"PHASE={phase_name} | LOSS(step={step_loss.item():.6f}, avg={avg_train_loss:.6f}) | "
                            f"PRED={pred_loss.item():.6f} | LATENT={latent_loss.item():.6f} | "
                            f"LR={optimizer.param_groups[0]['lr']:.3e}",
                        )
                del pred_loss, latent_loss, step_loss
                if reached_update_budget:
                    break
            except FloatingPointError as exc:
                print(f"Warning: {exc} at step {batch_idx}, skip.")
                optimizer.zero_grad(set_to_none=True)
                continue
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    raise
                if compiled and _is_compile_error(exc):
                    train_compile_exc = exc
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

        pbar.close()

        if train_compile_exc is not None:
            if (not compiled) or (not bool(getattr(TrainingConfig, "torch_compile_runtime_fallback", True))):
                raise train_compile_exc
            model = _unwrap_model(model)
            compiled = False
            optimizer.zero_grad(set_to_none=True)
            _clear_model_runtime_caches(model)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if is_main:
                print(f"torch.compile runtime failure detected; fallback to eager mode. Retry epoch {epoch + 1}.")
            continue

        if reached_update_budget:
            stop_after_validation = True

        if not schedule_by_updates:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss_sum = _distributed_reduce_sum(total_loss, device=device)
        train_pred_sum = _distributed_reduce_sum(total_pred, device=device)
        train_latent_sum = _distributed_reduce_sum(total_latent, device=device)
        train_batch_sum = _distributed_reduce_sum(num_batches, device=device)
        train_denom = max(train_batch_sum, 1.0)

        history["train_loss"].append(train_loss_sum / train_denom)
        history["train_pred_loss"].append(train_pred_sum / train_denom)
        history["train_latent_loss"].append(train_latent_sum / train_denom)
        history["lr"].append(current_lr)

        model.eval()
        val_sampler = getattr(val_loader, "sampler", None)
        if distributed and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)
        val_pred_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                features, time_features, encodings = _unpack_batch(batch_data)

                batch_tensors = _prepare_batch(
                    features, time_features, tokenizer,
                    device, transfer_non_blocking,
                    encoding_coarse=encodings["idx_coarse"] if encodings else None,
                    encoding_fine=encodings["idx_fine"] if encodings else None,
                )
                del features, time_features, encodings

                with _autocast_context(amp_enabled, amp_dtype):
                    pred_loss, _ = base_one_step_loss(
                        model=model, criterion=criterion,
                        input_coarse=batch_tensors["input_coarse"],
                        input_fine=batch_tensors["input_fine"],
                        target_coarse=batch_tensors["target_coarse"],
                        target_fine=batch_tensors["target_fine"],
                        t_min=batch_tensors["t_min"],
                        t_day=batch_tensors["t_day"],
                        t_month=batch_tensors["t_month"],
                        t_year=batch_tensors["t_year"],
                        with_latent_loss=False,
                    )
                val_pred_total += pred_loss.item()
                val_batches += 1
                del batch_tensors
                del pred_loss

        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_pred_sum = _distributed_reduce_sum(val_pred_total, device=device)
        val_batch_sum = _distributed_reduce_sum(val_batches, device=device)
        avg_val_loss = val_pred_sum / max(val_batch_sum, 1.0)
        history["val_loss"].append(avg_val_loss)
        history["val_1step_ce"].append(avg_val_loss)

        if is_main:
            print(
                f"Epoch {epoch + 1} [{phase_name}] - train: {history['train_loss'][-1]:.4f}, "
                f"val_seq_ce: {avg_val_loss:.4f}, lr: {current_lr:.2e}"
            )
            print(
                f"HPO_METRIC epoch={epoch + 1} val_seq_ce={avg_val_loss:.8f} "
                f"val_1step={avg_val_loss:.8f} "
                f"lr={current_lr:.8e} train={history['train_loss'][-1]:.8f} "
                f"updates={optimizer_updates_completed}"
            )
            _log_info(
                train_logger,
                f"EPOCH {epoch + 1}/{total_epochs} | PHASE={phase_name} | "
                f"TRAIN-LOSS={history['train_loss'][-1]:.6f} | "
                f"VAL-1STEP={avg_val_loss:.6f} | "
                f"TRAIN-PRED={history['train_pred_loss'][-1]:.6f} | "
                f"TRAIN-LATENT={history['train_latent_loss'][-1]:.6f} | "
                f"LR={current_lr:.3e}",
            )

        _save_epoch_base_model(
            model, tokenizer, epoch, avg_val_loss,
            model_config_overrides=runtime_model_config_overrides, logger=train_logger,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(
                model, tokenizer, optimizer, scheduler, epoch, avg_val_loss,
                model_config_overrides=runtime_model_config_overrides, logger=train_logger,
            )
        else:
            patience_counter += 1

        _clear_model_runtime_caches(model)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _write_training_history_snapshot()
        if stop_after_validation:
            if is_main:
                print(f"Reached max_train_updates={max_train_updates}. Stop training after epoch {epoch + 1}.")
                _log_info(train_logger, f"Reached max_train_updates={max_train_updates}. Stop training.")
            break
        if patience_counter >= early_stop_patience:
            if is_main:
                print(f"Early stop at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs).")
                _log_info(train_logger, f"Early stop at epoch {epoch + 1}.")
            break
        epoch += 1

    if is_main:
        _write_training_history_snapshot()
        print("-" * 50)
        print(f"Training done. Best val sequence CE: {best_val_loss:.4f}")
        _log_info(train_logger, f"Training done. Best val sequence CE: {best_val_loss:.6f}")
        if log_path:
            print(f"Training log saved: {log_path}")
    return model, tokenizer, demo_dataset


# ──────────────────────────────────────────────
# 分布式训练入口
# ──────────────────────────────────────────────

def _setup_distributed(local_rank, world_size):
    """初始化分布式进程组。"""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if torch.cuda.is_available() and dist.is_nccl_available():
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)


def _cleanup_distributed():
    """销毁分布式进程组。"""
    if _is_dist_ready():
        dist.destroy_process_group()


def _distributed_worker(local_rank, world_size):
    """分布式 worker 入口：初始化 → train_model → 清理。"""
    try:
        _setup_distributed(local_rank=local_rank, world_size=world_size)
        train_model(local_rank=local_rank, world_size=world_size, distributed=True)
    finally:
        _cleanup_distributed()


def _run_training_auto():
    """根据 GPU 数量自动选择单进程或 DDP 训练入口。"""
    auto_multi_gpu = bool(getattr(TrainingConfig, "auto_multi_gpu", True))
    max_gpus = int(getattr(TrainingConfig, "max_auto_gpus", 0))
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if auto_multi_gpu and gpu_count > 1:
        world_size = gpu_count if max_gpus <= 0 else min(gpu_count, max_gpus)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        print(f"Auto multi-GPU enabled. Launching {world_size} processes.")
        mp.spawn(_distributed_worker, nprocs=world_size, args=(world_size,), join=True)
        return

    train_model()


if __name__ == "__main__":
    _run_training_auto()
