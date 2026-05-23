"""训练数据集与 DataLoader。"""

import ctypes
import glob
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import DataConfig, PathConfig, TrainingConfig
from reproducibility import seed_worker

# Memmap utilities for handling large feature arrays efficiently.
_MEMORY_MAPPED_FILES = set()


def _register_memmap(path):
    _MEMORY_MAPPED_FILES.add(str(Path(path).resolve()))


def _cleanup_memmap_files():
    for path in list(_MEMORY_MAPPED_FILES):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        finally:
            _MEMORY_MAPPED_FILES.discard(path)


def load_memmap(path, dtype=np.float32):
    """Load an existing memmap file for read-only access."""
    mmap = np.memmap(path, dtype=dtype, mode='r')
    return mmap

warnings.filterwarnings("ignore")

TIME_KEYS = ("minute", "day", "month", "year")


class NpyMemmapBackend:
    """单个 .npy / .npz 文件的 mmap 后端，零 RAM 占用，按需 page-in。

    - 对每个文件调用 load_memmap / np.load(path, mmap_mode="r")
    - 返回的数组是 OS 级 mmap：不会立即占用物理内存
    - 多 DataLoader worker 共享同一份 kernel page cache
    """

    def __init__(self, path: str, dtype_override=None):
        self._path = path
        self._dtype_override = np.dtype(dtype_override) if dtype_override is not None else None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Memmap file not found: {path}")
        try:
            self._arr = load_memmap(path)
            self._shape = self._arr.shape
            self._dtype = self._arr.dtype
        except Exception:
            header = np.load(path, mmap_mode=None)
            self._shape = header.shape
            self._dtype = header.dtype
            self._arr = None

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype_override or self._dtype

    @property
    def num_samples(self) -> int:
        return int(self._shape[0])

    def open(self):
        if self._arr is None:
            self._arr = load_memmap(self._path)
        arr = self._arr
        if self._dtype_override is not None:
            arr = arr.view(self._dtype_override)
        return arr


class MemmapCacheWriter:
    """从现有 AShareDataset 导出分文件 mmap 格式。"""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_from_dataset(self, dataset, mode: str):
        """从 AShareDataset 实例导出所有数据到独立 memmap 文件。"""
        features = dataset._features_tensor.cpu().numpy()
        num_samples, seq_len, num_features = features.shape

        print(f"Writing {mode} memmap cache: {num_samples} samples → {self._output_dir}")

        _save_memmap(os.path.join(self._output_dir, "encoded_coarse.npy"),
                     dataset._encoded_indices_coarse.cpu().numpy() if dataset._encoded_indices_coarse is not None else np.zeros((num_samples, seq_len), dtype=np.int64))
        _save_memmap(os.path.join(self._output_dir, "encoded_fine.npy"),
                     dataset._encoded_indices_fine.cpu().numpy() if dataset._encoded_indices_fine is not None else np.zeros((num_samples, seq_len), dtype=np.int64))
        _save_memmap(os.path.join(self._output_dir, "sector_ids.npy"),
                     dataset._sector_ids_tensor.cpu().numpy())
        for key in TIME_KEYS:
            _save_memmap(os.path.join(self._output_dir, f"time_{key}.npy"),
                         dataset._time_tensors[key].cpu().numpy())
        _save_memmap(os.path.join(self._output_dir, "features.npy"), features)

        seq_stats = dataset.seq_stats
        if isinstance(seq_stats, list):
            cleaned = []
            for s in seq_stats:
                entry = {}
                if isinstance(s.get("mean"), np.ndarray):
                    entry["mean"] = s["mean"].tolist()
                else:
                    entry["mean"] = s.get("mean", [])
                if isinstance(s.get("std"), np.ndarray):
                    entry["std"] = s["std"].tolist()
                else:
                    entry["std"] = s.get("std", [])
                cleaned.append(entry)
            with open(os.path.join(self._output_dir, "seq_stats.json"), "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False)

        meta = {
            "mode": mode,
            "num_samples": num_samples,
            "seq_len": seq_len,
            "num_features": num_features,
            "data_cache_signature": dataset._cache_signature(),
            "encoding_cache_signature": dataset._encoding_cache_signature(),
        }
        with open(os.path.join(self._output_dir, "_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"Memmap cache written: {self._output_dir}")


class MemmapArrayDataset(Dataset):
    """基于 .npy mmap 文件的 Dataset，完全避免 torch.save/pickle。

    用法:
        ds = MemmapArrayDataset(cache_dir="dataset_cache/train")
        # ds[idx] 返回 dict，直接可用于 collate
    """

    def __init__(self, cache_dir: str, with_features: bool = False):
        self._cache_dir = cache_dir
        self._with_features = with_features

        meta_path = os.path.join(cache_dir, "_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        self.seq_len = self._meta["seq_len"]
        self.num_samples = self._meta["num_samples"]
        self.mode = self._meta.get("mode", "unknown")

        self._coarse = NpyMemmapBackend(os.path.join(cache_dir, "encoded_coarse.npy"))
        self._fine = NpyMemmapBackend(os.path.join(cache_dir, "encoded_fine.npy"))
        self._sector_ids = NpyMemmapBackend(os.path.join(cache_dir, "sector_ids.npy"))
        self._time = {
            key: NpyMemmapBackend(
                os.path.join(cache_dir, f"time_{key}.npy"),
                dtype_override="int64",
            )
            for key in TIME_KEYS
        }

        self._features_backend = None
        if with_features:
            feat_path = os.path.join(cache_dir, "features.npy")
            if os.path.exists(feat_path):
                self._features_backend = NpyMemmapBackend(feat_path)

        self._coarse_arr = None
        self._fine_arr = None
        self._sector_arr = None
        self._time_arr = {}
        self._features_arr = None

    def _ensure_open(self):
        if self._coarse_arr is None:
            self._coarse_arr = self._coarse.open()
            self._fine_arr = self._fine.open()
            self._sector_arr = self._sector_ids.open()
            for key in TIME_KEYS:
                self._time_arr[key] = self._time[key].open()
            if self._features_backend is not None:
                self._features_arr = self._features_backend.open()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        self._ensure_open()
        result = {
            "idx_coarse": torch.from_numpy(self._coarse_arr[idx].astype(np.int64)),
            "idx_fine": torch.from_numpy(self._fine_arr[idx].astype(np.int64)),
            "sector_ids": torch.tensor(int(self._sector_arr[idx]), dtype=torch.long),
            "t_min": torch.from_numpy(self._time_arr["minute"][idx].astype(np.int64)),
            "t_day": torch.from_numpy(self._time_arr["day"][idx].astype(np.int64)),
            "t_month": torch.from_numpy(self._time_arr["month"][idx].astype(np.int64)),
            "t_year": torch.from_numpy(self._time_arr["year"][idx].astype(np.int64)),
        }
        if self._with_features and self._features_arr is not None:
            result["features"] = torch.from_numpy(self._features_arr[idx].astype(np.float32))
        return result

    def precompute_encodings(self, tokenizer, device, force=False, batch_size=64):
        """Batched 预计算 tokenizer 编码，写入 memmap 文件。"""
        if not force and os.path.exists(os.path.join(self._cache_dir, "encoded_coarse.npy")):
            coarse = load_memmap(os.path.join(self._cache_dir, "encoded_coarse.npy"))
            fine = load_memmap(os.path.join(self._cache_dir, "encoded_fine.npy"))
            if coarse.shape[0] == self.num_samples and fine.shape[0] == self.num_samples:
                print(f"Encodings already cached for {self.mode}, skip.")
                self._coarse_arr = coarse
                self._fine_arr = fine
                return

        if self.num_samples == 0:
            _save_memmap(os.path.join(self._cache_dir, "encoded_coarse.npy"),
                         np.empty((0, self.seq_len), dtype=np.int64))
            _save_memmap(os.path.join(self._cache_dir, "encoded_fine.npy"),
                         np.empty((0, self.seq_len), dtype=np.int64))
            return

        print(
            f"Precomputing encodings for {self.mode} "
            f"({self.num_samples} samples, batch_size={batch_size})..."
        )

        coarse_path = os.path.join(self._cache_dir, "encoded_coarse.npy")
        fine_path = os.path.join(self._cache_dir, "encoded_fine.npy")
        coarse_mmap = np.memmap(coarse_path, dtype=np.int64, mode='w+',
                                shape=(self.num_samples, self.seq_len))
        fine_mmap = np.memmap(fine_path, dtype=np.int64, mode='w+',
                              shape=(self.num_samples, self.seq_len))
        _register_memmap(coarse_path)
        _register_memmap(fine_path)

        features_path = os.path.join(self._cache_dir, "features.npy")
        features_mmap = load_memmap(features_path)

        tokenizer.eval()
        with torch.no_grad():
            for start in tqdm(range(0, self.num_samples, batch_size),
                              desc=f"Encoding {self.mode}"):
                end = min(start + batch_size, self.num_samples)
                batch = torch.from_numpy(features_mmap[start:end].astype(np.float32)).to(device)
                idx_c, idx_f = tokenizer.encode(batch)
                coarse_mmap[start:end] = idx_c.cpu().numpy()
                fine_mmap[start:end] = idx_f.cpu().numpy()

        coarse_mmap.flush()
        fine_mmap.flush()
        self._coarse_arr = load_memmap(coarse_path)
        self._fine_arr = load_memmap(fine_path)
        print(f"Batched precompute done: {self.mode}")


def _save_memmap(path: str, arr: np.ndarray):
    mmap = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    mmap[:] = arr
    mmap.flush()
    _register_memmap(path)


def collate_fn_v2(batch):
    """精简版 collate：仅搬运 encodings + time + sector，不搬运 features。"""
    idx_coarse = torch.stack([item["idx_coarse"] for item in batch], dim=0)
    idx_fine = torch.stack([item["idx_fine"] for item in batch], dim=0)
    sector_ids = torch.stack([item["sector_ids"] for item in batch], dim=0)
    time_features = {
        "minute": torch.stack([item["t_min"] for item in batch], dim=0),
        "day": torch.stack([item["t_day"] for item in batch], dim=0),
        "month": torch.stack([item["t_month"] for item in batch], dim=0),
        "year": torch.stack([item["t_year"] for item in batch], dim=0),
    }
    features = None
    if "features" in batch[0]:
        features = torch.stack([item["features"] for item in batch], dim=0)
    encodings = {"idx_coarse": idx_coarse, "idx_fine": idx_fine}
    return features, sector_ids, time_features, encodings


def migrate_cache_to_memmap(cache_dir=None):
    """将现有 .pt cache 文件迁移为 memmap 分文件格式。

    自动检测 dataset_cache/ 下的所有 .pt 文件，逐一迁移。
    迁移完成后可通过 MemmapArrayDataset 直接加载。
    """
    if cache_dir is None:
        cache_dir = str(Path(PathConfig.checkpoint_dir).parent / "dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)

    checkpoint_dir = PathConfig.checkpoint_dir
    pt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "dataset_cache*.pt")))
    if not pt_files:
        print("No dataset_cache*.pt files found in checkpoint directory.")
        return False

    for pt_path in pt_files:
        print(f"\nMigrating: {pt_path}")
        try:
            cached_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  Failed to load {pt_path}: {exc}")
            continue

        features = cached_data.get("features")
        if features is None:
            print("  No 'features' key found, skip.")
            continue

        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(features, list):
            features = np.stack([np.asarray(f, dtype=np.float32) for f in features], axis=0)

        num_samples, seq_len, num_features = features.shape

        sig = cached_data.get("_data_cache_signature", {})
        mode = sig.get("mode", Path(pt_path).stem.split("_")[0] if "_" in Path(pt_path).stem else "unknown")
        encoding_sig = cached_data.get("_encoding_cache_signature", {})

        mode_dir = os.path.join(cache_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        _save_memmap(os.path.join(mode_dir, "features.npy"), features.astype(np.float32))

        for key in TIME_KEYS:
            time_data = None
            time_features_dict = cached_data.get("time_features", {})
            if isinstance(time_features_dict, dict) and key in time_features_dict:
                time_data = time_features_dict[key]
            if time_data is not None:
                if isinstance(time_data, torch.Tensor):
                    time_data = time_data.numpy()
                _save_memmap(os.path.join(mode_dir, f"time_{key}.npy"),
                             time_data.astype(np.int16))
            else:
                _save_memmap(os.path.join(mode_dir, f"time_{key}.npy"),
                             np.zeros((num_samples, seq_len), dtype=np.int16))

        sector_data = cached_data.get("sector_ids")
        if sector_data is not None:
            if isinstance(sector_data, torch.Tensor):
                sector_data = sector_data.numpy()
            _save_memmap(os.path.join(mode_dir, "sector_ids.npy"),
                         sector_data.astype(np.int64).ravel()[:num_samples])
        else:
            _save_memmap(os.path.join(mode_dir, "sector_ids.npy"),
                         np.zeros(num_samples, dtype=np.int64))

        encoded_coarse = cached_data.get("encoded_indices_coarse")
        encoded_fine = cached_data.get("encoded_indices_fine")
        if encoded_coarse is not None:
            if isinstance(encoded_coarse, torch.Tensor):
                encoded_coarse = encoded_coarse.numpy()
            _save_memmap(os.path.join(mode_dir, "encoded_coarse.npy"),
                         encoded_coarse.astype(np.int64))
        else:
            _save_memmap(os.path.join(mode_dir, "encoded_coarse.npy"),
                         np.zeros((num_samples, seq_len), dtype=np.int64))
        if encoded_fine is not None:
            if isinstance(encoded_fine, torch.Tensor):
                encoded_fine = encoded_fine.numpy()
            _save_memmap(os.path.join(mode_dir, "encoded_fine.npy"),
                         encoded_fine.astype(np.int64))
        else:
            _save_memmap(os.path.join(mode_dir, "encoded_fine.npy"),
                         np.zeros((num_samples, seq_len), dtype=np.int64))

        seq_stats = cached_data.get("seq_stats", [])
        if isinstance(seq_stats, list):
            cleaned = []
            for s in seq_stats:
                entry = {}
                if isinstance(s.get("mean"), np.ndarray):
                    entry["mean"] = s["mean"].tolist()
                else:
                    entry["mean"] = s.get("mean", [])
                if isinstance(s.get("std"), np.ndarray):
                    entry["std"] = s["std"].tolist()
                else:
                    entry["std"] = s.get("std", [])
                cleaned.append(entry)
            with open(os.path.join(mode_dir, "seq_stats.json"), "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False)

        meta = {
            "mode": mode,
            "num_samples": num_samples,
            "seq_len": seq_len,
            "num_features": num_features,
            "data_cache_signature": sig,
            "encoding_cache_signature": encoding_sig,
        }
        with open(os.path.join(mode_dir, "_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  → {mode_dir} ({num_samples} samples)")

    print(f"\nMigration complete. Cache dir: {cache_dir}")
    return True


def get_memmap_datasets(cache_base_dir=None, include_demo=True):
    """构建基于 memmap 的 train/val/demo 数据集。"""
    if cache_base_dir is None:
        cache_base_dir = os.path.join(os.path.dirname(PathConfig.checkpoint_dir), "dataset_cache")
    train_ds = MemmapArrayDataset(os.path.join(cache_base_dir, "train"))
    val_ds = MemmapArrayDataset(os.path.join(cache_base_dir, "val"))
    demo_ds = MemmapArrayDataset(os.path.join(cache_base_dir, "demo")) if include_demo else None
    return train_ds, val_ds, demo_ds


_TOKENIZER_FINGERPRINT_CACHE = {}


def _stable_payload_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _dataset_source_fingerprint(data_dir):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    manifest = []
    for path in csv_files:
        record = _stat_fingerprint(path)
        if record is not None:
            manifest.append(record)
    resolved_dir = os.path.abspath(data_dir)
    if os.name == "nt":
        resolved_dir = resolved_dir.lower()
    return {
        "data_dir": resolved_dir,
        "num_files": len(manifest),
        "manifest_sha256": _stable_payload_hash(manifest),
    }


def _resolve_tokenizer_fingerprint():
    tokenizer_path = str(getattr(TrainingConfig, "tokenizer_path", "")).strip()
    if not tokenizer_path:
        return None

    resolved_path = os.path.abspath(os.path.expanduser(tokenizer_path))
    stat_info = _stat_fingerprint(resolved_path)
    if stat_info is None:
        return {"path": resolved_path, "missing": True}

    cache_key = (
        stat_info["path"],
        stat_info["size"],
        stat_info["mtime_ns"],
    )
    cached = _TOKENIZER_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    payload = dict(stat_info)
    payload["sha256"] = _file_sha256(resolved_path)
    _TOKENIZER_FINGERPRINT_CACHE.clear()
    _TOKENIZER_FINGERPRINT_CACHE[cache_key] = dict(payload)
    return payload


def _get_available_ram_bytes():
    try:
        import psutil  # type: ignore

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
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
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


def _as_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features.to(dtype=torch.float32).contiguous()
    if len(features) == 0:
        return torch.empty(
            (0, DataConfig.seq_len, len(DataConfig.feature_cols)),
            dtype=torch.float32,
        )
    return torch.from_numpy(np.asarray(features, dtype=np.float32))


def _as_sector_tensor(sector_ids):
    if isinstance(sector_ids, torch.Tensor):
        return sector_ids.to(dtype=torch.long).contiguous()
    if len(sector_ids) == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.from_numpy(np.asarray(sector_ids, dtype=np.int64))


def _extract_time_tensors(time_features):
    if isinstance(time_features, dict):
        tensors = {}
        for key in TIME_KEYS:
            value = time_features.get(key)
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                tensors[key] = value.to(dtype=torch.long).contiguous()
            else:
                tensors[key] = torch.from_numpy(np.asarray(value, dtype=np.int64))
        return tensors

    if len(time_features) == 0:
        return {key: torch.empty((0, DataConfig.seq_len), dtype=torch.long) for key in TIME_KEYS}

    tensors = {}
    for key in TIME_KEYS:
        values = [item[key] for item in time_features]
        tensors[key] = torch.from_numpy(np.asarray(values, dtype=np.int64))
    return tensors


def _time_tensors_to_list(time_tensors):
    num_samples = int(time_tensors["day"].shape[0])
    result = []
    for idx in range(num_samples):
        result.append(
            {
                "minute": time_tensors["minute"][idx].cpu().numpy(),
                "day": time_tensors["day"][idx].cpu().numpy(),
                "month": time_tensors["month"][idx].cpu().numpy(),
                "year": time_tensors["year"][idx].cpu().numpy(),
            }
        )
    return result


def _clamp_int(value, low, high):
    return max(low, min(high, int(value)))


def _resolve_loader_config(dataset_len, batch_size, use_cuda, overrides=None):
    overrides = overrides or {}

    auto_tune = bool(getattr(TrainingConfig, "auto_tune_resources", False))
    num_workers = int(getattr(TrainingConfig, "num_workers", 0))
    pin_memory = bool(getattr(TrainingConfig, "pin_memory", True) and use_cuda)
    persistent_workers = bool(getattr(TrainingConfig, "persistent_workers", False))
    prefetch_factor = int(getattr(TrainingConfig, "prefetch_factor", 2))

    if auto_tune:
        cpu_total = max(1, os.cpu_count() or 1)
        cpu_target = int(cpu_total * float(getattr(TrainingConfig, "cpu_worker_utilization", 0.9)))
        cpu_target = max(1, cpu_target - 1)
        cpu_cap = max(0, cpu_total - (1 if cpu_total > 1 else 0))

        workers_min = int(getattr(TrainingConfig, "min_num_workers", 2))
        workers_max = int(getattr(TrainingConfig, "max_num_workers", cpu_target))
        num_workers = _clamp_int(cpu_target, workers_min, workers_max)
        num_workers = min(num_workers, cpu_cap)

        available_ram_bytes = _get_available_ram_bytes()
        if available_ram_bytes is not None:
            worker_ram_bytes = int(
                max(0.25, float(getattr(TrainingConfig, "estimated_ram_per_worker_gb", 0.75)))
                * (1024**3)
            )
            ram_cap = max(1, available_ram_bytes // max(1, worker_ram_bytes))
            num_workers = min(num_workers, int(ram_cap))

            available_ram_gb = available_ram_bytes / (1024**3)
            prefetch_by_ram = int(max(1, available_ram_gb // max(1, num_workers)))
            prefetch_factor = _clamp_int(
                prefetch_by_ram,
                int(getattr(TrainingConfig, "min_prefetch_factor", 2)),
                int(getattr(TrainingConfig, "max_prefetch_factor", 8)),
            )
        else:
            prefetch_factor = _clamp_int(
                prefetch_factor,
                int(getattr(TrainingConfig, "min_prefetch_factor", 2)),
                int(getattr(TrainingConfig, "max_prefetch_factor", 8)),
            )

        if dataset_len <= batch_size:
            num_workers = 0

        persistent_workers = bool(num_workers > 0 and persistent_workers)

    if "num_workers" in overrides:
        num_workers = int(overrides["num_workers"])
    if "pin_memory" in overrides:
        pin_memory = bool(overrides["pin_memory"])
    if "persistent_workers" in overrides:
        persistent_workers = bool(overrides["persistent_workers"])
    if "prefetch_factor" in overrides:
        prefetch_factor = int(overrides["prefetch_factor"])

    config = {
        "batch_size": int(batch_size),
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and int(num_workers) > 0),
    }
    if config["num_workers"] > 0:
        config["prefetch_factor"] = max(1, int(prefetch_factor))
    return config


class AShareDataset(Dataset):
    """A 股序列数据集。"""

    def __init__(self, mode="train"):
        self.seq_len = DataConfig.seq_len
        self.data_dir = DataConfig.data_dir
        self.mode = mode
        self.demo_ratio = DataConfig.demo_ratio
        self.demo_days = max(1, int(getattr(DataConfig, "demo_days", 30)))
        self.max_stocks = DataConfig.max_stocks

        self.features = []
        self.sector_ids = []
        self.time_features = []
        self.seq_stats = []
        self.dates = []
        self.symbols = []
        self.raw_data = {}

        self._features_tensor = None
        self._sector_ids_tensor = None
        self._time_tensors = None
        self._length = 0
        self._cache_file = self._resolve_cache_file()
        self._source_fingerprint = _dataset_source_fingerprint(self.data_dir)
        
        # 预计算的 tokenizer 编码缓存。
        self._encoded_indices_coarse = None
        self._encoded_indices_fine = None

        expected_sig = self._cache_signature()
        expected_encoding_sig = self._encoding_cache_signature()

        cache_loaded = False
        if os.path.exists(self._cache_file):
            print(f"Loading {mode} dataset cache: {self._cache_file}")
            try:
                cached_data = torch.load(self._cache_file, weights_only=False)
                cached_seq_stats = cached_data.get("seq_stats")
                cached_features = cached_data.get("features", [])
                cached_sig = cached_data.get("_data_cache_signature", cached_data.get("_cache_signature"))

                if (
                    cached_sig == expected_sig
                    and cached_seq_stats is not None
                    and len(cached_seq_stats) == len(cached_features)
                ):
                    self.features = cached_features
                    self.sector_ids = cached_data.get("sector_ids", [])
                    self.time_features = cached_data.get("time_features", [])
                    self.seq_stats = cached_seq_stats
                    self.dates = cached_data.get("dates", [])
                    self.symbols = cached_data.get("symbols", [])
                    self.raw_data = cached_data.get("raw_data", {})
                    
                    # 加载预计算的 tokenizer 编码缓存。
                    cached_encoding_sig = cached_data.get("_encoding_cache_signature")
                    cached_coarse = cached_data.get("encoded_indices_coarse")
                    cached_fine = cached_data.get("encoded_indices_fine")
                    if (
                        cached_encoding_sig == expected_encoding_sig
                        and self._has_valid_cached_encodings(
                            coarse=cached_coarse,
                            fine=cached_fine,
                            expected_samples=len(cached_features),
                        )
                    ):
                        self._encoded_indices_coarse = cached_coarse
                        self._encoded_indices_fine = cached_fine
                    elif cached_coarse is not None or cached_fine is not None:
                        print(
                            f"{mode} tokenizer encodings are outdated or incomplete. "
                            "Will recompute with the current tokenizer."
                        )
                    
                    cache_loaded = True
                else:
                    print(f"{mode} cache is outdated. Rebuilding cache.")
            except Exception as exc:
                print(f"Failed to load {mode} cache ({exc}). Rebuilding cache.")

        if not cache_loaded:
            print(f"Processing {mode} dataset from: {self.data_dir}")
            self._process_data()

        self._build_runtime_tensors()

        if not cache_loaded:
            self._persist_cache(reason="dataset rebuilt")

    def _cache_signature(self):
        return {
            "mode": self.mode,
            "seq_len": int(self.seq_len),
            "stride_ratio": float(DataConfig.stride_ratio),
            "feature_cols": tuple(DataConfig.feature_cols),
            "random_seed": int(getattr(DataConfig, "random_seed", 42)),
            "train_val_split": float(DataConfig.train_val_split),
            "demo_days": int(self.demo_days),
            "demo_ratio": float(self.demo_ratio),
            "max_stocks": int(self.max_stocks) if self.max_stocks else None,
            "source_fingerprint": self._source_fingerprint,
        }

    def _encoding_cache_signature(self):
        tokenizer_path = str(getattr(TrainingConfig, "tokenizer_path", "")).strip()
        return {
            "tokenizer_path": os.path.abspath(os.path.expanduser(tokenizer_path))
            if tokenizer_path
            else None,
            "tokenizer_fingerprint": _resolve_tokenizer_fingerprint(),
        }

    def _resolve_cache_file(self):
        return f"dataset_{self.mode}.pt"

    def _build_runtime_tensors(self):
        self._features_tensor = _as_feature_tensor(self.features)
        self._sector_ids_tensor = _as_sector_tensor(self.sector_ids)
        self._time_tensors = _extract_time_tensors(self.time_features)

        self._length = int(self._features_tensor.shape[0])

        if self.mode == "demo":
            if not isinstance(self.time_features, list):
                self.time_features = _time_tensors_to_list(self._time_tensors)
        else:
            # Release Python-object heavy structures for train/val to reduce memory and CPU overhead.
            self.features = self._features_tensor
            self.sector_ids = self._sector_ids_tensor
            self.time_features = self._time_tensors

    def _has_valid_cached_encodings(self, coarse, fine, expected_samples):
        if coarse is None or fine is None:
            return False
        if not isinstance(coarse, torch.Tensor) or not isinstance(fine, torch.Tensor):
            return False
        if coarse.ndim != 2 or fine.ndim != 2:
            return False
        if int(coarse.shape[0]) != int(expected_samples) or int(fine.shape[0]) != int(expected_samples):
            return False
        if int(coarse.shape[1]) != int(self.seq_len) or int(fine.shape[1]) != int(self.seq_len):
            return False
        return True

    def _export_cache_payload(self):
        payload = {
            "features": self._features_tensor.cpu(),
            "sector_ids": self._sector_ids_tensor.cpu(),
            "time_features": {k: v.cpu().to(torch.int16) for k, v in self._time_tensors.items()},
            "seq_stats": self.seq_stats,
            "dates": self.dates,
            "symbols": self.symbols,
            "_data_cache_signature": self._cache_signature(),
            "_encoding_cache_signature": self._encoding_cache_signature(),
        }
        # 保存预计算的 tokenizer 编码缓存。
        if self._encoded_indices_coarse is not None:
            payload["encoded_indices_coarse"] = self._encoded_indices_coarse.cpu()
        if self._encoded_indices_fine is not None:
            payload["encoded_indices_fine"] = self._encoded_indices_fine.cpu()
        if self.mode == "demo":
            payload["raw_data"] = self.raw_data
        return payload

    def _persist_cache(self, reason=None):
        cache_dir = os.path.dirname(self._cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{self._cache_file}.tmp.{os.getpid()}"
        try:
            torch.save(self._export_cache_payload(), tmp_path)
            os.replace(tmp_path, self._cache_file)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        if reason:
            print(f"Saving {self.mode} dataset cache ({reason}): {self._cache_file}")
    
    def precompute_encodings(self, tokenizer, device, force=False):
        """预计算并缓存整个数据集的 tokenizer 编码。"""
        if self._encoded_indices_coarse is not None and self._encoded_indices_fine is not None and not force:
            print(f"Tokenizer encodings already cached for {self.mode} dataset, skipping precompute.")
            return

        if len(self) == 0:
            self._encoded_indices_coarse = torch.empty((0, int(self.seq_len)), dtype=torch.long)
            self._encoded_indices_fine = torch.empty((0, int(self.seq_len)), dtype=torch.long)
            print(f"Tokenizer encoding precompute skipped for empty {self.mode} dataset.")
            self._persist_cache(reason="tokenizer encodings updated")
            return
        
        print(f"Precomputing tokenizer encodings for {self.mode} dataset ({len(self)} samples)...")
        
        indices_coarse_list = []
        indices_fine_list = []
        tokenizer.eval()
        
        with torch.no_grad():
            for idx in tqdm(range(len(self)), desc=f"Encoding {self.mode}"):
                # 逐样本编码，避免一次性占满显存。
                features_sample = self._features_tensor[idx: idx + 1].to(device)
                idx_coarse, idx_fine = tokenizer.encode(features_sample)
                indices_coarse_list.append(idx_coarse.cpu().squeeze(0))
                indices_fine_list.append(idx_fine.cpu().squeeze(0))
        
        # 合并为整批缓存张量。
        self._encoded_indices_coarse = torch.stack(indices_coarse_list, dim=0)
        self._encoded_indices_fine = torch.stack(indices_fine_list, dim=0)
        
        print(f"Tokenizer encoding precompute done: {self.mode}")
        print(f"  Shapes: coarse {self._encoded_indices_coarse.shape}, fine {self._encoded_indices_fine.shape}")
        self._persist_cache(reason="tokenizer encodings updated")

    def _process_data(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        print(f"Found {len(files)} CSV files")

        if self.max_stocks:
            np.random.seed(DataConfig.random_seed)
            files = list(np.random.choice(files, min(self.max_stocks, len(files)), replace=False))
            print(f"Sampled {len(files)} files")

        all_stock_data = []
        required_cols = {"open", "high", "low", "close", "volume", "amount"}

        for file_path in tqdm(files, desc="Loading stock data"):
            try:
                candidate_cols = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "symbol",
                    "sector_id",
                ]
                df = pd.read_csv(
                    file_path,
                    usecols=lambda col: col in candidate_cols,
                    low_memory=False,
                    dtype={"symbol": str},
                )

                if "date" not in df.columns:
                    continue
                if not required_cols.issubset(df.columns):
                    continue

                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

                if "volume" in df.columns:
                    df = df[df["volume"] > 0].reset_index(drop=True)
                if len(df) < self.seq_len:
                    continue

                if "symbol" in df.columns and len(df["symbol"]) > 0:
                    symbol = str(df["symbol"].iloc[0])
                else:
                    symbol = os.path.basename(file_path).split(".")[0]

                if "sector_id" in df.columns and len(df["sector_id"]) > 0:
                    sector_id = int(df["sector_id"].iloc[0])
                else:
                    sector_id = self._get_sector_id(symbol)

                all_stock_data.append({"symbol": symbol, "sector_id": sector_id, "df": df})
            except Exception:
                continue

        if not all_stock_data:
            print("No valid stock data found.")
            return

        all_dates = sorted(set(d for stock in all_stock_data for d in stock["df"]["date"].tolist()))
        total_dates = len(all_dates)
        if total_dates < 3:
            print("Date range is too small, skip dataset build.")
            return

        # demo 固定保留最近 demo_days 天，train/val 在更早的时间段内切分。
        demo_days = min(self.demo_days, total_dates - 1)
        split_demo_idx = max(2, total_dates - demo_days)
        split_train_val_idx = int(split_demo_idx * DataConfig.train_val_split)
        split_train_val_idx = min(max(split_train_val_idx, 1), split_demo_idx - 1)

        print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
        train_start, train_end = all_dates[0], all_dates[split_train_val_idx - 1]
        val_start, val_end = all_dates[split_train_val_idx], all_dates[split_demo_idx - 1]
        demo_start, demo_end = all_dates[split_demo_idx], all_dates[-1]
        print(f"Train end-date range: {train_start} to {train_end}")
        print(f"Val end-date range: {val_start} to {val_end}")
        print(f"Demo end-date range ({demo_days}d): {demo_start} to {demo_end}")

        def _mode_matches(end_date):
            if self.mode == "train":
                return train_start <= end_date <= train_end
            if self.mode == "val":
                return val_start <= end_date <= val_end
            return demo_start <= end_date <= demo_end

        stride = max(1, int(self.seq_len * DataConfig.stride_ratio))

        for stock in tqdm(all_stock_data, desc=f"Processing {self.mode} data"):
            try:
                df = stock["df"]
                symbol = stock["symbol"]
                sector_id = stock["sector_id"]

                df_processed = df.sort_values("date").reset_index(drop=True)
                prev_close = df_processed["close"].shift(1)

                df_processed["log_ret"] = np.log(df_processed["close"] / prev_close)
                df_processed["log_high"] = np.log(df_processed["high"] / prev_close)
                df_processed["log_low"] = np.log(df_processed["low"] / prev_close)
                df_processed["log_open"] = np.log(df_processed["open"] / prev_close)
                df_processed["log_vol"] = np.log1p(df_processed["volume"])
                df_processed["log_amt"] = np.log1p(df_processed["amount"])
                df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
                df_processed = df_processed.dropna(subset=DataConfig.feature_cols).reset_index(drop=True)

                if len(df_processed) < self.seq_len:
                    continue

                data = df_processed[DataConfig.feature_cols].values.astype(np.float32)
                if not np.isfinite(data).all():
                    continue

                num_seqs = (len(data) - self.seq_len) // stride + 1
                if num_seqs <= 0:
                    continue

                if self.mode == "demo":
                    self.raw_data[symbol] = {
                        "dates": df_processed["date"].tolist(),
                        "close": df_processed["close"].tolist(),
                        "open": df_processed["open"].tolist(),
                        "high": df_processed["high"].tolist(),
                        "low": df_processed["low"].tolist(),
                        "volume": df_processed["volume"].tolist(),
                        "amount": df_processed["amount"].tolist(),
                        "sector_id": sector_id,
                    }

                for i in range(num_seqs):
                    seq_start_idx = i * stride
                    seq_end_idx = seq_start_idx + self.seq_len
                    end_date = df_processed["date"].iloc[seq_end_idx - 1]
                    if not _mode_matches(end_date):
                        continue

                    seq = data[seq_start_idx:seq_end_idx]

                    mean = np.mean(seq, axis=0).astype(np.float32)
                    std = np.std(seq, axis=0).astype(np.float32)
                    std[std == 0] = 1.0
                    seq_norm = ((seq - mean) / std).astype(np.float32)

                    self.features.append(seq_norm)
                    self.sector_ids.append(sector_id)
                    self.seq_stats.append({"mean": mean, "std": std})

                    seq_dates = df_processed["date"].iloc[seq_start_idx:seq_end_idx].tolist()
                    self.time_features.append(self._extract_time_features(seq_dates))
                    if self.mode == "demo":
                        self.dates.append(seq_dates)
                        self.symbols.append(symbol)
            except Exception:
                continue

        print(f"Done: {self.mode} total sequences = {len(self.features)}")

    def _extract_time_features(self, dates):
        return {
            "minute": np.zeros(len(dates), dtype=np.int16),
            "day": np.clip(np.array([d.day for d in dates], dtype=np.int16) - 1, 0, 30),
            "month": np.clip(np.array([d.month for d in dates], dtype=np.int16) - 1, 0, 11),
            "year": np.clip(
                np.array([d.year - DataConfig.base_year for d in dates], dtype=np.int16), 0, 99
            ),
        }

    def _get_sector_id(self, symbol):
        sector_banking = 0
        sector_securities = 1
        sector_machinery = 11
        sector_electronics = 23
        sector_semiconductor = 24
        sector_new_energy = 41
        sector_other = 50

        try:
            code = int(symbol) if symbol.isdigit() else 0
        except Exception:
            code = 0

        if 601288 <= code <= 601398:
            return sector_banking
        if 600030 <= code <= 600999:
            return sector_securities
        if 600000 <= code <= 600029:
            return sector_banking
        if 300000 <= code <= 300749:
            return sector_electronics
        if 300750 <= code <= 300999:
            return sector_new_energy
        if 688000 <= code <= 688999:
            return sector_semiconductor
        if 2000 <= code <= 2999:
            return sector_machinery
        return sector_other

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        result = {
            "features": self._features_tensor[idx],
            "sector_ids": self._sector_ids_tensor[idx],
            "t_min": self._time_tensors["minute"][idx],
            "t_day": self._time_tensors["day"][idx],
            "t_month": self._time_tensors["month"][idx],
            "t_year": self._time_tensors["year"][idx],
        }
        # 返回可复用的预计算 tokenizer 编码。
        if self._encoded_indices_coarse is not None:
            result["idx_coarse"] = self._encoded_indices_coarse[idx]
        if self._encoded_indices_fine is not None:
            result["idx_fine"] = self._encoded_indices_fine[idx]
        return result


def collate_fn(batch):
    """组装 batch，并在可用时附带预计算编码。"""
    features_list = []
    sector_ids_list = []
    t_min_list = []
    t_day_list = []
    t_month_list = []
    t_year_list = []
    idx_coarse_list = []
    idx_fine_list = []
    has_encodings = False
    
    for item in batch:
        features_list.append(item["features"])
        sector_ids_list.append(item["sector_ids"])
        t_min_list.append(item["t_min"])
        t_day_list.append(item["t_day"])
        t_month_list.append(item["t_month"])
        t_year_list.append(item["t_year"])
        
        # 仅当整个 batch 都带编码时才返回编码张量。
        if "idx_coarse" in item and "idx_fine" in item:
            idx_coarse_list.append(item["idx_coarse"])
            idx_fine_list.append(item["idx_fine"])
            has_encodings = True
    
    features = torch.stack(features_list, dim=0)
    sector_ids = torch.stack(sector_ids_list, dim=0)
    time_features = {
        "minute": torch.stack(t_min_list, dim=0),
        "day": torch.stack(t_day_list, dim=0),
        "month": torch.stack(t_month_list, dim=0),
        "year": torch.stack(t_year_list, dim=0),
    }
    
    encodings = None
    if has_encodings:
        encodings = {
            "idx_coarse": torch.stack(idx_coarse_list, dim=0),
            "idx_fine": torch.stack(idx_fine_list, dim=0),
        }
    
    return features, sector_ids, time_features, encodings


def get_datasets(include_demo=True, use_memmap=False):
    """构建 train/val/demo 数据集。"""
    if use_memmap:
        return get_memmap_datasets(include_demo=include_demo)
    train_dataset = AShareDataset(mode="train")
    val_dataset = AShareDataset(mode="val")
    demo_dataset = AShareDataset(mode="demo") if include_demo else None
    return train_dataset, val_dataset, demo_dataset


def get_dataloaders(
    train_dataset=None,
    val_dataset=None,
    demo_dataset=None,
    batch_size=None,
    loader_overrides=None,
    include_demo=True,
    distributed=False,
    rank=0,
    world_size=1,
    seed=None,
    collate_fn_override=None,
):
    """构建训练、验证和 demo 的 DataLoader。"""
    if train_dataset is None or val_dataset is None:
        use_memmap = bool(collate_fn_override is not None)
        train_dataset, val_dataset, auto_demo_dataset = get_datasets(
            include_demo=include_demo,
            use_memmap=use_memmap,
        )
        if demo_dataset is None:
            demo_dataset = auto_demo_dataset

    resolved_batch_size = int(TrainingConfig.batch_size if batch_size is None else batch_size)
    if resolved_batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {resolved_batch_size}. "
            "Pass batch_size to get_dataloaders(...) or set TrainingConfig.batch_size > 0."
        )
    use_cuda = torch.cuda.is_available()
    dataset_lengths = [len(train_dataset), len(val_dataset)]
    if demo_dataset is not None:
        dataset_lengths.append(len(demo_dataset))

    loader_cfg = _resolve_loader_config(
        dataset_len=max(dataset_lengths),
        batch_size=resolved_batch_size,
        use_cuda=use_cuda,
        overrides=loader_overrides,
    )

    base_seed = int(DataConfig.random_seed if seed is None else seed)
    dataloader_seed = base_seed + (int(rank) if (distributed and world_size > 1) else 0)
    generator = torch.Generator()
    generator.manual_seed(dataloader_seed)

    resolved_collate = collate_fn_override if collate_fn_override is not None else collate_fn

    common_kwargs = {
        "batch_size": loader_cfg["batch_size"],
        "collate_fn": resolved_collate,
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": loader_cfg["pin_memory"],
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if loader_cfg["num_workers"] > 0:
        common_kwargs["persistent_workers"] = loader_cfg["persistent_workers"]
        common_kwargs["prefetch_factor"] = loader_cfg["prefetch_factor"]

    train_sampler = None
    val_sampler = None
    demo_sampler = None
    train_shuffle = True
    val_shuffle = False
    demo_shuffle = False

    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
            seed=base_seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=base_seed,
        )
        train_shuffle = False
        val_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=val_shuffle,
        sampler=val_sampler,
        **common_kwargs,
    )
    demo_loader = None
    if demo_dataset is not None:
        if distributed and world_size > 1:
            demo_sampler = DistributedSampler(
                demo_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
                seed=base_seed,
            )
            demo_shuffle = False
        demo_loader = DataLoader(
            demo_dataset,
            shuffle=demo_shuffle,
            sampler=demo_sampler,
            **common_kwargs,
        )

    print(
        "Loader config - "
        f"batch={loader_cfg['batch_size']}, workers={loader_cfg['num_workers']}, "
        f"pin_memory={loader_cfg['pin_memory']}, "
        f"prefetch={loader_cfg.get('prefetch_factor', 'n/a')}"
    )
    if demo_dataset is None:
        print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")
    else:
        print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, demo: {len(demo_dataset)}")
    return train_loader, val_loader, demo_loader, demo_dataset


if __name__ == "__main__":
    default_batch_size = max(1, int(getattr(TrainingConfig, "batch_size", 8) or 8))
    train_loader, _, _, _ = get_dataloaders(batch_size=default_batch_size)
    for batch_x, _, _, _ in train_loader:
        print("batch shape:", batch_x.shape)
        break
