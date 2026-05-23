# -*- coding: utf-8 -*-
"""Phase 5 Sup – Stage 0/1: unified evaluation, per-sample output, top-K oracle.

Usage::

    python -m hpo.phase5_sup_eval --stage audit
    python -m hpo.phase5_sup_eval --stage topk_oracle --top-k 4,8,16,32,64
    python -m hpo.phase5_sup_eval --stage paired \\
        --checkpoints ckpt_a.pt,ckpt_b.pt \\
        --cache dataset_val.pt
"""

from __future__ import annotations

import argparse, csv, json, os, sys, time, warnings
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, date

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import hpo.phase5_da as p5d
import hpo.phase5_hpo as p5hpo
from model.lora import has_lora_layers
from reproducibility import set_global_seed

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(_PROJECT_ROOT, "trials", "phase5_sup")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_NAMES = {0: "down", 1: "flat", 2: "up"}


# ═══════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════

def _load_any_checkpoint(ckpt_path: str, device: torch.device):
    """Load model from any checkpoint format (base, LoRA, full-FT HPO)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    is_base = "basemodel" in os.path.basename(ckpt_path) or "base_model" in ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Determine format
    hpo_params = ckpt.get("hpo_params", {})
    if hpo_params:
        # Phase 5 HPO checkpoint
        use_lora = hpo_params.get("use_lora", True)
        rank = hpo_params.get("lora_rank", 8)
        alpha = hpo_params.get("lora_alpha", 16)
        model = p5hpo.build_trainable_model_hpo(device, rank, alpha, use_lora=use_lora)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "lora_state_dict" in ckpt and ckpt["lora_state_dict"]:
        # LoRA-only checkpoint
        model = p5hpo.build_trainable_model_hpo(device, 8, 16, use_lora=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif is_base:
        # Plain base model (P3) — load weights from the checkpoint file
        model = p5d._build_model(device)
        p5d._load_base_weights(model, ckpt_path)
        # _load_base_weights already loaded model_state_dict, but check for extra keys
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        # Generic fallback
        model = p5d._build_model(device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════
# Per-sample inference
# ═══════════════════════════════════════════════

@torch.no_grad()
def run_inference_per_sample(model, tokenizer, loader, device, top_k: int = 0):
    """Run inference, return per-sample dicts + aggregate metrics.

    If *top_k* > 0, also return top-K token pairs and decoded returns.
    """
    model.eval()
    samples = []
    all_preds, all_labels, all_returns, all_pred_returns = [], [], [], []

    for raw_batch in tqdm(loader, desc="Inference", leave=False):
        batch = p5d._move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr,
                                       last_only=True)
        last_c = logits_c[:, -1, :].float()
        last_f = logits_f[:, -1, :].float()

        # Argmax prediction
        pred_c = last_c.argmax(dim=-1)
        pred_f = last_f.argmax(dim=-1)
        pred_ret = p5d._token_returns(tokenizer, pred_c.unsqueeze(1), pred_f.unsqueeze(1),
                                       batch["prompt_means"], batch["prompt_stds"]).squeeze(1)
        pred_dir = torch.where(pred_ret > 0,
                               torch.tensor(p5d.LABEL_UP, device=device),
                               torch.tensor(p5d.LABEL_DOWN, device=device))

        # Log-probs
        logp_c = F.log_softmax(last_c, dim=-1)
        logp_f = F.log_softmax(last_f, dim=-1)
        gold_c = batch["idx_c_full"][:, -1]
        gold_f = batch["idx_f_full"][:, -1]
        logp_gold = logp_c.gather(1, gold_c.unsqueeze(1)).squeeze(1) + \
                    logp_f.gather(1, gold_f.unsqueeze(1)).squeeze(1)
        logp_argmax = logp_c.gather(1, pred_c.unsqueeze(1)).squeeze(1) + \
                      logp_f.gather(1, pred_f.unsqueeze(1)).squeeze(1)

        # Margin up/down
        dir_logits, _ = model.forward_direction(idx_c, idx_f, t_min, t_day, t_mon, t_yr)
        margin_up_down = (dir_logits[:, 2] - dir_logits[:, 0]).cpu().numpy()

        # Top-K info
        topk_info = None
        if top_k > 0:
            topk_info = _extract_topk(tokenizer, last_c, last_f,
                                       batch["prompt_means"], batch["prompt_stds"],
                                       batch["real_returns"], top_k)

        # Per-sample records
        B = last_c.size(0)
        for i in range(B):
            lbl = int(batch["labels"][i].item())
            pred_lbl = int(pred_dir[i].item())
            is_correct = int(pred_lbl == lbl and lbl != p5d.LABEL_FLAT)
            sample = {
                "pred_label": pred_lbl,
                "actual_label": lbl,
                "actual_return": float(batch["real_returns"][i].item()),
                "pred_return": float(pred_ret[i].item()),
                "pred_coarse": int(pred_c[i].item()),
                "pred_fine": int(pred_f[i].item()),
                "gold_coarse": int(gold_c[i].item()),
                "gold_fine": int(gold_f[i].item()),
                "logp_gold": float(logp_gold[i].item()),
                "logp_argmax": float(logp_argmax[i].item()),
                "margin_up_down": float(margin_up_down[i]) if len(margin_up_down) > 0 else 0.0,
                "abs_error": float(abs(pred_ret[i].item() - batch["real_returns"][i].item())),
                "is_correct": is_correct,
            }
            if topk_info is not None:
                sample["topk_tokens_c"] = topk_info["tokens_c"][i]
                sample["topk_tokens_f"] = topk_info["tokens_f"][i]
                sample["topk_decoded_returns"] = topk_info["decoded_returns"][i]
                sample["topk_directions_correct"] = topk_info["directions_correct"][i]
            samples.append(sample)
            all_preds.append(pred_lbl)
            all_labels.append(lbl)
            all_returns.append(sample["actual_return"])
            all_pred_returns.append(sample["pred_return"])

    metrics = p5d._compute_metrics(
        np.array(all_preds), np.array(all_labels),
        np.array(all_returns), np.array(all_pred_returns), np.array(all_returns))
    return samples, metrics


def _extract_topk(tokenizer, logits_c, logits_f, means, stds, real_returns, k):
    """Extract top-K token pairs, decoded returns, and direction correctness."""
    B = logits_c.size(0)
    logp_c = F.log_softmax(logits_c, dim=-1)
    logp_f = F.log_softmax(logits_f, dim=-1)

    # Get top-K_c coarse and top-K_f fine independently, then compute joint
    K_c = min(k, logits_c.size(-1))
    K_f = min(k, logits_f.size(-1))
    topk_c_vals, topk_c_idx = torch.topk(logp_c, K_c, dim=-1)  # [B, K_c]
    topk_f_vals, topk_f_idx = torch.topk(logp_f, K_f, dim=-1)  # [B, K_f]

    tokens_c_list, tokens_f_list = [], []
    decoded_list, dir_correct_list = [], []

    real = real_returns.to(device=logits_c.device, dtype=torch.float32)

    for i in range(B):
        # Compute joint logp for all K_c × K_f pairs
        joint_logp = topk_c_vals[i].unsqueeze(1) + topk_f_vals[i].unsqueeze(0)  # [K_c, K_f]
        joint_flat = joint_logp.flatten()
        joint_topk = joint_flat.topk(min(k, joint_flat.size(0)))

        pair_indices = torch.stack([
            joint_topk.indices // K_f,  # coarse index within top-K_c
            joint_topk.indices % K_f,   # fine index within top-K_f
        ], dim=-1)  # [topK, 2]

        c_tokens = topk_c_idx[i, pair_indices[:, 0]]  # [topK]
        f_tokens = topk_f_idx[i, pair_indices[:, 1]]  # [topK]

        # Decode
        c_exp = c_tokens.unsqueeze(0)  # [1, topK]
        f_exp = f_tokens.unsqueeze(0)
        m = means[i:i+1].to(device=logits_c.device)
        s = stds[i:i+1].to(device=logits_c.device)
        decoded_ret = p5d._token_returns(tokenizer, c_exp, f_exp, m, s).squeeze(0)  # [topK]
        dir_correct = (torch.sign(decoded_ret) == torch.sign(real[i])).int().cpu().tolist()

        tokens_c_list.append(c_tokens.cpu().tolist())
        tokens_f_list.append(f_tokens.cpu().tolist())
        decoded_list.append(decoded_ret.cpu().tolist())
        dir_correct_list.append(dir_correct)

    return {
        "tokens_c": tokens_c_list,
        "tokens_f": tokens_f_list,
        "decoded_returns": decoded_list,
        "directions_correct": dir_correct_list,
    }


# ═══════════════════════════════════════════════
# Top-K Oracle upper bound
# ═══════════════════════════════════════════════

@torch.no_grad()
def compute_topk_oracle(model, tokenizer, loader, device, ks=(4, 8, 16, 32, 64)):
    """For each K, compute oracle_DA@K: fraction of samples where top-K contains
    a token pair with correct direction."""
    model.eval()
    max_k = max(ks)
    n_correct = {k: 0 for k in ks}
    n_directional = 0

    for raw_batch in tqdm(loader, desc="TopK Oracle", leave=False):
        batch = p5d._move_batch(raw_batch, device)
        batch["tokenizer"] = tokenizer
        idx_c, idx_f, t_min, t_day, t_mon, t_yr = p5d._prepare_inputs(batch)

        logits_c, logits_f, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr,
                                       last_only=True)
        last_c = logits_c[:, -1, :].float()
        last_f = logits_f[:, -1, :].float()

        info = _extract_topk(tokenizer, last_c, last_f,
                              batch["prompt_means"], batch["prompt_stds"],
                              batch["real_returns"], max_k)

        for i in range(last_c.size(0)):
            lbl = int(batch["labels"][i].item())
            if lbl == p5d.LABEL_FLAT:
                continue
            n_directional += 1
            correct = info["directions_correct"][i]
            for k in ks:
                if any(correct[:k]):
                    n_correct[k] += 1

    return {f"oracle_da@{k}": n_correct[k] / max(1, n_directional) for k in ks}, n_directional


# ═══════════════════════════════════════════════
# Paired metrics
# ═══════════════════════════════════════════════

def paired_significance(samples_a, samples_b):
    """McNemar test + bootstrap CI for paired DA comparison."""
    # Filter to directional samples where both models made predictions
    dir_idx = [i for i, (a, b) in enumerate(zip(samples_a, samples_b))
               if a["actual_label"] != p5d.LABEL_FLAT and b["actual_label"] != p5d.LABEL_FLAT]

    correct_a = np.array([samples_a[i]["is_correct"] for i in dir_idx])
    correct_b = np.array([samples_b[i]["is_correct"] for i in dir_idx])

    da_a = float(np.mean(correct_a))
    da_b = float(np.mean(correct_b))
    delta = da_b - da_a
    n = len(correct_a)

    # McNemar: b01 = correct in B but wrong in A; b10 = wrong in B but correct in A
    b01 = int(np.sum((1 - correct_a) & correct_b))
    b10 = int(np.sum(correct_a & (1 - correct_b)))
    if b01 + b10 > 0:
        from scipy.stats import chi2
        mcnemar_stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        mcnemar_p = 1.0 - chi2.cdf(mcnemar_stat, 1)
    else:
        mcnemar_p = 1.0

    # Bootstrap CI for delta
    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas.append(float(np.mean(correct_b[idx]) - np.mean(correct_a[idx])))
    boot_deltas = np.sort(boot_deltas)
    ci_lower = float(boot_deltas[int(0.025 * n_boot)])
    ci_upper = float(boot_deltas[int(0.975 * n_boot)])

    return {
        "n_directional": n,
        "da_a": da_a, "da_b": da_b, "delta_da": delta,
        "mcnemar_b01": b01, "mcnemar_b10": b10,
        "mcnemar_p": float(mcnemar_p),
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "binomial_se": float(np.sqrt(da_a * (1 - da_a) / n + da_b * (1 - da_b) / n)),
    }


# ═══════════════════════════════════════════════
# Walk-forward val split
# ═══════════════════════════════════════════════

def build_walkforward_val_splits(val_cache_path: str, n_splits: int = 4):
    """Split val cache into *n_splits* chronologically-ordered segments.

    Returns list of (indices, date_range) for each segment.
    """
    payload = torch.load(val_cache_path, map_location="cpu", weights_only=False)
    cache_dates = payload.get("dates", [])
    if not cache_dates:
        # Fallback: use sequential indices
        n = len(payload["features"])
        chunk = n // n_splits
        return [(np.arange(i * chunk, min((i + 1) * chunk, n), dtype=np.int64),
                 f"chunk_{i}") for i in range(n_splits)]

    # Parse dates and sort
    parsed = []
    for i, d in enumerate(cache_dates):
        if isinstance(d, (int, float)):
            parsed.append((i, datetime.fromordinal(int(d))))
        elif isinstance(d, str):
            parsed.append((i, datetime.fromisoformat(d[:10])))
        elif isinstance(d, datetime):
            parsed.append((i, d))
        elif isinstance(d, date):
            parsed.append((i, datetime(d.year, d.month, d.day)))
        else:
            parsed.append((i, datetime(2000, 1, 1)))

    parsed.sort(key=lambda x: x[1])
    sorted_indices = np.array([p[0] for p in parsed], dtype=np.int64)
    sorted_dates = [p[1] for p in parsed]

    n = len(sorted_indices)
    splits = []
    for i in range(n_splits):
        start = i * n // n_splits
        end = (i + 1) * n // n_splits if i < n_splits - 1 else n
        indices = sorted_indices[start:end]
        date_range = f"{sorted_dates[start].date()} → {sorted_dates[end-1].date()}"
        splits.append((indices, date_range))
    return splits


# ═══════════════════════════════════════════════
# Stages
# ═══════════════════════════════════════════════

def _get_cache_loader(cache_path: str, eps_override=None):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    returns = p5d._denorm_last_returns(payload)
    if eps_override is not None:
        eps = eps_override
    else:
        abs_r = np.abs(returns); abs_r = abs_r[np.isfinite(abs_r)]
        eps = max(1e-5, float(np.median(abs_r)) * 0.5)
    ds = p5d.DirectionDataset(payload, np.arange(len(returns), dtype=np.int64),
                               returns, eps, "class")
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=p5d.collate_fn)
    return loader, ds, payload, eps


def stage_audit(args):
    """Stage 0: unified evaluation — per-sample CSV + paired metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = p5d._load_tokenizer(device)

    caches = {"val": "dataset_val.pt", "demo": "dataset_demo.pt"}

    all_samples = {}
    all_metrics = {}

    # ── Evaluate all checkpoints ──
    model_names = []
    model_paths = []

    # Always include base model
    model_names.append("basemodel")
    model_paths.append(p5d.P3_CKPT)

    if args.checkpoints:
        for cp in args.checkpoints.split(","):
            cp = cp.strip()
            if not cp: continue
            name = os.path.splitext(os.path.basename(cp))[0]
            model_names.append(name)
            model_paths.append(cp)

    for cache_name, cache_path in caches.items():
        if not os.path.exists(cache_path):
            print(f"Cache not found: {cache_path}, skipping")
            continue

        loader, ds, payload, eps = _get_cache_loader(cache_path)
        print(f"\n{cache_name}: {len(ds)} items, eps={eps:.6f}, "
              f"class dist: down={ds.class_counts[0]} flat={ds.class_counts[1]} up={ds.class_counts[2]}")

        for mname, mpath in zip(model_names, model_paths):
            key = f"{cache_name}/{mname}"
            print(f"  Evaluating {key}...")
            model = _load_any_checkpoint(mpath, device)
            t0 = time.time()
            samples, metrics = run_inference_per_sample(model, tokenizer, loader, device)
            elapsed = time.time() - t0
            metrics["eval_time_s"] = elapsed
            print(f"    DA={metrics['direction_accuracy']:.4f}  "
                  f"BalAcc={metrics['balanced_accuracy']:.4f}  "
                  f"MAPE={metrics['mape']:.4f}  time={elapsed:.1f}s")

            all_samples[key] = samples
            all_metrics[key] = metrics

    # ── Paired comparison: each checkpoint vs base on same split ──
    paired_results = {}
    for cache_name in caches:
        base_key = f"{cache_name}/basemodel"
        if base_key not in all_samples:
            continue
        for mname in model_names[1:]:
            key = f"{cache_name}/{mname}"
            if key not in all_samples:
                continue
            paired = paired_significance(all_samples[base_key], all_samples[key])
            paired_results[key] = paired
            sig = "SIGNIFICANT" if paired["mcnemar_p"] < 0.05 else "not sig"
            print(f"\n  {mname} vs basemodel on {cache_name}:")
            print(f"    Delta DA = {paired['delta_da']:+.4f}  "
                  f"McNemar p = {paired['mcnemar_p']:.4f} ({sig})  "
                  f"Bootstrap 95% CI = [{paired['bootstrap_ci_95'][0]:+.4f}, {paired['bootstrap_ci_95'][1]:+.4f}]")

    # ── Walk-forward val splits ──
    wf_splits = build_walkforward_val_splits("dataset_val.pt", n_splits=4)
    print(f"\nWalk-forward val splits:")
    for i, (indices, dr) in enumerate(wf_splits):
        print(f"  Q{i+1}: {len(indices)} samples, {dr}")

    # ── Save ──
    out = {
        "timestamp": datetime.now().isoformat(),
        "caches": {k: {"n_items": len(v)} for k, v in all_samples.items() if v},
        "metrics": all_metrics,
        "paired": paired_results,
        "walkforward_splits": [{"n": len(idx), "range": dr} for idx, dr in wf_splits],
    }
    with open(os.path.join(OUT_DIR, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    # Per-sample CSV (demo split only, all models merged)
    demo_models = [k for k in all_samples if k.startswith("demo/")]
    if demo_models:
        _write_per_sample_csv(all_samples, demo_models, os.path.join(OUT_DIR, "per_sample_demo.csv"))

    print(f"\nAudit saved to {OUT_DIR}/audit.json")
    return out


def stage_topk_oracle(args):
    """Stage 1: Top-K oracle upper bound."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = p5d._load_tokenizer(device)
    model = _load_any_checkpoint(p5d.P3_CKPT, device)

    ks = [int(x) for x in args.top_k.split(",")]

    for cache_name, cache_path in [("val", "dataset_val.pt"), ("demo", "dataset_demo.pt")]:
        if not os.path.exists(cache_path):
            continue
        loader, ds, _, eps = _get_cache_loader(cache_path)
        print(f"\n{cache_name}: {len(ds)} items")
        oracle, n_dir = compute_topk_oracle(model, tokenizer, loader, device, ks)
        base_da = all_metrics_cache.get(f"{cache_name}/basemodel", {}).get("direction_accuracy", 0) \
                  if 'all_metrics_cache' in dir() else 0
        print(f"  Directional samples: {n_dir}")
        for k in ks:
            od = oracle[f"oracle_da@{k}"]
            margin = od - (base_da if base_da > 0 else 0)
            print(f"  oracle_DA@{k:2d} = {od:.4f}  (margin vs base: {margin:+.4f})")

        out = {
            "cache": cache_name,
            "n_items": len(ds),
            "n_directional": n_dir,
            "ks": ks,
            "oracle": oracle,
            "base_da": base_da if base_da > 0 else None,
        }
        with open(os.path.join(OUT_DIR, f"oracle_{cache_name}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nOracle results saved to {OUT_DIR}/oracle_*.json")


def _write_per_sample_csv(all_samples, model_keys, path):
    """Write merged per-sample CSV for multiple models."""
    if not model_keys:
        return
    first = model_keys[0]
    n = len(all_samples[first])
    fieldnames = ["sample_idx"]
    for mk in model_keys:
        short = mk.split("/")[-1]
        fieldnames += [f"{short}_pred_label", f"{short}_pred_return",
                       f"{short}_abs_error", f"{short}_is_correct",
                       f"{short}_logp_argmax", f"{short}_margin_up_down"]
    fieldnames += ["actual_label", "actual_return"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i in range(n):
            row = {"sample_idx": i}
            for mk in model_keys:
                short = mk.split("/")[-1]
                s = all_samples[mk][i]
                row[f"{short}_pred_label"] = s["pred_label"]
                row[f"{short}_pred_return"] = s["pred_return"]
                row[f"{short}_abs_error"] = s["abs_error"]
                row[f"{short}_is_correct"] = s["is_correct"]
                row[f"{short}_logp_argmax"] = s["logp_argmax"]
                row[f"{short}_margin_up_down"] = s["margin_up_down"]
            row["actual_label"] = all_samples[first][i]["actual_label"]
            row["actual_return"] = all_samples[first][i]["actual_return"]
            writer.writerow(row)


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 5 Sup – eval + oracle")
    p.add_argument("--stage", choices=["audit", "topk_oracle", "paired"], default="audit")
    p.add_argument("--checkpoints", type=str, default="",
                   help="Comma-separated checkpoint paths to evaluate (beyond basemodel)")
    p.add_argument("--top-k", type=str, default="4,8,16,32,64",
                   help="Comma-separated K values for oracle (default: 4,8,16,32,64)")
    p.add_argument("--cache", type=str, default="",
                   help="Specific cache path (default: val + demo)")
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _out_dir = args.output_dir if args.output_dir else OUT_DIR
    os.makedirs(_out_dir, exist_ok=True)
    # Patch the module-level OUT_DIR for functions that reference it
    import hpo.phase5_sup_eval as _mod
    _mod.OUT_DIR = _out_dir

    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.stage == "audit":
        stage_audit(args)
    elif args.stage == "topk_oracle":
        stage_topk_oracle(args)
