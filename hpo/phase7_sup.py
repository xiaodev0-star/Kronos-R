"""Phase 7 Supplementary Experiments (Round 2).

After Round 1 HPO identifies the best hyperparameters, this script runs
targeted follow-up experiments:

  A. Best configs + self-rollout (not just GT context)
  B. Combined: CI-trained model + temperature sampling at inference
  C. Fine-grained temperature calibration around best T
  D. Cross-confidence-level generalisation test
  E. Full-epoch training for top-3 configs

Designed to run unattended after Round 1 completes.
Total budget: ~4-5 hours.

Usage:
    python -m hpo.phase7_sup
"""

import copy, json, os, sys, time
from argparse import Namespace
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate
from posttrain.ci.eval_ci import compute_ci_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE7_DIR = os.path.join(PROJECT_ROOT, "trials", "phase7_ci")
SUP_DIR = os.path.join(PHASE7_DIR, "sup")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")

TOKENIZER_VOCAB = 1 << 10
PREFIX_LEN = 1023
HORIZON = 10

BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512],
    "position_encoding": "rope", "rope_base": 10000.0,
    "dropout": 0.1323, "use_revin": False, "num_factor_tokens": 0,
}


def _load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f:
            cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval()
    tok.requires_grad_(False)
    return tok


def _load_basemodel(device):
    bp = BACKBONE
    model = KronosReasoningGPT(
        dim=bp["dim"], depth=bp["depth"], heads=bp["heads"],
        num_kv_heads=bp["num_kv_heads"], dsa_windows=bp["dsa_windows"],
        dropout=bp["dropout"], vocab_size_coarse=TOKENIZER_VOCAB,
        vocab_size_fine=TOKENIZER_VOCAB,
        position_encoding=bp["position_encoding"], rope_base=bp["rope_base"],
        use_revin=bp["use_revin"], num_factor_tokens=bp["num_factor_tokens"],
    ).to(device)
    ckpt = torch.load(BASEMODEL_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _make_cfg():
    return Namespace(
        prefix_len=PREFIX_LEN, horizon=HORIZON,
        stride_ratio=DataConfig.stride_ratio,
        cache_dir=os.path.join(PROJECT_ROOT, "posttrain", "rollout", "cache"),
        max_stocks=0, cache_rebuild=False,
    )


def _build_val_loader(device, max_val_samples=500):
    cfg = _make_cfg()
    val_ds = RolloutWindowDataset("val", cfg=cfg, max_samples=max_val_samples, seed=59)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return val_loader


def _build_train_loader(device):
    cfg = _make_cfg()
    train_ds = RolloutWindowDataset("train", cfg=cfg, max_samples=0, seed=42)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True,
        collate_fn=rollout_collate, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader


# ═══════════════════════════════════════════════════════════════
# Experiment A: Best configs with self-rollout
# ═══════════════════════════════════════════════════════════════

def _train_with_self_rollout(model, tokenizer, train_loader, val_loader, device,
                              lr, conc_w, is_w, conf_level, top_k, kl_w,
                              gamma, max_updates, rollout_ratio, tdir):
    """Train with actual self-rollout (not GT context)."""
    import torch.optim as optim

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    ref_model.requires_grad_(False)

    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    total_updates = 0
    prefix_len = PREFIX_LEN
    horizon = HORIZON

    pbar = tqdm(total=max_updates, desc="  Self-rollout train")
    while total_updates < max_updates:
        for batch in train_loader:
            if total_updates >= max_updates:
                break

            feats = batch["features"].to(device=device, dtype=torch.float32)
            means = batch["means"].to(device=device, dtype=torch.float32)
            stds = batch["stds"].to(device=device, dtype=torch.float32)
            actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
            times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}

            B = feats.shape[0]
            if B == 0:
                continue

            idx_c_full, idx_f_full = tokenizer.encode(feats)

            # Self-rollout context construction
            context_c = idx_c_full[:, :prefix_len].clone()
            context_f = idx_f_full[:, :prefix_len].clone()

            was_training = model.training
            model.eval()
            with torch.no_grad():
                for step in range(horizon - 1):
                    cur_len = int(context_c.size(1))
                    cur_time = {k: times_f[k][:, :cur_len] for k in ("minute", "day", "month", "year")}
                    logits_c, logits_f, _ = model(
                        context_c, context_f,
                        cur_time["minute"], cur_time["day"],
                        cur_time["month"], cur_time["year"],
                        last_only=True,
                    )
                    use_self = rollout_ratio >= 1.0 or torch.rand(1, device=device).item() < rollout_ratio
                    if use_self:
                        pred_c = logits_c[:, -1, :].float().argmax(dim=-1)
                        pred_f = logits_f[:, -1, :].float().argmax(dim=-1)
                    else:
                        pred_c = idx_c_full[:, prefix_len + step]
                        pred_f = idx_f_full[:, prefix_len + step]
                    context_c = torch.cat([context_c, pred_c.unsqueeze(1)], dim=1)
                    context_f = torch.cat([context_f, pred_f.unsqueeze(1)], dim=1)
            if was_training:
                model.train()

            ctx_time = {k: times_f[k][:, :context_c.size(1)] for k in ("minute", "day", "month", "year")}
            target_c = idx_c_full[:, prefix_len:prefix_len + horizon]
            target_f = idx_f_full[:, prefix_len:prefix_len + horizon]

            opt.zero_grad(set_to_none=True)
            with (torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                logits_c, logits_f, _ = model(
                    context_c, context_f,
                    ctx_time["minute"], ctx_time["day"],
                    ctx_time["month"], ctx_time["year"],
                )
                r_c = logits_c[:, prefix_len - 1:prefix_len - 1 + horizon, :].float()
                r_f = logits_f[:, prefix_len - 1:prefix_len - 1 + horizon, :].float()

                # Step weights
                H_steps = int(r_c.size(1))
                steps_t = torch.arange(H_steps, device=r_c.device, dtype=torch.float32)
                step_w = 1.0 + gamma * steps_t / max(1, H_steps - 1)
                step_w = step_w / step_w.mean()

                # CE loss
                loss_c = F.cross_entropy(r_c.reshape(-1, r_c.size(-1)), target_c.reshape(-1), reduction="none").view(-1, H_steps)
                loss_f = F.cross_entropy(r_f.reshape(-1, r_f.size(-1)), target_f.reshape(-1), reduction="none").view(-1, H_steps)
                ce = ((loss_c + loss_f) * step_w.view(1, -1)).sum() / step_w.sum().clamp_min(1.0) / (loss_c.size(0))

                loss = ce

                # KL
                if kl_w > 0:
                    with torch.no_grad():
                        ref_lc, ref_lf, _ = ref_model(context_c, context_f,
                                                       ctx_time["minute"], ctx_time["day"],
                                                       ctx_time["month"], ctx_time["year"])
                    ref_rc = ref_lc[:, prefix_len - 1:prefix_len - 1 + horizon, :].float()
                    ref_rf = ref_lf[:, prefix_len - 1:prefix_len - 1 + horizon, :].float()
                    kl = F.kl_div(F.log_softmax(r_c, dim=-1), F.softmax(ref_rc, dim=-1), reduction="batchmean")
                    kl = kl + F.kl_div(F.log_softmax(r_f, dim=-1), F.softmax(ref_rf, dim=-1), reduction="batchmean")
                    loss = loss + kl_w * kl

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()

            total_updates += 1
            pbar.update(1)

    pbar.close()
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(tdir, "self_rollout_model.pt"))
    return model


# ═══════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_sampling_ci(model, tokenizer, val_loader, device, temp, num_samples, conf_level, feed_mode):
    """CI via temperature sampling."""
    model.eval()
    prefix_len = PREFIX_LEN
    horizon = HORIZON
    all_lower, all_upper, all_actual = [], [], []
    alpha = 1.0 - float(conf_level)
    low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    temp = max(float(temp), 1e-5)

    for batch in val_loader:
        feats = batch["features"].to(device=device, dtype=torch.float32)
        means = batch["means"].to(device=device, dtype=torch.float32)
        stds = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
        B = feats.shape[0]
        if B == 0: continue

        idx_c_full, idx_f_full = tokenizer.encode(feats)
        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()
        step_lower, step_upper = [], []

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {k: times_f[k][:, :cur_len] for k in ("minute", "day", "month", "year")}
            logits_c, logits_f, _ = model(context_c, context_f,
                                          cur_time["minute"], cur_time["day"],
                                          cur_time["month"], cur_time["year"], last_only=True)
            last_c = logits_c[:, -1, :].float()
            last_f = logits_f[:, -1, :].float()
            probs_c = torch.softmax(last_c / temp, dim=-1)
            probs_f = torch.softmax(last_f / temp, dim=-1)
            sc = torch.multinomial(probs_c, num_samples=int(num_samples), replacement=True)
            sf = torch.multinomial(probs_f, num_samples=int(num_samples), replacement=True)
            decoded = tokenizer.decode(sc, sf)
            pred_rets = decoded[:, :, 0].float() * stds[:, 0:1] + means[:, 0:1]
            sorted_r = pred_rets.sort(dim=1).values
            step_lower.append(sorted_r[:, max(0, min(int(num_samples)-1, int(low_q*int(num_samples))))].cpu())
            step_upper.append(sorted_r[:, max(0, min(int(num_samples)-1, int(high_q*int(num_samples))))].cpu())

            if step < horizon - 1:
                if feed_mode == "argmax":
                    next_c, next_f = last_c.argmax(dim=-1), last_f.argmax(dim=-1)
                else:
                    mp = int(num_samples) // 2
                    si = pred_rets.argsort(dim=1)
                    next_c = sc[torch.arange(B, device=device), si[:, mp]]
                    next_f = sf[torch.arange(B, device=device), si[:, mp]]
                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        all_lower.append(torch.stack(step_lower, dim=1))
        all_upper.append(torch.stack(step_upper, dim=1))
        all_actual.append(actual.cpu())

    if not all_lower: return {}
    pl = torch.cat(all_lower, dim=0).numpy()
    pu = torch.cat(all_upper, dim=0).numpy()
    aa = torch.cat(all_actual, dim=0).numpy()
    return compute_ci_metrics(pl, pu, aa, confidence_level=float(conf_level))


@torch.no_grad()
def _eval_dist_ci(model, tokenizer, val_loader, device, conf_level, top_k):
    """CI via distribution quantiles."""
    model.eval()
    prefix_len = PREFIX_LEN
    horizon = HORIZON
    all_lower, all_upper, all_actual = [], [], []
    alpha = 1.0 - float(conf_level)
    low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    K = min(int(top_k), TOKENIZER_VOCAB)

    for batch in val_loader:
        feats = batch["features"].to(device=device, dtype=torch.float32)
        means = batch["means"].to(device=device, dtype=torch.float32)
        stds = batch["stds"].to(device=device, dtype=torch.float32)
        actual = batch["actual_returns"].to(device=device, dtype=torch.float32)
        times_f = {k: v.to(device=device, dtype=torch.long) for k, v in batch["time"].items()}
        B = feats.shape[0]
        if B == 0: continue

        idx_c_full, idx_f_full = tokenizer.encode(feats)
        context_c = idx_c_full[:, :prefix_len].clone()
        context_f = idx_f_full[:, :prefix_len].clone()
        step_lower, step_upper = [], []

        for step in range(horizon):
            cur_len = int(context_c.size(1))
            cur_time = {k: times_f[k][:, :cur_len] for k in ("minute", "day", "month", "year")}
            logits_c, logits_f, _ = model(context_c, context_f,
                                          cur_time["minute"], cur_time["day"],
                                          cur_time["month"], cur_time["year"], last_only=True)
            last_c = logits_c[:, -1, :].float()
            last_f = logits_f[:, -1, :].float()
            probs_c = F.softmax(last_c, dim=-1)
            probs_f = F.softmax(last_f, dim=-1)
            top_pc, top_ic = torch.topk(probs_c, k=K, dim=-1)
            top_pf, top_if = torch.topk(probs_f, k=K, dim=-1)
            top_pc = top_pc / top_pc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            top_pf = top_pf / top_pf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pair_probs = top_pc.unsqueeze(-1) * top_pf.unsqueeze(-2)
            B_s, Kc, Kf = pair_probs.shape

            pc_flat = top_ic.unsqueeze(-1).expand(B_s, Kc, Kf).reshape(B_s, Kc * Kf)
            pf_flat = top_if.unsqueeze(-2).expand(B_s, Kc, Kf).reshape(B_s, Kc * Kf)
            with torch.no_grad():
                decoded = tokenizer.decode(pc_flat, pf_flat)[..., 0].float()
                returns = decoded.view(B_s, Kc, Kf)
                ret_denorm = returns * stds[:, 0].view(B_s, 1, 1) + means[:, 0].view(B_s, 1, 1)

            ret_flat = ret_denorm.view(B_s, -1)
            prob_flat = pair_probs.view(B_s, -1)
            sort_idx = ret_flat.argsort(dim=-1)
            sorted_ret = ret_flat.gather(-1, sort_idx)
            sorted_prob = prob_flat.gather(-1, sort_idx)
            cum_prob = sorted_prob.cumsum(dim=-1)
            cum_prob = cum_prob / cum_prob[..., -1:].clamp_min(1e-8)
            Np = cum_prob.shape[-1]
            idx_low = (cum_prob >= low_q).float().argmax(dim=-1).clamp(0, Np-1)
            idx_high = (cum_prob >= high_q).float().argmax(dim=-1).clamp(0, Np-1)
            rows = torch.arange(B_s, device=ret_flat.device)
            step_lower.append(sorted_ret[rows, idx_low].cpu())
            step_upper.append(sorted_ret[rows, idx_high].cpu())

            if step < horizon - 1:
                next_c = last_c.argmax(dim=-1)
                next_f = last_f.argmax(dim=-1)
                context_c = torch.cat([context_c, next_c.unsqueeze(1)], dim=1)
                context_f = torch.cat([context_f, next_f.unsqueeze(1)], dim=1)

        all_lower.append(torch.stack(step_lower, dim=1))
        all_upper.append(torch.stack(step_upper, dim=1))
        all_actual.append(actual.cpu())

    if not all_lower: return {}
    pl = torch.cat(all_lower, dim=0).numpy()
    pu = torch.cat(all_upper, dim=0).numpy()
    aa = torch.cat(all_actual, dim=0).numpy()
    return compute_ci_metrics(pl, pu, aa, confidence_level=float(conf_level))


# ═══════════════════════════════════════════════════════════════
# Main sup experiment runner
# ═══════════════════════════════════════════════════════════════

def load_round1_best():
    """Load best configs from Round 1 results."""
    results = {"sampling": None, "training": None}

    # Best sampling config
    sampling_csv = os.path.join(PHASE7_DIR, "summary_sampling.csv")
    if os.path.exists(sampling_csv):
        import csv
        with open(sampling_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in list(r.keys()):
                try: r[k] = float(r[k])
                except (ValueError, TypeError): pass
        if rows:
            results["sampling"] = min(rows, key=lambda r: r.get("avg_interval_score", 999))

    # Best training config
    training_csv = os.path.join(PHASE7_DIR, "summary_training.csv")
    if os.path.exists(training_csv):
        import csv
        with open(training_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in list(r.keys()):
                try: r[k] = float(r[k])
                except (ValueError, TypeError): pass
            if "value" in r:
                r["avg_interval_score"] = r["value"]
        if rows:
            results["training"] = min(rows, key=lambda r: r.get("avg_interval_score", 999))

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 7 Supplementary Experiments")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    os.makedirs(SUP_DIR, exist_ok=True)

    # ── Load Round 1 results ──
    best = load_round1_best()
    print(f"\nBest Sampling: {best['sampling']}")
    print(f"Best Training: {best['training']}")

    if best["sampling"] is None and best["training"] is None:
        print("ERROR: No Round 1 results found. Run phase7_ci_sampling and phase7_ci first.")
        return

    tokenizer = _load_tokenizer(device)
    val_loader = _build_val_loader(device, max_val_samples=500)

    all_sup_results = {}

    # ═══════════════════════════════════════════════════════════
    # Sup A: Fine-grained temperature sweep around best
    # ═══════════════════════════════════════════════════════════
    if best["sampling"]:
        bs = best["sampling"]
        best_t = bs.get("temperature", 1.0)
        print(f"\n{'='*60}")
        print(f"Sup A: Fine-grained T sweep around T={best_t}")
        print(f"{'='*60}")

        model = _load_basemodel(device)
        fine_temps = sorted(set([
            max(0.1, best_t * m) for m in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
        ]))
        # Also add best sampling params
        best_n = int(bs.get("num_samples", 64))
        best_conf = bs.get("confidence_level", 0.80)
        best_fm = bs.get("feed_mode", "argmax")

        fine_results = []
        for t in fine_temps:
            t = round(t, 2)
            tag = f"supA_T{t}_N{best_n}_C{best_conf}_{best_fm}"
            rpath = os.path.join(SUP_DIR, f"{tag}.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    fine_results.append(json.load(f))
                continue

            t0 = time.time()
            m = _eval_sampling_ci(model, tokenizer, val_loader, device,
                                   temp=t, num_samples=best_n,
                                   conf_level=best_conf, feed_mode=best_fm)
            elapsed = time.time() - t0
            row = {"temperature": t, "num_samples": best_n,
                   "confidence_level": best_conf, "feed_mode": best_fm,
                   **{k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))},
                   "elapsed_s": round(elapsed, 1)}
            fine_results.append(row)
            with open(rpath, "w") as f:
                json.dump(row, f, indent=2)
            print(f"  T={t:.2f}  IS={m.get('avg_interval_score', '?'):.6f}  "
                  f"cov={m.get('coverage', '?'):.4f}  w={m.get('avg_width', '?'):.6f}")

        best_fine = min(fine_results, key=lambda r: r.get("avg_interval_score", 999))
        all_sup_results["supA_fine_temp"] = {
            "best_T": best_fine["temperature"],
            "best_IS": best_fine["avg_interval_score"],
            "all": fine_results,
        }
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Sup B: Cross-confidence generalisation
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Sup B: Cross-confidence generalisation")
    print(f"{'='*60}")

    # Test best sampling config at all confidence levels
    if best["sampling"]:
        bs = best["sampling"]
        model = _load_basemodel(device)
        cross_conf_results = []
        for conf in [0.68, 0.80, 0.90, 0.95]:
            tag = f"supB_sampling_C{conf}"
            rpath = os.path.join(SUP_DIR, f"{tag}.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    cross_conf_results.append(json.load(f))
                continue
            t0 = time.time()
            m = _eval_sampling_ci(model, tokenizer, val_loader, device,
                                   temp=bs.get("temperature", 1.0),
                                   num_samples=int(bs.get("num_samples", 64)),
                                   conf_level=conf,
                                   feed_mode=bs.get("feed_mode", "argmax"))
            elapsed = time.time() - t0
            row = {"confidence_level": conf, **{k: v for k, v in m.items()
                   if isinstance(v, (int, float, str, bool))}, "elapsed_s": round(elapsed, 1)}
            cross_conf_results.append(row)
            with open(rpath, "w") as f:
                json.dump(row, f, indent=2)
            print(f"  C={conf:.0%}  IS={m.get('avg_interval_score', '?'):.6f}  "
                  f"cov={m.get('coverage', '?'):.4f}  w={m.get('avg_width', '?'):.6f}")
        all_sup_results["supB_cross_conf_sampling"] = cross_conf_results
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Sup C: Combined — best CI-trained model + sampling inference
    # ═══════════════════════════════════════════════════════════
    if best["training"]:
        bt = best["training"]
        print(f"\n{'='*60}")
        print(f"Sup C: Best CI-trained model + sampling inference")
        print(f"{'='*60}")

        # Find the best training trial directory
        trial_dir = bt.get("trial_dir") or bt.get("dir_name")
        if trial_dir:
            ci_model_path = os.path.join(PHASE7_DIR, str(trial_dir), "ci_model.pt")
        else:
            ci_model_path = None
            for d in os.listdir(PHASE7_DIR):
                if d.startswith("trial_train_"):
                    p = os.path.join(PHASE7_DIR, d, "ci_model.pt")
                    if os.path.exists(p):
                        ci_model_path = p
                        break

        if ci_model_path and os.path.exists(ci_model_path):
            model = _load_basemodel(device)
            ckpt = torch.load(ci_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()

            combined_results = []
            # Test the CI-trained model with distribution quantiles AND sampling
            for method in ["dist", "sampling"]:
                for conf in [0.68, 0.80, 0.90]:
                    tag = f"supC_{method}_C{conf}"
                    rpath = os.path.join(SUP_DIR, f"{tag}.json")
                    if os.path.exists(rpath):
                        with open(rpath) as f:
                            combined_results.append(json.load(f))
                        continue
                    t0 = time.time()
                    if method == "dist":
                        m = _eval_dist_ci(model, tokenizer, val_loader, device,
                                           conf_level=conf,
                                           top_k=int(bt.get("ci_top_k", 32)))
                    else:
                        m = _eval_sampling_ci(model, tokenizer, val_loader, device,
                                               temp=bt.get("oracle_temp", 1.0) if "oracle_temp" in bt else 1.0,
                                               num_samples=64,
                                               conf_level=conf, feed_mode="argmax")
                    elapsed = time.time() - t0
                    row = {"method": method, "confidence_level": conf,
                           **{k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))},
                           "elapsed_s": round(elapsed, 1)}
                    combined_results.append(row)
                    with open(rpath, "w") as f:
                        json.dump(row, f, indent=2)
                    print(f"  {method} C={conf:.0%}  IS={m.get('avg_interval_score', '?'):.6f}  "
                          f"cov={m.get('coverage', '?'):.4f}")
            all_sup_results["supC_combined"] = combined_results
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
        else:
            print("  SKIP: no CI-trained model checkpoint found")

    # ═══════════════════════════════════════════════════════════
    # Sup D: Self-rollout training for top-3 training configs
    # ═══════════════════════════════════════════════════════════
    if best["training"]:
        print(f"\n{'='*60}")
        print(f"Sup D: Self-rollout for top configs")
        print(f"{'='*60}")

        # Load top training configs from summary
        training_csv = os.path.join(PHASE7_DIR, "summary_training.csv")
        if os.path.exists(training_csv):
            import csv
            with open(training_csv, newline="") as f:
                all_train = list(csv.DictReader(f))
            for r in all_train:
                for k in list(r.keys()):
                    try: r[k] = float(r[k])
                    except (ValueError, TypeError): pass
                if "value" in r:
                    r["avg_interval_score"] = r["value"]
            top3 = sorted(all_train, key=lambda r: r.get("avg_interval_score", 999))[:3]

            train_loader = _build_train_loader(device)
            for i, cfg_row in enumerate(top3):
                tag = f"supD_top{i+1}_selfrollout"
                tdir = os.path.join(SUP_DIR, tag)
                rpath = os.path.join(tdir, "result.json")
                if os.path.exists(rpath):
                    print(f"  Top{i+1}: already done, skipping")
                    continue

                os.makedirs(tdir, exist_ok=True)
                print(f"\n  Top{i+1}: conc_w={cfg_row.get('concentration_weight','?')} "
                      f"is_w={cfg_row.get('interval_score_weight','?')} "
                      f"topk={cfg_row.get('ci_top_k','?')} lr={cfg_row.get('lr','?'):.2e}")

                model = _load_basemodel(device)
                lr = float(cfg_row.get("lr", 2e-5))
                conc_w = float(cfg_row.get("concentration_weight", 1.0))
                is_w = float(cfg_row.get("interval_score_weight", 0.3))
                conf_level = float(cfg_row.get("ci_confidence_level", 0.80))
                top_k = int(cfg_row.get("ci_top_k", 32))
                kl_w = float(cfg_row.get("kl_weight", 0.02))
                gamma = float(cfg_row.get("step_weight_gamma", 0.5))
                max_up = min(480, int(cfg_row.get("max_updates", 480)))

                t0 = time.time()
                model = _train_with_self_rollout(
                    model, tokenizer, train_loader, val_loader, device,
                    lr=lr, conc_w=conc_w, is_w=is_w, conf_level=conf_level,
                    top_k=top_k, kl_w=kl_w, gamma=gamma, max_updates=max_up,
                    rollout_ratio=0.85, tdir=tdir,
                )
                elapsed = time.time() - t0

                # Evaluate
                m_dist = _eval_dist_ci(model, tokenizer, val_loader, device, conf_level, top_k)
                m_samp = _eval_sampling_ci(model, tokenizer, val_loader, device,
                                            temp=1.0, num_samples=64,
                                            conf_level=conf_level, feed_mode="argmax")

                result = {
                    "config": cfg_row,
                    "elapsed_min": round(elapsed / 60, 1),
                    "ci_dist": {k: v for k, v in m_dist.items() if isinstance(v, (int, float, str, bool))},
                    "ci_sampling_T1.0": {k: v for k, v in m_samp.items() if isinstance(v, (int, float, str, bool))},
                }
                with open(rpath, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    dist IS={m_dist.get('avg_interval_score','?'):.6f}  "
                      f"samp IS={m_samp.get('avg_interval_score','?'):.6f}  "
                      f"time={elapsed/60:.1f}min")

                del model
                if device.type == "cuda": torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Save final summary
    # ═══════════════════════════════════════════════════════════
    summary_path = os.path.join(SUP_DIR, "sup_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_sup_results, f, indent=2, ensure_ascii=False)
    print(f"\nSup experiments complete. Summary: {summary_path}")

    # Quick cross-method comparison
    print(f"\n{'='*60}")
    print(f"FINAL CROSS-METHOD COMPARISON")
    print(f"{'='*60}")

    if best["sampling"]:
        print(f"Best Sampling: IS={best['sampling'].get('avg_interval_score','?'):.6f}  "
              f"T={best['sampling'].get('temperature','?')}  "
              f"cov={best['sampling'].get('coverage','?'):.4f}")
    if best["training"]:
        print(f"Best Training: IS={best['training'].get('avg_interval_score','?'):.6f}  "
              f"conc_w={best['training'].get('concentration_weight','?')}  "
              f"cov={best['training'].get('coverage','?'):.4f}")

    if all_sup_results.get("supC_combined"):
        for r in all_sup_results["supC_combined"]:
            print(f"Combined [{r.get('method','?')} C={r.get('confidence_level','?'):.0%}]: "
                  f"IS={r.get('avg_interval_score','?'):.6f}  "
                  f"cov={r.get('coverage','?'):.4f}")


if __name__ == "__main__":
    main()
