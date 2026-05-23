"""Compare BaseModel vs Rollout vs DA models on Demo set.

Metrics: 1-step MAPE/DA + 10-step AR path_mape.
"""

import json, os, sys
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from config import DataConfig
from data_processor import get_datasets, collate_fn
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.pt")
TOKENIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer_config.json")
BASEMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")
ROLLOUT_PATH = os.path.join(PROJECT_ROOT, "trials", "phase6_rollout", "trial_010", "rollout_model.pt")
DAMODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "base_model.pt")  # fallback

BACKBONE = {
    "dim": 384, "depth": 3, "heads": 4, "num_kv_heads": 1,
    "dsa_windows": [None, 512, 512], "position_encoding": "rope",
    "rope_base": 10000.0, "dropout": 0.1323,
    "use_revin": False, "num_factor_tokens": 0,
}
TOKENIZER_BITS = 10
VOCAB = 1 << TOKENIZER_BITS


def _load_tokenizer(device):
    ckpt = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg and os.path.exists(TOKENIZER_CONFIG_PATH):
        with open(TOKENIZER_CONFIG_PATH) as f: cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"], strict=False)
    tok.eval(); tok.requires_grad_(False)
    return tok


def _load_model(device, checkpoint_path):
    model = KronosReasoningGPT(
        dim=BACKBONE["dim"], depth=BACKBONE["depth"], heads=BACKBONE["heads"],
        num_kv_heads=BACKBONE["num_kv_heads"], dsa_windows=BACKBONE["dsa_windows"],
        dropout=BACKBONE["dropout"], vocab_size_coarse=VOCAB, vocab_size_fine=VOCAB,
        position_encoding=BACKBONE["position_encoding"], rope_base=BACKBONE["rope_base"],
        use_revin=BACKBONE["use_revin"], num_factor_tokens=BACKBONE["num_factor_tokens"],
    ).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _load_demo_seq_stats():
    """Load per-sequence mean/std from demo cache for denormalization."""
    cache = torch.load(os.path.join(PROJECT_ROOT, "dataset_demo.pt"),
                       map_location="cpu", weights_only=False)
    ss = cache["seq_stats"]
    N = len(ss)
    means = torch.zeros(N, 6)
    stds = torch.zeros(N, 6)
    for i, s in enumerate(ss):
        means[i] = torch.as_tensor(s["mean"], dtype=torch.float32)
        stds[i] = torch.as_tensor(s["std"], dtype=torch.float32)
    return means, stds


@torch.no_grad()
def eval_1step(model, tokenizer, loader, device, means, stds):
    """1-step prediction on demo set."""
    all_preds, all_actuals = [], []
    sample_idx = 0
    for batch_data in tqdm(loader, desc="  1-step", leave=False):
        feats, _, tf, enc = batch_data
        feats = feats.to(device)
        B = feats.shape[0]
        if B == 0: continue

        idx_c, idx_f = tokenizer.encode(feats[:, :1023, :])
        actual_norm = feats[:, 1023, 0]

        t_min = tf["minute"][:, :1023].to(device).long()
        t_day = tf["day"][:, :1023].to(device).long()
        t_mon = tf["month"][:, :1023].to(device).long()
        t_yr  = tf["year"][:, :1023].to(device).long()

        lc, lf, _ = model(idx_c, idx_f, t_min, t_day, t_mon, t_yr, last_only=True)
        pc = lc[:, -1, :].argmax(dim=-1)
        pf = lf[:, -1, :].argmax(dim=-1)
        dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
        pred_norm = dec[:, 0, 0]

        # Denormalize using original seq_stats
        bm = means[sample_idx:sample_idx+B, 0].to(device)
        bs = stds[sample_idx:sample_idx+B, 0].clamp(min=1e-6).to(device)
        pred_log_ret = pred_norm * bs + bm
        actual_log_ret = actual_norm * bs + bm

        all_preds.append(pred_log_ret.cpu())
        all_actuals.append(actual_log_ret.cpu())
        sample_idx += B

    preds = torch.cat(all_preds).numpy().astype(np.float64)
    actuals = torch.cat(all_actuals).numpy().astype(np.float64)
    fin = np.isfinite(preds) & np.isfinite(actuals)
    preds, actuals = preds[fin], actuals[fin]

    pr = np.exp(np.clip(preds, -20, 20)); ar = np.exp(np.clip(actuals, -20, 20))
    mape = float(np.mean(np.abs((pr - ar) / np.maximum(np.abs(ar), 1e-6))) * 100)
    ps = np.where(preds >= 0, 1, -1); as_ = np.where(actuals >= 0, 1, -1)
    da = float(np.mean(ps == as_) * 100)
    return {"mape": round(mape, 4), "da": round(da, 2)}


@torch.no_grad()
def eval_ar10(model, tokenizer, loader, device, means, stds):
    """10-step autoregressive rollout → path_mape."""
    all_path_mape = []
    sample_idx = 0
    for batch_data in tqdm(loader, desc="  AR10", leave=False):
        feats, _, tf, enc = batch_data
        feats = feats.to(device)
        B = feats.shape[0]
        if B == 0: continue

        idx_c, idx_f = tokenizer.encode(feats)
        cur_c = idx_c[:, :1023]; cur_f = idx_f[:, :1023]
        tgt_c = idx_c[:, 1023:]; tgt_f = idx_f[:, 1023:]
        H = tgt_c.shape[1]
        if H < 1: continue

        # Denorm params for this batch
        bm = means[sample_idx:sample_idx+B, 0].unsqueeze(1)  # [B, 1]
        bs = stds[sample_idx:sample_idx+B, 0].clamp(min=1e-6).unsqueeze(1)
        sample_idx += B

        tz = torch.zeros(B, 1033, dtype=torch.long, device=device)
        tgt_dec = tokenizer.decode(tgt_c, tgt_f)
        actual_norm = tgt_dec[:, :, 0].cpu()
        actual_rets = (actual_norm * bs + bm)  # denormalized

        pred_rets = []
        for step in range(H):
            sl = cur_c.shape[1]
            lc, lf, _ = model(cur_c, cur_f, tz[:,:sl], tz[:,:sl], tz[:,:sl], tz[:,:sl], last_only=True)
            if not torch.isfinite(lc).all(): break
            pc = lc[:, -1, :].argmax(dim=-1)
            pf = lf[:, -1, :].argmax(dim=-1)
            dec = tokenizer.decode(pc.unsqueeze(1), pf.unsqueeze(1))
            pred_norm = dec[:, 0, 0].cpu()
            pred_rets.append(pred_norm * bs.squeeze(1) + bm.squeeze(1))
            cur_c = torch.cat([cur_c, pc.unsqueeze(1)], dim=1)
            cur_f = torch.cat([cur_f, pf.unsqueeze(1)], dim=1)

        if len(pred_rets) < H: continue
        pred_rets = torch.stack(pred_rets, dim=1)
        cum_pred = torch.cumsum(pred_rets, dim=1)
        cum_actual = torch.cumsum(actual_rets, dim=1)

        for step in range(H):
            pr = torch.exp(torch.clamp(cum_pred[:, step].float(), -20, 20))
            ar = torch.exp(torch.clamp(cum_actual[:, step].float(), -20, 20))
            denom = torch.clamp(torch.abs(ar), min=1e-6)
            valid = torch.isfinite(pr) & torch.isfinite(ar) & (denom > 0)
            if valid.sum() > 0:
                m = (torch.abs(pr[valid] - ar[valid]) / denom[valid]).mean().item() * 100
                all_path_mape.append(m)

    return {"path_mape": round(float(np.mean(all_path_mape)), 2) if all_path_mape else 999.0,
            "num_steps": len(all_path_mape)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model Comparison on Demo Set")
    print(f"  Device: {device}")

    # Load demo data (first 50 stocks for speed)
    print("Loading demo data...")
    _, _, demo_ds = get_datasets(include_demo=True)
    print(f"  Demo samples: {len(demo_ds)}")

    tok = _load_tokenizer(device)
    demo_ds.precompute_encodings(tok, device)

    # Load per-sequence mean/std for proper denormalization
    means, stds = _load_demo_seq_stats()

    loader = torch.utils.data.DataLoader(
        demo_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    results = {}

    # 1. BaseModel
    print("\n[1/3] BaseModel...")
    bm = _load_model(device, BASEMODEL_PATH)
    r1 = eval_1step(bm, tok, loader, device, means, stds)
    r10 = eval_ar10(bm, tok, loader, device, means, stds)
    results["BaseModel"] = {**r1, **r10}
    print(f"  MAPE={r1['mape']:.2f}%  DA={r1['da']:.1f}%  path_mape={r10['path_mape']:.1f}%")
    del bm

    # 2. Rollout model
    print("\n[2/3] Rollout (Phase 6 trial 010)...")
    if os.path.exists(ROLLOUT_PATH):
        rm = _load_model(device, ROLLOUT_PATH)
        r1 = eval_1step(rm, tok, loader, device, means, stds)
        r10 = eval_ar10(rm, tok, loader, device, means, stds)
        results["Rollout"] = {**r1, **r10}
        print(f"  MAPE={r1['mape']:.2f}%  DA={r1['da']:.1f}%  path_mape={r10['path_mape']:.1f}%")
        del rm
    else:
        print(f"  SKIP: {ROLLOUT_PATH} not found")

    # 3. DA model — same as BaseModel for now (DA training not yet adapted to DSA)
    print("\n[3/3] DA model (same as BaseModel — DA training pending DSA adaptation)...")
    results["DA"] = results.get("BaseModel", {}).copy()

    # Summary table
    print(f"\n{'='*70}")
    print(f"Demo Set Comparison")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'1-Step MAPE':>12} {'1-Step DA':>10} {'10-Step path_mape':>16}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<15} {r.get('mape', 0):>8.2f}%   {r.get('da', 0):>6.1f}%   {r.get('path_mape', 0):>10.1f}%")


if __name__ == "__main__":
    main()
