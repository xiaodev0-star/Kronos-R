"""Phase 8 V4: Bold Prediction — 统一实验框架

支持 Phase8(旧loss) 和 V4(新loss) 的训练、评估、可视化对比。
用法:
    python experiment_runner.py --mode phase8   # Phase8 基线
    python experiment_runner.py --mode v4       # V4 Bold Prediction
    python experiment_runner.py --mode compare  # 对比已有结果
"""
from __future__ import annotations
import copy, json, math, os, sys, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch, torch.nn.functional as F, torch.optim as optim
from contextlib import nullcontext
from tqdm import tqdm

from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import resolve_project_path
from reproducibility import set_global_seed

set_global_seed(42, deterministic=False)

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
PREFIX_LEN, HORIZON = 1023, 10
BATCH_SIZE, GA = 2, 8
LR = 9.59e-6
CKPT_EVERY = 20
EVAL_BATCHES_VAL = 200
EVAL_BATCHES_DEMO = 200
ADA_THRESHOLDS = [0.001, 0.003, 0.005, 0.010, 0.020]
ZC_RATIO = 0.1

EXPL_TEMP, NEFTUNE_A, N_TRAJ, TOP_K = 0.414, 2.5, 4, 16
ORACLE_MAG_PEN = 3.99

BACKBONE = dict(dim=384, depth=3, heads=4, num_kv_heads=1,
                dsa_windows=[None, 512, 512], position_encoding="rope",
                rope_base=10000.0, dropout=0.1323, use_revin=False, num_factor_tokens=0)
VOCAB = 1024

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def amp_dt(dev):
    if dev.type != "cuda": return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def autocast(dev, dt, enabled):
    if not enabled: return nullcontext()
    try: return torch.amp.autocast(device_type="cuda", dtype=dt)
    except: return torch.cuda.amp.autocast(dtype=dt)

def load_tokenizer(dev):
    p = resolve_project_path("checkpoints/tokenizer.pt")
    cp = resolve_project_path("checkpoints/tokenizer_config.json")
    ck = torch.load(p, map_location=dev, weights_only=False)
    cfg = ck.get("config", {})
    if not cfg and os.path.exists(cp):
        with open(cp) as f: cfg = json.load(f)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(dev)
    tok.load_state_dict(ck["model_state_dict"], strict=False)
    tok.eval(); tok.requires_grad_(False)
    return tok

def load_basemodel(dev):
    m = KronosReasoningGPT(vocab_size_coarse=VOCAB, vocab_size_fine=VOCAB, **BACKBONE).to(dev)
    p = resolve_project_path("checkpoints/base_model.pt")
    if os.path.exists(p):
        ck = torch.load(p, map_location=dev, weights_only=False)
        m.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    return m

class CachedDS(torch.utils.data.Dataset):
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.features = payload["features"].to(torch.float32)
        self.sector_ids = payload["sector_ids"].to(torch.long)
        self.time = {k: v.to(torch.long) for k, v in payload["time_features"].items()}
        self.actual = payload["actual_returns"].to(torch.float32)
        stats = payload["seq_stats"]
        self.means = torch.stack([torch.as_tensor(s["mean"], dtype=torch.float32) for s in stats])
        self.stds = torch.stack([torch.as_tensor(s["std"], dtype=torch.float32) for s in stats])
    def __len__(self): return len(self.features)
    def __getitem__(self, i):
        return {"features": self.features[i], "time": {k: v[i] for k, v in self.time.items()},
                "actual_returns": self.actual[i], "means": self.means[i], "stds": self.stds[i]}

def make_loaders(dev, cache_dir, bs=BATCH_SIZE):
    ds_tr = CachedDS(os.path.join(cache_dir, "rollout_train.pt"))
    ds_va = CachedDS(os.path.join(cache_dir, "rollout_val.pt"))
    lk = dict(num_workers=0, pin_memory=(dev.type == "cuda"))
    return (torch.utils.data.DataLoader(ds_tr, bs, shuffle=True, **lk),
            torch.utils.data.DataLoader(ds_va, bs*2, shuffle=False, **lk))

# ═══════════════════════════════════════════════════════════════════════
# Expected Returns
# ═══════════════════════════════════════════════════════════════════════
def expected_returns(tok, lc, lf, means, stds, top_k=16, sharp=1.0):
    B, H, _ = lc.shape
    K = min(top_k, lc.size(-1))
    tlc, tix = torch.topk(lc.float(), K, dim=-1)
    tlf, fix = torch.topk(lf.float(), K, dim=-1)
    pc = F.softmax(tlc / sharp, dim=-1); pc = pc / pc.sum(-1, keepdim=True).clamp_min(1e-8)
    pf = F.softmax(tlf / sharp, dim=-1); pf = pf / pf.sum(-1, keepdim=True).clamp_min(1e-8)
    jp = pc.unsqueeze(-1) * pf.unsqueeze(-2)
    p_c = tix.unsqueeze(-1).expand(B,H,K,K).reshape(B*H, K*K)
    p_f = fix.unsqueeze(-2).expand(B,H,K,K).reshape(B*H, K*K)
    with torch.no_grad():
        dec = tok.decode(p_c, p_f)[..., 0].float()
        rg = dec.view(B,H,K,K) * stds[:,0:1,None,None] + means[:,0:1,None,None]
    return (jp * rg).sum((-1, -2))

# ═══════════════════════════════════════════════════════════════════════
# Oracle Filter
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def oracle_explore(model, tok, ic, iff, times, means, stds, actual, dev, amp_en, amp_dt_):
    B, N, H = ic.size(0), N_TRAJ, HORIZON
    BN = B * N
    tc = ic[:,:PREFIX_LEN].unsqueeze(1).expand(B,N,PREFIX_LEN).reshape(BN, PREFIX_LEN)
    tf = iff[:,:PREFIX_LEN].unsqueeze(1).expand(B,N,PREFIX_LEN).reshape(BN, PREFIX_LEN)
    tr = torch.zeros(B, N, H, device=dev)
    te = {}
    for k in ("minute","day","month","year"):
        te[k] = times[k][:,:PREFIX_LEN].unsqueeze(1).expand(B,N,PREFIX_LEN).reshape(BN, PREFIX_LEN)
    for s in range(H):
        cl = PREFIX_LEN + s
        cc, cf = tc[:,:cl], tf[:,:cl]
        ct = {k: v[:,:cl] for k,v in te.items()}
        with autocast(dev, amp_dt_, amp_en):
            lc, lf, _ = model(cc, cf, ct["minute"], ct["day"], ct["month"], ct["year"],
                              last_only=True, neftune_alpha=NEFTUNE_A)
        pc = torch.multinomial(F.softmax(lc[:,-1,:].float()/max(1e-4,EXPL_TEMP), -1), 1)
        pf = torch.multinomial(F.softmax(lf[:,-1,:].float()/max(1e-4,EXPL_TEMP), -1), 1)
        dec = tok.decode(pc, pf)[...,0].float()
        sr = dec * stds.unsqueeze(1).expand(B,N,6).reshape(BN,6)[:,0:1] + means.unsqueeze(1).expand(B,N,6).reshape(BN,6)[:,0:1]
        tr[:,:,s] = sr.view(B,N)
        tc = torch.cat([tc, pc], 1); tf = torch.cat([tf, pf], 1)
        for k in te:
            nt = times[k][:,PREFIX_LEN+s:PREFIX_LEN+s+1]
            te[k] = torch.cat([te[k], nt.unsqueeze(1).expand(B,N,1).reshape(BN,1)], 1)
    pr = tr.sum(2); ap = actual[:,:H].sum(1)
    cd = (pr * ap.unsqueeze(1)) > 0
    iv = cd.any(1) & (ap.abs() > 1e-6)
    err = torch.abs(pr - ap.unsqueeze(1))
    mp = torch.clamp(ap.abs().unsqueeze(1) - pr.abs(), min=0)
    err = err + ORACLE_MAG_PEN * mp; err[~cd] = float('inf')
    bi = err.argmin(1)
    fl = PREFIX_LEN + H
    gc = tc.view(B,N,fl).gather(1, bi.view(B,1,1).expand(B,1,fl)).squeeze(1)
    gf = tf.view(B,N,fl).gather(1, bi.view(B,1,1).expand(B,1,fl)).squeeze(1)
    gtc, gtf = ic[:,:fl], iff[:,:fl]
    m = iv.float().view(B,1)
    gc = (gc.float()*m + gtc.float()*(1-m)).long()
    gf = (gf.float()*m + gtf.float()*(1-m)).long()
    return gc, gf, iv

# ═══════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════
# --- Phase 8 (old asymmetric) ---
def loss_phase8(exp, act, alpha=3.0, beta=10.0, tw=2.0, tr_=0.5):
    ae = torch.abs(exp - act)
    dp = exp * act
    is_d = act.abs() > 1e-4
    is_w = (dp < 0) & is_d
    is_t = (dp > 0) & (exp.abs() < act.abs() * tr_) & is_d
    w = torch.ones_like(ae)
    w = torch.where(is_w, alpha + beta * exp.abs(), w)
    w = torch.where(is_t, tw, w)
    return (ae * w).mean()

# --- V4 Bold Prediction ---
def loss_mag(exp, act, uw=2.0):
    """L2 magnitude matching — pushes away from zero."""
    pm, am = exp.abs(), act.abs()
    w = torch.where(pm < am, uw, 1.0)
    return (w * (pm - am).pow(2)).mean()

def loss_dir(exp, act, dw=1.0, mw=0.5, scale=100.0, mr=0.3):
    """Direction calibration — BCE-style, gradient at 0 is non-zero."""
    sig = exp * act * scale
    d_loss = F.softplus(-sig).mean()
    ic = (exp * act) > 0
    ib = exp.abs() > act.abs() * mr
    m = ic & ib
    if m.any():
        c_loss = (exp[m].abs() - act[m].abs()).pow(2).mean()
    else:
        c_loss = torch.tensor(0.0, device=exp.device)
    return dw * d_loss + mw * c_loss

# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, tok, loader, dev, dt, max_batches=EVAL_BATCHES_VAL):
    model.eval(); amp_en = dev.type == "cuda"
    all_p, all_a = [], []
    n = 0
    for b in loader:
        feats = b["features"].to(dev, dtype=torch.float32)
        means, stds = b["means"].to(dev, dtype=torch.float32), b["stds"].to(dev, dtype=torch.float32)
        times = {k: v.to(dev, dtype=torch.long) for k,v in b["time"].items()}
        ic, iff = tok.encode(feats)
        cc, cf = ic[:,:PREFIX_LEN].clone(), iff[:,:PREFIX_LEN].clone()
        pr = []
        for s in range(HORIZON):
            sl = cc.size(1)
            ct = {k: v[:,:sl] for k,v in times.items()}
            with autocast(dev, dt, amp_en):
                lc, lf, _ = model(cc, cf, ct["minute"], ct["day"], ct["month"], ct["year"], last_only=True)
            pc, pf_ = lc[:,-1,:].argmax(-1), lf[:,-1,:].argmax(-1)
            dec = tok.decode(pc.unsqueeze(1), pf_.unsqueeze(1))
            ret = dec[:,0,0].cpu().float() * stds[:,0].cpu() + means[:,0].cpu()
            pr.append(ret)
            if s < HORIZON - 1:
                cc = torch.cat([cc, pc.unsqueeze(1)], 1)
                cf = torch.cat([cf, pf_.unsqueeze(1)], 1)
        all_p.append(torch.stack(pr, 1)); all_a.append(b["actual_returns"].cpu())
        n += 1
        if n >= max_batches: break
    pred = torch.cat(all_p, 0).numpy()
    actual = torch.cat(all_a, 0).numpy()
    pt, at = torch.from_numpy(pred), torch.from_numpy(actual)
    # Legacy
    pc, ac = np.cumsum(pred, 1), np.cumsum(actual, 1)
    prr = np.exp(np.clip(pc[:,-1], -50, 50))
    arr = np.exp(np.clip(ac[:,-1], -50, 50))
    path_mape = float(np.mean(np.abs(prr - arr) / np.maximum(np.abs(arr), 1e-4)) * 100)
    ps = np.where(pred >= 0, 1, -1); as_ = np.where(actual >= 0, 1, -1)
    da = float(np.mean(ps == as_) * 100)
    # Bold metrics
    pa, aa = pt.abs(), at.abs()
    mag_ratio = float(pa.std() / aa.std().clamp_min(1e-8))
    boldness = float(pa.mean() / aa.mean().clamp_min(1e-8))
    is_d = aa > 1e-4
    zc = float((pa < aa * ZC_RATIO)[is_d].float().mean()) if is_d.any() else 0.0
    pst = (pt >= 0).float() * 2 - 1; ast = (at >= 0).float() * 2 - 1
    ada = {}
    for tau in ADA_THRESHOLDS:
        conf = pa > tau; cov = float(conf.float().mean())
        if conf.sum() > 0:
            av = float((pst[conf] == ast[conf]).float().mean().item() * 100)
        else: av = float('nan')
        pct = int(tau * 10000)
        ada[f"ada_{pct:03d}"] = round(av, 2); ada[f"ada_{pct:03d}_cov"] = round(cov * 100, 1)
    ic_ = (pt * at) > 0
    dir_err = float((~ic_).float().mean() * 100)
    mag_err = float((pa[ic_] - aa[ic_]).abs().mean().item()) if ic_.any() else float('nan')
    # Per-step
    psa = pa.mean(0).numpy(); asa = aa.mean(0).numpy()
    psda = [float((ps[:,s] == as_[:,s]).mean() * 100) for s in range(HORIZON)]
    return {"path_mape": path_mape, "da": da, "mag_ratio": mag_ratio, "boldness": boldness,
            "zc_ratio": zc, "dir_err": dir_err, "mag_err": mag_err, **ada,
            "pred_flat": pred.flatten(), "actual_flat": actual.flatten(),
            "pred_abs": pa.numpy().flatten(), "actual_abs": aa.numpy().flatten(),
            "ps_mag": psa, "as_mag": asa, "ps_da": psda}

# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════
def plot_diag(m, upd, path, tag=""):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    # Scatter
    ax = fig.add_subplot(gs[0,0])
    ax.scatter(m["actual_flat"], m["pred_flat"], alpha=0.12, s=3, c='steelblue')
    lim = max(abs(m["actual_flat"]).max(), abs(m["pred_flat"]).max(), 0.01) * 1.1
    ax.plot([-lim,lim],[-lim,lim],'r--',lw=0.8); ax.plot([-lim,lim],[0,0],'k--',lw=0.8,alpha=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set(xlabel='Actual', ylabel='Predicted', title=f'{tag} Pred vs Actual (upd={upd})')
    # Magnitude hist
    ax2 = fig.add_subplot(gs[0,1])
    mx = max(np.percentile(m["actual_abs"], 99), 0.01)
    bins = np.linspace(0, mx, 40)
    ax2.hist(m["actual_abs"], bins, alpha=0.5, label='|actual|', color='green', density=True)
    ax2.hist(m["pred_abs"], bins, alpha=0.5, label='|pred|', color='orange', density=True)
    ax2.set(xlabel='|Return|', ylabel='Density',
            title=f'|Pred| vs |Actual| (boldness={m["boldness"]:.3f})')
    ax2.legend(fontsize=8)
    # Per-step mag
    ax3 = fig.add_subplot(gs[0,2])
    s = np.arange(1, HORIZON+1)
    ax3.bar(s-0.2, m["as_mag"]*100, 0.35, label='|actual|%', color='green', alpha=0.7)
    ax3.bar(s+0.2, m["ps_mag"]*100, 0.35, label='|pred|%', color='orange', alpha=0.7)
    ax3.set(xlabel='Step', ylabel='Mean |Return| (%)', title='Per-Step Magnitude')
    ax3.legend(fontsize=8)
    # Per-step DA
    ax4 = fig.add_subplot(gs[1,0])
    ax4.bar(s, m["ps_da"], color='steelblue', alpha=0.7)
    ax4.axhline(50, color='red', ls='--', lw=1, label='random')
    ax4.set(xlabel='Step', ylabel='DA (%)', ylim=(40, 65),
            title=f'Per-Step DA (overall={m["da"]:.1f}%)')
    ax4.legend(fontsize=8)
    # ADA
    ax5 = fig.add_subplot(gs[1,1])
    avs, cvs, tls = [], [], []
    for tau in ADA_THRESHOLDS:
        pct = int(tau*10000)
        avs.append(m.get(f"ada_{pct:03d}", float('nan')))
        cvs.append(m.get(f"ada_{pct:03d}_cov", 0))
        tls.append(f"{tau*100:.1f}%")
    xp = np.arange(len(tls))
    ax5.bar(xp, avs, color='coral', alpha=0.7, label='ADA%')
    ax5.axhline(50, color='red', ls='--', lw=1)
    ax5t = ax5.twinx()
    ax5t.plot(xp, cvs, 'go-', lw=2, ms=6, label='Coverage%')
    ax5.set_xticks(xp); ax5.set_xticklabels(tls)
    ax5.set(xlabel='Threshold', ylabel='ADA (%)', ylim=(30, 70))
    ax5t.set_ylabel('Coverage (%)'); ax5t.set_ylim(0, 100)
    l1, lb1 = ax5.get_legend_handles_labels(); l2, lb2 = ax5t.get_legend_handles_labels()
    ax5.legend(l1+l2, lb1+lb2, fontsize=8, loc='lower right')
    # Summary
    ax6 = fig.add_subplot(gs[1,2]); ax6.axis('off')
    txt = (f"{'═'*42}\n  {tag} Metrics (update={upd})\n{'═'*42}\n\n"
           f"  Magnitude Ratio : {m['mag_ratio']:.3f}  (ideal=1.0)\n"
           f"  Boldness        : {m['boldness']:.3f}  (ideal=1.0)\n"
           f"  Zero-Collapse % : {m['zc_ratio']*100:.1f}% (ideal=0%)\n\n"
           f"  Dir Error       : {m['dir_err']:.1f}%\n"
           f"  Mag Error       : {m['mag_err']:.5f}\n\n"
           f"  DA              : {m['da']:.1f}%\n"
           f"  path_MAPE       : {m['path_mape']:.3f}%\n\n"
           f"  ADA @0.5%       : {m.get('ada_0050','N/A')}%\n"
           f"  Cov @0.5%       : {m.get('ada_0050_cov','N/A')}%\n{'═'*42}")
    ax6.text(0.05, 0.95, txt, transform=ax6.transAxes, fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    fig.suptitle(f'{tag} — Update {upd}', fontsize=14, fontweight='bold')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════
def train(mode, max_updates, cache_dir, out_dir, val_loader=None, demo_loader=None,
          v4_mag_w=2.0, v4_dir_w=1.0, v4_under_w=2.0, v4_dir_mw=0.5):
    dev = torch.device("cuda"); dt = amp_dt(dev)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    tok = load_tokenizer(dev)
    base = load_basemodel(dev)
    tr_loader, va_loader_local = make_loaders(dev, cache_dir)
    if val_loader is None: val_loader = va_loader_local

    model = copy.deepcopy(base)
    opt = optim.AdamW(model.parameters(), lr=LR, fused=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))
    ws = max(2, max_updates // 10)
    w_sched = optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, ws)
    c_sched = optim.lr_scheduler.CosineAnnealingLR(opt, max(1, max_updates - ws), eta_min=LR * 0.05)

    os.makedirs(out_dir, exist_ok=True)
    tag = "Phase8" if mode == "phase8" else "V4-Bold"
    print(f"\n{'='*60}\n  {tag} Training: {max_updates} updates, GA={GA}\n{'='*60}")

    uc = 0; mc = 0; losses = []
    model.train(); opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_updates, desc=tag)

    while uc < max_updates:
        for batch in tr_loader:
            if uc >= max_updates: break
            feats = batch["features"].to(dev, dtype=torch.float32)
            means, stds_ = batch["means"].to(dev, dtype=torch.float32), batch["stds"].to(dev, dtype=torch.float32)
            actual = batch["actual_returns"].to(dev, dtype=torch.float32)
            times = {k: v.to(dev, dtype=torch.long) for k,v in batch["time"].items()}
            B = feats.size(0)
            if B == 0: continue

            ic, iff = tok.encode(feats)
            gc, gf, hv = oracle_explore(model, tok, ic, iff, times, means, stds_, actual, dev, True, dt)

            model.train()
            tl = gc.size(1); tt = {k: v[:,:tl] for k,v in times.items()}
            with autocast(dev, dt, True):
                lc, lf, _ = model(gc[:,:-1], gf[:,:-1],
                                  tt["minute"][:,:tl-1], tt["day"][:,:tl-1],
                                  tt["month"][:,:tl-1], tt["year"][:,:tl-1], neftune_alpha=0.0)
                st = PREFIX_LEN - 1
                rc, rf = lc[:,st:st+HORIZON,:], lf[:,st:st+HORIZON,:]
                ah = actual[:,:HORIZON]
                exp = expected_returns(tok, rc, rf, means, stds_, TOP_K)

                if mode == "phase8":
                    e1 = loss_phase8(exp, ah)
                    ep = loss_phase8(torch.cumsum(exp,1), torch.cumsum(ah,1), alpha=4.0, beta=15.0)
                    loss_main = e1 + 1.5 * ep
                else:  # v4
                    lm = loss_mag(exp, ah, v4_under_w)
                    ld = loss_dir(exp, ah, v4_dir_w, v4_dir_mw)
                    loss_main = v4_mag_w * lm + v4_dir_w * ld

                if hv.any():
                    tc = gc[hv, PREFIX_LEN:PREFIX_LEN+HORIZON]
                    tf = gf[hv, PREFIX_LEN:PREFIX_LEN+HORIZON]
                    ce = F.cross_entropy(rc[hv].reshape(-1, rc.size(-1)).float(), tc.reshape(-1))
                    cf_ = F.cross_entropy(rf[hv].reshape(-1, rf.size(-1)).float(), tf.reshape(-1))
                    star = ce + cf_
                else:
                    star = torch.tensor(0.0, device=dev)

                loss = (loss_main + 0.334 * star) / GA

            if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward(); mc += 1

            if mc % GA == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if uc < ws: w_sched.step()
                else: c_sched.step()
                uc += 1
                rl = loss.item() * GA
                losses.append(rl)
                pbar.update(1)
                pbar.set_postfix(loss=f"{rl:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

                if uc % CKPT_EVERY == 0:
                    # Eval on VAL
                    mv = evaluate(model, tok, val_loader, dev, dt)
                    img = os.path.join(out_dir, f"{tag.lower()}_upd{uc}.png")
                    plot_diag(mv, uc, img, tag)
                    ada_k = "ada_050"
                    print(f"\n  [{tag} upd={uc}] VAL: path_MAPE={mv['path_mape']:.3f}% DA={mv['da']:.1f}% "
                          f"mag={mv['mag_ratio']:.3f} bold={mv['boldness']:.3f} "
                          f"zc={mv['zc_ratio']*100:.1f}% ADA@0.5%={mv.get(ada_k,'?')}% cov={mv.get(ada_k+'_cov','?')}%")
                    # Skip mid-training Demo eval to save time (80% of eval time is Demo)
                    # Demo eval only at final checkpoint
                    # Save checkpoint
                    torch.save({"model_state_dict": model.state_dict(), "update_count": uc},
                               os.path.join(out_dir, f"{tag.lower()}_upd{uc}.pt"))
                    model.train()
            if uc >= max_updates: break
    pbar.close()

    # Final eval
    mv = evaluate(model, tok, val_loader, dev, dt)
    plot_diag(mv, uc, os.path.join(out_dir, f"{tag.lower()}_final.png"), tag)
    torch.save({"model_state_dict": model.state_dict(), "update_count": uc},
               os.path.join(out_dir, f"{tag.lower()}_final.pt"))
    res = {k: v for k,v in mv.items() if not isinstance(v, np.ndarray)}
    res["update_count"] = uc; res["mode"] = mode
    with open(os.path.join(out_dir, "results.json"), "w") as f: json.dump(res, f, indent=2, default=str)

    if demo_loader is not None:
        md = evaluate(model, tok, demo_loader, dev, dt, EVAL_BATCHES_DEMO)
        plot_diag(md, uc, os.path.join(out_dir, f"{tag.lower()}_final_demo.png"), f"{tag}-Demo")
        res_d = {k: v for k,v in md.items() if not isinstance(v, np.ndarray)}
        with open(os.path.join(out_dir, "results_demo.json"), "w") as f: json.dump(res_d, f, indent=2, default=str)

    print(f"\n{'='*60}\n  {tag} Final VAL Report\n{'='*60}")
    for k in ("path_mape","da","mag_ratio","boldness","zc_ratio","dir_err","mag_err"):
        print(f"  {k}: {mv[k]}")
    for tau in ADA_THRESHOLDS:
        pct = int(tau*10000)
        print(f"  ADA @{tau*100:.1f}%: {mv.get(f'ada_{pct:03d}','N/A')}%  cov={mv.get(f'ada_{pct:03d}_cov','N/A')}%")
    return model, mv

# ═══════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════
def compare(results_dirs, out_dir):
    """Generate side-by-side comparison plot from multiple result directories."""
    all_res = {}
    for d in results_dirs:
        rp = os.path.join(d, "results.json")
        if os.path.exists(rp):
            with open(rp) as f: all_res[os.path.basename(d)] = json.load(f)
    if len(all_res) < 2:
        print("Need at least 2 results to compare."); return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    names = list(all_res.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # ADA bar chart
    ax = axes[0, 0]
    for i, (nm, r) in enumerate(all_res.items()):
        vals = [r.get(f"ada_{int(t*10000):03d}", 0) for t in ADA_THRESHOLDS]
        x = np.arange(len(ADA_THRESHOLDS)) + i * 0.25
        ax.bar(x, vals, 0.2, label=nm, color=colors[i], alpha=0.8)
    ax.axhline(50, color='red', ls='--', lw=1)
    ax.set_xticks(np.arange(len(ADA_THRESHOLDS)) + 0.12)
    ax.set_xticklabels([f"{t*100:.1f}%" for t in ADA_THRESHOLDS])
    ax.set_ylabel('ADA (%)'); ax.set_title('Actionable DA'); ax.legend(fontsize=8)

    # Coverage
    ax = axes[0, 1]
    for i, (nm, r) in enumerate(all_res.items()):
        vals = [r.get(f"ada_{int(t*10000):03d}_cov", 0) for t in ADA_THRESHOLDS]
        ax.plot(np.arange(len(ADA_THRESHOLDS)), vals, 'o-', color=colors[i], label=nm, lw=2)
    ax.set_xticks(np.arange(len(ADA_THRESHOLDS)))
    ax.set_xticklabels([f"{t*100:.1f}%" for t in ADA_THRESHOLDS])
    ax.set_ylabel('Coverage (%)'); ax.set_title('ADA Coverage'); ax.legend(fontsize=8)

    # Magnitude health
    ax = axes[0, 2]
    metrics = ['mag_ratio', 'boldness', 'zc_ratio']
    labels = ['Mag Ratio', 'Boldness', 'Zero-Collapse %']
    ideals = [1.0, 1.0, 0.0]
    x = np.arange(len(metrics))
    for i, (nm, r) in enumerate(all_res.items()):
        vals = [r.get(m, 0) for m in metrics]
        if metrics[2] in r: vals[2] *= 100  # zc_ratio to percentage
        ax.bar(x + i * 0.25, vals, 0.2, label=nm, color=colors[i], alpha=0.8)
    for j, (l, iv) in enumerate(zip(labels, ideals)):
        ax.plot(j, iv, 'r*', ms=12)
    ax.set_xticks(x + 0.12); ax.set_xticklabels(labels)
    ax.set_title('Magnitude Health'); ax.legend(fontsize=8)

    # DA & path_MAPE
    ax = axes[1, 0]
    for i, (nm, r) in enumerate(all_res.items()):
        ax.bar(i, r.get("da", 0), 0.4, label=nm, color=colors[i], alpha=0.8)
    ax.axhline(50, color='red', ls='--', lw=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('DA (%)'); ax.set_title('Directional Accuracy')

    ax = axes[1, 1]
    for i, (nm, r) in enumerate(all_res.items()):
        ax.bar(i, r.get("path_mape", 0), 0.4, label=nm, color=colors[i], alpha=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('path_MAPE (%)'); ax.set_title('Path MAPE (reference)')

    # Summary text
    ax = axes[1, 2]; ax.axis('off')
    lines = [f"{'═'*50}", f"  Comparison Summary", f"{'═'*50}", ""]
    for nm, r in all_res.items():
        lines.append(f"  [{nm}]")
        lines.append(f"    DA={r.get('da',0):.1f}%  path_MAPE={r.get('path_mape',0):.3f}%")
        lines.append(f"    mag_ratio={r.get('mag_ratio',0):.3f}  boldness={r.get('boldness',0):.3f}")
        lines.append(f"    zc={r.get('zc_ratio',0)*100:.1f}%  ADA@0.5%={r.get('ada_050','?')}%")
        lines.append("")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9, va='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    plt.suptitle('Experiment Comparison', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison saved to {os.path.join(out_dir, 'comparison.png')}")

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", choices=["phase8", "v4", "compare"], required=True)
    pa.add_argument("--updates", type=int, default=120)
    pa.add_argument("--cache", default="posttrain/rollout/cache/v4_n300")
    pa.add_argument("--out", default=None)
    pa.add_argument("--demo-cache", default="posttrain/rollout/cache", help="Full cache for demo eval")
    pa.add_argument("--compare-dirs", nargs="+", default=None)
    # V4 params
    pa.add_argument("--v4-mag-w", type=float, default=2.0)
    pa.add_argument("--v4-dir-w", type=float, default=1.0)
    pa.add_argument("--v4-under-w", type=float, default=2.0)
    pa.add_argument("--v4-dir-mw", type=float, default=0.5)
    args = pa.parse_args()

    if args.mode == "compare":
        if not args.compare_dirs: print("--compare-dirs required"); sys.exit(1)
        compare(args.compare_dirs, args.compare_dirs[0])
    else:
        dev = torch.device("cuda")
        demo_loader = None
        if args.demo_cache:
            try:
                _, dl = make_loaders(dev, args.demo_cache)
                demo_loader = dl
                print(f"Demo loader: {len(dl)} batches from {args.demo_cache}")
            except Exception as e:
                print(f"Could not load demo cache: {e}")

        out = args.out or f"outputs/v4_exp/{args.mode}"
        train(args.mode, args.updates, args.cache, out,
              demo_loader=demo_loader,
              v4_mag_w=args.v4_mag_w, v4_dir_w=args.v4_dir_w,
              v4_under_w=args.v4_under_w, v4_dir_mw=args.v4_dir_mw)
