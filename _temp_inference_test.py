"""Quick test: argmax vs temperature sampling at inference."""
import os, sys; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace
from evaluate_predictions import load_model
from posttrain.rollout.data import RolloutWindowDataset, rollout_collate, resolve_project_path
from posttrain.rollout.train_rollout import _amp_dtype, _autocast_context, _move_batch, _encode_features, compute_rollout_metrics

device = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high'); torch.cuda.empty_cache()
amp_dtype = _amp_dtype('bfloat16'); amp_enabled = True

cfg = Namespace(prefix_len=1023, horizon=10, stride_ratio=0.5,
    cache_dir=resolve_project_path('posttrain/rollout/cache'), max_stocks=0, cache_rebuild=False, mape_eps=1e-4)
demo_dataset = RolloutWindowDataset('val', cfg=cfg, max_samples=0, seed=999)
demo_loader = DataLoader(demo_dataset, batch_size=2, shuffle=False, collate_fn=rollout_collate, num_workers=0)

@torch.inference_mode()
def eval_temp(ckpt_path, temp, max_batches=100):
    torch.cuda.empty_cache()
    model, tokenizer = load_model(device=device, checkpoint_path=ckpt_path, strict_checkpoint_compat=False)
    tokenizer.eval(); tokenizer.requires_grad_(False); model.eval()
    ap_data, aa_data = [], []; n = 0
    for batch in demo_loader:
        batch = _move_batch(batch, device)
        idx_c, idx_f = _encode_features(tokenizer, batch['features'])
        cur_c = idx_c[:, :1023].clone(); cur_f = idx_f[:, :1023].clone()
        preds = []
        for step in range(10):
            sl = cur_c.size(1); ct = {k: v[:, :sl] for k, v in batch['time'].items()}
            with _autocast_context(device, amp_enabled, amp_dtype):
                lc, lf, _ = model(cur_c, cur_f, ct['minute'], ct['day'], ct['month'], ct['year'], last_only=True)
            if temp > 0:
                pc = torch.multinomial(F.softmax(lc[:, -1, :].float() / temp, dim=-1), num_samples=1)
                pf = torch.multinomial(F.softmax(lf[:, -1, :].float() / temp, dim=-1), num_samples=1)
            else:
                pc = lc[:, -1, :].argmax(dim=-1, keepdim=True)
                pf = lf[:, -1, :].argmax(dim=-1, keepdim=True)
            dec = tokenizer.decode(pc, pf)
            ret = dec[:, 0, 0].cpu().float() * batch['stds'][:, 0].cpu() + batch['means'][:, 0].cpu()
            preds.append(ret)
            if step < 9:
                cur_c = torch.cat([cur_c, pc], dim=1); cur_f = torch.cat([cur_f, pf], dim=1)
        ap_data.append(torch.stack(preds, dim=1)); aa_data.append(batch['actual_returns'].cpu())
        n += 1
        if n >= max_batches: break
    pred = torch.cat(ap_data, dim=0).numpy(); actual = torch.cat(aa_data, dim=0).numpy()
    m = compute_rollout_metrics(pred, actual, mape_eps=1e-4)
    ap = np.abs(pred.ravel())
    ps = (torch.from_numpy(pred) >= 0).float()*2-1; _as = (torch.from_numpy(actual) >= 0).float()*2-1
    cm = torch.abs(torch.from_numpy(pred)) > 0.005
    m['act_da'] = float((ps[cm]==_as[cm]).float().mean().item()*100) if cm.sum()>0 else 0
    m['act_ratio'] = float(cm.float().mean().item()*100)
    m['abs_mean'] = float(np.mean(ap)); m['abs_med'] = float(np.median(ap))
    m['pct_zero'] = float(np.mean(ap<0.001)*100)
    del model; torch.cuda.empty_cache()
    return m

soft_ckpt = resolve_project_path('outputs/experiment_Soft-Baseline.pt')
gumbel_ckpt = resolve_project_path('outputs/experiment_Gumbel-tau0.5.pt')

with open('_temp_results.txt', 'w') as f:
    f.write(f'{"Model":<22} {"Temp":<8} {"MAPE":>8} {"|Pred|Mean":>12} {"|Pred|Med":>12} {"%<0.001":>8} {"DA":>8} {"ActDA":>8} {"ActRatio":>8}\n')
    f.write('-'*95 + '\n')

    for label, ckpt in [('Soft-Baseline', soft_ckpt), ('Gumbel-tau=0.5', gumbel_ckpt)]:
        for t, tlabel in [(0, 'argmax'), (0.414, 'T=0.414'), (0.5, 'T=0.5')]:
            m = eval_temp(ckpt, t)
            line = f'{label:<22} {tlabel:<8} {m["path_mape"]:>8.4f} {m["abs_mean"]:>12.6f} {m["abs_med"]:>12.6f} {m["pct_zero"]:>8.1f} {m["da"]:>8.2f} {m["act_da"]:>8.2f} {m["act_ratio"]:>8.2f}'
            print(line); f.write(line + '\n'); f.flush()

print('\nDone! Results saved to _temp_results.txt')
