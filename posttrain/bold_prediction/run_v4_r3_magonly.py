"""V4-R3: Minimalist — only magnitude matching loss, no direction loss, no CE, no oracle.

Uses base model's own argmax predictions as golden (self-distillation).
Goal: prove that pure L_mag can preserve or improve magnitude without destroying DA.
"""
import copy, json, math, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.gridspec as gridspec
import numpy as np; import torch, torch.nn.functional as F; import torch.optim as optim
from contextlib import nullcontext; from tqdm import tqdm; from argparse import Namespace

from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_reasoning import KronosReasoningGPT
from posttrain.rollout.data import resolve_project_path
from reproducibility import set_global_seed; set_global_seed(42)

PREFIX_LEN, HORIZON, BS, GA = 1023, 10, 2, 8
LR, CKPT_EVERY, MAX_UPDATES = 9.59e-6, 20, 120
MAG_W, UNDER_W = 1.0, 2.0  # magnitude loss weight
STAR_CE_W = 0.0  # zero CE — pure magnitude
DIR_W = 0.0  # zero direction
BB = dict(dim=384, depth=3, heads=4, num_kv_heads=1, dsa_windows=[None,512,512],
          position_encoding="rope", rope_base=10000.0, dropout=0.1323, use_revin=False, num_factor_tokens=0)

def amp_dt(d): return torch.bfloat16 if (d.type=="cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if d.type=="cuda" else None)
def autocast(d,dt,e):
    if not e: return nullcontext()
    try: return torch.amp.autocast(device_type="cuda", dtype=dt)
    except: return torch.cuda.amp.autocast(dtype=dt)

def lt(dev):
    p=resolve_project_path("checkpoints/tokenizer.pt"); cp=resolve_project_path("checkpoints/tokenizer_config.json")
    ck=torch.load(p,map_location=dev,weights_only=False); cfg=ck.get("config",{})
    if not cfg and os.path.exists(cp):
        with open(cp) as f: cfg=json.load(f)
    tok=HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(dev); tok.load_state_dict(ck["model_state_dict"],strict=False)
    tok.eval(); tok.requires_grad_(False); return tok

def lb(dev):
    m=KronosReasoningGPT(vocab_size_coarse=1024,vocab_size_fine=1024,**BB).to(dev)
    p=resolve_project_path("checkpoints/base_model.pt")
    if os.path.exists(p): ck=torch.load(p,map_location=dev,weights_only=False); m.load_state_dict(ck.get("model_state_dict",ck),strict=False)
    return m

class CDS(torch.utils.data.Dataset):
    def __init__(s,path):
        pl=torch.load(path,map_location="cpu",weights_only=False); s.f=pl["features"].to(torch.float32)
        s.t={k:v.to(torch.long) for k,v in pl["time_features"].items()}; s.a=pl["actual_returns"].to(torch.float32)
        st=pl["seq_stats"]; s.m=torch.stack([torch.as_tensor(x["mean"],dtype=torch.float32) for x in st])
        s.s=torch.stack([torch.as_tensor(x["std"],dtype=torch.float32) for x in st])
    def __len__(s): return len(s.f)
    def __getitem__(s,i): return {"features":s.f[i],"time":{k:v[i] for k,v in s.t.items()},"actual_returns":s.a[i],"means":s.m[i],"stds":s.s[i]}

def ml(d,cache_dir):
    tr=CDS(os.path.join(cache_dir,"rollout_train.pt")); va=CDS(os.path.join(cache_dir,"rollout_val.pt"))
    lk=dict(num_workers=0,pin_memory=True)
    return (torch.utils.data.DataLoader(tr,BS,shuffle=True,**lk), torch.utils.data.DataLoader(va,BS*2,shuffle=False,**lk))

def er(tok, lc, lf, means, stds, top_k=16):
    B,H,_=lc.shape; K=min(top_k,lc.size(-1))
    tlc,tic=torch.topk(lc.float(),K,-1); tlf,fic=torch.topk(lf.float(),K,-1)
    pc=F.softmax(tlc/1.0,-1); pc=pc/pc.sum(-1,keepdim=True).clamp_min(1e-8)
    pf=F.softmax(tlf/1.0,-1); pf=pf/pf.sum(-1,keepdim=True).clamp_min(1e-8)
    jp=pc.unsqueeze(-1)*pf.unsqueeze(-2)
    p_c=tic.unsqueeze(-1).expand(B,H,K,K).reshape(B*H,K*K); p_f=fic.unsqueeze(-2).expand(B,H,K,K).reshape(B*H,K*K)
    with torch.no_grad(): dec=tok.decode(p_c,p_f)[...,0].float(); rg=dec.view(B,H,K,K)*stds[:,0:1,None,None]+means[:,0:1,None,None]
    return (jp*rg).sum((-1,-2))

def loss_mag(exp, act, uw=2.0):
    pm,am=exp.abs(),act.abs(); w=torch.where(pm<am,uw,1.0); return (w*(pm-am).pow(2)).mean()

@torch.no_grad()
def ev(model,tok,loader,dev,dt,mb=200):
    model.eval(); ae=dev.type=="cuda"
    ap,aa=[],[]; n=0
    for b in loader:
        feats=b["features"].to(dev,dtype=torch.float32); ms=b["means"].to(dev,dtype=torch.float32)
        ss=b["stds"].to(dev,dtype=torch.float32); times={k:v.to(dev,dtype=torch.long) for k,v in b["time"].items()}
        ic,iff=tok.encode(feats); cc,cf=ic[:,:PREFIX_LEN].clone(),iff[:,:PREFIX_LEN].clone(); pr=[]
        for s in range(HORIZON):
            sl=cc.size(1); ct={k:v[:,:sl] for k,v in times.items()}
            with autocast(dev,dt,ae): lc,lf,_=model(cc,cf,ct["minute"],ct["day"],ct["month"],ct["year"],last_only=True)
            pc,pf=lc[:,-1,:].argmax(-1),lf[:,-1,:].argmax(-1); dec=tok.decode(pc.unsqueeze(1),pf.unsqueeze(1))
            pr.append(dec[:,0,0].cpu().float()*ss[:,0].cpu()+ms[:,0].cpu())
            if s<HORIZON-1: cc=torch.cat([cc,pc.unsqueeze(1)],1); cf=torch.cat([cf,pf.unsqueeze(1)],1)
        ap.append(torch.stack(pr,1)); aa.append(b["actual_returns"].cpu()); n+=1
        if n>=mb: break
    pred=torch.cat(ap,0).numpy(); actual=torch.cat(aa,0).numpy()
    pt,at=torch.from_numpy(pred),torch.from_numpy(actual)
    pc_c=np.cumsum(pred,1); ac_c=np.cumsum(actual,1)
    prr=np.exp(np.clip(pc_c[:,-1],-50,50)); arr=np.exp(np.clip(ac_c[:,-1],-50,50))
    pm=float(np.mean(np.abs(prr-arr)/np.maximum(np.abs(arr),1e-4))*100)
    ps=np.where(pred>=0,1,-1); a_s=np.where(actual>=0,1,-1); da=float(np.mean(ps==a_s)*100)
    pa,aa=pt.abs(),at.abs(); mr=float(pa.std()/aa.std().clamp_min(1e-8))
    bd=float(pa.mean()/aa.mean().clamp_min(1e-8)); is_d=aa>1e-4
    zc=float((pa<aa*0.1)[is_d].float().mean()) if is_d.any() else 0.0
    pst=(pt>=0).float()*2-1; ast=(at>=0).float()*2-1; ada={}
    for tau in [0.001,0.003,0.005,0.010,0.020]:
        conf=pa>tau; cov=float(conf.float().mean())
        av=float((pst[conf]==ast[conf]).float().mean().item()*100) if conf.sum()>0 else float('nan')
        pct=int(tau*10000); ada[f"ada_{pct:03d}"]=round(av,2); ada[f"ada_{pct:03d}_cov"]=round(cov*100,1)
    is_c=(pt*at)>0; de=float((~is_c).float().mean()*100)
    me=float((pa[is_c]-aa[is_c]).abs().mean().item()) if is_c.any() else float('nan')
    return {"path_mape":pm,"da":da,"mag_ratio":mr,"boldness":bd,"zc_ratio":zc,"dir_err":de,"mag_err":me,**ada,
            "pred_flat":pred.flatten(),"actual_flat":actual.flatten(),"pred_abs":pa.numpy().flatten(),"actual_abs":aa.numpy().flatten(),
            "ps_mag":pa.mean(0).numpy(),"as_mag":aa.mean(0).numpy(),"ps_da":[float((ps[:,s]==a_s[:,s]).mean()*100) for s in range(HORIZON)]}

def plot_diag(m,upd,path,tag=""):
    fig=plt.figure(figsize=(18,12)); gs=gridspec.GridSpec(2,3,hspace=0.35,wspace=0.3)
    ax=fig.add_subplot(gs[0,0])
    ax.scatter(m["actual_flat"],m["pred_flat"],alpha=0.12,s=3,c='steelblue')
    lim=max(abs(m["actual_flat"]).max(),abs(m["pred_flat"]).max(),0.01)*1.1
    ax.plot([-lim,lim],[-lim,lim],'r--',lw=0.8); ax.plot([-lim,lim],[0,0],'k--',lw=0.8,alpha=0.5)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set(xlabel='Actual',ylabel='Predicted',title=f'{tag} Pred vs Actual (upd={upd})')
    ax2=fig.add_subplot(gs[0,1]); mx=max(np.percentile(m["actual_abs"],99),0.01); bins=np.linspace(0,mx,40)
    ax2.hist(m["actual_abs"],bins,alpha=0.5,label='|actual|',color='green',density=True)
    ax2.hist(m["pred_abs"],bins,alpha=0.5,label='|pred|',color='orange',density=True)
    ax2.set(xlabel='|Return|',ylabel='Density',title=f'|Pred| vs |Actual| (boldness={m["boldness"]:.3f})'); ax2.legend(fontsize=8)
    ax3=fig.add_subplot(gs[0,2]); s=np.arange(1,HORIZON+1)
    ax3.bar(s-0.2,m["as_mag"]*100,0.35,label='|actual|%',color='green',alpha=0.7)
    ax3.bar(s+0.2,m["ps_mag"]*100,0.35,label='|pred|%',color='orange',alpha=0.7)
    ax3.set(xlabel='Step',ylabel='Mean |Return| (%)',title='Per-Step Magnitude'); ax3.legend(fontsize=8)
    ax4=fig.add_subplot(gs[1,0])
    ax4.bar(s,m["ps_da"],color='steelblue',alpha=0.7); ax4.axhline(50,color='red',ls='--',lw=1,label='random')
    ax4.set(xlabel='Step',ylabel='DA (%)',ylim=(40,65),title=f'Per-Step DA (overall={m["da"]:.1f}%)'); ax4.legend(fontsize=8)
    ax5=fig.add_subplot(gs[1,1]); avs,cvs,tls=[],[],[]
    for tau in [0.001,0.003,0.005,0.010,0.020]:
        pct=int(tau*10000); avs.append(m.get(f"ada_{pct:03d}",float('nan')))
        cvs.append(m.get(f"ada_{pct:03d}_cov",0)); tls.append(f"{tau*100:.1f}%")
    xp=np.arange(len(tls)); ax5.bar(xp,avs,color='coral',alpha=0.7,label='ADA%'); ax5.axhline(50,color='red',ls='--',lw=1)
    ax5t=ax5.twinx(); ax5t.plot(xp,cvs,'go-',lw=2,ms=6,label='Coverage%')
    ax5.set_xticks(xp); ax5.set_xticklabels(tls); ax5.set(xlabel='Threshold',ylabel='ADA (%)',ylim=(30,70))
    ax5t.set_ylabel('Coverage (%)'); ax5t.set_ylim(0,100); l1,lb1=ax5.get_legend_handles_labels(); l2,lb2=ax5t.get_legend_handles_labels()
    ax5.legend(l1+l2,lb1+lb2,fontsize=8,loc='lower right')
    ax6=fig.add_subplot(gs[1,2]); ax6.axis('off')
    txt=(f"{'═'*42}\n  {tag} Metrics (update={upd})\n{'═'*42}\n\n  Mag Ratio: {m['mag_ratio']:.3f}\n  Boldness: {m['boldness']:.3f}\n  Zero-Collapse: {m['zc_ratio']*100:.1f}%\n\n  DA: {m['da']:.1f}%\n  path_MAPE: {m['path_mape']:.3f}%\n\n  ADA@0.5%: {m.get('ada_050','?')}%\n  Cov@0.5%: {m.get('ada_050_cov','?')}%\n{'═'*42}")
    ax6.text(0.05,0.95,txt,transform=ax6.transAxes,fontsize=10,va='top',fontfamily='monospace',bbox=dict(boxstyle='round',fc='lightyellow',alpha=0.8))
    fig.suptitle(f'{tag} — Update {upd}',fontsize=14,fontweight='bold'); plt.savefig(path,dpi=120,bbox_inches='tight'); plt.close(fig)

def main():
    dev=torch.device("cuda"); dt=amp_dt(dev)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision("high")
    tok=lt(dev); base=lb(dev)
    tr_loader,va_loader=ml(dev,"posttrain/rollout/cache/v4_n300")
    model=copy.deepcopy(base); opt=optim.AdamW(model.parameters(),lr=LR,fused=True)
    scaler=torch.cuda.amp.GradScaler(enabled=(dt==torch.float16))
    ws=max(2,MAX_UPDATES//10); w_sched=optim.lr_scheduler.LinearLR(opt,0.1,1.0,ws)
    c_sched=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,MAX_UPDATES-ws),eta_min=LR*0.05)
    out_dir="outputs/v4_exp/v4_r3_magonly"; os.makedirs(out_dir,exist_ok=True)
    print(f"V4-R3: Pure magnitude loss (no direction, no CE, no oracle)")
    print(f"  mag_w={MAG_W}, under_w={UNDER_W}, ce_w={STAR_CE_W}, dir_w={DIR_W}")
    uc,mc=0,0; model.train(); opt.zero_grad(set_to_none=True); pbar=tqdm(total=MAX_UPDATES,desc="V4-R3-MagOnly")
    while uc<MAX_UPDATES:
        for batch in tr_loader:
            if uc>=MAX_UPDATES: break
            feats=batch["features"].to(dev,dtype=torch.float32); ms=batch["means"].to(dev,dtype=torch.float32)
            ss=batch["stds"].to(dev,dtype=torch.float32); actual=batch["actual_returns"].to(dev,dtype=torch.float32)
            times={k:v.to(dev,dtype=torch.long) for k,v in batch["time"].items()}; B=feats.size(0)
            if B==0: continue
            # Deterministic argmax — no Oracle!
            ic,iff=tok.encode(feats); cc,cf=ic[:,:PREFIX_LEN].clone(),iff[:,:PREFIX_LEN].clone()
            full_c,full_f=cc.clone(),cf.clone()
            for s in range(HORIZON):
                sl=cc.size(1); ct={k:v[:,:sl] for k,v in times.items()}
                with autocast(dev,dt,True): lc,lf,_=model(cc,cf,ct["minute"],ct["day"],ct["month"],ct["year"],last_only=True,neftune_alpha=0.0)
                pc=lc[:,-1,:].argmax(-1,keepdim=True); pf=lf[:,-1,:].argmax(-1,keepdim=True)
                cc=torch.cat([cc,pc],1); cf=torch.cat([cf,pf],1)
            golden_c,golden_f=cc,cf  # [B, PREFIX_LEN+HORIZON]
            # Forward pass on golden
            model.train(); tl=golden_c.size(1); tt={k:v[:,:tl] for k,v in times.items()}
            with autocast(dev,dt,True):
                lc,lf,_=model(golden_c[:,:-1],golden_f[:,:-1],tt["minute"][:,:tl-1],tt["day"][:,:tl-1],tt["month"][:,:tl-1],tt["year"][:,:tl-1],neftune_alpha=0.0)
                st=PREFIX_LEN-1; rc,rf=lc[:,st:st+HORIZON,:],lf[:,st:st+HORIZON,:]; ah=actual[:,:HORIZON]
                exp=er(tok,rc,rf,ms,ss,16)
                # Only magnitude loss
                loss=MAG_W*loss_mag(exp,ah,UNDER_W)/GA
            if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward(); mc+=1
            if mc%GA==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if uc<ws: w_sched.step()
                else: c_sched.step()
                uc+=1; pbar.update(1); pbar.set_postfix(loss=f"{loss.item()*GA:.6f}",lr=f"{opt.param_groups[0]['lr']:.2e}")
                if uc%CKPT_EVERY==0:
                    mv=ev(model,tok,va_loader,dev,dt,200); img=os.path.join(out_dir,f"r3_upd{uc}.png")
                    plot_diag(mv,uc,img,"V4-R3-MagOnly")
                    print(f"\n  [R3 upd={uc}] path_MAPE={mv['path_mape']:.3f}% DA={mv['da']:.1f}% mag={mv['mag_ratio']:.3f} bold={mv['boldness']:.3f} zc={mv['zc_ratio']*100:.1f}% ADA@0.5%={mv.get('ada_050','?')}% cov={mv.get('ada_050_cov','?')}%")
                    torch.save({"model_state_dict":model.state_dict(),"update_count":uc},os.path.join(out_dir,f"r3_upd{uc}.pt"))
                    model.train()
    pbar.close()
    mv=ev(model,tok,va_loader,dev,dt,200); plot_diag(mv,uc,os.path.join(out_dir,"r3_final.png"),"V4-R3-MagOnly")
    torch.save({"model_state_dict":model.state_dict(),"update_count":uc},os.path.join(out_dir,"r3_final.pt"))
    res={k:v for k,v in mv.items() if not isinstance(v,np.ndarray)}; res["update_count"]=uc
    with open(os.path.join(out_dir,"results.json"),"w") as f: json.dump(res,f,indent=2,default=str)
    print(f"\nR3 Final: path_MAPE={mv['path_mape']:.3f}% DA={mv['da']:.1f}% mag={mv['mag_ratio']:.3f} bold={mv['boldness']:.3f} zc={mv['zc_ratio']*100:.1f}%")
    for t in [0.001,0.003,0.005,0.010,0.020]:
        pct=int(t*10000); print(f'  ADA@{t*100:.1f}%={mv.get(f"ada_{pct:03d}","N/A")}% cov={mv.get(f"ada_{pct:03d}_cov","N/A")}%')

if __name__=="__main__": main()
