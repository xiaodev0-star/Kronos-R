"""V4 grid search — simple parameter sweep."""
import copy,json,os,torch,torch.nn.functional as F,torch.optim as optim,numpy as np
from contextlib import nullcontext
from model.tokenizer import HierarchicalQuantizer as HQ
from model.tokenizer_config import build_tokenizer_kwargs as btk
from model.kronos_reasoning import KronosReasoningGPT as KRG
from posttrain.rollout.data import resolve_project_path as rpp
from reproducibility import set_global_seed; set_global_seed(42)

PL,H,BS,GA=1023,10,2,8; MU=60  # fast trials
BB=dict(dim=384,depth=3,heads=4,num_kv_heads=1,dsa_windows=[None,512,512],position_encoding='rope',rope_base=10000.0,dropout=0.1323,use_revin=False,num_factor_tokens=0)
def ad(d): return torch.bfloat16 if(d.type=='cuda'and torch.cuda.is_bf16_supported())else(torch.float16 if d.type=='cuda'else None)
def ac(d,dt,e):
    if not e: return nullcontext()
    try: return torch.amp.autocast(device_type='cuda',dtype=dt)
    except: return torch.cuda.amp.autocast(dtype=dt)
def lt(dev):
    p=rpp('checkpoints/tokenizer.pt'); cp=rpp('checkpoints/tokenizer_config.json')
    ck=torch.load(p,map_location=dev,weights_only=False); cfg=ck.get('config',{}) or (json.load(open(cp)) if os.path.exists(cp) else {})
    tok=HQ(**btk(cfg)).to(dev); tok.load_state_dict(ck['model_state_dict'],strict=False); tok.eval(); tok.requires_grad_(False); return tok
def lb(dev):
    m=KRG(vocab_size_coarse=1024,vocab_size_fine=1024,**BB).to(dev)
    p=rpp('checkpoints/base_model.pt')
    if os.path.exists(p): ck=torch.load(p,map_location=dev,weights_only=False); m.load_state_dict(ck.get('model_state_dict',ck),strict=False)
    return m
class CDS(torch.utils.data.Dataset):
    def __init__(s,path):
        pl=torch.load(path,map_location='cpu',weights_only=False); s.f=pl['features'].to(torch.float32); s.t={k:v.to(torch.long) for k,v in pl['time_features'].items()}; s.a=pl['actual_returns'].to(torch.float32)
        st=pl['seq_stats']; s.m=torch.stack([torch.as_tensor(x['mean'],dtype=torch.float32) for x in st]); s.s=torch.stack([torch.as_tensor(x['std'],dtype=torch.float32) for x in st])
    def __len__(s): return len(s.f)
    def __getitem__(s,i): return {'features':s.f[i],'time':{k:v[i] for k,v in s.t.items()},'actual_returns':s.a[i],'means':s.m[i],'stds':s.s[i]}
def er(tok,lc,lf,ms,ss,top_k=16):
    B,H,_=lc.shape; K=min(top_k,lc.size(-1))
    tlc,tic=torch.topk(lc.float(),K,-1); tlf,fic=torch.topk(lf.float(),K,-1)
    pc=F.softmax(tlc/1.0,-1); pc=pc/pc.sum(-1,keepdim=True).clamp_min(1e-8)
    pf=F.softmax(tlf/1.0,-1); pf=pf/pf.sum(-1,keepdim=True).clamp_min(1e-8)
    jp=pc.unsqueeze(-1)*pf.unsqueeze(-2); p_c=tic.unsqueeze(-1).expand(B,H,K,K).reshape(B*H,K*K); p_f=fic.unsqueeze(-2).expand(B,H,K,K).reshape(B*H,K*K)
    with torch.no_grad(): dec=tok.decode(p_c,p_f)[...,0].float(); rg=dec.view(B,H,K,K)*ss[:,0:1,None,None]+ms[:,0:1,None,None]
    return(jp*rg).sum((-1,-2))
def lm(exp,act,uw=2.0):
    pm,am=exp.abs(),act.abs(); w=torch.where(pm<am,uw,1.0); return(w*(pm-am).pow(2)).mean()
def ld(exp,act,dw=1.0,mw=0.5,sc=100.0,mr=0.3):
    sig=exp*act*sc; dl=F.softplus(-sig).mean(); ic=(exp*act)>0; ib=exp.abs()>act.abs()*mr; m=ic&ib
    cl=(exp[m].abs()-act[m].abs()).pow(2).mean() if m.any() else torch.tensor(0.0,device=exp.device); return dw*dl+mw*cl
@torch.no_grad()
def ev(model,tok,loader,dev,dt,mb=200):
    model.eval(); ae=dev.type=='cuda'; ap,aa=[],[]; n=0
    for b in loader:
        feats=b['features'].to(dev,dtype=torch.float32); ms=b['means'].to(dev,dtype=torch.float32); ss_=b['stds'].to(dev,dtype=torch.float32)
        times={k:v.to(dev,dtype=torch.long) for k,v in b['time'].items()}; ic,iff=tok.encode(feats); cc,cf=ic[:,:PL].clone(),iff[:,:PL].clone(); pr=[]
        for s in range(H):
            sl=cc.size(1); ct={k:v[:,:sl] for k,v in times.items()}
            with ac(dev,dt,ae): lc,lf,_=model(cc,cf,ct['minute'],ct['day'],ct['month'],ct['year'],last_only=True)
            pc,pf=lc[:,-1,:].argmax(-1),lf[:,-1,:].argmax(-1); dec=tok.decode(pc.unsqueeze(1),pf.unsqueeze(1))
            pr.append(dec[:,0,0].cpu().float()*ss_[:,0].cpu()+ms[:,0].cpu())
            if s<H-1: cc=torch.cat([cc,pc.unsqueeze(1)],1); cf=torch.cat([cf,pf.unsqueeze(1)],1)
        ap.append(torch.stack(pr,1)); aa.append(b['actual_returns'].cpu()); n+=1
        if n>=mb: break
    pred=torch.cat(ap,0).numpy(); actual=torch.cat(aa,0).numpy(); pt,at=torch.from_numpy(pred),torch.from_numpy(actual)
    pc_c=np.cumsum(pred,1); ac_c=np.cumsum(actual,1)
    prr=np.exp(np.clip(pc_c[:,-1],-50,50)); arr=np.exp(np.clip(ac_c[:,-1],-50,50))
    pm=float(np.mean(np.abs(prr-arr)/np.maximum(np.abs(arr),1e-4))*100)
    ps=np.where(pred>=0,1,-1); a_s=np.where(actual>=0,1,-1); da=float(np.mean(ps==a_s)*100)
    pa,aa_t=pt.abs(),at.abs(); mr=float(pa.std()/aa_t.std().clamp_min(1e-8)); bd=float(pa.mean()/aa_t.mean().clamp_min(1e-8))
    is_d=aa_t>1e-4; zc=float((pa<aa_t*0.1)[is_d].float().mean()) if is_d.any() else 0.0
    pst=(pt>=0).float()*2-1; ast=(at>=0).float()*2-1; ada={}
    for tau in[0.001,0.003,0.005,0.010,0.020]:
        conf=pa>tau; cov=float(conf.float().mean()); av=float((pst[conf]==ast[conf]).float().mean().item()*100) if conf.sum()>0 else float('nan')
        pct=int(tau*10000); ada['ada_{:03d}'.format(pct)]=round(av,2); ada['ada_{:03d}_cov'.format(pct)]=round(cov*100,1)
    return{'path_mape':pm,'da':da,'mag_ratio':mr,'boldness':bd,'zc_ratio':zc,**ada}

# Main
dev=torch.device('cuda'); dt=ad(dev); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision('high')
tok=lt(dev)
tr_ds=CDS('posttrain/rollout/cache/v4_n300/rollout_train.pt'); va_ds=CDS('posttrain/rollout/cache/v4_n300/rollout_val.pt')
lk=dict(num_workers=0,pin_memory=True)
tr_loader=torch.utils.data.DataLoader(tr_ds,BS,shuffle=True,**lk); va_loader=torch.utils.data.DataLoader(va_ds,BS*2,shuffle=False,**lk)
lr=9.59e-6

grids=[(0.5,2.0,0.2,0.3,100.0),(1.0,2.0,0.1,0.2,50.0),(1.0,3.0,0.3,0.3,50.0),(2.0,2.0,0.05,0.1,30.0),(1.5,2.0,0.15,0.2,50.0),(0.8,4.0,0.3,0.3,80.0)]
os.makedirs('outputs/v4_exp/grid',exist_ok=True)

print('Grid search: 6 configs x 60 updates...')
results={}
for i,(mw,uw,dw,dmw,dsc) in enumerate(grids):
    name='H{}'.format(i)
    print('\n{}: mag_w={}, under_w={}, dir_w={}, dir_mw={}, dir_sc={}'.format(name,mw,uw,dw,dmw,dsc))
    base=lb(dev); model=copy.deepcopy(base)
    opt=optim.AdamW(model.parameters(),lr=lr,fused=True)
    scaler=torch.cuda.amp.GradScaler(enabled=(dt==torch.float16))
    ws=max(2,MU//10); w_sched=optim.lr_scheduler.LinearLR(opt,0.1,1.0,ws)
    c_sched=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,MU-ws),eta_min=lr*0.05)
    uc,mc=0,0; model.train(); opt.zero_grad(set_to_none=True)
    while uc<MU:
        for batch in tr_loader:
            if uc>=MU: break
            feats=batch['features'].to(dev,dtype=torch.float32); ms=batch['means'].to(dev,dtype=torch.float32)
            ss_=batch['stds'].to(dev,dtype=torch.float32); actual=batch['actual_returns'].to(dev,dtype=torch.float32)
            times={k:v.to(dev,dtype=torch.long) for k,v in batch['time'].items()}; B=feats.size(0)
            if B==0: continue
            ic,iff=tok.encode(feats); cc,cf=ic[:,:PL].clone(),iff[:,:PL].clone()
            for s in range(H):
                sl=cc.size(1); ct={k:v[:,:sl] for k,v in times.items()}
                with ac(dev,dt,True): lc,lf,_=model(cc,cf,ct['minute'],ct['day'],ct['month'],ct['year'],last_only=True,neftune_alpha=0.0)
                pc=lc[:,-1,:].argmax(-1,keepdim=True); pf=lf[:,-1,:].argmax(-1,keepdim=True)
                cc=torch.cat([cc,pc],1); cf=torch.cat([cf,pf],1)
            gc,gf=cc[:,:PL+H],cf[:,:PL+H]; model.train(); tl=gc.size(1); tt={k:v[:,:tl] for k,v in times.items()}
            with ac(dev,dt,True):
                lc,lf,_=model(gc[:,:-1],gf[:,:-1],tt['minute'][:,:tl-1],tt['day'][:,:tl-1],tt['month'][:,:tl-1],tt['year'][:,:tl-1],neftune_alpha=0.0)
                st_=PL-1; rc,rf=lc[:,st_:st_+H,:],lf[:,st_:st_+H,:]; ah=actual[:,:H]
                exp=er(tok,rc,rf,ms,ss_,16)
                loss=(mw*lm(exp,ah,uw)+dw*ld(exp,ah,dw=dw,mw=dmw,sc=dsc))/GA
            if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward(); mc+=1
            if mc%GA==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if uc<ws: w_sched.step()
                else: c_sched.step()
                uc+=1
    mv=ev(model,tok,va_loader,dev,dt,200); results[name]=(mw,uw,dw,dmw,dsc,mv)
    mr=mv['mag_ratio']; da=mv['da']; bd=mv['boldness']; zc=mv['zc_ratio']*100
    ada05=mv.get('ada_050',0); cov05=mv.get('ada_050_cov',0); pm=mv['path_mape']
    print('  -> DA={:.1f}% mag={:.3f} bold={:.3f} zc={:.1f}% ADA@0.5%={:.1f}% cov={:.1f}% MAPE={:.3f}%'.format(da,mr,bd,zc,ada05,cov05,pm))
    torch.save({'model_state_dict':model.state_dict()},'outputs/v4_exp/grid/{}_final.pt'.format(name))
    rr={k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in mv.items()}
    with open('outputs/v4_exp/grid/{}_results.json'.format(name),'w') as f: json.dump(rr,f,indent=2,default=float)
    del model,base; torch.cuda.empty_cache()

print('\n===== GRID RESULTS =====')
best_score=-999; best_name=''
for name,(mw,uw,dw,dmw,dsc,mv) in sorted(results.items()):
    mr=mv['mag_ratio']; da=mv['da']; bd=mv['boldness']; zc=mv['zc_ratio']*100
    ada05=mv.get('ada_050',0); cov05=mv.get('ada_050_cov',0); pm=mv['path_mape']
    score=da+10*mr+2*cov05-5*zc
    print('{}: mag_w={:.1f} uw={:.1f} dw={:.2f} -> DA={:.1f}% mag={:.3f} bold={:.3f} zc={:.1f}% ADA@0.5%={:.1f}% cov={:.1f}% score={:.1f}'.format(name,mw,uw,dw,da,mr,bd,zc,ada05,cov05,score))
    if score>best_score: best_score=score; best_name=name; best_mv=mv; best_params=(mw,uw,dw,dmw,dsc)
print('\nBest: {} (score={:.1f})'.format(best_name,best_score))
print('Params: mag_w={}, under_w={}, dir_w={}'.format(best_params[0],best_params[1],best_params[2]))
print('DA={:.1f}% mag={:.3f} bold={:.3f} zc={:.1f}%'.format(best_mv['da'],best_mv['mag_ratio'],best_mv['boldness'],best_mv['zc_ratio']*100))
