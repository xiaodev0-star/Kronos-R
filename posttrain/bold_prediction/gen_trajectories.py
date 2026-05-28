"""Generate trajectory plots from best V4 model vs BaseModel."""
import json,os,torch,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import nullcontext
from model.tokenizer import HierarchicalQuantizer as HQ
from model.tokenizer_config import build_tokenizer_kwargs as btk
from model.kronos_reasoning import KronosReasoningGPT as KRG
from posttrain.rollout.data import resolve_project_path as rpp
from reproducibility import set_global_seed; set_global_seed(42)

PL,H=1023,10; BS=2
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

class CDS(torch.utils.data.Dataset):
    def __init__(s,path):
        pl=torch.load(path,map_location='cpu',weights_only=False); s.f=pl['features'].to(torch.float32); s.t={k:v.to(torch.long) for k,v in pl['time_features'].items()}; s.a=pl['actual_returns'].to(torch.float32)
        st=pl['seq_stats']; s.m=torch.stack([torch.as_tensor(x['mean'],dtype=torch.float32) for x in st]); s.s=torch.stack([torch.as_tensor(x['std'],dtype=torch.float32) for x in st])
    def __len__(s): return len(s.f)
    def __getitem__(s,i): return {'features':s.f[i],'time':{k:v[i] for k,v in s.t.items()},'actual_returns':s.a[i],'means':s.m[i],'stds':s.s[i]}

@torch.no_grad()
def rollout_trajectory(model,tok,batch,dev,dt):
    """Generate full 10-step autoregressive trajectory."""
    feats=batch['features'].to(dev,dtype=torch.float32)
    ms=batch['means'].to(dev,dtype=torch.float32); ss=batch['stds'].to(dev,dtype=torch.float32)
    times={k:v.to(dev,dtype=torch.long) for k,v in batch['time'].items()}
    ic,iff=tok.encode(feats); cc,cf=ic[:,:PL].clone(),iff[:,:PL].clone()
    preds=[]; ae=dev.type=='cuda'
    for s in range(H):
        sl=cc.size(1); ct={k:v[:,:sl] for k,v in times.items()}
        with ac(dev,dt,ae): lc,lf,_=model(cc,cf,ct['minute'],ct['day'],ct['month'],ct['year'],last_only=True)
        pc=lc[:,-1,:].argmax(-1,keepdim=True); pf=lf[:,-1,:].argmax(-1,keepdim=True)
        dec=tok.decode(pc,pf); ret=dec[:,0,0].cpu().float()*ss[:,0].cpu()+ms[:,0].cpu()
        preds.append(ret)
        if s<H-1: cc=torch.cat([cc,pc],1); cf=torch.cat([cf,pf],1)
    return torch.stack(preds,1)  # [B, 10]

# Main
dev=torch.device('cuda'); dt=ad(dev)
torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision('high')
tok=lt(dev)

# Load val data
va_ds=CDS('posttrain/rollout/cache/v4_n300/rollout_val.pt')
va_loader=torch.utils.data.DataLoader(va_ds,1,shuffle=True,num_workers=0,pin_memory=True)

# Load BaseModel and best V4 model (H2_150upd)
def load_model(path):
    m=KRG(vocab_size_coarse=1024,vocab_size_fine=1024,**BB).to(dev)
    ck=torch.load(path,map_location='cpu',weights_only=False); m.load_state_dict(ck.get('model_state_dict',ck),strict=False)
    m.eval(); return m

bm=load_model('checkpoints/base_model.pt')
h2=load_model('outputs/v4_exp/H2_final_150upd_final.pt')

# Collect trajectories
N_SAMPLES=16
all_base=[]; all_h2=[]; all_actual=[]
for i,batch in enumerate(va_loader):
    if i>=N_SAMPLES: break
    actual=batch['actual_returns'][:,:H].cpu()
    pb=rollout_trajectory(bm,tok,batch,dev,dt)
    ph=rollout_trajectory(h2,tok,batch,dev,dt)
    all_base.append(pb); all_h2.append(ph); all_actual.append(actual)

all_base=torch.cat(all_base,0).numpy()    # [N, 10]
all_h2=torch.cat(all_h2,0).numpy()         # [N, 10]
all_actual=torch.cat(all_actual,0).numpy()  # [N, 10]

# Cumulative (path) returns
cum_base=np.cumsum(all_base,1); cum_h2=np.cumsum(all_h2,1); cum_actual=np.cumsum(all_actual,1)

# Convert log-returns to price ratios for intuitive display
price_base=np.exp(np.clip(cum_base,-50,50)); price_h2=np.exp(np.clip(cum_h2,-50,50)); price_actual=np.exp(np.clip(cum_actual,-50,50))

# ===== FIGURE 1: 16 individual trajectories =====
fig,axes=plt.subplots(4,4,figsize=(20,18))
fig.suptitle('V4 Bold Prediction — 10-Step Trajectories (16 samples)',fontsize=16,fontweight='bold')
for i,ax in enumerate(axes.flat):
    steps=np.arange(1,H+1)
    ax.plot(steps,price_actual[i],'k-o',lw=2,ms=6,label='Actual',zorder=3)
    ax.plot(steps,price_base[i],'s--',color='#2ca02c',lw=1.5,ms=5,label='BaseModel',alpha=0.8)
    ax.plot(steps,price_h2[i],'D-',color='#1f77b4',lw=2,ms=6,label='V4-H2',alpha=0.9)
    ax.axhline(1.0,color='gray',ls=':',lw=0.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Price Ratio'); ax.set_title('Sample {}'.format(i+1))
    if i==0: ax.legend(fontsize=7)
    ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/v4_exp/trajectories_16.png',dpi=150,bbox_inches='tight')
plt.close()
print('Saved: outputs/v4_exp/trajectories_16.png')

# ===== FIGURE 2: 4 highlighted trajectories with step returns =====
fig,axes=plt.subplots(2,4,figsize=(22,10))
fig.suptitle('V4 Bold Prediction — Trajectory Details (4 samples)',fontsize=16,fontweight='bold')

# Pick 4 diverse samples
scores=all_actual[:,-1]-all_actual[:,0]  # total return
idx=np.argsort(scores)
picks=[idx[0],idx[len(idx)//3],idx[2*len(idx)//3],idx[-1]]  # diverse

for j,(ax1,ax2) in enumerate(zip(axes[0],axes[1])):
    i=picks[j]
    steps=np.arange(1,H+1)
    # Row 1: cumulative price
    ax1.plot(steps,price_actual[i],'k-o',lw=2.5,ms=7,label='Actual',zorder=3)
    ax1.plot(steps,price_base[i],'s--',color='#2ca02c',lw=1.5,ms=6,label='BaseModel')
    ax1.plot(steps,price_h2[i],'D-',color='#1f77b4',lw=2.5,ms=7,label='V4-H2')
    ax1.axhline(1.0,color='gray',ls=':',lw=0.5)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Cumulative Price Ratio')
    ax1.set_title('Sample {} — Cumulative Path'.format(i+1)); ax1.grid(True,alpha=0.3)
    if j==0: ax1.legend(fontsize=8)
    # Row 2: per-step returns
    w=0.25
    ax2.bar(steps-w,all_actual[i]*100,w,color='black',alpha=0.7,label='Actual')
    ax2.bar(steps,all_base[i]*100,w,color='#2ca02c',alpha=0.7,label='BaseModel')
    ax2.bar(steps+w,all_h2[i]*100,w,color='#1f77b4',alpha=0.7,label='V4-H2')
    ax2.axhline(0,color='gray',ls='-',lw=0.5)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Log Return (%)')
    ax2.set_title('Sample {} — Per-Step Returns'.format(i+1)); ax2.grid(True,alpha=0.3)
    if j==0: ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/v4_exp/trajectories_4detail.png',dpi=150,bbox_inches='tight')
plt.close()
print('Saved: outputs/v4_exp/trajectories_4detail.png')

# ===== FIGURE 3: Magnitude comparison — per-step |pred| vs |actual| =====
fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle('V4 Bold Prediction — Magnitude Analysis',fontsize=16,fontweight='bold')

steps=np.arange(1,H+1)
# 3a. Mean |pred| per step
ax=axes[0,0]
abs_base=np.abs(all_base).mean(0)*100; abs_h2=np.abs(all_h2).mean(0)*100; abs_act=np.abs(all_actual).mean(0)*100
w=0.25
ax.bar(steps-w,abs_act,w,color='black',alpha=0.7,label='|actual|')
ax.bar(steps,abs_base,w,color='#2ca02c',alpha=0.7,label='|pred| BaseModel')
ax.bar(steps+w,abs_h2,w,color='#1f77b4',alpha=0.7,label='|pred| V4-H2')
ax.set_xlabel('Step'); ax.set_ylabel('Mean |Return| (%)'); ax.set_title('Per-Step Mean Magnitude')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 3b. |pred| distribution (all steps pooled)
ax=axes[0,1]
mx=max(np.percentile(np.abs(all_actual).flatten(),99)*100,0.01)
bins=np.linspace(0,mx,40)
ax.hist(np.abs(all_actual).flatten()*100,bins,alpha=0.4,label='|actual|',color='black',density=True)
ax.hist(np.abs(all_base).flatten()*100,bins,alpha=0.4,label='|pred| BaseModel',color='#2ca02c',density=True)
ax.hist(np.abs(all_h2).flatten()*100,bins,alpha=0.4,label='|pred| V4-H2',color='#1f77b4',density=True)
ax.set_xlabel('|Return| (%)'); ax.set_ylabel('Density'); ax.set_title('Magnitude Distribution (all steps)')
ax.legend(fontsize=8)

# 3c. Pred vs Actual scatter (all steps)
ax=axes[1,0]
ax.scatter(all_actual.flatten()*100,all_base.flatten()*100,alpha=0.3,s=10,color='#2ca02c',label='BaseModel')
ax.scatter(all_actual.flatten()*100,all_h2.flatten()*100,alpha=0.3,s=10,color='#1f77b4',label='V4-H2')
lim=max(abs(all_actual.flatten()).max()*100*1.2,0.5)
ax.plot([-lim,lim],[-lim,lim],'k--',lw=1); ax.plot([-lim,lim],[0,0],'k:',lw=0.5); ax.plot([0,0],[-lim,lim],'k:',lw=0.5)
ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
ax.set_xlabel('Actual Return (%)'); ax.set_ylabel('Predicted Return (%)'); ax.set_title('Pred vs Actual (all steps)')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 3d. Magnitude Ratio per step
ax=axes[1,1]
mr_base=[]; mr_h2=[]
for s in range(H):
    pb=np.abs(all_base[:,s]); pa=np.abs(all_actual[:,s])
    mr_base.append(pb.std()/max(pa.std(),1e-8))
    ph=np.abs(all_h2[:,s]); mr_h2.append(ph.std()/max(pa.std(),1e-8))
ax.plot(steps,mr_base,'s-',color='#2ca02c',lw=2,ms=6,label='BaseModel mag_ratio')
ax.plot(steps,mr_h2,'D-',color='#1f77b4',lw=2,ms=6,label='V4-H2 mag_ratio')
ax.axhline(1.0,color='red',ls='--',lw=1.5,label='ideal=1.0')
ax.set_xlabel('Step'); ax.set_ylabel('mag_ratio'); ax.set_title('Per-Step Magnitude Ratio')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/v4_exp/magnitude_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print('Saved: outputs/v4_exp/magnitude_analysis.png')

# ===== FIGURE 4: Zero-collapse visualization =====
fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle('V4 Bold Prediction — Zero-Collapse Analysis',fontsize=16,fontweight='bold')

# 4a. Prediction magnitude vs threshold
ax=axes[0]
thresholds=np.linspace(0,0.05,50)
for name,data,color in [('BaseModel',all_base,'#2ca02c'),('V4-H2',all_h2,'#1f77b4'),('Actual',all_actual,'black')]:
    above=[np.mean(np.abs(data)>t)*100 for t in thresholds]
    ax.plot(thresholds*100,above,lw=2,color=color,label=name)
ax.set_xlabel('Threshold (%)'); ax.set_ylabel('% Predictions Above Threshold')
ax.set_title('Coverage vs Threshold'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 4b. Timidity histogram: |pred|/|actual| ratio
ax=axes[1]
for name,data,color in [('BaseModel',all_base,'#2ca02c'),('V4-H2',all_h2,'#1f77b4')]:
    ratio=np.abs(data)/(np.abs(all_actual)+1e-6)
    ratio=np.clip(ratio.flatten(),0,3)
    ax.hist(ratio,bins=50,alpha=0.4,color=color,label=name,density=True)
ax.axvline(0.1,color='red',ls='--',lw=1.5,label='zc threshold (0.1)')
ax.axvline(1.0,color='green',ls='--',lw=1.5,label='perfect=1.0')
ax.set_xlabel('|pred| / |actual|'); ax.set_ylabel('Density'); ax.set_title('Prediction/Actual Magnitude Ratio')
ax.legend(fontsize=8)

# 4c. Per-step zero-collapse ratio
ax=axes[2]
zc_base=[]; zc_h2=[]
for s in range(H):
    is_dir=np.abs(all_actual[:,s])>1e-4
    zc_base.append(np.mean(np.abs(all_base[is_dir,s])<np.abs(all_actual[is_dir,s])*0.1)*100)
    zc_h2.append(np.mean(np.abs(all_h2[is_dir,s])<np.abs(all_actual[is_dir,s])*0.1)*100)
ax.plot(steps,zc_base,'s-',color='#2ca02c',lw=2,ms=6,label='BaseModel')
ax.plot(steps,zc_h2,'D-',color='#1f77b4',lw=2,ms=6,label='V4-H2')
ax.fill_between(steps,0,zc_base,alpha=0.1,color='#2ca02c')
ax.fill_between(steps,0,zc_h2,alpha=0.1,color='#1f77b4')
ax.set_xlabel('Step'); ax.set_ylabel('Zero-Collapse %'); ax.set_title('Per-Step Zero-Collapse Ratio')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/v4_exp/zero_collapse_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print('Saved: outputs/v4_exp/zero_collapse_analysis.png')

print('\nAll 4 figures generated in outputs/v4_exp/')
