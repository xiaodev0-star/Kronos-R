"""Generate final comparison plot from saved results."""
import json,os,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.gridspec as gridspec

# Collect all available results
results={}
# H5 150upd (best magnitude)
p='outputs/v4_exp/H5_final_150upd_results.json'
if os.path.exists(p):
    with open(p) as f: results['H5_150upd']=json.load(f)

# H2 150upd (best balance)
p='outputs/v4_exp/H2_final_150upd_results.json'
if os.path.exists(p):
    with open(p) as f: results['H2_150upd']=json.load(f)

# H5 60upd Demo
p='outputs/v4_exp/grid/H5_demo.json'
if os.path.exists(p):
    with open(p) as f: r=json.load(f); results['H5_Demo']=r

# H2 60upd Demo
p='outputs/v4_exp/grid/H2_demo.json'
if os.path.exists(p):
    with open(p) as f: r=json.load(f); results['H2_Demo']=r

# BaseModel known values from earlier Demo eval
results['BaseModel']={'da':49.1,'mag_ratio':0.925,'boldness':0.549,'zc_ratio':0.392,'path_mape':8.523,
    'ada_010':49.48,'ada_010_cov':62.5,'ada_030':49.96,'ada_030_cov':48.6,'ada_050':49.86,'ada_050_cov':43.9,
    'ada_100':49.97,'ada_100_cov':36.0,'ada_200':50.42,'ada_200_cov':21.1}

print('Loaded {} models'.format(len(results)))
for nm in results:
    r=results[nm]; print('{}: DA={:.1f}% mag={:.3f} bold={:.3f} zc={:.1f}%'.format(nm,r['da'],r['mag_ratio'],r['boldness'],r['zc_ratio']*100))

# Generate simple summary bar chart
fig,axes=plt.subplots(2,3,figsize=(18,10))
names=list(results.keys())
colors={'BaseModel':'#2ca02c','H2_150upd':'#1f77b4','H5_150upd':'#ff7f0e','H2_Demo':'#1f77b4','H5_Demo':'#ff7f0e'}

# 1. mag_ratio
ax=axes[0,0]; x=np.arange(len(names)); vals=[results[n]['mag_ratio'] for n in names]
bars=ax.bar(x,vals,color=[colors.get(n,'gray') for n in names],alpha=0.8)
ax.axhline(1.0,color='red',ls='--',lw=1,label='ideal=1.0')
ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha='right'); ax.set_ylabel('mag_ratio'); ax.set_title('Magnitude Ratio')
for i,v in enumerate(vals): ax.text(i,v+0.02,'{:.3f}'.format(v),ha='center',fontsize=8)

# 2. boldness
ax=axes[0,1]; vals=[results[n]['boldness'] for n in names]
ax.bar(x,vals,color=[colors.get(n,'gray') for n in names],alpha=0.8)
ax.axhline(1.0,color='red',ls='--',lw=1,label='ideal=1.0')
ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha='right'); ax.set_ylabel('boldness'); ax.set_title('Prediction Boldness')
for i,v in enumerate(vals): ax.text(i,v+0.02,'{:.3f}'.format(v),ha='center',fontsize=8)

# 3. zc_ratio
ax=axes[0,2]; vals=[results[n]['zc_ratio']*100 for n in names]
ax.bar(x,vals,color=[colors.get(n,'gray') for n in names],alpha=0.8)
ax.axhline(0,color='green',ls='--',lw=1,label='ideal=0%')
ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha='right'); ax.set_ylabel('Zero-Collapse %'); ax.set_title('Zero-Collapse Ratio')
for i,v in enumerate(vals): ax.text(i,v+0.5,'{:.1f}%'.format(v),ha='center',fontsize=8)

# 4. DA
ax=axes[1,0]; vals=[results[n]['da'] for n in names]
ax.bar(x,vals,color=[colors.get(n,'gray') for n in names],alpha=0.8)
ax.axhline(50,color='red',ls='--',lw=1,label='random=50%')
ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha='right'); ax.set_ylabel('DA (%)'); ax.set_title('Directional Accuracy')
for i,v in enumerate(vals): ax.text(i,v+0.3,'{:.1f}%'.format(v),ha='center',fontsize=8)

# 5. ADA@0.5% with coverage
ax=axes[1,1]
ada_vals=[results[n].get('ada_050',0) for n in names]
cov_vals=[results[n].get('ada_050_cov',0) for n in names]
x2=np.arange(len(names)); w=0.35
ax.bar(x2-w/2,ada_vals,w,label='ADA@0.5%',color='coral',alpha=0.8)
ax.bar(x2+w/2,cov_vals,w,label='Coverage%',color='steelblue',alpha=0.8)
ax.set_xticks(x2); ax.set_xticklabels(names,rotation=15,ha='right'); ax.set_title('ADA@0.5% & Coverage'); ax.legend(fontsize=8)

# 6. Summary text
ax=axes[1,2]; ax.axis('off')
bm=results.get('BaseModel',results.get(list(results.keys())[0]))
best_name=[n for n in names if 'H2' in n and '150' in n]
if best_name: best=results[best_name[0]]
elif 'H2_Demo' in results: best=results['H2_Demo']
else: best=results.get(list(results.keys())[-1])

lines=['='*50,'  V4 Bold Prediction — Final Report','='*50,'',
    '  Core Innovation: L_mag magnitude matching','  breaks the zero-collapse trap.','',
    '  Best Config (H2): mag_w=1.0, under_w=3.0, dir_w=0.3','',
    '  Key Improvements over BaseModel:']
dm=best['mag_ratio']-bm['mag_ratio']; db=best['boldness']-bm['boldness']
dz=bm['zc_ratio']*100-best['zc_ratio']*100; dc=best['da']-bm['da']
lines.append('  mag_ratio: {:.3f} -> {:.3f} ({:+.0f}%)'.format(bm['mag_ratio'],best['mag_ratio'],dm*100))
lines.append('  boldness:  {:.3f} -> {:.3f} ({:+.0f}%)'.format(bm['boldness'],best['boldness'],db*100))
lines.append('  zc_ratio:  {:.1f}% -> {:.1f}% ({:+.0f}pp)'.format(bm['zc_ratio']*100,best['zc_ratio']*100,dz))
lines.append('  DA:        {:.1f}% -> {:.1f}% ({:+.1f}pp)'.format(bm['da'],best['da'],dc))
lines.append(''); lines.append('  Methodology:')
lines.append('  - No Oracle exploration (argmax rollout)')
lines.append('  - No CE/STaR loss (pure V4 loss)')
lines.append('  - Deterministic self-distillation')
lines.append('  - L_mag: L2 on magnitudes with under_weight')
lines.append('  - L_dir: BCE-style direction calibration')
lines.append('='*50)
ax.text(0.05,0.95,'\n'.join(lines),transform=ax.transAxes,fontsize=8.5,va='top',fontfamily='monospace',bbox=dict(boxstyle='round',fc='lightyellow',alpha=0.8))

plt.suptitle('Phase 8 V4 — Breaking Zero-Collapse with Bold Prediction Loss',fontsize=15,fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/v4_exp/final_comparison.png',dpi=150,bbox_inches='tight')
plt.close()
print('Saved: outputs/v4_exp/final_comparison.png')

# Print final table
print('\n'+'='*70)
print('  FINAL RESULTS TABLE')
print('='*70)
print('  {:<18} {:>6} {:>10} {:>10} {:>8} {:>8}'.format('Model','DA%','mag_ratio','boldness','zc%','ADA@0.5%'))
print('  '+'-'*62)
for nm in names:
    r=results[nm]; ada=r.get('ada_050','N/A')
    print('  {:<18} {:>6.1f} {:>10.3f} {:>10.3f} {:>8.1f} {:>8}'.format(nm,r['da'],r['mag_ratio'],r['boldness'],r['zc_ratio']*100,ada))
print('='*70)
