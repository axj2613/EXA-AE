"""For each behavioral construct: a reconstruction-CORRECTED brain map (residualized on
reconstruction quality) + a Yeo-7 per-network contribution chart. PNG outputs only."""
import importlib.util, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(_HERE,"..","..")))
_s=importlib.util.spec_from_file_location("E",os.path.join(_HERE,"evaluate_final_model.py")); E=importlib.util.module_from_spec(_s); _s.loader.exec_module(E)

OUT="eval_out"
fp=np.load(f"{OUT}/all_readout_features.npz",allow_pickle=True); Xe=np.asarray(fp["Xe"]); ids=[str(s) for s in fp["subject_ids"]]; rowi={s:i for i,s in enumerate(ids)}
coords=np.loadtxt("datasets/hcp/atlases/A424_Coordinates.dat")[:,1:4]
recon_q=1.0-Xe.mean(0)
net=np.load(f"{OUT}/_yeo_net.npy")
NAMES={1:"Visual",2:"Somatomotor",3:"Dorsal Attn",4:"Salience/VentAttn",5:"Limbic",6:"Frontoparietal",7:"Default"}

_c={}
def series(f,c):
    if f not in _c:
        d=pd.read_csv(f"datasets/hcp/HCP_YA_subjects_{f}.csv"); d["Subject"]=d["Subject"].astype(str); _c[f]=d.set_index("Subject")
    s=_c[f][c].reindex(ids)
    if not pd.api.types.is_numeric_dtype(s):  # encode string binaries (e.g. Gender M/F) as 0/1 (point-biserial)
        s=s.astype("category").cat.codes.replace(-1,np.nan)
    return s.to_numpy(float)
def pmap(varlist):
    M=np.column_stack([series(f,c) for f,c in varlist]); mu=np.nanmean(M,0); sd=np.nanstd(M,0)
    comp=np.nanmean((M-mu)/sd,1); k=np.isfinite(comp); idx=np.where(k)[0]; yv=comp[k]
    return np.array([abs(pearsonr(Xe[idx,p],yv)[0]) for p in range(424)])
def resid(m):
    b=np.polyfit(recon_q,m,1); r=m-np.polyval(b,recon_q); return (r-r.mean())/r.std()
def z(a): return (a-a.mean())/a.std()

def diverging_map(values,title,out):
    lim=np.percentile(np.abs(values),98); fig,axes=plt.subplots(1,2,figsize=(13.2,5.6))
    for ax,(i,j,xl,yl,t) in zip(axes,[(0,1,"x (L→R)","y (P→A)","Axial view"),(1,2,"y (P→A)","z (I→S)","Sagittal view")]):
        sc=ax.scatter(coords[:,i],coords[:,j],c=values,cmap="RdBu_r",vmin=-lim,vmax=lim,s=42,edgecolor=E.SURFACE,linewidth=0.3); E.style(ax,t,xl,yl)
    cb=fig.colorbar(sc,ax=axes,fraction=0.025,pad=0.02); cb.set_label("contribution beyond reconstruction (z)",color=E.INK_2,fontsize=9)
    cb.ax.tick_params(colors=E.MUTED,labelsize=8); cb.outline.set_visible(False)
    fig.suptitle(title,color=E.INK,fontsize=13.5,fontweight="700",x=0.02,ha="left"); fig.patch.set_facecolor(E.SURFACE)
    fig.savefig(out,dpi=140,bbox_inches="tight",facecolor=E.SURFACE); plt.close(fig)

def yeo_chart(raw,cor,title,out):
    rz,cz=z(raw),z(cor); nets=list(range(1,8))
    r=[rz[net==k].mean() for k in nets]; c=[cz[net==k].mean() for k in nets]
    o=np.argsort(r)[::-1]; nets=[nets[i] for i in o]; r=[r[i] for i in o]; c=[c[i] for i in o]
    xx=np.arange(7); w=.38; fig,ax=plt.subplots(figsize=(10.2,5.0))
    ax.bar(xx-w/2,r,w,color=E.S1,edgecolor=E.SURFACE,linewidth=1.2,label="raw map")
    ax.bar(xx+w/2,c,w,color=E.S2,edgecolor=E.SURFACE,linewidth=1.2,label="corrected (beyond reconstruction)")
    ax.axhline(0,color=E.AXIS,lw=1.1); ax.set_xticks(xx); ax.set_xticklabels([NAMES[k] for k in nets],rotation=20,ha="right",fontsize=9.5)
    E.style(ax,title,"","mean contribution (z across parcels)"); ax.legend(frameon=False,fontsize=9.5,loc="upper right")
    E.save(fig,out)
    return [(NAMES[nets[i]],round(r[i],2)) for i in range(3)]

COG=[("cognition","CogCrystalComp_Unadj"),("cognition","CogFluidComp_Unadj"),("cognition","PMAT24_A_CR"),("cognition","ReadEng_Unadj"),("cognition","PicVocab_Unadj"),("task_performance","WM_Task_Acc"),("task_performance","Language_Task_Acc")]
CRYST=[("cognition","CogCrystalComp_Unadj"),("cognition","ReadEng_Unadj"),("cognition","PicVocab_Unadj")]
CONSTRUCTS=[
 ("sex","Sex (M vs F)",[("info","Gender")],0.79),
 ("age","Age (years)",[("info","Age_in_Yrs")],0.38),
 ("gfactor","General cognition (g)",COG,0.44),
 ("cryst","Crystallized cognition",CRYST,0.40),
 ("fluid","Fluid reasoning",[("cognition","PMAT24_A_CR")],0.31),
 ("lang","Language task accuracy",[("task_performance","Language_Task_Acc")],0.31),
 ("wm","Working-memory task",[("task_performance","WM_Task_Acc")],0.27),
 ("ddisc","Delay discounting",[("cognition","DDisc_AUC_40K")],0.25),
 ("pain","Pain intensity",[("sensory","PainIntens_RawScore")],0.27),
 ("rule","Rule-breaking",[("psychiatric","ASR_Rule_T")],0.25),
]
print("construct              r     top-3 Yeo networks (raw contribution z)")
for tag,name,vl,r in CONSTRUCTS:
    m=pmap(vl); mc=resid(m)
    diverging_map(mc,f"{name} — contribution beyond reconstruction quality",f"{OUT}/beh_brainmap_{tag}_corrected.png")
    top=yeo_chart(m,mc,f"Which Yeo-7 networks carry {name}",f"{OUT}/beh_yeo_{tag}.png")
    print(f"{name:<22}{r:.2f}  " + ", ".join(f"{n} {v:+.2f}" for n,v in top))
print("\nwrote beh_brainmap_<tag>_corrected.png and beh_yeo_<tag>.png for:", ", ".join(t for t,*_ in CONSTRUCTS))
