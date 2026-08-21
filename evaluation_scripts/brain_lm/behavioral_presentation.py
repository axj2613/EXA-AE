"""Presentation-ready behavioral-prediction figures from the reconstruction fingerprint:
 (1) individual predicted-vs-actual scatters (r AND R^2) per construct, held-out TEST;
 (2) a per-construct brain map of where the signal lives (|corr(parcel error, target)|);
 (3) the positive-negative mode -- a cognition(+) / impulsivity-substance(-) composite -- with its
     scatter, PC1 loadings, and brain map.
All outputs: eval_out/beh_*.png"""
import importlib.util, json, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(_HERE,"..","..")))
_s=importlib.util.spec_from_file_location("E",os.path.join(_HERE,"evaluate_final_model.py")); E=importlib.util.module_from_spec(_s); _s.loader.exec_module(E)

fp=np.load("eval_out/all_readout_features.npz",allow_pickle=True)
Xe=np.asarray(fp["Xe"]); ids=[str(s) for s in fp["subject_ids"]]; rowi={s:i for i,s in enumerate(ids)}
split=json.load(open("subject_split.json")); tr=[s for s in map(str,split["train"]) if s in rowi]; te=[s for s in map(str,split["test"]) if s in rowi]
coords=np.loadtxt("datasets/hcp/atlases/A424_Coordinates.dat")[:,1:4]; assert coords.shape==(424,3)
OUT="eval_out"

_cache={}
def col(file,c):
    if file not in _cache:
        d=pd.read_csv(f"datasets/hcp/HCP_YA_subjects_{file}.csv"); d["Subject"]=d["Subject"].astype(str); _cache[file]=d
    return dict(zip(_cache[file]["Subject"],_cache[file][c]))

def ridge(ftr,ytr,fte): return make_pipeline(StandardScaler(),PCA(50,random_state=0),RidgeCV(alphas=np.logspace(-1,3,7))).fit(ftr,ytr).predict(fte)

def scatter(y,pred,title,out,xlab="actual score",ylab="predicted score"):
    r=pearsonr(y,pred)[0]; r2=r2_score(y,pred)
    fig,ax=plt.subplots(figsize=(5.8,5.6))
    ax.scatter(y,pred,s=20,color=E.S1,alpha=0.6,edgecolor=E.SURFACE,linewidth=0.4)
    b=np.polyfit(y,pred,1); xs=np.array([y.min(),y.max()]); ax.plot(xs,np.polyval(b,xs),color=E.CRITICAL,linewidth=2.0)
    E.style(ax,title,xlab,ylab)
    ax.text(0.04,0.96,f"r = {r:.2f}\nR² = {r2:.3f}\nn = {len(y)}",transform=ax.transAxes,va="top",color=E.INK,fontsize=12,fontweight="700")
    E.save(fig,out); return r,r2

def target_all(getval):
    """(row indices into Xe, target values) for every subject with a finite label."""
    idx,y=[],[]
    for s in ids:
        v=getval(s)
        if v is None or (isinstance(v,float) and np.isnan(v)): continue
        idx.append(rowi[s]); y.append(float(v))
    return np.array(idx),np.array(y)

def brainmap(getval,title,out):
    idx,y=target_all(getval)
    vals=np.array([abs(pearsonr(Xe[idx,p],y)[0]) for p in range(424)])
    E.plot_brain_map(coords,vals,title,"|corr(parcel error, score)|",out)

def split_xy(getval):
    def grab(subs):
        xs,ys=[],[]
        for s in subs:
            v=getval(s)
            if v is None or (isinstance(v,float) and np.isnan(v)): continue
            xs.append(Xe[rowi[s]]); ys.append(float(v))
        return np.array(xs),np.array(ys)
    return grab(tr),grab(te)

# ---------------- single-variable constructs: scatter (+ brain map for the cognition/impulsivity ones)
SINGLE=[("cognition","CogCrystalComp_AgeAdj","Crystallized cognition","cryst",1),
        ("cognition","PMAT24_A_CR","Fluid reasoning (Penn matrices)","fluid",1),
        ("cognition","CogTotalComp_AgeAdj","General cognition","total",1),
        ("task_performance","WM_Task_Acc","Working-memory task accuracy","wm",1),
        ("task_performance","Language_Task_Acc","Language task accuracy","lang",0),
        ("cognition","DDisc_AUC_40K","Delay discounting (patience)","ddisc",1),
        ("sensory","PainIntens_RawScore","Pain intensity","pain",0),
        ("psychiatric","ASR_Rule_T","Rule-breaking (externalizing)","rule",0)]
for file,c,name,tag,mapit in SINGLE:
    gv=(lambda mm: (lambda s: mm.get(s)))(col(file,c))
    (Xtr,ytr),(Xte,yte)=split_xy(gv); pred=ridge(Xtr,ytr,Xte)
    r,r2=scatter(yte,pred,f"{name} — held-out TEST",f"{OUT}/beh_scatter_{tag}.png")
    if mapit: brainmap(gv,f"Where the {name.split(' (')[0].lower()} signal lives",f"{OUT}/beh_brainmap_{tag}.png")
    print(f"scatter {name:<34} r={r:+.3f} R2={r2:+.3f}")

# ---------------- cognition g-factor (average of z-scored cognitive measures)
COG=[("cognition","CogCrystalComp_Unadj"),("cognition","CogFluidComp_Unadj"),("cognition","PMAT24_A_CR"),
     ("cognition","ReadEng_Unadj"),("cognition","PicVocab_Unadj"),("task_performance","WM_Task_Acc"),
     ("task_performance","Language_Task_Acc")]
def zmat(varlist,subs,mu=None,sd=None):
    M=np.array([[col(f,c).get(s,np.nan) for f,c in varlist] for s in subs],float)
    if mu is None: mu=np.nanmean(M,0); sd=np.nanstd(M,0)
    return (M-mu)/sd, mu, sd
Ztr,mu,sd=zmat(COG,tr); Zte,_,_=zmat(COG,te,mu,sd)
gtr=np.nanmean(Ztr,1); gte=np.nanmean(Zte,1)
ktr=np.isfinite(gtr); kte=np.isfinite(gte)
Xtr=np.array([Xe[rowi[s]] for s,k in zip(tr,ktr) if k]); Xte=np.array([Xe[rowi[s]] for s,k in zip(te,kte) if k])
predg=ridge(Xtr,gtr[ktr],Xte); rg,r2g=scatter(gte[kte],predg,"General cognitive factor (g) — held-out TEST",
                                               f"{OUT}/beh_scatter_gfactor.png","actual g (z)","predicted g (z)")
# brain map of g across ALL subjects
Zall,_,_=zmat(COG,ids,mu,sd); gall=np.nanmean(Zall,1); ka=np.isfinite(gall)
idxa=np.array([rowi[s] for s,k in zip(ids,ka) if k]); ya=gall[ka]
valg=np.array([abs(pearsonr(Xe[idxa,p],ya)[0]) for p in range(424)])
E.plot_brain_map(coords,valg,"Where the general-cognition (g) signal lives",f"|corr(parcel error, g)|",f"{OUT}/beh_brainmap_gfactor.png")
print(f"g-factor r={rg:+.3f} R2={r2g:+.3f}")

# ---------------- positive-negative mode: PCA over cognition(+) and impulsivity/substance/externalizing(-)
POS=COG+[("cognition","DDisc_AUC_40K")]
NEG=[("psychiatric","ASR_Rule_Raw"),("psychiatric","DSM_Antis_Raw"),("emotion","AngAggr_Unadj"),
     ("substance_use","SSAGA_Alc_12_Frq_Drk"),("substance_use","Total_Drinks_7days"),
     ("substance_use","Num_Days_Used_Any_Tobacco_7days")]
VARS=POS+NEG
NAMES=["Crystallized","Fluid comp","Penn matrices","Reading","PicVocab","WM task","Language task","Patience (DDisc)",
       "Rule-breaking","Antisocial","Aggression","Alcohol freq","Drinks/wk","Tobacco days"]
Ztr,mu,sd=zmat(VARS,tr); Zte,_,_=zmat(VARS,te,mu,sd)
Ztr_i=np.nan_to_num(Ztr); Zte_i=np.nan_to_num(Zte)
pca=PCA(3,random_state=0).fit(Ztr_i)
load=pca.components_[0].copy()
if load[0]<0: load=-load; sgn=-1
else: sgn=1
pctr=sgn*pca.transform(Ztr_i)[:,0]; pcte=sgn*pca.transform(Zte_i)[:,0]
Xtr=np.array([Xe[rowi[s]] for s in tr]); Xte=np.array([Xe[rowi[s]] for s in te])
predp=ridge(Xtr,pctr,Xte); rp,r2p=scatter(pcte,predp,"Positive–negative mode — held-out TEST",
                                          f"{OUT}/beh_scatter_posneg.png","actual mode score","predicted mode score")
# loadings bar
order=np.argsort(load); fig,ax=plt.subplots(figsize=(8.6,6.0))
cols=[E.S2 if load[i]>=0 else E.CRITICAL for i in order]
ax.barh(np.arange(len(order)),load[order],color=cols,edgecolor=E.SURFACE,linewidth=1.2)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([NAMES[i] for i in order],fontsize=9.5)
ax.axvline(0,color=E.AXIS,lw=1.2)
E.style(ax,"Positive–negative mode: what loads on the axis","PC1 loading",None)
ax.text(0.98,0.04,"green = positive pole (cognition)\nred = negative pole (impulsivity / substance)",transform=ax.transAxes,ha="right",color=E.MUTED,fontsize=9)
E.save(fig,f"{OUT}/beh_posneg_loadings.png")
# brain map of the mode across all subjects
Zall,_,_=zmat(VARS,ids,mu,sd); pcall=sgn*pca.transform(np.nan_to_num(Zall))[:,0]
# only subjects that actually had >=half the vars, to avoid imputation-dominated scores
frac=np.isfinite(Zall).mean(1); ka=frac>=0.5
idxa=np.array([rowi[s] for s,k in zip(ids,ka) if k]); ya=pcall[ka]
valp=np.array([abs(pearsonr(Xe[idxa,p],ya)[0]) for p in range(424)])
E.plot_brain_map(coords,valp,"Where the positive–negative mode lives","|corr(parcel error, mode)|",f"{OUT}/beh_brainmap_posneg.png")
print(f"pos-neg mode r={rp:+.3f} R2={r2p:+.3f} | variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print("PC1 loadings:", {NAMES[i]:round(load[i],2) for i in order})
