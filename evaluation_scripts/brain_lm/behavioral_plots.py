"""Plots for the interesting, FDR-significant, beyond-demographics behavioral predictions from the
reconstruction fingerprint (held-out TEST). Trivial demographic-proxy hits (Height/Strength/etc.,
which age+sex predict better) and within-file redundant duplicates are deliberately excluded."""
import importlib.util, json, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(_HERE,"..","..")))
E = importlib.util.module_from_spec(importlib.util.spec_from_file_location("E", os.path.join(_HERE,"evaluate_final_model.py")))
importlib.util.spec_from_file_location("E", os.path.join(_HERE,"evaluate_final_model.py")).loader.exec_module(E)

fp = np.load("eval_out/all_readout_features.npz", allow_pickle=True)
Xe=np.asarray(fp["Xe"]); ids=[str(s) for s in fp["subject_ids"]]; rowi={s:i for i,s in enumerate(ids)}
meanfp=Xe.mean(axis=1)
split=json.load(open("subject_split.json")); tr=[s for s in map(str,split["train"]) if s in rowi]; te=[s for s in map(str,split["test"]) if s in rowi]
info=pd.read_csv("datasets/hcp/HCP_YA_subjects_info.csv"); info["Subject"]=info["Subject"].astype(str)
age=dict(zip(info["Subject"],info["Age_in_Yrs"])); sex=dict(zip(info["Subject"],(info["Gender"]=="M").astype(float)))

# (file, column, label, interesting?)  -- last two rows are demographic-proxies shown for CONTRAST
PICKS=[("cognition","CogCrystalComp_AgeAdj","Crystallized cognition",1),
       ("cognition","ReadEng_AgeAdj","Reading / vocabulary",1),
       ("cognition","CogTotalComp_AgeAdj","Total cognition",1),
       ("cognition","PMAT24_A_CR","Fluid reasoning (Penn matrices)",1),
       ("task_performance","Language_Task_Acc","Language task accuracy",1),
       ("task_performance","WM_Task_Acc","Working-memory task accuracy",1),
       ("cognition","DDisc_AUC_40K","Delay discounting (AUC)",1),
       ("sensory","PainIntens_RawScore","Pain intensity",1),
       ("psychiatric","ASR_Rule_T","Rule-breaking (externalizing)",1),
       ("motor","Strength_Unadj","Grip strength  (sex proxy)",0),
       ("health_family","Height","Height  (sex proxy)",0)]

_colcache={}
def col(file,c):
    if file not in _colcache:
        d=pd.read_csv(f"datasets/hcp/HCP_YA_subjects_{file}.csv"); d["Subject"]=d["Subject"].astype(str); _colcache[file]=d
    return dict(zip(_colcache[file]["Subject"],_colcache[file][c]))

def ridge(ftr,ytr,fte,pca=None):
    steps=[StandardScaler()]+([PCA(pca,random_state=0)] if pca else [])+[RidgeCV(alphas=np.logspace(-1,3,7))]
    return make_pipeline(*steps).fit(ftr,ytr).predict(fte)

results=[]
for file,c,label,interesting in PICKS:
    m=col(file,c)
    def grab(subs):
        xs,ms,ds,ys=[],[],[],[]
        for s in subs:
            v=m.get(s)
            if v is None or (isinstance(v,float) and np.isnan(v)): continue
            xs.append(Xe[rowi[s]]); ms.append(meanfp[rowi[s]]); ds.append([age[s],sex[s]]); ys.append(float(v))
        return np.array(xs),np.array(ms),np.array(ds),np.array(ys)
    Xtr,Mtr,Dtr,ytr=grab(tr); Xte,Mte,Dte,yte=grab(te)
    pred=ridge(Xtr,ytr,Xte,pca=50); r=pearsonr(yte,pred)[0]; r2=r2_score(yte,pred)
    dr=pearsonr(yte,ridge(Dtr,ytr,Dte))[0]                                   # age+sex baseline
    mr=pearsonr(yte,ridge(Mtr.reshape(-1,1),ytr,Mte.reshape(-1,1)))[0]       # overall-error only
    b=np.polyfit(Mtr,ytr,1); resr=pearsonr(yte-np.polyval(b,Mte),ridge(Xtr,ytr-np.polyval(b,Mtr),Xte,pca=50))[0]
    results.append(dict(label=label,interesting=interesting,r=r,r2=r2,dr=dr,mr=mr,resr=resr,y=yte,pred=pred,n=len(yte)))

# ---- Chart 24: fingerprint vs age+sex baseline (the interesting-vs-trivial separation)
order=sorted(results,key=lambda z:z["r"],reverse=True)
labels=[x["label"] for x in order]; fpr=[x["r"] for x in order]; dr=[x["dr"] for x in order]
y=np.arange(len(labels))[::-1]
fig,ax=plt.subplots(figsize=(9.8,6.2))
ax.barh(y+0.2,fpr,0.38,color=E.S1,edgecolor=E.SURFACE,linewidth=1.2,label="Reconstruction fingerprint")
ax.barh(y-0.2,dr,0.38,color=E.MUTED,edgecolor=E.SURFACE,linewidth=1.2,label="Age + sex only")
for yi,x in zip(y,order):
    ax.text(x["r"]+0.006,yi+0.2,f"{x['r']:.2f}",va="center",fontsize=9,fontweight="600",color=E.INK)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.axvline(0,color=E.AXIS,lw=1.2)
E.style(ax,"Predicting behavior: fingerprint vs age + sex baseline","held-out TEST correlation (r)",None)
ax.legend(frameon=False,fontsize=9.5,loc="lower right")
ax.text(0.99,0.02,"cognition: fingerprint >> age+sex   |   physical: age+sex wins",transform=ax.transAxes,ha="right",color=E.MUTED,fontsize=9)
E.save(fig,"eval_out/24_behavioral_beyond_demographics.png")

# ---- Chart 25: predicted-vs-actual for 6 distinct interesting domains
grid=[x for x in results if x["label"] in
      ["Crystallized cognition","Fluid reasoning (Penn matrices)","Working-memory task accuracy",
       "Delay discounting (AUC)","Pain intensity","Rule-breaking (externalizing)"]]
fig,axes=plt.subplots(2,3,figsize=(13.5,8.2))
for ax,x in zip(axes.ravel(),grid):
    ax.scatter(x["y"],x["pred"],s=16,color=E.S1,alpha=0.6,edgecolor=E.SURFACE,linewidth=0.4)
    b=np.polyfit(x["y"],x["pred"],1); xs=np.array([x["y"].min(),x["y"].max()])
    ax.plot(xs,np.polyval(b,xs),color=E.CRITICAL,lw=1.8)
    E.style(ax,x["label"],"actual","predicted")
    ax.text(0.04,0.96,f"r = {x['r']:.2f}\nn = {x['n']}",transform=ax.transAxes,va="top",color=E.INK,fontsize=10,fontweight="600")
fig.suptitle("Held-out TEST predictions — beyond age, sex, and data quality",color=E.INK,fontsize=14,fontweight="700",x=0.02,ha="left")
E.save(fig,"eval_out/25_behavioral_scatter.png")

# ---- Chart 26: data-quality robustness (spatial fingerprint vs overall-error-only vs residual)
rob=[x for x in order if x["interesting"]]
labels=[x["label"] for x in rob]; x=np.arange(len(labels))
fig,ax=plt.subplots(figsize=(11.5,5.2)); w=0.27
ax.bar(x-w,[v["r"] for v in rob],w,color=E.S1,edgecolor=E.SURFACE,lw=1.2,label="spatial fingerprint (PCA-50)")
ax.bar(x,[v["mr"] for v in rob],w,color=E.MUTED,edgecolor=E.SURFACE,lw=1.2,label="overall error only (data quality)")
ax.bar(x+w,[v["resr"] for v in rob],w,color=E.S2,edgecolor=E.SURFACE,lw=1.2,label="residual (quality removed)")
ax.axhline(0,color=E.AXIS,lw=1.1)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha="right",fontsize=8.5)
E.style(ax,"Robustness: the signal is spatial FC, not global data quality","","held-out TEST r")
ax.legend(frameon=False,fontsize=9,loc="upper right")
E.save(fig,"eval_out/26_behavioral_dataquality.png")
print("saved eval_out/24_behavioral_beyond_demographics.png, 25_behavioral_scatter.png, 26_behavioral_dataquality.png")
for x in order: print(f"  {x['label']:<40} r={x['r']:+.3f}  age+sex={x['dr']:+.3f}  resid={x['resr']:+.3f}")
