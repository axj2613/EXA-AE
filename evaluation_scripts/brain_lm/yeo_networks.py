"""Assign each A424 parcel to a Yeo-7 network (by its MNI coordinate) and show which networks carry
the general-cognition signal -- raw and after correcting for reconstruction quality."""
import importlib.util, os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_atlas_yeo_2011
import nibabel as nib
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(_HERE,"..","..")))
_s=importlib.util.spec_from_file_location("E",os.path.join(_HERE,"evaluate_final_model.py")); E=importlib.util.module_from_spec(_s); _s.loader.exec_module(E)

coords=np.loadtxt("datasets/hcp/atlases/A424_Coordinates.dat")[:,1:4]
d=np.load("eval_out/_corrected_maps.npz"); g=d["g"]; gc=d["gc"]

img=nib.load(fetch_atlas_yeo_2011()["maps"]); vol=np.asarray(img.dataobj).squeeze().astype(int); inv=np.linalg.inv(img.affine)
def net_at(xyz, rad=4):
    v=inv@np.array([*xyz,1.0]); i,j,k=np.round(v[:3]).astype(int)
    best=0
    for r in range(0,rad+1):  # expand until a labeled voxel is found (parcels can sit just off the cortical ribbon)
        sub=vol[max(0,i-r):i+r+1, max(0,j-r):j+r+1, max(0,k-r):k+r+1]
        lab=sub[sub>0]
        if lab.size:
            vals,ct=np.unique(lab,return_counts=True); best=int(vals[ct.argmax()]); break
    return best
net=np.array([net_at(c) for c in coords])
NAMES={1:"Visual",2:"Somatomotor",3:"Dorsal Attn",4:"Salience/VentAttn",5:"Limbic",6:"Frontoparietal",7:"Default"}
YEOCOL={1:"#781286",2:"#4682B4",3:"#00760E",4:"#C43AFA",5:"#DCF8A4",6:"#E69422",7:"#CD3E4E"}
assigned=net[net>0]
print("parcels/network:", {NAMES[k]:int((net==k).sum()) for k in range(1,8)}, "| unassigned:", int((net==0).sum()))

# ---- per-network contribution: z-score each map across parcels, mean per network (comparable)
def z(a): return (a-a.mean())/a.std()
gz=z(g); gcz=z(gc)
nets=[k for k in range(1,8)]
raw=[gz[net==k].mean() for k in nets]; cor=[gcz[net==k].mean() for k in nets]
order=np.argsort(raw)[::-1]; nets=[nets[i] for i in order]; raw=[raw[i] for i in order]; cor=[cor[i] for i in order]
xx=np.arange(len(nets)); w=0.38
fig,ax=plt.subplots(figsize=(10.2,5.2))
ax.bar(xx-w/2,raw,w,color=E.S1,edgecolor=E.SURFACE,linewidth=1.2,label="raw g-map")
ax.bar(xx+w/2,cor,w,color=E.S2,edgecolor=E.SURFACE,linewidth=1.2,label="corrected (beyond reconstruction)")
ax.axhline(0,color=E.AXIS,lw=1.1)
ax.set_xticks(xx); ax.set_xticklabels([NAMES[k] for k in nets],rotation=20,ha="right",fontsize=9.5)
E.style(ax,"Which Yeo-7 networks carry the general-cognition signal","","mean contribution (z across parcels)")
ax.legend(frameon=False,fontsize=9.5,loc="upper right")
E.save(fig,"eval_out/beh_yeo_networks.png")

# ---- network-colored parcel map (context)
x,y,zc=coords.T
fig,axes=plt.subplots(1,2,figsize=(13.2,5.6))
for ax,(i,j,xl,yl,t) in zip(axes,[(0,1,"x (L→R)","y (P→A)","Axial view"),(1,2,"y (P→A)","z (I→S)","Sagittal view")]):
    for k in range(1,8):
        mask=net==k
        ax.scatter(coords[mask,i],coords[mask,j],c=YEOCOL[k],s=40,edgecolor=E.SURFACE,linewidth=0.3,label=NAMES[k] if ax is axes[0] else None)
    E.style(ax,t,xl,yl)
axes[0].legend(frameon=False,fontsize=8.5,loc="upper left",ncol=2)
fig.suptitle("A424 parcels colored by Yeo-7 network",color=E.INK,fontsize=13.5,fontweight="700",x=0.02,ha="left")
E.save(fig,"eval_out/beh_brainmap_yeo.png")

np.save("eval_out/_yeo_net.npy",net)
print("\nraw g-map network ranking (z):")
for k,r in zip(nets,raw): print(f"  {NAMES[k]:<18}{r:+.2f}")
print("saved eval_out/beh_yeo_networks.png, beh_brainmap_yeo.png")
