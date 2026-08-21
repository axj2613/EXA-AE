"""'Purified' contribution maps: each cognition prediction map residualized on the reconstruction-
quality map (across parcels), so a parcel is credited for carrying cognition signal ABOVE what its
baseline reconstruction quality would predict -- rather than the raw map partly reflecting which
parcels reconstruct well to begin with. Diverging colormap: red = contributes MORE than expected,
blue = less."""
import importlib.util, os, sys, numpy as np
import matplotlib.pyplot as plt
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(_HERE,"..","..")))
_s=importlib.util.spec_from_file_location("E",os.path.join(_HERE,"evaluate_final_model.py")); E=importlib.util.module_from_spec(_s); _s.loader.exec_module(E)

coords=np.loadtxt("datasets/hcp/atlases/A424_Coordinates.dat")[:,1:4]
d=np.load("eval_out/_corrected_maps.npz")

def diverging_map(values, title, out):
    x,yy,z=coords.T
    lim=np.percentile(np.abs(values),98)
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.6))
    for ax,(i,j,xl,yl,ttl) in zip(axes,[(0,1,"x (L→R)","y (P→A)","Axial view"),(1,2,"y (P→A)","z (I→S)","Sagittal view")]):
        sc=ax.scatter(coords[:,i],coords[:,j],c=values,cmap="RdBu_r",vmin=-lim,vmax=lim,s=42,edgecolor=E.SURFACE,linewidth=0.3)
        E.style(ax,ttl,xl,yl)
    cb=fig.colorbar(sc,ax=axes,fraction=0.025,pad=0.02); cb.set_label("contribution beyond reconstruction (z)",color=E.INK_2,fontsize=9)
    cb.ax.tick_params(colors=E.MUTED,labelsize=8); cb.outline.set_visible(False)
    fig.suptitle(title,color=E.INK,fontsize=13.5,fontweight="700",x=0.02,ha="left")
    fig.patch.set_facecolor(E.SURFACE)
    fig.savefig(out,dpi=140,bbox_inches="tight",facecolor=E.SURFACE); plt.close(fig); print("saved",out)

diverging_map(d["gc"],"General cognition — contribution beyond reconstruction quality","eval_out/beh_brainmap_gfactor_corrected.png")
diverging_map(d["crc"],"Crystallized cognition — contribution beyond reconstruction quality","eval_out/beh_brainmap_cryst_corrected.png")
