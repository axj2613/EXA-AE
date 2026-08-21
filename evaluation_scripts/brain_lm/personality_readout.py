"""Big-Five (NEO-FFI) personality prediction from a trained VisionTransformerBlockGenome, using
BOTH subject readouts and comparing them head-to-head:

  * CLS embedding            (eval_out/subject_embeddings.npz, 128-d learned summary)
  * reconstruction fingerprint (eval_out/readout_features.npz, 424-d per-parcel masked-recon MSE)

Both feature caches are produced by evaluate_final_model.py / fingerprint_readout.py on the SAME
final_model.pkl and the SAME held-out TEST split, so this script needs no model forward passes -- it
only runs the identical cross-validation recipe used elsewhere (KFold-5 + StandardScaler + RidgeCV)
on each cached feature set, for each of the five NEOFAC traits.

Caveat, stated up front: personality is only weakly (often not at all) decodable from resting-state
fMRI; near-zero R^2 is the expected literature result, not a bug. We report R^2 / Pearson r / MAE
honestly for every trait and readout.

Emits into --output_dir (default eval_out):
  18_personality_comparison.png          per-trait CV R^2, fingerprint vs CLS (chance = 0)
  19_personality_scatter_fingerprint.png predicted-vs-actual, 5 traits (fingerprint)
  20_personality_scatter_cls.png         predicted-vs-actual, 5 traits (CLS embedding)

Usage:
    python evaluation_scripts/brain_lm/personality_readout.py <personality_csv> \
        [--cls_cache eval_out/subject_embeddings.npz] \
        [--fingerprint_cache eval_out/readout_features.npz] [--output_dir eval_out]
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

# reuse the eval module's validated design tokens + style/save helpers
_spec = importlib.util.spec_from_file_location("evalmod", os.path.join(_HERE, "evaluate_final_model.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

TRAITS = [
    ("NEOFAC_O", "Openness"),
    ("NEOFAC_C", "Conscientiousness"),
    ("NEOFAC_E", "Extraversion"),
    ("NEOFAC_A", "Agreeableness"),
    ("NEOFAC_N", "Neuroticism"),
]


def cv_regress(X, y):
    """Identical recipe to evaluate_final_model.py's clinical regression, plus a parametric p-value
    on the correlation between held-out predictions and truth (a screen for above-chance signal;
    slightly anticonservative because CV folds aren't fully independent -- permutation-test anything
    that looks promising)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy.stats import pearsonr
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    yhat = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
    if np.std(yhat) > 0:
        r, pval = pearsonr(y, yhat)
    else:
        r, pval = 0.0, 1.0
    return yhat, r2_score(y, yhat), float(r), float(pval), mean_absolute_error(y, yhat)


def plot_comparison(results, out):
    """results[label] = {'fingerprint': (r2,r,mae,y,yhat), 'cls': (...)}. Grouped R^2 bars."""
    import matplotlib.pyplot as plt
    labels = list(results.keys())
    fp = [results[l]["fingerprint"][0] for l in labels]
    cl = [results[l]["cls"][0] for l in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    b1 = ax.bar(x - w / 2, fp, w, color=E.S1, edgecolor=E.SURFACE, linewidth=1.5,
                label="Reconstruction fingerprint")
    b2 = ax.bar(x + w / 2, cl, w, color=E.S2, edgecolor=E.SURFACE, linewidth=1.5,
                label="CLS embedding")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (0.004 if v >= 0 else -0.004),
                    f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    color=E.INK, fontsize=8.5, fontweight="600")
    ax.axhline(0, color=E.AXIS, linewidth=1.4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    lo = min(0.0, min(fp + cl)); hi = max(0.02, max(fp + cl))
    ax.set_ylim(lo - 0.02, hi + 0.03)
    E.style(ax, "Big-Five personality from resting-state fMRI  (cross-validated R²)", None,
            "R²   (0 = no better than predicting the mean)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.text(0.01, 0.02, "R² ≤ 0 → the readout carries no usable signal for that trait",
            transform=ax.transAxes, color=E.MUTED, fontsize=9)
    E.save(fig, out)


def plot_scatter_grid(results, key, title, out):
    import matplotlib.pyplot as plt
    labels = list(results.keys())
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8))
    for ax, label in zip(axes, labels):
        r2, r, pval, mae, y, yhat = results[label][key]
        ax.scatter(y, yhat, s=16, color=E.S1 if key == "fingerprint" else E.S2, alpha=0.6,
                   edgecolor=E.SURFACE, linewidth=0.4)
        lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
        pad = (hi - lo) * 0.06
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=E.MUTED, linewidth=1.3,
                linestyle="--")
        E.style(ax, label, "actual", "predicted")
        ax.text(0.04, 0.96, f"r = {r:.2f} (p={pval:.2f})\nR² = {r2:.3f}", transform=ax.transAxes,
                va="top", color=E.INK, fontsize=9.5, fontweight="600")
    fig.suptitle(title, color=E.INK, fontsize=13.5, fontweight="700", x=0.01, ha="left")
    E.save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("personality_csv")
    p.add_argument("--cls_cache", default="eval_out/subject_embeddings.npz")
    p.add_argument("--fingerprint_cache", default="eval_out/readout_features.npz")
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--tag", default="", help="suffix for output filenames, e.g. _pooled")
    args = p.parse_args()

    cls = np.load(args.cls_cache, allow_pickle=True)
    fp = np.load(args.fingerprint_cache, allow_pickle=True)
    cls_ids = [str(s) for s in cls["subject_ids"]]
    fp_ids = [str(s) for s in fp["subject_ids"]]
    assert cls_ids == fp_ids, "CLS and fingerprint caches are not subject-aligned; re-run the evals"
    Xc, Xe = np.asarray(cls["X"]), np.asarray(fp["Xe"])

    per = pd.read_csv(args.personality_csv); per["Subject"] = per["Subject"].astype(str)
    df = pd.DataFrame({"Subject": cls_ids}).merge(per, on="Subject", how="left")

    print(f"features: CLS {Xc.shape}, fingerprint {Xe.shape}  |  subjects {len(cls_ids)}")
    print(f"{'trait':17s} {'readout':13s} {'R2':>7s} {'r':>6s} {'p':>7s} {'MAE':>6s}   (n)")
    results = {}
    for col, label in TRAITS:
        y_all = df[col].values.astype(float)
        m = ~np.isnan(y_all)
        y = y_all[m]
        entry = {}
        for key, X in (("fingerprint", Xe), ("cls", Xc)):
            yhat, r2, r, pval, mae = cv_regress(X[m], y)
            entry[key] = (r2, r, pval, mae, y, yhat)
            star = " *" if pval < 0.05 else ""
            print(f"{label:17s} {key:13s} {r2:>7.3f} {r:>6.2f} {pval:>7.3f} {mae:>6.2f}   ({m.sum()}){star}")
        results[label] = entry

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag
    plot_comparison(results, f"{args.output_dir}/18_personality_comparison{tag}.png")
    plot_scatter_grid(results, "fingerprint", "Personality from the reconstruction-error fingerprint",
                      f"{args.output_dir}/19_personality_scatter_fingerprint{tag}.png")
    plot_scatter_grid(results, "cls", "Personality from the CLS embedding",
                      f"{args.output_dir}/20_personality_scatter_cls{tag}.png")
    print(f"\npersonality charts written to {args.output_dir}/ (18-20{tag})")


if __name__ == "__main__":
    main()
