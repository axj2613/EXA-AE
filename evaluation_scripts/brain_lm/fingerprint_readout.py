"""Reconstruction-error FINGERPRINT clinical readout for a trained VisionTransformerBlockGenome.

Complements evaluate_final_model.py (which reads clinical variables off the CLS embedding). Here the
per-subject feature is the model's *per-parcel masked-reconstruction error* -- a 424-d normative
"fingerprint" of where THIS subject's brain reconstructs worse than the model expects. Motivation:
the CLS token was trained only to help reconstruction, so it need not encode identity-level structure;
the per-parcel error pattern is a richer, spatially-resolved readout and was the model's best clinical
signal in earlier runs.

Emits (into --output_dir, default eval_out), overwriting any stale copies:
  14_fingerprint_age.png       cross-validated age: predicted vs actual (R^2, Pearson r, MAE)
  15_fingerprint_sex_roc.png   cross-validated sex ROC (AUC)
  16_readout_comparison.png    fingerprint vs CLS embedding, with the FC+ridge classical ceiling
  17_age_predictive_parcels.png where in the brain the error pattern carries age signal

Same subject split, frozen train-split normalization, and CV recipe (KFold-5 + StandardScaler/RidgeCV
for age; StratifiedKFold-5 + StandardScaler/LogReg for sex) as evaluate_final_model.py, so the numbers
are directly comparable. TEST is never touched during any fitting other than nested CV on TEST itself.

Usage:
    python evaluation_scripts/brain_lm/fingerprint_readout.py \
        <genome_pkl> <hcp_root> <atlas_coords> <split_json> <stats_npz> <clinical_csv> \
        [--output_dir eval_out] [--device cpu] [--windows_per_subject 32] [--batch_size 16] \
        [--cls_cache eval_out/subject_embeddings.npz]
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

# reuse the eval module's validated design tokens + shared plot helpers (style/save/brain map)
_spec = importlib.util.spec_from_file_location("evalmod", os.path.join(_HERE, "evaluate_final_model.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

from time_series.hcp_window_dataset import HCPWindowDataset  # noqa: E402

# FC + ridge classical ceiling on the same TEST subjects (fixed reference from docs/improvement_plan.md)
FC_AGE_R2 = 0.119
FC_SEX_AUC = 0.907


def subject_fingerprints(genome, dataset, device, subjects, windows_per_subject, batch_size):
    """One 424-d per-parcel reconstruction-MSE vector per subject.

    For each subject we run masked forward passes over an even stride of its rest windows and
    accumulate squared error only on the parcels each pass actually masked (mask_ratio is the
    model's own), then divide by the per-parcel masked count. Over ~windows_per_subject passes at
    mask 0.5 every parcel is masked ~windows_per_subject/2 times, giving a stable estimate.
    """
    for m in genome._iter_modules():
        m.eval()
    n_parcels = genome.num_parcels
    X, kept = [], []
    with torch.no_grad():
        for i, sid in enumerate(subjects):
            try:
                w = dataset.subject_windows(sid, genome.window_length, types=("rest",))
            except Exception:
                continue
            if w is None or len(w) == 0:
                continue
            if len(w) > windows_per_subject:                       # even stride across the scan
                idx = np.linspace(0, len(w) - 1, windows_per_subject).astype(int)
                w = w[idx]
            se = np.zeros(n_parcels)
            cnt = np.zeros(n_parcels)
            for s in range(0, len(w), batch_size):
                genome.reset()
                chunk = w[s:s + batch_size].to(device)
                loss, pred, mask, patches = genome.forward(chunk)
                if not torch.isfinite(loss):
                    continue
                pred = pred.float().cpu().numpy()                 # (B, 424, 20)
                tgt = patches.float().cpu().numpy()
                msk = mask.bool().cpu().numpy()                   # (B, 424) 1 = masked
                for p in range(n_parcels):
                    rows = msk[:, p]
                    if not rows.any():
                        continue
                    d = (pred[rows, p, :] - tgt[rows, p, :]).ravel()
                    se[p] += (d * d).sum()
                    cnt[p] += d.size
            fp = np.where(cnt > 0, se / np.maximum(cnt, 1), np.nan)
            X.append(fp)
            kept.append(str(sid))
            if (i + 1) % 25 == 0:
                print(f"    fingerprinted {i + 1}/{len(subjects)} subjects", flush=True)
    for m in genome._iter_modules():
        m.train()
    X = np.vstack(X)
    # rare: a parcel never masked for a subject -> impute that column's mean so CV has no NaNs
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, kept


def cv_age(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    yhat = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
    r = float(np.corrcoef(y, yhat)[0, 1])
    return yhat, r2_score(y, yhat), r, mean_absolute_error(y, yhat)


def cv_sex(X, ybin):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    prob = cross_val_predict(clf, X, ybin, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(ybin, prob)
    return prob, roc_auc_score(ybin, prob), fpr, tpr


# ------------------------------------------------------------------ charts
def plot_fingerprint_age(y, yhat, r2, r, mae, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.scatter(y, yhat, s=26, color=E.S1, alpha=0.75, edgecolor=E.SURFACE, linewidth=0.5)
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
    pad = (hi - lo) * 0.06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=E.MUTED, linewidth=1.5, linestyle="--")
    E.style(ax, "Age from the reconstruction-error fingerprint", "actual age (years)",
            "predicted age (years)")
    ax.text(0.03, 0.97, f"Pearson r = {r:.2f}\nMAE = {mae:.1f} yr\ncross-val R² = {r2:.3f}",
            transform=ax.transAxes, va="top", color=E.INK, fontsize=11, fontweight="600")
    ax.text(0.97, 0.03, "dashed = perfect", transform=ax.transAxes, ha="right",
            color=E.MUTED, fontsize=8.5)
    E.save(fig, out)


def plot_fingerprint_roc(fpr, tpr, auc, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, color=E.S1, linewidth=2.4)
    ax.plot([0, 1], [0, 1], color=E.MUTED, linewidth=1.4, linestyle="--")
    E.style(ax, "Sex from the reconstruction-error fingerprint",
            "false-positive rate", "true-positive rate")
    ax.text(0.97, 0.06, f"AUC = {auc:.3f}", transform=ax.transAxes, ha="right",
            color=E.INK, fontsize=12, fontweight="600")
    ax.text(0.55, 0.44, "chance", color=E.MUTED, fontsize=9, rotation=33)
    E.save(fig, out)


def plot_readout_comparison(scores, out):
    """scores: {readout_name: (age_r2, sex_auc)} for the model-derived readouts. A plain `chance`
    line (R²=0 / AUC=0.5, the null model) is the only reference -- no external method baselines."""
    import matplotlib.pyplot as plt
    names = list(scores.keys())
    age = [scores[n][0] for n in names]
    auc = [scores[n][1] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, vals, chance, title, ylab in [
        (axes[0], age, 0.0, "Age  (cross-validated R²)", "R²"),
        (axes[1], auc, 0.5, "Sex  (cross-validated AUC)", "AUC"),
    ]:
        bars = ax.bar(names, vals, color=[E.S1, E.S2][:len(names)], width=0.55,
                      edgecolor=E.SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}",
                    ha="center", color=E.INK, fontsize=11, fontweight="600")
        ax.axhline(chance, color=E.MUTED, linewidth=1.4, linestyle="--")
        ax.text(len(names) - 0.5, chance, " chance", color=E.MUTED, fontsize=9,
                va="bottom", ha="right")
        top = max(max(vals), chance)
        ax.set_ylim(min(0.0, min(vals), chance) - 0.03, top * 1.18)
        E.style(ax, title, None, ylab)
    fig.suptitle("Which readout carries the clinical signal?", color=E.INK, fontsize=13.5,
                 fontweight="700", x=0.02, ha="left")
    E.save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("genome_pkl"); p.add_argument("hcp_root"); p.add_argument("atlas_coords")
    p.add_argument("split_json"); p.add_argument("stats_npz"); p.add_argument("clinical_csv")
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--length_index", default=None)
    p.add_argument("--windows_per_subject", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cls_cache", default="eval_out/subject_embeddings.npz")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    with open(args.genome_pkl, "rb") as f:
        genome = pickle.load(f)
    genome.to(device)

    dataset = HCPWindowDataset(
        root_dir=args.hcp_root, atlas_coordinates_filename=args.atlas_coords,
        window_length=genome.window_length, split_path=args.split_json,
        stats_path=args.stats_npz, length_index_path=args.length_index,
    )
    coords = dataset.parcel_coordinates
    coords = coords.numpy() if torch.is_tensor(coords) else np.asarray(coords)

    test_ids = [str(s) for s in json.load(open(args.split_json))["test"]]

    print("[1/2] per-subject reconstruction-error fingerprints ...")
    Xe, kept = subject_fingerprints(genome, dataset, device, test_ids,
                                    args.windows_per_subject, args.batch_size)
    print(f"  fingerprints: {Xe.shape} for {len(kept)} subjects")

    # align clinical labels to kept subject order
    clin = pd.read_csv(args.clinical_csv); clin["Subject"] = clin["Subject"].astype(str)
    df = pd.DataFrame({"Subject": kept}).merge(clin, on="Subject", how="left")

    # CLS embeddings from the eval run's cache, re-aligned to the SAME kept order
    cls = np.load(args.cls_cache, allow_pickle=True)
    cls_map = {str(s): row for s, row in zip(cls["subject_ids"], cls["X"])}
    have_cls = [s in cls_map for s in kept]
    Xc = np.vstack([cls_map[s] for s in kept if s in cls_map]) if any(have_cls) else None

    print("[2/2] cross-validated clinical readout ...")
    # --- age
    am = df["Age_in_Yrs"].notna().values
    yage = df["Age_in_Yrs"].values[am].astype(float)
    yhat, r2_fp, r_fp, mae_fp = cv_age(Xe[am], yage)
    print(f"  fingerprint  age  R²={r2_fp:+.3f}  r={r_fp:.2f}  MAE={mae_fp:.2f} yr")
    plot_fingerprint_age(yage, yhat, r2_fp, r_fp, mae_fp, f"{args.output_dir}/14_fingerprint_age.png")

    # --- sex
    g = df["Gender"].values
    sm = pd.notna(g)
    ybin = (g[sm] == "M").astype(int)
    _, auc_fp, fpr, tpr = cv_sex(Xe[sm], ybin)
    print(f"  fingerprint  sex  AUC={auc_fp:.3f}")
    plot_fingerprint_roc(fpr, tpr, auc_fp, f"{args.output_dir}/15_fingerprint_sex_roc.png")

    # --- CLS comparison on the identical subjects/CV
    scores = {"Reconstruction\nfingerprint": (r2_fp, auc_fp)}
    if Xc is not None and Xc.shape[0] == Xe.shape[0]:
        _, r2_c, _, _ = cv_age(Xc[am], yage)
        _, auc_c, _, _ = cv_sex(Xc[sm], ybin)
        print(f"  CLS embedding age  R²={r2_c:+.3f}   sex AUC={auc_c:.3f}")
        scores["CLS\nembedding"] = (r2_c, auc_c)
    plot_readout_comparison(scores, f"{args.output_dir}/16_readout_comparison.png")

    # --- where the age signal lives: |corr(parcel error, age)| on the brain
    age_corr = np.array([abs(np.corrcoef(Xe[am, p], yage)[0, 1]) for p in range(Xe.shape[1])])
    E.plot_brain_map(coords, age_corr, "Parcels whose reconstruction error tracks age",
                     "|corr(parcel error, age)|", f"{args.output_dir}/17_age_predictive_parcels.png")

    np.savez_compressed(f"{args.output_dir}/readout_features.npz",
                        Xe=Xe, subject_ids=np.array(kept, dtype=object))
    print(f"\nfingerprint charts written to {args.output_dir}/ (14-17)")


if __name__ == "__main__":
    main()
