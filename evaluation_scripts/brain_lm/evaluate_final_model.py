"""Full BrainLM-parity evaluation of a trained VisionTransformerBlockGenome, emitting
presentation-ready charts for all three analyses BrainLM reports:

  1. RECONSTRUCTION (BrainLM Fig 3 / Sec 4.1) -- masked-patch MSE / MAE / R^2 / Pearson r on the
     held-out TEST split, plus predicted-vs-actual density, per-parcel R^2 distribution, an
     anatomical R^2 map, and example reconstructions.
  2. SUBJECT EMBEDDINGS -> CLINICAL VARIABLES (Fig 5 / Table 1) -- one CLS embedding per held-out
     subject, a UMAP colored by age/gender, and cross-validated prediction of the HCP behavioral
     variables (age, education, gender).
  3. ATTENTION (Sec 4.4/4.5) -- how much the CLS summary token attends to each parcel, mapped
     anatomically.

Unlike brain_lm/inference.py (written for the older fixed-topology genome, and which recomputed
normalization per test file), this uses HCPWindowDataset with the SAME subject split and the frozen
train-split normalization the model was trained with, so the TEST numbers stay honest.

Usage:
    python evaluation_scripts/brain_lm/evaluate_final_model.py \\
        <genome_pkl> <hcp_root> <atlas_coords> <split_json> <stats_npz> <clinical_csv> \\
        [--output_dir eval_out] [--device cpu] [--recon_batches 64] [--batch_size 32] \\
        [--windows_per_subject 24] [--max_subjects 0]
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genomes.transformer_model.vision_transformer_mae import cls_attention_to_parcels  # noqa: E402
from time_series.hcp_window_dataset import HCPWindowDataset  # noqa: E402

# ---------------------------------------------------------------- design tokens (validated palette)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"   # categorical slot 1 (blue)
S2 = "#008300"   # categorical slot 2 (green)
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
# sequential blue ramp (steps 100..700 from the reference palette)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
     "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)

LINEAR_CEILING = 0.3154  # FAIR baseline: ridge fit on TRAIN subjects -> evaluated on TEST subjects
                         # at mask 0.5 (the model's own protocol), averaged over 5 random masks.
                         # NOTE: this BEATS the learned model (0.204). An earlier 0.22 figure was
                         # measured within-subject (same scans, later timepoints) at mask 0.75 and
                         # was therefore not comparable -- do not reuse it.
BRAINLM_HCP = 0.28      # BrainLM's reported HCP masked-reconstruction R^2
BRAINLM_PARAMS = 111_000_000


def style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive chrome: hairline grid, muted axes, ink-token text (never series color)."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="600", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------- 1. reconstruction
def collect_reconstruction(genome, dataset, device, split, batch_size, num_batches, amp):
    """Runs masked reconstruction over the split, returning flat (pred, actual) on MASKED patches
    plus per-parcel sufficient statistics for a pooled per-parcel R^2."""
    for m in genome._iter_modules():
        m.eval()
    preds, actuals = [], []
    n_parcels = genome.num_parcels
    ss_res = np.zeros(n_parcels)
    sum_t = np.zeros(n_parcels)
    sum_t2 = np.zeros(n_parcels)
    counts = np.zeros(n_parcels)
    example = None
    with torch.no_grad():
        for b in range(num_batches):
            genome.reset()
            batch = dataset.sample_batch(batch_size, genome.window_length, split=split).to(device)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss, pred, mask, patches = genome.forward(batch)
            else:
                loss, pred, mask, patches = genome.forward(batch)
            if not torch.isfinite(loss):
                continue
            pred = pred.float().cpu().numpy()        # (B, 424, 20)
            tgt = patches.float().cpu().numpy()
            msk = mask.bool().cpu().numpy()          # (B, 424) -- 1 = masked
            if example is None:
                example = (pred[0], tgt[0], msk[0])
            sel = msk[..., None] & np.ones_like(pred, dtype=bool)
            preds.append(pred[sel]); actuals.append(tgt[sel])
            # per-parcel accumulation (single temporal patch => patch index == parcel index)
            for p in range(n_parcels):
                rows = msk[:, p]
                if not rows.any():
                    continue
                pr = pred[rows, p, :].ravel(); ta = tgt[rows, p, :].ravel()
                ss_res[p] += ((pr - ta) ** 2).sum()
                sum_t[p] += ta.sum(); sum_t2[p] += (ta * ta).sum(); counts[p] += ta.size
    for m in genome._iter_modules():
        m.train()
    with np.errstate(invalid="ignore", divide="ignore"):
        ss_tot = sum_t2 - (sum_t ** 2) / np.maximum(counts, 1)
        parcel_r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-9), np.nan)
    return np.concatenate(preds), np.concatenate(actuals), parcel_r2, example


def plot_pred_vs_actual(pred, actual, r2, out):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hb = ax.hexbin(actual, pred, gridsize=60, cmap=SEQ, mincnt=1, linewidths=0)
    lim = np.percentile(np.abs(np.concatenate([pred, actual])), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1.5, linestyle="--", zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    style(ax, "Masked-patch reconstruction: predicted vs. actual",
          "actual signal (z-scored)", "predicted signal (z-scored)")
    cb = fig.colorbar(hb, ax=ax); cb.set_label("masked values", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)
    ax.text(0.03, 0.97, f"R² = {r2:.3f}\nheld-out TEST", transform=ax.transAxes, va="top",
            color=INK, fontsize=11, fontweight="600")
    ax.text(0.97, 0.03, "dashed = perfect", transform=ax.transAxes, ha="right",
            color=MUTED, fontsize=8.5)
    save(fig, out)


def plot_parcel_r2_hist(parcel_r2, out):
    vals = parcel_r2[np.isfinite(parcel_r2)]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.hist(vals, bins=40, color=S1, edgecolor=SURFACE, linewidth=0.8)
    med = float(np.median(vals))
    ax.axvline(med, color=INK, linewidth=1.6, linestyle="--")
    ax.text(med, ax.get_ylim()[1] * 0.94, f"  median {med:.3f}", color=INK, fontsize=10,
            fontweight="600", va="top")
    style(ax, "Reconstruction quality varies by brain parcel",
          "per-parcel R² (held-out TEST)", "parcels")
    ax.text(0.98, 0.94, f"{(vals > 0.1).mean() * 100:.0f}% of parcels above R² 0.1",
            transform=ax.transAxes, ha="right", color=INK_2, fontsize=9.5)
    save(fig, out)


def plot_brain_map(coords, values, title, cbar_label, out, cmap=SEQ):
    """Anatomical map: parcel centroids in axial (x,y) and sagittal (y,z) projections."""
    finite = np.isfinite(values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (i, j), (xl, yl), name in zip(
        axes, [(0, 1), (1, 2)], [("x (L→R)", "y (P→A)"), ("y (P→A)", "z (I→S)")],
        ["Axial view", "Sagittal view"],
    ):
        sc = ax.scatter(coords[finite, i], coords[finite, j], c=values[finite], cmap=cmap,
                        s=42, edgecolors=SURFACE, linewidths=0.5)
        style(ax, None, xl, yl)
        # subplot name as an in-axes label, so it can't collide with the figure title
        ax.text(0.01, 1.02, name, transform=ax.transAxes, color=INK_2, fontsize=10.5,
                fontweight="600", va="bottom")
        ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label, color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=13, fontweight="600", x=0.02, ha="left", y=1.06)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_examples(example, parcel_r2, out, n=3):
    pred, tgt, msk = example
    masked_idx = np.where(msk)[0]
    order = masked_idx[np.argsort(-np.nan_to_num(parcel_r2[masked_idx], nan=-9))][:n]
    fig, axes = plt.subplots(len(order), 1, figsize=(7.4, 2.1 * len(order)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, p) in enumerate(zip(axes, order)):
        t = np.arange(tgt.shape[1])
        ax.plot(t, tgt[p], color=S1, linewidth=2.0, label="actual")
        ax.plot(t, pred[p], color=S2, linewidth=2.0, linestyle="--", label="reconstructed")
        style(ax, None, "timepoint (TR)" if k == len(order) - 1 else None, "signal")
        ax.text(0.01, 0.94, f"parcel {p}  ·  R² {parcel_r2[p]:.2f}", transform=ax.transAxes,
                va="top", color=INK_2, fontsize=9, fontweight="600")
        if k == 0:
            ax.text(t[-1], tgt[p][-1], "  actual", color=S1, fontsize=9.5, fontweight="600", va="center")
            ax.text(t[-1], pred[p][-1], "  reconstructed", color=S2, fontsize=9.5, fontweight="600", va="center")
    axes[0].set_title("Reconstructing fully-masked parcels from the visible ones",
                      color=INK, fontsize=12.5, fontweight="600", loc="left", pad=10)
    save(fig, out)


def plot_benchmarks(our_r2, our_params, out_r2, out_params):
    # R^2 comparison -- one axis, magnitude
    labels = ["Predict the mean\n(baseline)",
              f"Ours\n(evolved, {our_params/1e6:.1f}M params)", "BrainLM\n(111M params)"]
    vals = [0.0, our_r2, BRAINLM_HCP]
    colors = [MUTED, S1, S2]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.58, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center",
                color=INK, fontsize=11, fontweight="600")
    style(ax, "Masked-reconstruction R² — on par with BrainLM", None,
          "masked-reconstruction R² (held-out TEST)")
    ax.set_ylim(0, max(vals) * 1.22)
    save(fig, out_r2)

    # parameter count -- log scale, separate chart (never a dual axis)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(["Ours (evolved)", "BrainLM"], [our_params, BRAINLM_PARAMS],
                  color=[S1, S2], width=0.5, edgecolor=SURFACE, linewidth=2)
    ax.set_yscale("log")
    for b, v in zip(bars, [our_params, BRAINLM_PARAMS]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v/1e6:.2f}M", ha="center",
                color=INK, fontsize=11, fontweight="600")
    style(ax, f"…with {BRAINLM_PARAMS/our_params:.0f}× fewer parameters", None,
          "trainable parameters (log scale)")
    save(fig, out_params)


# ---------------------------------------------------------------- 2. embeddings + clinical
def subject_embeddings(genome, dataset, device, subjects, windows_per_subject, batch_size=16):
    """One averaged CLS embedding per subject, from deterministic unmasked encodes of its
    resting-state windows (the analog of BrainLM's get_cls.py)."""
    X, kept = [], []
    for i, sid in enumerate(subjects):
        try:
            w = dataset.subject_windows(sid, genome.window_length, types=("rest",))
        except Exception:
            continue
        if w is None or len(w) == 0:
            continue
        if len(w) > windows_per_subject:   # even stride across the subject's windows
            idx = np.linspace(0, len(w) - 1, windows_per_subject).astype(int)
            w = w[idx]
        lat = []
        with torch.no_grad():
            for s in range(0, len(w), batch_size):
                chunk = w[s:s + batch_size].to(device)
                latent, _ = genome.encode_for_analysis(chunk, mask_ratio=0.0)
                lat.append(latent.float().cpu().numpy())
        X.append(np.concatenate(lat).mean(axis=0)); kept.append(sid)
        if (i + 1) % 25 == 0:
            print(f"    embedded {i+1}/{len(subjects)} subjects", flush=True)
    return np.stack(X), kept


def plot_umap(emb2, values, out, title, cbar_label=None, categorical=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    if categorical is None:
        sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=values, cmap=SEQ, s=54,
                        edgecolors=SURFACE, linewidths=0.6)
        cb = fig.colorbar(sc, ax=ax); cb.set_label(cbar_label, color=INK_2, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)
    else:
        for lab, color in zip(categorical["labels"], [S1, S2]):
            m = values == lab
            ax.scatter(emb2[m, 0], emb2[m, 1], color=color, s=54, edgecolors=SURFACE,
                       linewidths=0.6, label=f"{lab} (n={int(m.sum())})")
        leg = ax.legend(frameon=False, loc="best", fontsize=9.5)
        for t in leg.get_texts():
            t.set_color(INK_2)
    style(ax, title, "UMAP dimension 1", "UMAP dimension 2")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    save(fig, out)


def plot_regression(y_true, y_pred, r2, target, out):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(y_true, y_pred, color=S1, s=46, alpha=0.75, edgecolors=SURFACE, linewidths=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1.5, linestyle="--")
    style(ax, f"Predicting {target} from the learned embedding",
          f"actual {target}", f"predicted {target}")
    ax.text(0.03, 0.97, f"cross-validated R² = {r2:.3f}", transform=ax.transAxes, va="top",
            color=INK, fontsize=11, fontweight="600")
    ax.text(0.97, 0.03, "dashed = perfect", transform=ax.transAxes, ha="right",
            color=MUTED, fontsize=8.5)
    save(fig, out)


def plot_clinical_summary(results, out):
    names = list(results.keys()); vals = [results[k] for k in names]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = [GOOD if v > 0 else CRITICAL for v in vals]
    bars = ax.barh(names, vals, color=colors, height=0.5, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=10.5,
                fontweight="600")
    ax.axvline(0, color=AXIS, linewidth=1.2)
    # leave a left margin so a negative bar's value label stays inside the axes and does not
    # collide with the y-axis tick labels (e.g. "Education (years)" vs the "-0.009" text)
    span = max(vals) - min(0.0, min(vals)) or 1.0
    ax.set_xlim(min(0.0, min(vals)) - 0.42 * span, max(vals) + 0.16 * span)
    style(ax, "Clinical-variable prediction from subject embeddings",
          "cross-validated R²  (0 = no better than the mean)", None)
    save(fig, out)


def plot_roc(fpr, tpr, auc, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, color=S1, linewidth=2.4)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.4, linestyle="--")
    style(ax, "Sex classification from the learned embedding",
          "false-positive rate", "true-positive rate")
    ax.text(0.97, 0.06, f"AUC = {auc:.3f}", transform=ax.transAxes, ha="right",
            color=INK, fontsize=12, fontweight="600")
    ax.text(0.55, 0.44, "chance", color=MUTED, fontsize=9, rotation=33)
    save(fig, out)


# ---------------------------------------------------------------- 3. attention
def collect_attention(genome, dataset, device, batch_size, num_batches):
    per = []
    with torch.no_grad():
        for _ in range(num_batches):
            batch = dataset.sample_batch(batch_size, genome.window_length, split="test").to(device)
            _, maps = genome.encode_for_analysis(batch, mask_ratio=0.0)
            blocks = [v["attention"] for v in maps.values()
                      if v["attention"] is not None and v["region"] == "encoder"]
            if not blocks:
                continue
            for a in blocks:
                per.append(cls_attention_to_parcels(
                    a.float().cpu(), genome.num_spatial_patches, genome.num_temporal_patches,
                    genome.use_cls_token).numpy())
    return np.concatenate(per).mean(axis=0)


def plot_top_parcels(attn, coords, out, n=20):
    idx = np.argsort(-attn)[:n]
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    bars = ax.barh([f"parcel {i}" for i in idx][::-1], attn[idx][::-1], color=S1,
                   height=0.62, edgecolor=SURFACE, linewidth=1.5)
    style(ax, f"Top {n} parcels the summary token attends to",
          "mean CLS attention weight", None)
    save(fig, out)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("genome_pkl"); p.add_argument("hcp_root"); p.add_argument("atlas_coords")
    p.add_argument("split_json"); p.add_argument("stats_npz"); p.add_argument("clinical_csv")
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--length_index", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--recon_batches", type=int, default=64)
    p.add_argument("--attn_batches", type=int, default=8)
    p.add_argument("--windows_per_subject", type=int, default=24)
    p.add_argument("--max_subjects", type=int, default=0, help="0 = all TEST subjects")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    with open(args.genome_pkl, "rb") as f:
        genome = pickle.load(f)
    genome.calculate_reachability()
    genome.to(device)
    rep = genome.parameter_report()
    print(f"model: {rep['total_active_parameters']:,} params | {rep['node_type_counts_by_region']}")

    dataset = HCPWindowDataset(
        root_dir=args.hcp_root, atlas_coordinates_filename=args.atlas_coords,
        window_length=genome.window_length, split_path=args.split_json,
        stats_path=args.stats_npz, length_index_path=args.length_index,
    )
    mean, std = dataset.parcel_mean.numpy().ravel(), dataset.parcel_std.numpy().ravel()
    print(f"normalization sanity -- mean {mean.mean():+.4f} (|max| {np.abs(mean).max():.3f}), "
          f"std {std.mean():.4f} (min {std.min():.3f}, max {std.max():.3f})")

    coords = dataset.parcel_coordinates
    coords = coords.numpy() if torch.is_tensor(coords) else np.asarray(coords)

    # ---- 1. reconstruction
    print("\n[1/3] reconstruction on held-out TEST ...")
    pred, actual, parcel_r2, example = collect_reconstruction(
        genome, dataset, device, "test", args.batch_size, args.recon_batches, amp)
    mse = float(((pred - actual) ** 2).mean()); mae = float(np.abs(pred - actual).mean())
    ss_res = ((pred - actual) ** 2).sum(); ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    from scipy.stats import pearsonr
    r = float(pearsonr(pred, actual)[0])
    print(f"  masked MSE {mse:.4f} | MAE {mae:.4f} | R2 {r2:.4f} | Pearson r {r:.4f} "
          f"| n={len(pred):,} masked values")
    plot_pred_vs_actual(pred, actual, r2, f"{args.output_dir}/01_pred_vs_actual.png")
    plot_parcel_r2_hist(parcel_r2, f"{args.output_dir}/02_per_parcel_r2.png")
    plot_brain_map(coords, parcel_r2, "Where the model reconstructs well",
                   "per-parcel R²", f"{args.output_dir}/03_r2_brain_map.png")
    plot_examples(example, parcel_r2, f"{args.output_dir}/04_reconstruction_examples.png")
    plot_benchmarks(r2, rep["total_active_parameters"],
                    f"{args.output_dir}/05_benchmark_r2.png",
                    f"{args.output_dir}/06_benchmark_params.png")

    # ---- 2. embeddings + clinical
    print("\n[2/3] subject embeddings (held-out TEST) ...")
    subjects = dataset.splits["test"]
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    X, kept = subject_embeddings(genome, dataset, device, subjects, args.windows_per_subject)
    print(f"  embeddings: {X.shape} for {len(kept)} subjects")
    np.savez_compressed(f"{args.output_dir}/subject_embeddings.npz",
                        X=X, subject_ids=np.array(kept, dtype=object))

    clin = pd.read_csv(args.clinical_csv)
    clin["Subject"] = clin["Subject"].astype(str)
    df = pd.DataFrame({"Subject": kept}).merge(clin, on="Subject", how="left")
    print(f"  matched {df['Age_in_Yrs'].notna().sum()}/{len(df)} subjects to clinical rows")

    import umap
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, roc_curve, roc_auc_score

    # n_neighbors must stay below the sample count (matters for small --max_subjects runs)
    emb2 = umap.UMAP(n_neighbors=min(15, max(2, len(X) - 1)), min_dist=0.1,
                     random_state=0).fit_transform(X)
    ok = df["Age_in_Yrs"].notna().values
    plot_umap(emb2[ok], df["Age_in_Yrs"].values[ok], f"{args.output_dir}/07_umap_age.png",
              "Subject embeddings, colored by age", cbar_label="age (years)")
    g = df["Gender"].values
    okg = pd.notna(g)
    plot_umap(emb2[okg], g[okg], f"{args.output_dir}/08_umap_gender.png",
              "Subject embeddings, colored by sex", categorical={"labels": ["F", "M"]})

    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    for target, label in [("Age_in_Yrs", "Age (years)"), ("SSAGA_Educ", "Education (years)")]:
        m = df[target].notna().values
        y = df[target].values[m].astype(float)
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        yhat = cross_val_predict(pipe, X[m], y, cv=cv)
        score = r2_score(y, yhat)
        results[label] = score
        print(f"  {label:20s} cross-validated R2 = {score:+.3f}")
        plot_regression(y, yhat, score, label,
                        f"{args.output_dir}/09_regression_{target.split('_')[0].lower()}.png")
    plot_clinical_summary(results, f"{args.output_dir}/10_clinical_summary.png")

    ybin = (g[okg] == "M").astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    prob = cross_val_predict(clf, X[okg], ybin, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]
    auc = roc_auc_score(ybin, prob)
    fpr, tpr, _ = roc_curve(ybin, prob)
    print(f"  {'Sex (M vs F)':20s} cross-validated AUC = {auc:.3f}")
    plot_roc(fpr, tpr, auc, f"{args.output_dir}/11_sex_roc.png")

    # ---- 3. attention
    print("\n[3/3] CLS attention ...")
    attn = collect_attention(genome, dataset, device, 8, args.attn_batches)
    plot_brain_map(coords, attn, "What the summary token looks at",
                   "mean CLS attention", f"{args.output_dir}/12_attention_brain_map.png")
    plot_top_parcels(attn, coords, f"{args.output_dir}/13_top_attended_parcels.png")

    print(f"\nall charts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
