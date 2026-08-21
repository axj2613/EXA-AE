"""Big-Five personality as a HIGH/LOW CLASSIFICATION from the reconstruction-error fingerprint.

Each trait is binarized at the TRAIN-split median (a leak-free threshold: derived without any test
labels), the same threshold binarizes the held-out TEST subjects. A classifier is fit on the TRAIN
subjects' 424-d fingerprints and evaluated on the TEST subjects -- a genuine train->held-out-test
protocol (not cross-validation).

Caveat surfaced in the output: the cached fingerprints are IN-sample for train subjects (the model
was pretrained to reconstruct them) but OUT-of-sample for test subjects, so train fingerprints have
systematically lower error. We print that shift; feature standardization (fit on train) absorbs a
uniform part of it, but it remains a caveat for the train->test transfer.

Emits into --output_dir (default eval_out):
  21_personality_clf_summary.png   per-trait held-out AUC + balanced accuracy (chance = 0.5)
  22_personality_clf_roc.png       ROC curves, all five traits
  23_personality_clf_confusion.png confusion matrices, all five traits

Usage:
    python evaluation_scripts/brain_lm/personality_classification.py \
        <personality_csv> <split_json> [--features eval_out/all_readout_features.npz] \
        [--output_dir eval_out] [--n_pca 50]
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
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


def classify(Xtr, ytr, Xte, n_pca):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    k = max(2, min(n_pca, Xtr.shape[0] // 3, Xtr.shape[1]))
    clf = make_pipeline(StandardScaler(), PCA(k, random_state=0),
                        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def plot_summary(rows, out):
    import matplotlib.pyplot as plt
    labels = [r["label"] for r in rows]
    auc = [r["auc"] for r in rows]
    bacc = [r["bacc"] for r in rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    b1 = ax.bar(x - w / 2, auc, w, color=E.S1, edgecolor=E.SURFACE, linewidth=1.5, label="ROC AUC")
    b2 = ax.bar(x + w / 2, bacc, w, color=E.S2, edgecolor=E.SURFACE, linewidth=1.5,
                label="balanced accuracy")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.2f}",
                    ha="center", color=E.INK, fontsize=9, fontweight="600")
    ax.axhline(0.5, color=E.MUTED, linewidth=1.4, linestyle="--")
    ax.text(len(labels) - 0.5, 0.5, " chance", color=E.MUTED, fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(0.7, max(auc + bacc) + 0.06))
    E.style(ax, "High vs low personality from the fingerprint — held-out TEST", None, "score")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    E.save(fig, out)


def plot_roc(rows, out):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    colors = [E.SEQ(v) for v in np.linspace(0.35, 0.95, len(rows))]
    for r, c in zip(rows, colors):
        ax.plot(r["fpr"], r["tpr"], color=c, linewidth=2.2,
                label=f"{r['label']}  (AUC {r['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=E.MUTED, linewidth=1.4, linestyle="--")
    E.style(ax, "Personality high/low ROC — held-out TEST", "false-positive rate", "true-positive rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    E.save(fig, out)


def plot_confusion(rows, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.9))
    for ax, r in zip(axes, rows):
        cm = r["cm"].astype(float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=E.SEQ, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]*100:.0f}%", ha="center", va="center",
                        color=E.SURFACE if cmn[i, j] > 0.5 else E.INK, fontsize=10, fontweight="600")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["low", "high"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["low", "high"])
        ax.set_title(f"{r['label']}\nacc {r['acc']:.2f} (base {r['base']:.2f})", color=E.INK,
                     fontsize=11, fontweight="600", pad=8)
        ax.set_xlabel("predicted", color=E.INK_2, fontsize=9)
        if r is rows[0]:
            ax.set_ylabel("actual", color=E.INK_2, fontsize=9)
    fig.suptitle("Confusion matrices (row-normalized) — held-out TEST", color=E.INK, fontsize=13,
                 fontweight="700", x=0.01, ha="left")
    E.save(fig, out)


def main():
    from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                                 balanced_accuracy_score, f1_score, confusion_matrix)
    p = argparse.ArgumentParser()
    p.add_argument("personality_csv")
    p.add_argument("split_json")
    p.add_argument("--features", default="eval_out/all_readout_features.npz")
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--n_pca", type=int, default=50)
    args = p.parse_args()

    fp = np.load(args.features, allow_pickle=True)
    Xe = np.asarray(fp["Xe"])
    ids = [str(s) for s in fp["subject_ids"]]
    row = {s: i for i, s in enumerate(ids)}
    split = json.load(open(args.split_json))
    tr_ids = [s for s in map(str, split["train"]) if s in row]
    te_ids = [s for s in map(str, split["test"]) if s in row]
    va_ids = [s for s in map(str, split.get("val", [])) if s in row]

    # reconstruction shift: in-sample (train) vs out-of-sample (val/test) mean fingerprint MSE
    mean_fp = Xe.mean(axis=1)
    print("mean fingerprint MSE by split (in-sample train vs out-of-sample val/test):")
    print(f"  train {mean_fp[[row[s] for s in tr_ids]].mean():.4f} | "
          f"val {mean_fp[[row[s] for s in va_ids]].mean():.4f} | "
          f"test {mean_fp[[row[s] for s in te_ids]].mean():.4f}\n")

    per = pd.read_csv(args.personality_csv); per["Subject"] = per["Subject"].astype(str)
    lab = {t: dict(zip(per["Subject"], per[t])) for t, _ in TRAITS}

    print(f"train->TEST classification (threshold = TRAIN median); PCA-{args.n_pca} + logreg")
    print(f"{'Trait':<17}{'thr':>5}{'test n':>7}{'test hi/lo':>12}{'AUC':>6}{'acc':>6}"
          f"{'bal_acc':>8}{'F1':>6}{'base_acc':>9}")
    print("-" * 82)
    rows = []
    for col, label in TRAITS:
        def build(idlist):
            xs, ys, keep = [], [], []
            for s in idlist:
                v = lab[col].get(s)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                xs.append(Xe[row[s]]); ys.append(float(v)); keep.append(s)
            return np.array(xs), np.array(ys)
        Xtr, ytr_raw = build(tr_ids)
        Xte, yte_raw = build(te_ids)
        thr = float(np.median(ytr_raw))               # leak-free threshold from TRAIN only
        ytr = (ytr_raw >= thr).astype(int)
        yte = (yte_raw >= thr).astype(int)
        prob = classify(Xtr, ytr, Xte, args.n_pca)
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(yte, prob)
        acc = accuracy_score(yte, pred)
        bacc = balanced_accuracy_score(yte, pred)
        f1 = f1_score(yte, pred)
        base = max(yte.mean(), 1 - yte.mean())        # majority-class baseline accuracy
        fpr, tpr, _ = roc_curve(yte, prob)
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        rows.append(dict(label=label, auc=auc, acc=acc, bacc=bacc, base=base,
                         fpr=fpr, tpr=tpr, cm=cm))
        star = " *" if auc > 0.58 else ""
        print(f"{label:<17}{thr:>5.0f}{len(yte):>7}{f'{int(yte.sum())}/{int((1-yte).sum())}':>12}"
              f"{auc:>6.2f}{acc:>6.2f}{bacc:>8.2f}{f1:>6.2f}{base:>9.2f}{star}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_summary(rows, f"{args.output_dir}/21_personality_clf_summary.png")
    plot_roc(rows, f"{args.output_dir}/22_personality_clf_roc.png")
    plot_confusion(rows, f"{args.output_dir}/23_personality_clf_confusion.png")
    print(f"\ncharts written to {args.output_dir}/ (21-23)")


if __name__ == "__main__":
    main()
