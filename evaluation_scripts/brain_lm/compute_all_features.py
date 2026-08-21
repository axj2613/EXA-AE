"""Compute BOTH subject readouts -- the CLS embedding and the per-parcel reconstruction-error
fingerprint -- for EVERY subject in the split (train + val + test), in a single pass over each
subject's windows, and cache them for a higher-N downstream probe (e.g. personality_readout.py).

Why pool train+val+test: a frozen readout on the 219 TEST subjects alone is underpowered for weak
targets (personality). Pooling to ~1,050 subjects raises power ~5x. IMPORTANT caveats, surfaced so
the numbers are read correctly:
  * The model was PRETRAINED (self-supervised reconstruction) on the train subjects, so this is a
    representation PROBE, not a held-out-generalization estimate. Personality labels were never used
    in pretraining, so there is no label leakage -- but train subjects reconstruct slightly better,
    making their fingerprints in-sample-optimistic (a nuisance orthogonal to the labels).
  * Use k-fold CV downstream (no subject in both folds); report this separately from the clean
    TEST-only clinical numbers.

Outputs (into --output_dir, default eval_out), both in the SAME subject order:
  all_subject_embeddings.npz   X  (n, d_model)   mean unmasked CLS embedding per subject
  all_readout_features.npz     Xe (n, 424)       per-parcel masked-reconstruction MSE per subject

Usage:
    python evaluation_scripts/brain_lm/compute_all_features.py \
        <genome_pkl> <hcp_root> <atlas_coords> <split_json> <stats_npz> \
        [--output_dir eval_out] [--device cpu] [--windows_per_subject 24] [--batch_size 16]
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from time_series.hcp_window_dataset import HCPWindowDataset  # noqa: E402


def compute_both(genome, dataset, device, subjects, windows_per_subject, batch_size):
    """Per subject, load its rest windows once and compute (a) the mean unmasked CLS embedding and
    (b) the per-parcel masked-reconstruction MSE fingerprint. Returns (X_cls, X_fp, kept_ids)."""
    for m in genome._iter_modules():
        m.eval()
    n_parcels = genome.num_parcels
    Xc, Xe, kept = [], [], []
    with torch.no_grad():
        for i, sid in enumerate(subjects):
            try:
                w = dataset.subject_windows(sid, genome.window_length, types=("rest",))
            except Exception:
                continue
            if w is None or len(w) == 0:
                continue
            if len(w) > windows_per_subject:                      # even stride across the scan
                idx = np.linspace(0, len(w) - 1, windows_per_subject).astype(int)
                w = w[idx]

            # (a) CLS embedding -- deterministic unmasked encode, averaged over the subject's windows
            lat = []
            for s in range(0, len(w), batch_size):
                chunk = w[s:s + batch_size].to(device)
                latent, _ = genome.encode_for_analysis(chunk, mask_ratio=0.0)
                lat.append(latent.float().cpu().numpy())
            cls_vec = np.concatenate(lat).mean(axis=0)

            # (b) fingerprint -- masked reconstruction, per-parcel MSE over masked patches
            se = np.zeros(n_parcels); cnt = np.zeros(n_parcels)
            for s in range(0, len(w), batch_size):
                genome.reset()
                chunk = w[s:s + batch_size].to(device)
                loss, pred, mask, patches = genome.forward(chunk)
                if not torch.isfinite(loss):
                    continue
                pred = pred.float().cpu().numpy(); tgt = patches.float().cpu().numpy()
                msk = mask.bool().cpu().numpy()
                for p in range(n_parcels):
                    rows = msk[:, p]
                    if not rows.any():
                        continue
                    d = (pred[rows, p, :] - tgt[rows, p, :]).ravel()
                    se[p] += (d * d).sum(); cnt[p] += d.size
            fp_vec = np.where(cnt > 0, se / np.maximum(cnt, 1), np.nan)

            Xc.append(cls_vec); Xe.append(fp_vec); kept.append(str(sid))
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(subjects)} subjects", flush=True)
    for m in genome._iter_modules():
        m.train()
    Xc = np.vstack(Xc); Xe = np.vstack(Xe)
    # impute any never-masked parcel column (rare) with its column mean so downstream CV has no NaN
    col_mean = np.nanmean(Xe, axis=0)
    nan = np.where(np.isnan(Xe)); Xe[nan] = np.take(col_mean, nan[1])
    return Xc, Xe, kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("genome_pkl"); p.add_argument("hcp_root"); p.add_argument("atlas_coords")
    p.add_argument("split_json"); p.add_argument("stats_npz")
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--length_index", default=None)
    p.add_argument("--windows_per_subject", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=16)
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

    split = json.load(open(args.split_json))
    subjects = [str(s) for s in split.get("train", []) + split.get("val", []) + split.get("test", [])]
    print(f"computing CLS + fingerprint for {len(subjects)} subjects "
          f"(train {len(split.get('train', []))} + val {len(split.get('val', []))} "
          f"+ test {len(split.get('test', []))}) ...")

    Xc, Xe, kept = compute_both(genome, dataset, device, subjects,
                                args.windows_per_subject, args.batch_size)
    print(f"done: CLS {Xc.shape}, fingerprint {Xe.shape} for {len(kept)} subjects")

    ids = np.array(kept, dtype=object)
    np.savez_compressed(f"{args.output_dir}/all_subject_embeddings.npz", X=Xc, subject_ids=ids)
    np.savez_compressed(f"{args.output_dir}/all_readout_features.npz", Xe=Xe, subject_ids=ids)
    print(f"saved all_subject_embeddings.npz + all_readout_features.npz to {args.output_dir}/")


if __name__ == "__main__":
    main()
