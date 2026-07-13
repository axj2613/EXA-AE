"""Extracts one CLS-embedding row per subject from a trained evolved genome, the exact analog of
get_cls.py but using our own VisionTransformerBlockGenome instead of pretrained BrainLM. Saves an
.npz (X: (num_subjects, d_model), subject_ids) in the format eval_cls.py consumes for clinical /
behavioral variable regression.

Per subject: tile each of the subject's recordings (of the chosen type, default resting-state)
into windows, run them through encode_for_analysis with mask_ratio=0 (deterministic, whole-window
encoding -- the same reason get_cls.py disables masking), take the CLS latent per window, and
average across all of the subject's windows into a single embedding. Uses the frozen train-split
normalization and the SAME subject split file used during training, so embeddings are extracted
only for held-out TEST subjects by default (never seen during pretraining).

Usage:
    python evaluation_scripts/brain_lm/subject_embeddings.py \\
        <genome_pkl> <hcp_root> <atlas_coords> <split_json> <stats_npz> [output.npz] \\
        [--split test] [--types rest] [--device cuda]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

# make the repo importable regardless of CWD (Kaggle runs from /kaggle/working, repo is in
# /kaggle/working/exa-star); this script is at <repo>/evaluation_scripts/brain_lm/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from time_series.hcp_window_dataset import HCPWindowDataset  # noqa: E402


@torch.no_grad()
def subject_embedding(genome, windows: torch.Tensor, device, batch_size: int = 8) -> np.ndarray:
    """Mean CLS latent over all of a subject's windows. windows: (num_windows, num_parcels,
    window_length), already normalized."""
    latents = []
    for start in range(0, windows.shape[0], batch_size):
        chunk = windows[start:start + batch_size].to(device)
        latent, _ = genome.encode_for_analysis(chunk, mask_ratio=0.0)
        latents.append(latent.float().cpu().numpy())
    return np.concatenate(latents, axis=0).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("genome_pkl")
    parser.add_argument("hcp_root")
    parser.add_argument("atlas_coords")
    parser.add_argument("split_json")
    parser.add_argument("stats_npz")
    parser.add_argument("output", nargs="?", default="subject_embeddings.npz")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--types", default="rest", help="comma-separated recording types (rest,task)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())

    with open(args.genome_pkl, "rb") as genome_file:
        genome = pickle.load(genome_file)
    genome.to(device)

    dataset = HCPWindowDataset(
        root_dir=args.hcp_root,
        atlas_coordinates_filename=args.atlas_coords,
        window_length=genome.window_length,
        split_path=args.split_json,   # reuse the SAME split used during training
        stats_path=args.stats_npz,    # reuse the frozen train-split normalization
    )

    subject_ids, embeddings = [], []
    subjects = dataset.splits[args.split]
    print(f"extracting embeddings for {len(subjects)} '{args.split}' subjects (types={types})")

    for subject_id in subjects:
        windows = dataset.subject_windows(subject_id, window_length=genome.window_length, types=types)
        if windows.shape[0] == 0:
            continue
        embeddings.append(subject_embedding(genome, windows, device))
        subject_ids.append(subject_id)

    if not embeddings:
        raise RuntimeError("no subject embeddings were extracted (no matching recordings?)")

    X = np.asarray(embeddings, dtype=np.float32)
    print(f"X shape: {X.shape}")
    np.savez_compressed(args.output, X=X, subject_ids=np.array(subject_ids, dtype=object))
    print(f"saved subject embeddings to {args.output}")


if __name__ == "__main__":
    main()
