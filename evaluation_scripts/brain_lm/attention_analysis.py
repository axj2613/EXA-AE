"""Attention/latent interpretability analysis for a trained ViT-MAE genome (either the fixed
VisionTransformerMAEGenome or the evolved VisionTransformerBlockGenome), following BrainLM's
attention-map and latent analyses (paper Sections 4.2/4.4/4.5).

For each sampled fMRI window it runs the genome's deterministic, unmasked encode_for_analysis
pass, then:
  - aggregates the CLS-token attention into a per-parcel attention vector (how much the summary
    token attends to each brain parcel), averaged over attention blocks and windows, and saves it
    (CSV with parcel name + xyz + attention, plus a plot);
  - collects the per-window CLS latent embeddings and saves them (.npy) for downstream UMAP /
    clinical-variable regression (BrainLM Figure 5 / Table 1).

A k-NN functional-network classifier (BrainLM Table 3) is provided but requires a per-parcel
network-label array (e.g. a Yeo-7-network mapping of the atlas), which is not shipped with this
repo; pass one via --network_labels to run it.

Usage:
    python evaluation_scripts/brain_lm/attention_analysis.py \\
        <genome_pkl> <test_npz> <atlas_coords> [output_prefix] [--network_labels labels.npy]
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# make the repo importable regardless of CWD (Kaggle runs from /kaggle/working, repo is in
# /kaggle/working/exa-star); this script is at <repo>/evaluation_scripts/brain_lm/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genomes.transformer_model.vision_transformer_mae import cls_attention_to_parcels  # noqa: E402
from time_series.fmri_patch_dataset import FMRIPatchDataset  # noqa: E402


def grid_dims(genome):
    """Reads (num_spatial_patches, num_temporal_patches, use_cls_token) from either genome type
    (the block genome holds them directly; the fixed genome holds them on its .model)."""
    obj = genome if hasattr(genome, "num_spatial_patches") else genome.model
    return obj.num_spatial_patches, obj.num_temporal_patches, obj.use_cls_token


def collect_attention_and_latents(genome, dataset, num_windows, window_length, batch_size=4):
    """Runs encode_for_analysis over sampled windows, returning per-window per-parcel attention
    (num_windows, num_spatial_patches) averaged over all attention blocks, and per-window latents
    (num_windows, d_model)."""
    num_spatial, num_temporal, has_cls = grid_dims(genome)

    per_window_parcel_attention = []
    latents = []

    collected = 0
    while collected < num_windows:
        this_batch = min(batch_size, num_windows - collected)
        batch = dataset.sample_batch(batch_size=this_batch, window_length=window_length)
        latent, attention_maps = genome.encode_for_analysis(batch, mask_ratio=0.0)
        latents.append(latent.cpu())

        if attention_maps:
            # average per-parcel attention across every attention block/layer
            parcel_maps = [
                cls_attention_to_parcels(info["attention"], num_spatial, num_temporal, has_cls)
                for info in attention_maps.values()
            ]
            per_window_parcel_attention.append(torch.stack(parcel_maps, dim=0).mean(dim=0).cpu())
        else:
            per_window_parcel_attention.append(torch.zeros(this_batch, num_spatial))

        collected += this_batch

    return torch.cat(per_window_parcel_attention, dim=0).numpy(), torch.cat(latents, dim=0).numpy()


def functional_network_knn(parcel_attention, network_labels, n_neighbors=5, train_frac=0.8):
    """k-NN classification of parcels into functional networks from their attention profiles
    across windows (BrainLM Table 3). parcel_attention is (num_windows, num_parcels); each parcel
    is a sample whose features are its attention values across windows. Requires scikit-learn.

    Args:
        parcel_attention: (num_windows, num_parcels) attention matrix.
        network_labels: (num_parcels,) integer network id per parcel.
    Returns:
        classification accuracy on the held-out parcels.
    """
    from sklearn.neighbors import KNeighborsClassifier

    features = parcel_attention.T  # (num_parcels, num_windows): each parcel is a sample
    num_parcels = features.shape[0]
    rng = np.random.default_rng(0)
    order = rng.permutation(num_parcels)
    split = int(num_parcels * train_frac)
    train_idx, test_idx = order[:split], order[split:]

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(features[train_idx], network_labels[train_idx])
    return classifier.score(features[test_idx], network_labels[test_idx])


def save_parcel_attention(dataset, mean_parcel_attention, output_prefix):
    roi_names = dataset.roi_names or [f"parcel_{i}" for i in range(len(mean_parcel_attention))]
    coords = dataset.parcel_coordinates

    csv_path = f"{output_prefix}_parcel_attention.csv"
    with open(csv_path, "w") as csv_file:
        csv_file.write("parcel,x,y,z,attention\n")
        for i, name in enumerate(roi_names):
            xyz = coords[i].tolist() if coords is not None else [float("nan")] * 3
            csv_file.write(f"{name},{xyz[0]},{xyz[1]},{xyz[2]},{mean_parcel_attention[i]}\n")
    print(f"saved per-parcel attention to {csv_path}")

    order = np.argsort(mean_parcel_attention)[::-1]
    top = min(30, len(order))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(range(top), mean_parcel_attention[order[:top]])
    axis.set_xticks(range(top))
    axis.set_xticklabels([roi_names[order[i]] for i in range(top)], rotation=90, fontsize=7)
    axis.set_ylabel("mean CLS attention")
    axis.set_title(f"Top {top} parcels by CLS attention")
    figure.tight_layout()
    figure.savefig(f"{output_prefix}_parcel_attention.png")
    print(f"saved attention plot to {output_prefix}_parcel_attention.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("genome_pkl")
    parser.add_argument("test_npz")
    parser.add_argument("atlas_coords")
    parser.add_argument("output_prefix", nargs="?", default="vit_mae_attention")
    parser.add_argument("--num_windows", type=int, default=64)
    parser.add_argument("--network_labels", default=None, help="optional .npy of per-parcel network ids")
    args = parser.parse_args()

    with open(args.genome_pkl, "rb") as genome_file:
        genome = pickle.load(genome_file)

    dataset = FMRIPatchDataset(npz_filenames=[args.test_npz], atlas_coordinates_filename=args.atlas_coords)
    window_length = genome.window_length

    parcel_attention, latents = collect_attention_and_latents(
        genome, dataset, num_windows=args.num_windows, window_length=window_length
    )
    mean_parcel_attention = parcel_attention.mean(axis=0)

    np.save(f"{args.output_prefix}_latents.npy", latents)
    print(f"saved {latents.shape} CLS latents to {args.output_prefix}_latents.npy")
    save_parcel_attention(dataset, mean_parcel_attention, args.output_prefix)

    if args.network_labels is not None:
        labels = np.load(args.network_labels)
        accuracy = functional_network_knn(parcel_attention, labels)
        print(f"functional-network k-NN accuracy: {accuracy:.3f}")
    else:
        print("no --network_labels provided; skipping functional-network k-NN classification")


if __name__ == "__main__":
    main()
