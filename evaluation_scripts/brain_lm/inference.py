"""Evaluates a trained VisionTransformerMAEGenome (see genomes/vit_mae_genome.py) on held-out
fMRI data, mirroring BrainLM's reconstruction evaluation (Figure 3 / Section 4.1 of the BrainLM
paper): reconstructs a masked window and reports how well the model recovers the masked patches
versus simply copying the visible ones.

Usage:
    python evaluation_scripts/brain_lm/inference.py <genome_pkl> <test_npz> [output_prefix]

Note: this script builds a fresh FMRIPatchDataset from the test .npz file to get its per-parcel
normalization statistics, rather than reusing the exact statistics the genome was trained with.
For the current single/few-recording prototyping scope (see exa_vit_mae_config.ini) this is a
reasonable approximation; once training moves to many subjects, normalization statistics should
be persisted alongside the genome and reused here instead of being recomputed per test file.
"""

import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")  # allow running this script directly from evaluation_scripts/brain_lm/

from genomes.transformer_model.vision_transformer_mae import unpatchify  # noqa: E402
from time_series.fmri_patch_dataset import FMRIPatchDataset  # noqa: E402


def reconstruct_window(genome, window: torch.Tensor):
    """Runs one masked-autoencoder forward pass and fills in the visible patches with their
    ground-truth values (only the masked patches are the model's actual prediction), matching
    how MAE reconstructions are conventionally visualized.

    Args:
        genome: a VisionTransformerMAEGenome.
        window: a single normalized window, shape (1, num_parcels, window_length).

    Returns:
        reconstructed: shape (1, num_parcels, window_length), masked regions from the model's
            prediction and visible regions copied from the ground truth.
        mask: shape (1, num_patches), 1 for masked patches.
        loss: scalar masked-patch reconstruction MSE (normalized-signal units).
    """
    genome.model.eval()
    with torch.no_grad():
        loss, pred, mask, patches = genome.model(window)

        mask_expanded = mask.unsqueeze(-1).bool().expand_as(pred)
        reconstructed_patches = torch.where(mask_expanded, pred, patches)

        reconstructed = unpatchify(
            reconstructed_patches,
            num_parcels=genome.model.num_parcels,
            window_length=genome.model.window_length,
            parcel_patch_size=genome.model.parcel_patch_size,
            time_patch_size=genome.model.time_patch_size,
        )

    return reconstructed, mask, loss.item()


def masked_patch_r2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """R^2 computed only over masked patches, matching BrainLM's reported reconstruction R^2."""
    mask_expanded = mask.unsqueeze(-1).bool().expand_as(pred)
    predicted_values = pred[mask_expanded].numpy()
    actual_values = target[mask_expanded].numpy()

    residual_ss = np.sum((actual_values - predicted_values) ** 2)
    total_ss = np.sum((actual_values - actual_values.mean()) ** 2)
    return 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")


def plot_reconstruction(
    dataset, original: torch.Tensor, reconstructed: torch.Tensor, output_filename: str, num_parcels_to_plot: int = 4
):
    """Plots ground-truth vs reconstructed time series for a handful of parcels, in the
    original (denormalized) data scale, mirroring BrainLM Figure 3."""
    original_raw = dataset.denormalize(original[0]).numpy()
    reconstructed_raw = dataset.denormalize(reconstructed[0]).numpy()

    parcel_indices = np.linspace(0, original_raw.shape[0] - 1, num_parcels_to_plot, dtype=int)

    figure, axes = plt.subplots(len(parcel_indices), 1, figsize=(8, 2 * len(parcel_indices)), sharex=True)
    if len(parcel_indices) == 1:
        axes = [axes]

    for axis, parcel_idx in zip(axes, parcel_indices):
        parcel_name = dataset.roi_names[parcel_idx] if dataset.roi_names else f"parcel {parcel_idx}"
        axis.plot(original_raw[parcel_idx], "k.", label="data", markersize=3)
        axis.plot(reconstructed_raw[parcel_idx], "r-", label="reconstruction", linewidth=1)
        axis.set_ylabel(parcel_name)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time")
    figure.suptitle("VisionTransformerMAE reconstruction (masked patches filled in)")
    figure.tight_layout()
    figure.savefig(output_filename)
    print(f"saved reconstruction plot to {output_filename}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluation_scripts/brain_lm/inference.py <genome_pkl> <test_npz> [output_prefix]")
        return

    genome_filename = sys.argv[1]
    test_npz_filename = sys.argv[2]
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else "vit_mae_reconstruction"

    with open(genome_filename, "rb") as genome_file:
        genome = pickle.load(genome_file)

    dataset = FMRIPatchDataset(npz_filenames=[test_npz_filename])

    if dataset.num_parcels != genome.num_parcels:
        raise ValueError(
            f"test data has {dataset.num_parcels} parcels but genome was trained with "
            f"{genome.num_parcels} parcels"
        )

    # deterministic window (start of the recording) so repeated eval runs are comparable
    window_length = genome.window_length
    window = dataset.normalize(dataset.recordings[0][:, :window_length]).unsqueeze(0)

    reconstructed, mask, masked_mse = reconstruct_window(genome, window)

    with torch.no_grad():
        _, pred, mask, patches = genome.model(window)
    r2 = masked_patch_r2(pred, patches, mask)

    print(f"genome: {genome}")
    print(f"masked patches: {int(mask.sum().item())} / {mask.numel()} ({genome.model.mask_ratio:.0%} target ratio)")
    print(f"masked-patch reconstruction MSE (normalized units): {masked_mse:.6f}")
    print(f"masked-patch reconstruction R^2: {r2:.4f}")

    plot_reconstruction(dataset, window, reconstructed, output_filename=f"{output_prefix}.png")


if __name__ == "__main__":
    main()
