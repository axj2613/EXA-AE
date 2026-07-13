"""Proxy-vs-full fidelity experiment: does the cheap per-genome fitness the evolution uses (a few
gradient steps on a small random slice) rank architectures the SAME way full training would?

The evolutionary search evaluates each genome with a tiny training budget, so it is fast -- but
that only produces a good foundation model if a genome's cheap "proxy" validation MSE is a
faithful *ranking* of its fully-trained potential. This script measures that directly:

  1. Build N varied topologies (grow the seed with random AddNode mutations -> a spread of sizes
     and block-type mixes).
  2. Train two independent copies of each topology from identical starting weights -- one at the
     PROXY budget, one at the FULL budget -- and record the validation MSE (fitness) and wall-clock
     of each, plus the parameter count.
  3. Report the Spearman rank correlation between proxy and full fitness (high => the cheap signal
     is trustworthy, so evolution can be scaled with confidence) and the proxy speedup (the
     efficiency gain). Saves a CSV and a scatter plot.

Usage:
    python evaluation_scripts/proxy_fidelity_experiment.py \\
        <hcp_root> <atlas_coords> <split_json> <stats_npz> [output_prefix] \\
        [--num_topologies 12] [--full_iters 10 --full_bpi 50] [--device cuda]
"""

import argparse
import copy
import csv
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")

from evolution.vision_transformer_block_edge_generator import VisionTransformerBlockEdgeGenerator  # noqa: E402
from evolution.vision_transformer_block_node_generator import VisionTransformerBlockNodeGenerator  # noqa: E402
from genomes.vision_transformer_block_genome import VisionTransformerBlockGenome  # noqa: E402
from reproduction.add_node import AddNode  # noqa: E402
from time_series.hcp_window_dataset import HCPWindowDataset  # noqa: E402
from weight_generators.lamarckian_block_weight_generator import LamarckianBlockWeightGenerator  # noqa: E402


def build_topologies(seed, node_generator, edge_generator, weight_generator, num_topologies, max_mutations, rng):
    """Grows `num_topologies` genomes from the seed, each by applying a random number (1..
    max_mutations) of AddNode mutations, giving a spread of sizes and block-type compositions."""
    add_node = AddNode(node_generator, edge_generator, weight_generator, autoencoder=True)
    topologies = []
    for _ in range(num_topologies):
        genome = copy.deepcopy(seed)
        for _ in range(rng.randint(1, max_mutations)):
            child = add_node([genome])
            child.calculate_reachability()
            if child.viable:
                genome = child
        genome.calculate_reachability()
        topologies.append(genome)
    return topologies


def train_copy(topology, dataset, device, learning_rate, budget):
    """Trains a fresh copy of a topology (same starting weights) at the given budget; returns
    (validation fitness, active parameter count, wall-clock seconds)."""
    genome = copy.deepcopy(topology)
    genome.to(device)
    optimizer = torch.optim.Adam(genome.parameters(), lr=learning_rate)
    start = time.time()
    genome.train(dataset=dataset, optimizer=optimizer, **budget)
    return genome.fitness, genome.complexity["total_active_parameters"], time.time() - start


def spearman(a, b) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        # rank-correlation fallback (Pearson on ranks) if scipy is unavailable
        ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
        return float(np.corrcoef(ar, br)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hcp_root")
    parser.add_argument("atlas_coords")
    parser.add_argument("split_json")
    parser.add_argument("stats_npz")
    parser.add_argument("output_prefix", nargs="?", default="proxy_fidelity")
    parser.add_argument("--length_index", default=None)
    parser.add_argument("--num_topologies", type=int, default=12)
    parser.add_argument("--max_mutations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model config (match the evolution run)
    parser.add_argument("--window_length", type=int, default=120)
    parser.add_argument("--time_patch_size", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fitness_batches", type=int, default=8)
    # budgets
    parser.add_argument("--proxy_iters", type=int, default=2)
    parser.add_argument("--proxy_bpi", type=int, default=20)
    parser.add_argument("--full_iters", type=int, default=10)
    parser.add_argument("--full_bpi", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset = HCPWindowDataset(
        root_dir=args.hcp_root, atlas_coordinates_filename=args.atlas_coords,
        window_length=args.window_length, split_path=args.split_json,
        stats_path=args.stats_npz, length_index_path=args.length_index,
    )

    weight_generator = LamarckianBlockWeightGenerator()
    node_generator = VisionTransformerBlockNodeGenerator(num_heads=args.num_heads, d_ff=args.d_ff, dropout=args.dropout)
    edge_generator = VisionTransformerBlockEdgeGenerator()
    seed = VisionTransformerBlockGenome(
        generation_number=0, num_parcels=dataset.num_parcels, window_length=args.window_length,
        parcel_coordinates=dataset.parcel_coordinates, d_model=args.d_model, num_heads=args.num_heads,
        d_ff=args.d_ff, dropout=args.dropout, time_patch_size=args.time_patch_size,
        mask_ratio=args.mask_ratio, weight_generator=weight_generator,
    )

    topologies = build_topologies(
        seed, node_generator, edge_generator, weight_generator, args.num_topologies, args.max_mutations, rng
    )

    proxy_budget = dict(iterations=args.proxy_iters, batches_per_iteration=args.proxy_bpi,
                        batch_size=args.batch_size, fitness_batches=args.fitness_batches, use_amp=device.type == "cuda")
    full_budget = dict(iterations=args.full_iters, batches_per_iteration=args.full_bpi,
                       batch_size=args.batch_size, fitness_batches=args.fitness_batches, use_amp=device.type == "cuda")

    rows = []
    for i, topology in enumerate(topologies):
        proxy_fit, params, proxy_time = train_copy(topology, dataset, device, args.lr, proxy_budget)
        full_fit, _, full_time = train_copy(topology, dataset, device, args.lr, full_budget)
        rows.append((i, params, proxy_fit, full_fit, proxy_time, full_time))
        print(f"topology {i:2d}: params={params:>9,} proxy_MSE={proxy_fit:.5f} full_MSE={full_fit:.5f} "
              f"(proxy {proxy_time:.1f}s, full {full_time:.1f}s)")

    proxy_fits = [r[2] for r in rows]
    full_fits = [r[3] for r in rows]
    rho = spearman(proxy_fits, full_fits)
    speedup = sum(r[5] for r in rows) / max(1e-9, sum(r[4] for r in rows))

    print(f"\nSpearman rank correlation (proxy vs full fitness): {rho:.3f}")
    print(f"  >0.7 = the cheap fitness ranks architectures reliably; scale generations with confidence.")
    print(f"proxy speedup (full_time / proxy_time): {speedup:.1f}x")

    csv_path = f"{args.output_prefix}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["topology", "active_params", "proxy_fitness", "full_fitness", "proxy_time_s", "full_time_s"])
        writer.writerows(rows)
    print(f"saved {csv_path}")

    figure, axis = plt.subplots(figsize=(6, 6))
    scatter = axis.scatter(proxy_fits, full_fits, c=[r[1] for r in rows], cmap="viridis", edgecolors="k")
    axis.set_xlabel("proxy-budget validation MSE")
    axis.set_ylabel("full-budget validation MSE")
    axis.set_title(f"Proxy vs full fitness (Spearman rho={rho:.3f}, {speedup:.1f}x faster)")
    figure.colorbar(scatter, label="active parameters")
    figure.tight_layout()
    figure.savefig(f"{args.output_prefix}.png", dpi=150)
    print(f"saved {args.output_prefix}.png")


if __name__ == "__main__":
    main()
