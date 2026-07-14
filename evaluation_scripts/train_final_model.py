"""Finalize an evolved architecture into a competitive model.

The evolutionary search only trains each genome briefly, so the "best" genome it outputs is a
lightly-trained *topology*, and (given a noisy proxy fitness) not necessarily even the best
topology. This script turns that into a finished model in two stages:

  1. RE-RANK (only with --checkpoint): the final population is the top-K genomes by the cheap proxy
     fitness. Re-train each at a higher budget and re-score on the validation split, so the winner
     is chosen by a less-noisy signal -- multi-fidelity selection, robust to a weak proxy.

  2. FULL TRAINING: freeze the winning topology and train it for a long, dedicated run -- warmup +
     cosine-decayed LR, periodic validation, early stopping, and keeping the best-VALIDATION
     checkpoint. This is the model that competes on reconstruction; the search only found the
     architecture. (train() as used in the search is a one-shot brief-budget loop; this is a
     separate, proper training loop.)

Usage:
    # from an evolution checkpoint (re-rank the population, then full-train the winner):
    python evaluation_scripts/train_final_model.py \\
        --checkpoint /kaggle/working/evolution_checkpoint.pkl \\
        <hcp_root> <atlas_coords> <split_json> <stats_npz> [--length_index ...] \\
        [--output final_model.pkl] [--total_steps 10000 --batch_size 64 ...]

    # or full-train one specific genome:
    python evaluation_scripts/train_final_model.py --genome best_genome.pkl <hcp_root> ...
"""

import argparse
import copy
import math
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evolution.checkpoint import load_checkpoint  # noqa: E402
from time_series.hcp_window_dataset import HCPWindowDataset  # noqa: E402


def set_dropout(genome, probability):
    """Override the dropout probability of every nn.Dropout in the genome (dropout is baked into
    the modules at construction; bump it for the long final run since 769 train subjects is a much
    smaller pool than BrainLM's 61k, so overfitting risk is higher)."""
    for module in genome._iter_modules():
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Dropout):
                submodule.p = probability


def _accumulate_masked_r2(stats, pred, target, mask):
    """Accumulates sufficient statistics for a masked-patch R^2 over many batches (a per-batch R^2
    would be a biased, high-variance estimate)."""
    masked = mask.bool().unsqueeze(-1).expand_as(pred)
    predicted = pred[masked].float()
    actual = target[masked].float()
    stats["ss_res"] += ((predicted - actual) ** 2).sum().item()
    stats["sum_t"] += actual.sum().item()
    stats["sum_t2"] += (actual * actual).sum().item()
    stats["n"] += actual.numel()


def _finalize_r2(stats):
    if stats["n"] == 0:
        return float("nan")
    ss_tot = stats["sum_t2"] - (stats["sum_t"] ** 2) / stats["n"]
    return 1.0 - stats["ss_res"] / ss_tot if ss_tot > 0 else float("nan")


def evaluate(genome, dataset, device, split, batch_size, num_batches, amp_enabled):
    """Returns (mean masked-reconstruction MSE, masked-patch R^2) over `num_batches` batches of the
    given split, eval mode, no grad. R^2 over masked patches is BrainLM's reported reconstruction
    metric (their held-out ~0.46 UKB / ~0.28 HCP).

    A rare fp16 overflow on one outlier window (large-but-finite trained weights pushing an
    attention logit past fp16's ~65504 range) shouldn't nuke an entire report after a long run --
    such a batch is skipped and excluded from the average/R^2, with a warning printed. This is a
    different policy than the genome-selection fitness guard in vision_transformer_block_genome.py,
    which treats ANY non-finite validation batch as full genome failure: that guard is choosing
    among many candidate architectures, so it should be pessimistic; this call is reporting the
    honest quality of the one already-chosen winner, so it should stay robust to a single glitch."""
    for module in genome._iter_modules():
        module.eval()
    total_loss = 0.0
    valid_batches = 0
    stats = {"ss_res": 0.0, "sum_t": 0.0, "sum_t2": 0.0, "n": 0}
    try:
        with torch.no_grad():
            for _ in range(num_batches):
                genome.reset()
                batch = dataset.sample_batch(batch_size, genome.window_length, split=split).to(device)
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss, pred, mask, patches = genome.forward(batch)
                else:
                    loss, pred, mask, patches = genome.forward(batch)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print(f"WARNING: non-finite loss on a '{split}' evaluation batch -- skipped")
                    continue
                total_loss += loss_value
                valid_batches += 1
                _accumulate_masked_r2(stats, pred, patches, mask)
    finally:
        for module in genome._iter_modules():
            module.train()
    if valid_batches == 0:
        # every batch was non-finite (e.g. the model has diverged to NaN weights) -- report failure
        # rather than a spurious MSE 0.0, which would otherwise register as a new "best" and save the
        # broken model.
        return float("inf"), float("nan")
    return total_loss / valid_batches, _finalize_r2(stats)


def rerank(candidates, dataset, device, args):
    """Re-train each survivor at the higher re-rank budget, re-score on validation, and return the
    best. Continues from each genome's search weights (more training of the same topology)."""
    amp_enabled = device.type == "cuda"
    print(f"\n=== re-ranking {len(candidates)} survivors at {args.rerank_iters}x{args.rerank_bpi} steps ===")
    for genome in candidates:
        genome.to(device)
        optimizer = torch.optim.Adam(genome.parameters(), lr=args.lr)
        genome.train(
            dataset=dataset, optimizer=optimizer, iterations=args.rerank_iters,
            batches_per_iteration=args.rerank_bpi, batch_size=args.rerank_batch_size,
            fitness_batches=args.val_batches, use_amp=amp_enabled,
        )

    candidates.sort(key=lambda g: g.fitness)
    print("\nleaderboard (re-rank validation MSE, best first):")
    for rank, genome in enumerate(candidates):
        params = genome.complexity["total_active_parameters"]
        print(f"  {rank + 1:2d}. gen {genome.generation_number:<5d} val MSE {genome.fitness:.5f}  "
              f"{params:,} params  {genome.complexity['node_type_counts']}")
    return candidates[0]


def full_train(genome, dataset, device, args):
    """Dedicated long training with warmup+cosine LR, periodic validation, early stopping, and
    best-validation checkpointing to args.output."""
    amp_enabled = device.type == "cuda"
    genome.to(device)
    for module in genome._iter_modules():
        module.train()

    optimizer = torch.optim.Adam(genome.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    # linear warmup then cosine decay to min_lr (as a fraction of lr)
    min_ratio = args.min_lr / args.lr

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps))
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n=== full training: {args.total_steps} steps, batch {args.batch_size}, "
          f"val every {args.val_every}, patience {args.patience} ===")
    best_val = float("inf")
    checks_without_improvement = 0

    for step in range(1, args.total_steps + 1):
        genome.reset()
        batch = dataset.sample_batch(args.batch_size, genome.window_length, split="train").to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _, _, _ = genome.forward(batch)
            scaler.scale(loss).backward()
            # unscale before clipping so max_norm is measured in true (not fp16-scaled) gradient
            # units. Without clipping, fp16 training of the deeper seed diverges to NaN as warmup
            # ramps the LR to its peak (mirrors the clip already in genome.train's search loop).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(genome.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _, _, _ = genome.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(genome.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if step % args.val_every == 0 or step == args.total_steps:
            val_mse, val_r2 = evaluate(genome, dataset, device, "val", args.batch_size, args.val_batches, amp_enabled)
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {step:6d}  lr {lr_now:.2e}  train {loss.item():.5f}  "
                  f"val MSE {val_mse:.5f}  val R2 {val_r2:.4f}"
                  f"{'  <- best' if math.isfinite(val_mse) and val_mse < best_val - 1e-6 else ''}")
            # require a FINITE val: a diverged model reports inf (see evaluate), which must never
            # count as an improvement or get saved as the best model.
            if math.isfinite(val_mse) and val_mse < best_val - 1e-6:
                best_val = val_mse
                checks_without_improvement = 0
                _save_best(genome, val_mse, args.output)
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= args.patience:
                    print(f"early stopping: no val improvement in {args.patience} checks")
                    break

    print(f"\ndone. best validation MSE: {best_val:.5f}  ->  {args.output}")


def _save_best(genome, val_fitness, output_path):
    """Saves a portable CPU copy of the genome as the current best-validation model. Deepcopies so
    the live genome (and its optimizer's parameter references) keep training on-device untouched."""
    genome.reset()  # clear non-leaf node values so the genome is deepcopy-able
    best = copy.deepcopy(genome)
    best.to("cpu")
    best.fitness = val_fitness
    best.complexity = best.parameter_report()
    for parameter in best.parameters():
        parameter.grad = None
    tmp = output_path + ".tmp"
    with open(tmp, "wb") as best_file:
        pickle.dump(best, best_file)
    os.replace(tmp, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hcp_root")
    parser.add_argument("atlas_coords")
    parser.add_argument("split_json")
    parser.add_argument("stats_npz")
    parser.add_argument("--length_index", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="evolution checkpoint: re-rank its population, then train the winner")
    source.add_argument("--genome", help="a single genome .pkl to full-train directly")
    parser.add_argument("--output", default="final_model.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dropout", type=float, default=None, help="override dropout for the final run")
    # final-training schedule
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_batches", type=int, default=32)
    parser.add_argument("--test_batches", type=int, default=64, help="batches for the final held-out test R^2")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=300)
    # re-rank budget (only with --checkpoint)
    parser.add_argument("--rerank_iters", type=int, default=4)
    parser.add_argument("--rerank_bpi", type=int, default=100)
    parser.add_argument("--rerank_batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)

    # load the genome / population first to learn the window length, then build the dataset
    candidates = None
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint)
        candidates = list(state["population_strategy"].population)
        window_length = candidates[0].window_length
    else:
        with open(args.genome, "rb") as genome_file:
            winner = pickle.load(genome_file)
        window_length = winner.window_length

    dataset = HCPWindowDataset(
        root_dir=args.hcp_root, atlas_coordinates_filename=args.atlas_coords,
        window_length=window_length, split_path=args.split_json,
        stats_path=args.stats_npz, length_index_path=args.length_index,
    )

    if candidates is not None:
        winner = rerank(candidates, dataset, device, args)
        print(f"\nselected winner: gen {winner.generation_number}, "
              f"{winner.complexity['total_active_parameters']:,} params")

    if args.dropout is not None:
        set_dropout(winner, args.dropout)
        print(f"dropout overridden to {args.dropout} for the final run")

    full_train(winner, dataset, device, args)

    # honest held-out number: reconstruction R^2 on the TEST split (untouched during training/
    # selection), computed once on the best-validation model, for comparison against BrainLM.
    with open(args.output, "rb") as best_file:
        best = pickle.load(best_file)
    best.to(device)
    test_mse, test_r2 = evaluate(
        best, dataset, device, "test", args.batch_size, args.test_batches, device.type == "cuda"
    )
    print(f"\n=== HELD-OUT TEST (best-val model, {args.test_batches} batches) ===")
    print(f"reconstruction R^2 = {test_r2:.4f}   (MSE {test_mse:.5f})")
    print("BrainLM reference: masked-reconstruction R^2 ~0.46 (UKB) / ~0.28 (HCP)")


if __name__ == "__main__":
    main()
