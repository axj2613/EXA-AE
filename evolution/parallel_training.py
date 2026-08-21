from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import math

import torch


def _train_one(genome, dataset, device, learning_rate, train_kwargs, pdh=None, probe=None):
    """Moves a genome to its assigned device, builds a fresh optimizer, and trains it. Runs in a
    worker thread. Distinct genomes share no mutable state (each owns its own modules/tensors on
    its own device); the only shared object is the dataset, whose sampling is lock-guarded.

    The whole body runs inside `torch.cuda.device(device)`: a worker thread does NOT inherit the
    genome's device as its current CUDA device, so without this the allocator, GradScaler internals,
    and any implicit-device kernels would target the process-default device (cuda:0) while the
    tensors live on cuda:1 -- a cross-device launch that raises cudaErrorIllegalAddress and poisons
    the whole CUDA context.

    If `pdh` (a ProgressiveDynamicHurdles) is given, the genome is trained with hurdle-escalated
    budgeting instead of the fixed train_kwargs budget: it reuses the same optimizer across stages
    (so momentum + inherited weights accumulate) and reads batch_size/fitness_batches/use_amp from
    train_kwargs (iterations/batches_per_iteration are ignored -- PDH supplies the step schedule).

    OOM-resilient: because evolved genomes grow without bound, a large enough one (mainly its
    decoder attention over the full token sequence) can exceed GPU memory. Rather than crashing the
    whole run, an OOM'd genome is assigned infinite fitness -- so selection drops it, which softly
    caps genome size -- its memory is released, and the search continues on the next generation.
    """
    device = torch.device(device)
    device_context = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
    optimizer = None
    with device_context:
        try:
            genome.to(device)
            optimizer = torch.optim.Adam(genome.parameters(), lr=learning_rate)
            if pdh is not None:
                pdh.train(
                    genome, dataset=dataset, optimizer=optimizer,
                    batch_size=train_kwargs["batch_size"],
                    fitness_batches=train_kwargs["fitness_batches"],
                    use_amp=train_kwargs.get("use_amp", False),
                )
            else:
                genome.train(dataset=dataset, optimizer=optimizer, **train_kwargs)

            # Phase 3: fold a downstream CLINICAL score into the selection fitness. genome.fitness is
            # currently pure reconstruction MSE -- preserve it as recon_fitness (PDH keeps escalating
            # on THAT), then set the selection fitness to recon_MSE - weight*clinical. Only score
            # genomes that reconstruct finitely and (under PDH) cleared >= min_stage, so only
            # hurdle-clearers pay the probe cost. See evolution.downstream_probe.
            if probe is not None and math.isfinite(genome.fitness):
                genome.recon_fitness = float(genome.fitness)
                stage = getattr(genome, "pdh_stage", None)
                if pdh is None or stage is None or stage >= probe.min_stage:
                    s = probe.score(genome, device)
                    genome.probe_sex_auc = s["sex_auc"]
                    genome.probe_age_r2 = s["age_r2"]
                    genome.probe_score = s["combined"]
                    genome.fitness = genome.recon_fitness - probe.fitness_weight * s["combined"]
                    print(f"[probe] genome {genome.generation_number}: sex_auc {s['sex_auc']:.3f} "
                          f"age_r2 {s['age_r2']:+.3f} -> fitness {genome.recon_fitness:.4f} "
                          f"- {probe.fitness_weight}*{s['combined']:.3f} = {genome.fitness:.4f}")
                else:
                    genome.probe_score = 0.0  # not deep enough to earn a probe; ranked on MSE
                genome.reset()  # clear node .value set by the probe's forward passes before pickling
        except torch.cuda.OutOfMemoryError:
            genome.fitness = float("inf")
            genome.complexity = genome.parameter_report()
            genome.reset()
            genome.to("cpu")
            optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"OOM: genome {genome.generation_number} "
                  f"({genome.complexity['total_active_parameters']:,} params) penalized and skipped")
    return genome


def evolve_parallel(population, dataset, devices, learning_rate, num_genomes=None, pdh=None,
                    probe=None, **train_kwargs):
    """Generates a small batch of genomes and trains them concurrently, one per device -- turning
    the two idle T4s into ~2x evolution throughput. Genome TRAINING is independent and
    embarrassingly parallel, so it is threaded across the GPUs; genome GENERATION and INSERTION
    stay in the calling (main) thread, where they run sequentially so the global innovation
    counter, the shared RNG, and the population state remain consistent and deterministic.

    Threads (not processes) are used because PyTorch releases the GIL during CUDA kernels, so two
    genomes training on cuda:0 / cuda:1 genuinely overlap on the hardware, and there is no need to
    pickle genomes/CUDA tensors across a process boundary. HCPWindowDataset.sample_batch is
    lock-guarded so concurrent sampling from the shared dataset is safe.

    Args:
        population: a SinglePopulation (or compatible) -- generate_genome()/insert_genome() are
            called on it in the main thread.
        dataset: shared, thread-safe dataset (HCPWindowDataset).
        devices: list of torch devices to spread this batch of genomes across (e.g.
            [cuda:0, cuda:1]).
        learning_rate: Adam learning rate for each genome.
        num_genomes: how many genomes to generate+train this call (default: len(devices)).
        pdh: optional ProgressiveDynamicHurdles. If given, genomes are trained with hurdle-escalated
            budgeting and a new hurdle may be created after each insertion; iterations/
            batches_per_iteration in train_kwargs are then ignored.
        **train_kwargs: forwarded to genome.train (iterations, batches_per_iteration, batch_size,
            fitness_batches, use_amp).

    Returns:
        The list of trained genomes (already inserted into the population).
    """
    num_genomes = num_genomes or len(devices)

    # generate sequentially in the main thread (keeps innovation numbers unique + monotonic)
    genomes = [population.generate_genome() for _ in range(num_genomes)]
    assigned_devices = [devices[i % len(devices)] for i in range(num_genomes)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(_train_one, genome, dataset, device, learning_rate, train_kwargs, pdh, probe)
            for genome, device in zip(genomes, assigned_devices)
        ]
        trained = [future.result() for future in futures]  # re-raises any worker exception here

    # insert sequentially (population is sorted/truncated by fitness on each insert). With PDH, let
    # the controller observe each insertion so it can create the next hurdle after models_per_hurdle
    # genomes (main-thread only, so hurdle state stays consistent).
    for genome in trained:
        population.insert_genome(genome)
        if pdh is not None:
            pdh.observe(population.population)

    # release freed-but-cached GPU memory and defragment between generations -- across hundreds of
    # generations of varying-sized genomes the caching allocator fragments, which alone can cause
    # OOM even when enough total memory is free. empty_cache() only affects the current device, so
    # loop over all of them.
    if torch.cuda.is_available():
        gc.collect()
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()

    return trained


def resolve_devices() -> list[torch.device]:
    """Returns one torch.device per available CUDA GPU (so evolve_parallel spreads genomes across
    all of them), or [cpu] if there is no GPU."""
    count = torch.cuda.device_count()
    if count == 0:
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{i}") for i in range(count)]
