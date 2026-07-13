from __future__ import annotations

import concurrent.futures
import gc

import torch


def _train_one(genome, dataset, device, learning_rate, train_kwargs):
    """Moves a genome to its assigned device, builds a fresh optimizer, and trains it. Runs in a
    worker thread. Distinct genomes share no mutable state (each owns its own modules/tensors on
    its own device); the only shared object is the dataset, whose sampling is lock-guarded.

    OOM-resilient: because evolved genomes grow without bound, a large enough one (mainly its
    decoder attention over the full token sequence) can exceed GPU memory. Rather than crashing the
    whole run, an OOM'd genome is assigned infinite fitness -- so selection drops it, which softly
    caps genome size -- its memory is released, and the search continues on the next generation.
    """
    optimizer = None
    try:
        genome.to(device)
        optimizer = torch.optim.Adam(genome.parameters(), lr=learning_rate)
        genome.train(dataset=dataset, optimizer=optimizer, **train_kwargs)
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


def evolve_parallel(population, dataset, devices, learning_rate, num_genomes=None, **train_kwargs):
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
            executor.submit(_train_one, genome, dataset, device, learning_rate, train_kwargs)
            for genome, device in zip(genomes, assigned_devices)
        ]
        trained = [future.result() for future in futures]  # re-raises any worker exception here

    # insert sequentially (population is sorted/truncated by fitness on each insert)
    for genome in trained:
        population.insert_genome(genome)

    # release freed-but-cached GPU memory and defragment between generations -- across hundreds of
    # generations of varying-sized genomes the caching allocator fragments, which alone can cause
    # OOM even when enough total memory is free.
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return trained


def resolve_devices() -> list[torch.device]:
    """Returns one torch.device per available CUDA GPU (so evolve_parallel spreads genomes across
    all of them), or [cpu] if there is no GPU."""
    count = torch.cuda.device_count()
    if count == 0:
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{i}") for i in range(count)]
