from __future__ import annotations

import os
import pickle
import random

import numpy as np
import torch

from innovation.innovation_generator import InnovationGenerator


def save_checkpoint(path: str, population_strategy, generation: int, config: dict | None = None):
    """Pickles the full evolution state so a run can resume after a Kaggle session times out.

    Captures the population strategy (its genomes, seed, generated-genome counter, and
    reproduction selector), the GLOBAL InnovationGenerator counter (innovation numbers must stay
    unique and monotonic across a resume or crossover's innovation-number matching breaks), the
    completed-generation index, and the Python/NumPy/Torch RNG states (so sampling and mutation
    continue deterministically rather than repeating the pre-checkpoint random draws).

    Genomes are moved to CPU before pickling so the checkpoint loads on any device; the caller
    re-homes them with genome.to(device) after loading.
    """
    for genome in _all_genomes(population_strategy):
        genome.to("cpu")

    state = {
        "population_strategy": population_strategy,
        "innovation_counter": InnovationGenerator.innovation_counter,
        "generation": generation,
        "config": config,
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }

    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as checkpoint_file:
        pickle.dump(state, checkpoint_file)
    os.replace(tmp_path, path)  # atomic: a crash mid-write can't corrupt the last good checkpoint


def load_checkpoint(path: str) -> dict:
    """Loads a checkpoint saved by save_checkpoint and restores the global InnovationGenerator
    counter and the RNG states. Returns the state dict (population_strategy, generation, config);
    the caller should genome.to(device) the population and resume the loop from `generation`."""
    with open(path, "rb") as checkpoint_file:
        state = pickle.load(checkpoint_file)

    InnovationGenerator.innovation_counter = state["innovation_counter"]
    random.setstate(state["py_rng"])
    np.random.set_state(state["np_rng"])
    torch.set_rng_state(state["torch_rng"])

    return state


def _all_genomes(population_strategy):
    genomes = list(getattr(population_strategy, "population", []))
    seed = getattr(population_strategy, "seed_genome", None)
    if seed is not None:
        genomes.append(seed)
    return genomes
