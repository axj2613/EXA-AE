from __future__ import annotations

import copy
import random

import torch

from reproduction.reproduction_method import ReproductionMethod


class MutateViTHyperparameters(ReproductionMethod):
    """Creates a child VisionTransformerMAEGenome by randomly perturbing one macro-architecture
    hyperparameter of the parent (encoder/decoder depth, d_model, temporal patch size, mask
    ratio, or dropout).

    Because changing most of these hyperparameters changes the shape of the underlying model's
    weight tensors, the child's VisionTransformerMAE is rebuilt from scratch with the mutated
    hyperparameters rather than inheriting the parent's trained weights -- there is no Lamarckian
    warm-start here (unlike Clone, which deep-copies the parent's weights unchanged). True
    weight-preserving topology evolution is left for the future block-level graph-evolution phase.

    parcel_patch_size is deliberately not mutated: it stays fixed at 1 (every original parcel
    remains its own token), matching BrainLM's published implementation and the project's
    decision to keep full per-parcel resolution rather than grouping parcels into coarser patches.
    """

    def __init__(self, num_parcels: int, window_length: int, parcel_coordinates: torch.Tensor):
        super().__init__(node_generator=None, edge_generator=None, weight_generator=None)
        self.num_parcels = num_parcels
        self.window_length = window_length
        self.parcel_coordinates = parcel_coordinates

    def number_parents(self) -> int:
        return 1

    def __call__(self, parent_genomes: list) -> "VisionTransformerMAEGenome":  # noqa: F821
        from genomes.vit_mae_genome import VisionTransformerMAEGenome  # avoid circular import

        parent = parent_genomes[0]
        hyperparameters = copy.deepcopy(parent.hyperparameters)

        mutation = random.choice([
            self._mutate_encoder_depth,
            self._mutate_decoder_depth,
            self._mutate_d_model,
            self._mutate_decoder_d_model,
            self._mutate_mask_ratio,
            self._mutate_dropout,
            self._mutate_time_patch_size,
        ])
        mutation(hyperparameters)

        print(f"mutating ViT-MAE hyperparameters via {mutation.__name__}: {hyperparameters}")

        # generation_number is overwritten by the population strategy once the genome is accepted
        child_genome = VisionTransformerMAEGenome(
            generation_number=0,
            num_parcels=self.num_parcels,
            window_length=self.window_length,
            parcel_coordinates=self.parcel_coordinates,
            hyperparameters=hyperparameters,
        )
        return child_genome

    @staticmethod
    def _mutate_encoder_depth(hyperparameters: dict):
        delta = random.choice([-1, 1])
        hyperparameters["encoder_depth"] = max(1, hyperparameters["encoder_depth"] + delta)

    @staticmethod
    def _mutate_decoder_depth(hyperparameters: dict):
        delta = random.choice([-1, 1])
        hyperparameters["decoder_depth"] = max(1, hyperparameters["decoder_depth"] + delta)

    @staticmethod
    def _mutate_d_model(hyperparameters: dict):
        num_heads = hyperparameters["num_heads"]
        choices = [d for d in (32, 64, 128, 256, 512) if d % num_heads == 0 and d != hyperparameters["d_model"]]
        if choices:
            hyperparameters["d_model"] = random.choice(choices)

    @staticmethod
    def _mutate_decoder_d_model(hyperparameters: dict):
        num_heads = hyperparameters["decoder_num_heads"]
        choices = [
            d for d in (16, 32, 64, 128, 256) if d % num_heads == 0 and d != hyperparameters["decoder_d_model"]
        ]
        if choices:
            hyperparameters["decoder_d_model"] = random.choice(choices)

    @staticmethod
    def _mutate_mask_ratio(hyperparameters: dict):
        delta = random.uniform(-0.1, 0.1)
        hyperparameters["mask_ratio"] = min(0.9, max(0.1, hyperparameters["mask_ratio"] + delta))

    @staticmethod
    def _mutate_dropout(hyperparameters: dict):
        delta = random.uniform(-0.05, 0.05)
        hyperparameters["dropout"] = min(0.5, max(0.0, hyperparameters["dropout"] + delta))

    def _mutate_time_patch_size(self, hyperparameters: dict):
        # restrict to divisors that keep the number of temporal patches (and hence total token
        # count) in a sane range -- window_length // time_patch_size between 2 and 40
        candidates = [
            d for d in range(1, self.window_length + 1)
            if self.window_length % d == 0 and 2 <= self.window_length // d <= 40
        ]
        choices = [d for d in candidates if d != hyperparameters["time_patch_size"]]
        if choices:
            hyperparameters["time_patch_size"] = random.choice(choices)
