from __future__ import annotations

import random

import torch

from reproduction.clone import Clone
from reproduction.mutate_vit_hyperparameters import MutateViTHyperparameters
from reproduction.reproduction_method import ReproductionMethod
from reproduction.reproduction_selector import ReproductionSelector


class ViTMAEReproductionSelector(ReproductionSelector):
    """Selects uniformly at random between hyperparameter mutation and cloning when generating
    new VisionTransformerMAEGenome children. Unlike EXAGPReproductionSelector, there is no
    node/edge graph here, so none of the graph-mutation operators (AddNode, SplitEdge, Crossover,
    ...) apply -- only macro-architecture hyperparameter search.
    """

    def __init__(self, num_parcels: int, window_length: int, parcel_coordinates: torch.Tensor):
        super().__init__(node_generator=None, edge_generator=None, weight_generator=None)

        self.reproduction_methods = [
            MutateViTHyperparameters(
                num_parcels=num_parcels, window_length=window_length, parcel_coordinates=parcel_coordinates
            ),
            Clone(node_generator=None, edge_generator=None, weight_generator=None),
        ]

    def __call__(self) -> ReproductionMethod:
        random.shuffle(self.reproduction_methods)
        return self.reproduction_methods[0]
