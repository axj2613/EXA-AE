from __future__ import annotations

import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from reproduction.add_edge import AddEdge
from reproduction.enable_edge import EnableEdge
from reproduction.disable_edge import DisableEdge
from reproduction.split_edge import SplitEdge

from reproduction.add_node import AddNode
from reproduction.enable_node import EnableNode
from reproduction.disable_node import DisableNode
from reproduction.split_node import SplitNode
from reproduction.merge_node import MergeNode

from reproduction.clone import Clone
from reproduction.crossover import Crossover

from reproduction.reproduction_method import ReproductionMethod
from reproduction.reproduction_selector import ReproductionSelector

from weight_generators.weight_generator import WeightGenerator


class VisionTransformerBlockReproductionSelector(ReproductionSelector):
    """Sets up a reproduction selector for evolving VisionTransformerBlockGenomes, reusing the
    existing generic mutation/crossover operators (see genomes.vision_transformer_block_genome's
    docstring for why they're safe to reuse unmodified) with autoencoder=True throughout, exactly
    as evolution.exagp_reproduction_selector.EXAGPReproductionSelector does for AutoencoderGenome.

    AddRecurrentEdge is deliberately excluded: it specifically requests a NEW recurrent (time_skip
    > 0) connection between two existing nodes, and that operation has no meaning for BlockEdges,
    which are always feed-forward (see BlockEdge's docstring).

    Recurrent edge-wiring is also disabled inside AddNode and Crossover via `allow_recurrent=False`.
    Those operators otherwise wire each new/orphaned node with both a recurrent and a non-recurrent
    edge pass; since BlockEdges have no recurrence, the recurrent pass only ever produces
    backward/same-depth edges that VisionTransformerBlockGenome's forward graph drops as inert --
    graph clutter with no functional effect. Skipping it keeps evolved genomes strictly feed-forward
    (the non-recurrent pass still guarantees at least one input and one output edge). Pass
    allow_recurrent=True to restore the old both-passes behavior.
    """

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
        allow_recurrent: bool = False,
    ):
        super().__init__(
            node_generator=node_generator,
            edge_generator=edge_generator,
            weight_generator=weight_generator,
        )

        autoencoder = True
        self.reproduction_methods = [
            AddEdge(node_generator, edge_generator, weight_generator, autoencoder),
            DisableEdge(node_generator, edge_generator, weight_generator),
            EnableEdge(node_generator, edge_generator, weight_generator),
            SplitEdge(node_generator, edge_generator, weight_generator),
            AddNode(node_generator, edge_generator, weight_generator, autoencoder,
                    allow_recurrent=allow_recurrent),
            EnableNode(node_generator, edge_generator, weight_generator),
            DisableNode(node_generator, edge_generator, weight_generator, autoencoder),
            MergeNode(node_generator, edge_generator, weight_generator, autoencoder),
            SplitNode(node_generator, edge_generator, weight_generator, autoencoder),
            Clone(node_generator, edge_generator, weight_generator),
            # number_parents=2 (standard crossover arity), not the scalar EXAGP selector's 10:
            # Crossover returns None whenever fewer than number_parents genomes are available, so
            # 10 would silently never fire for the small populations this genome type uses (the
            # scalar path only gets away with 10 because its population is hardcoded to 50).
            Crossover(node_generator, edge_generator, weight_generator, autoencoder, number_parents=2,
                      allow_recurrent=allow_recurrent),
        ]

    def __call__(self) -> ReproductionMethod:
        random.shuffle(self.reproduction_methods)
        return self.reproduction_methods[0]
