from __future__ import annotations

import math

import torch

from genomes.genome import Genome
from weight_generators.weight_generator import WeightGenerator


class LamarckianBlockWeightGenerator(WeightGenerator):
    """Weight generator for VisionTransformerBlockGenome. It plays the same two roles as the
    scalar LamarckianWeightGenerator -- initialize new weights after a mutation, and recombine
    parent weights after a crossover -- but is shape-agnostic so it works with the block nodes'
    multi-element parameter tensors (e.g. a whole EncoderLayer's Linear/LayerNorm weights), which
    the scalar generator cannot handle: LamarckianWeightGenerator crashes on such tensors via
    Genome.get_weight_distribution's .item() call, and KaimingWeightGenerator would only ever
    write scalars.

    Two representation facts drive the implementation:

    1. Block NODE weights are a VIEW of the node's internal nn.Module parameters (e.g.
       list(encoder_layer.parameters())); the module is the source of truth used in forward().
       So a blended value must be written IN-PLACE into the existing parameter tensor
       (weight.data.copy_(...)), never by replacing the list entry -- replacing would desync the
       list from the module, so forward() would keep using the old weights. Node weights are
       always pre-populated by the node's __init__ (including warm-start), so they are never None
       and never need fresh initialization here.

    2. Block EDGE weights are a single scalar that IS the source of truth (BlockEdge.forward reads
       self.weights[0] directly), matching RecurrentEdge's [None] pattern. So edge weights are
       set by assigning a fresh leaf tensor, and None entries are filled with a fan-in-scaled
       scalar (like KaimingWeightGenerator).
    """

    def __init__(self, c1: float = -0.5, c2: float = 1.5):
        """Args:
            c1, c2: bounds of the randomized line-search coefficient r for crossover
                recombination (child = wp1 + r*(wp2 - wp1)); the defaults match the [-0.5, 1.5]
                range described in the EXAMM paper.
        """
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def _edge_fan_in(edge) -> int:
        return max(1, len(edge.output_node.input_edges))

    def __call__(self, genome: Genome, **kwargs: dict):
        if "parent_genomes" not in kwargs:
            self._initialize_after_mutation(genome)
        else:
            self._recombine_after_crossover(genome, sorted(kwargs["parent_genomes"]))

    def _initialize_after_mutation(self, genome: Genome):
        """Fill only the None edge-gate weights created by the mutation; every block node's
        weights were already initialized (near-identity if warm-started) by its __init__."""
        for edge in genome.edges:
            if edge.weights[0] is None:
                edge.weights[0] = torch.tensor(
                    torch.randn(1).item() / math.sqrt(self._edge_fan_in(edge)),
                    requires_grad=True,
                )

    def _recombine_after_crossover(self, genome: Genome, parents: list[Genome]):
        r = (torch.rand(1).item() * (self.c2 - self.c1)) + self.c1

        # --- node parameters: blend in-place to keep the nn.Module view consistent ---
        for node in genome.nodes:
            parent_weight_lists = [
                parent.node_map[node.innovation_number].weights
                for parent in parents
                if node.innovation_number in parent.node_map
            ]
            for i, existing in enumerate(node.weights):
                if existing is None:
                    continue
                # move every parent's weight onto the child weight's device before blending:
                # parents may live on different devices (e.g. a checkpoint-restored genome on CPU
                # alongside a freshly trained one on cuda), and mixing devices in the arithmetic
                # below would raise "Expected all tensors to be on the same device".
                candidates = [
                    weights[i].detach().to(existing.device)
                    for weights in parent_weight_lists
                    if weights[i] is not None and weights[i].shape == existing.shape
                ]
                if len(candidates) >= 2:
                    more_fit = candidates[0]
                    others_avg = torch.stack(candidates[1:], dim=0).mean(dim=0)
                    blended = more_fit + r * (others_avg - more_fit)
                    existing.data.copy_(blended)

        # --- edge gates: scalar, assign fresh leaf tensors ---
        for edge in genome.edges:
            parent_weight_lists = [
                parent.edge_map[edge.innovation_number].weights
                for parent in parents
                if edge.innovation_number in parent.edge_map
            ]
            for i in range(len(edge.weights)):
                # target device: the child edge's own weight if set, else the first parent's --
                # move all parent candidates onto it so mixed-device parents don't crash the blend
                # (the child genome is moved to its final device by genome.to(device) afterward).
                target_device = edge.weights[i].device if edge.weights[i] is not None else None
                candidates = [
                    weights[i].detach()
                    for weights in parent_weight_lists
                    if weights[i] is not None
                ]
                if candidates and target_device is None:
                    target_device = candidates[0].device
                candidates = [candidate.to(target_device) for candidate in candidates] if candidates else candidates
                if len(candidates) >= 2:
                    more_fit = candidates[0]
                    others_avg = torch.stack(candidates[1:], dim=0).mean(dim=0)
                    blended = more_fit + r * (others_avg - more_fit)
                    edge.weights[i] = blended.clone().requires_grad_(True)
                elif edge.weights[i] is None:
                    if candidates:
                        edge.weights[i] = candidates[0].clone().requires_grad_(True)
                    else:
                        edge.weights[i] = torch.tensor(
                            torch.randn(1).item() / math.sqrt(self._edge_fan_in(edge)),
                            requires_grad=True,
                        )
