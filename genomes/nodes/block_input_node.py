from __future__ import annotations

import torch

from genomes.nodes.block_node import BlockNode


class BlockInputNode(BlockNode):
    """The evolvable graph's single entry point. Its .value is set directly (bypassing edges) by
    VisionTransformerBlockGenome.forward to the patch-embedded, positionally-encoded, masked
    visible token sequence -- forward() is never actually invoked on this node in practice (see
    the special case in VisionTransformerBlockGenome.forward) but is implemented as identity for
    interface completeness.
    """

    def __init__(self, innovation_number: int, depth: float, d_model: int):
        super().__init__(innovation_number, depth, d_model)
        self.is_boundary_node = True

    def forward(self, x: torch.Tensor, context: dict) -> torch.Tensor:
        return x
