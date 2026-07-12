from __future__ import annotations

import torch

from genomes.nodes.block_node import BlockNode


class BlockOutputNode(BlockNode):
    """The evolvable graph's single exit point. Its .value (after the fixed, non-evolved
    mask-token-insertion/unshuffle/decoder-positional-embedding step has already run at the
    depth=0.5 bottleneck node -- see VisionTransformerBlockGenome.forward) is read by
    VisionTransformerBlockGenome.forward_mae and fed into the fixed decoder prediction head.
    Implemented as identity, matching BlockInputNode.
    """

    def __init__(self, innovation_number: int, depth: float, d_model: int):
        super().__init__(innovation_number, depth, d_model)
        self.is_boundary_node = True

    def forward(self, x: torch.Tensor, context: dict) -> torch.Tensor:
        return x
