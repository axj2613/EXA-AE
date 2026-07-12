from __future__ import annotations

import torch
import torch.nn as nn

from genomes.nodes.block_node import BlockNode


class SimpleBlockNode(BlockNode):
    """The vector-token analog of the scalar "simple neuron" (genomes.nodes.node.Node): a
    per-token, non-mixing transform (no attention across tokens, no recurrence across time). It
    is residual -- output = x + tanh(linear(x)) -- with a tanh nonlinearity matching the
    phi_s(.) = tanh(.) activation the EXAMM/EXALT paper uses for simple neurons, generalized from
    a scalar weighted sum to a per-token linear layer. Gives evolution a cheap, local,
    non-mixing building block to contrast against AttentionBlockNode (mixes across all tokens)
    and the recurrent block node (mixes sequentially across tokens).

    The residual (x +) form exists so warm_start=True (which zero-initializes the linear layer)
    makes the node an exact identity at insertion: tanh(linear(x)) becomes tanh(0) = 0, so
    output = x. This makes a freshly mutated-in block non-destructive to the parent genome's
    learned function (see AttentionBlockNode's docstring for the rationale); gradients still flow
    to the zeroed linear, so training can grow the block into usefulness.
    """

    def __init__(self, innovation_number: int, depth: float, d_model: int, warm_start: bool = False):
        super().__init__(innovation_number, depth, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

        if warm_start:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        self.weights = list(self.linear.parameters())

    def forward(self, x: torch.Tensor, context: dict) -> torch.Tensor:
        return x + self.activation(self.linear(x))
