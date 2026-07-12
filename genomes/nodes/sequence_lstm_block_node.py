from __future__ import annotations

import torch
import torch.nn as nn

from genomes.nodes.block_node import BlockNode


class SequenceLSTMBlockNode(BlockNode):
    """A vector-valued LSTM (generalizing the scalar LSTMNode's hidden_size=1 cell to a full
    d_model-wide hidden state), applied as a residual block: output = x + out_proj(lstm(x)).

    It recurs over whatever ordering the tokens currently have in the graph -- the shuffled
    visible-token order in the encoder region, or the restored spatial-major order in the decoder
    region. Neither is the true per-parcel temporal axis, so this is deliberately named a generic
    order-dependent SEQUENCE-mixing block, NOT a temporal model: it gives evolution a distinct
    sequential-mixing operator to contrast against AttentionBlockNode (full cross-token mixing)
    and SimpleBlockNode (no cross-token mixing). For recurrence along the real per-parcel time
    axis, see TemporalLSTMBlockNode (restricted to the decoder region where the full patch grid
    is available). This node ignores the layout `context`.

    warm_start=True zero-initializes out_proj so the residual branch outputs zero and the block
    is an exact identity at insertion (see AttentionBlockNode's docstring for why); gradients
    still flow to the LSTM and out_proj so training can grow the block.
    """

    def __init__(self, innovation_number: int, depth: float, d_model: int, warm_start: bool = False):
        super().__init__(innovation_number, depth, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)

        if warm_start:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        self.weights = list(self.lstm.parameters()) + list(self.out_proj.parameters())

    def forward(self, x: torch.Tensor, context: dict) -> torch.Tensor:
        output, _ = self.lstm(x)
        return x + self.out_proj(output)
