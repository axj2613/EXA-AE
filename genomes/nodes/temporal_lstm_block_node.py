from __future__ import annotations

import torch
import torch.nn as nn

from genomes.nodes.block_node import BlockNode


class TemporalLSTMBlockNode(BlockNode):
    """A vector-valued LSTM that recurs along the TRUE per-parcel temporal axis, resolving the
    physical-fidelity gap of SequenceLSTMBlockNode (which recurs over whatever arbitrary token
    ordering the graph happens to present). Applied as a residual block:
    output = x + out_proj(lstm_over_time(x)).

    It requires the full (parcel x temporal-patch) grid, which only exists in the DECODER region
    (depth > 0.5): after the bottleneck's mask-token-insertion/unshuffle step, every one of the
    num_spatial x num_temporal patches is present in spatial-major order (token index =
    spatial_idx * num_temporal + temporal_idx), so the sequence can be reshaped back into a grid.
    VisionTransformerBlockNodeGenerator therefore only emits this node type at depth > 0.5. Given
    that grid, it reshapes to (batch * num_spatial, num_temporal, d_model) and runs the LSTM over
    the temporal axis INDEPENDENTLY for each parcel, so the recurrence models each parcel's own
    BOLD time course and never bleeds across parcel boundaries. The CLS token (if present) is
    passed through unchanged.

    In the encoder region the masked visible tokens do not form a clean per-parcel grid (a random
    subset per parcel, ragged), so a faithful temporal recurrence there is ill-defined; this node
    falls back to whole-sequence recurrence in that case, but the node generator avoids placing it
    there. warm_start zero-inits out_proj for exact-identity insertion (see AttentionBlockNode).
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
        num_spatial = context["num_spatial"]
        num_temporal = context["num_temporal"]
        has_cls = context["has_cls"]

        cls = None
        tokens = x
        if has_cls:
            cls = x[:, :1, :]
            tokens = x[:, 1:, :]

        batch_size, num_tokens, d_model = tokens.shape

        if context["region"] != "decoder" or num_tokens != num_spatial * num_temporal:
            # encoder region / no clean grid: fall back to whole-sequence recurrence
            recurred, _ = self.lstm(tokens)
        else:
            # (batch, spatial*temporal, d) -> (batch*spatial, temporal, d), recur over time per
            # parcel, then flatten back
            grid = tokens.reshape(batch_size * num_spatial, num_temporal, d_model)
            recurred, _ = self.lstm(grid)
            recurred = recurred.reshape(batch_size, num_spatial * num_temporal, d_model)

        out = tokens + self.out_proj(recurred)

        if cls is not None:
            out = torch.cat([cls, out], dim=1)
        return out
