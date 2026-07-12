from __future__ import annotations

import torch
import torch.nn as nn

from genomes.nodes.block_node import BlockNode
from genomes.transformer_model.encoder_layer import EncoderLayer


class AttentionBlockNode(BlockNode):
    """A full self-attention + feed-forward transformer block (reuses the repo's existing
    EncoderLayer: self-attention, residual+LayerNorm, position-wise FFN, residual+LayerNorm),
    mixing information across ALL tokens currently in the sequence -- the block-graph analog of a
    fully-connected recurrent memory cell, but attending rather than recurring.

    warm_start=True zero-initializes the two residual sub-blocks' output projections (the
    attention output projection W_o and the FFN's second linear fc2), a ReZero/Fixup-style
    "zero-branch" init: both residual branches then output zero, so the block reduces to
    norm2(norm1(x)) -- an (up-to-post-LayerNorm) near-identity. This makes a freshly MUTATED-IN
    block non-destructive to the parent genome's learned function, so inserting it doesn't
    corrupt fitness before the block has had a chance to train; gradients still flow to the
    zeroed layers, so training can grow the block into usefulness. Only newly generated nodes are
    warm-started (see VisionTransformerBlockNodeGenerator); the seed genome's own nodes and
    deepcopy-inherited nodes keep their trained weights.
    """

    def __init__(
        self, innovation_number: int, depth: float, d_model: int, num_heads: int, d_ff: int,
        dropout: float, warm_start: bool = False,
    ):
        super().__init__(innovation_number, depth, d_model)
        self.encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)

        if warm_start:
            for layer in (self.encoder_layer.self_attn.W_o, self.encoder_layer.feed_forward.fc2):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.weights = list(self.encoder_layer.parameters())

        # when capture_attention is set (by VisionTransformerBlockGenome.encode_for_analysis),
        # forward() stashes this block's self-attention probabilities on last_attention, shape
        # (batch, num_heads, num_query_tokens, num_key_tokens), for interpretability analysis.
        self.capture_attention = False
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, context: dict) -> torch.Tensor:
        if self.capture_attention:
            output, attn_probs = self.encoder_layer(x, None, output_attentions=True)
            self.last_attention = attn_probs.detach()
            return output
        return self.encoder_layer(x, None)
