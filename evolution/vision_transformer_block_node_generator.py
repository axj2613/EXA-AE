from __future__ import annotations

import random

from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.nodes.attention_block_node import AttentionBlockNode
from genomes.nodes.simple_block_node import SimpleBlockNode
from genomes.nodes.sequence_lstm_block_node import SequenceLSTMBlockNode
from genomes.nodes.temporal_lstm_block_node import TemporalLSTMBlockNode

from innovation.innovation_generator import InnovationGenerator


# "temporal_lstm" is intentionally EXCLUDED from the pool: it recurs over the per-parcel temporal
# axis, but the single-temporal-patch config (window == time_patch_size => num_temporal == 1) leaves
# no temporal axis to recur over, so the node is degenerate -- and the HCP signal carries ~no
# temporal structure at this patch size anyway (see docs/improvement_plan.md). Excluding it here
# means an explicit request also raises "unknown node type" below, so it can never be selected.
# Re-add "temporal_lstm" to re-enable if a run ever returns to multi-temporal-patch windows; the
# constructor branch and the depth gating just below are kept ready for exactly that.
ALL_NODE_TYPES = ("attention", "simple", "sequence_lstm")

# TemporalLSTMBlockNode (when re-enabled) is the only depth-restricted type: it needs the full
# (parcel x temporal-patch) grid that only exists in the decoder region (see its docstring), so it
# would be offered only at depth > 0.5.
_DECODER_ONLY_NODE_TYPES = frozenset({"temporal_lstm"})


class VisionTransformerBlockNodeGenerator(NodeGenerator):
    """Selects a block node type at random when generating new hidden nodes for a
    VisionTransformerBlockGenome -- mirrors evolution.exagp_node_generator.EXAGPNodeGenerator's
    role for the scalar RNN cell types, but at token-block granularity.

    `allowed_node_types` controls which block types evolution may introduce, so experiments can
    compare, e.g., an attention-block-only search (["attention"]) against a mixed-cell search
    (all four types) -- directly analogous to the EXAMM paper's single-cell-type vs. all-cell-type
    runs. Note this constrains only the types that MUTATIONS add; the seed genome's single fixed
    bottleneck node is always a SimpleBlockNode (it is the structural encoder/decoder pivot), so
    it is a shared constant across every configuration and does not bias an A/B comparison.

    The effective pool is `allowed_node_types` intersected with the depth rules (TemporalLSTMBlockNode
    only in the decoder region). num_heads/d_ff/dropout are fixed genome-level hyperparameters
    shared by every AttentionBlockNode this generator creates. All nodes are warm-started
    (near-identity init) since they are inserted into an existing genome by a mutation -- see
    AttentionBlockNode's docstring.
    """

    def __init__(self, num_heads: int, d_ff: int, dropout: float, allowed_node_types=None):
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        allowed = tuple(ALL_NODE_TYPES if allowed_node_types is None else allowed_node_types)
        unknown = [t for t in allowed if t not in ALL_NODE_TYPES]
        if unknown:
            raise ValueError(f"unknown node type(s) {unknown}; valid types are {list(ALL_NODE_TYPES)}")
        if not allowed:
            raise ValueError("allowed_node_types must contain at least one node type")
        # the encoder region (depth <= 0.5) cannot host decoder-only types, so at least one
        # encoder-valid type is required or the generator could be asked for an impossible node.
        if all(t in _DECODER_ONLY_NODE_TYPES for t in allowed):
            raise ValueError(
                f"allowed_node_types {list(allowed)} contains only decoder-region types; include at "
                "least one type valid in the encoder region (attention, simple, or sequence_lstm)"
            )
        self.allowed_node_types = allowed

    def __call__(self, depth: float, target_genome: Genome):
        d_model = target_genome.d_model
        innovation_number = InnovationGenerator.get_innovation_number()

        node_types = [
            t for t in self.allowed_node_types
            if depth > 0.5 or t not in _DECODER_ONLY_NODE_TYPES
        ]
        node_type = random.choice(node_types)

        if node_type == "attention":
            return AttentionBlockNode(
                innovation_number=innovation_number, depth=depth, d_model=d_model,
                num_heads=self.num_heads, d_ff=self.d_ff, dropout=self.dropout, warm_start=True,
            )
        elif node_type == "sequence_lstm":
            return SequenceLSTMBlockNode(
                innovation_number=innovation_number, depth=depth, d_model=d_model, warm_start=True,
            )
        elif node_type == "temporal_lstm":
            return TemporalLSTMBlockNode(
                innovation_number=innovation_number, depth=depth, d_model=d_model, warm_start=True,
            )
        else:
            return SimpleBlockNode(
                innovation_number=innovation_number, depth=depth, d_model=d_model, warm_start=True,
            )
