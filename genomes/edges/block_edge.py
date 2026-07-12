from __future__ import annotations

import torch


class BlockEdge:
    """A weighted connection between two BlockNodes: a single learnable scalar gate broadcast
    over the whole (batch, num_tokens, d_model) tensor passed along it, mirroring
    genomes.edges.recurrent_edge.RecurrentEdge's scalar-weight semantics but without any
    timestep/time_skip firing mechanics -- a BlockEdge's forward() is called exactly once per
    graph evaluation (see VisionTransformerBlockGenome.forward), not once per real timestep,
    since block nodes consume/produce a whole batch at once rather than firing per timestep.

    This intentionally does not subclass genomes.edges.edge.Edge: that ABC's forward() signature
    takes a time_step argument that has no meaning here, and __init__ requires
    max_sequence_length, which this edge type has no use for.

    time_skip is fixed at 0. It exists only so genomes.genome.Genome.get_edge_distributions
    (which reads edge.time_skip generically) and reproduction operators like AddNode (which
    always request both a "recurrent" and "non-recurrent" edge from the edge generator) work
    unmodified; a "recurrent" BlockEdge request is treated identically to a normal one. Recurrent
    (time_skip > 0) connections between arbitrary blocks don't have a well-defined meaning once
    processing has moved to whole-tensor-per-forward-call semantics -- any sequential recurrence
    lives inside a node's own implementation (see SequenceLSTMBlockNode / TemporalLSTMBlockNode),
    not in how edges route between nodes.
    """

    def __init__(self, innovation_number: int, input_node, output_node):
        self.innovation_number = innovation_number
        self.input_node = input_node
        self.output_node = output_node
        self.input_innovation_number = input_node.innovation_number
        self.output_innovation_number = output_node.innovation_number

        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

        self.time_skip = 0
        self.disabled = False
        # filled in by a WeightGenerator (e.g. weight_generators.kaiming_weight_generator), which
        # only overwrites None entries -- matches RecurrentEdge's [None] pattern exactly.
        self.weights: list[torch.Tensor] = [None]

    def __repr__(self) -> str:
        return (
            f"BlockEdge {self.innovation_number} from node {self.input_innovation_number} to "
            f"node {self.output_innovation_number}, weight: {self.weights}"
        )

    def reset(self):
        for weight in self.weights:
            if weight is not None:
                weight.grad = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weights[0]
