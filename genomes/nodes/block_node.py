from __future__ import annotations

import torch

from abc import ABC, abstractmethod


class BlockNode(ABC):
    """Base class for vision-transformer "block" nodes: unlike the scalar, per-timestep Node
    hierarchy (genomes/nodes/node.py), a BlockNode's forward() is called ONCE per batch,
    consuming and producing a whole (batch, num_tokens, d_model) tensor -- appropriate for
    self-attention blocks, which need the entire token sequence present at once, and used
    uniformly by other block types (vector-LSTM, simple pointwise) too for structural consistency.

    A BlockNode intentionally does NOT subclass the scalar Node: it has no per-timestep value
    list, no input-firing counters, and no time_skip/recurrent-edge concept -- topological
    (depth-ordered) evaluation with a single forward call per node is sufficient since there is no
    inner timestep loop to synchronize (see VisionTransformerBlockGenome.forward). It duck-types
    the structural subset of Node's interface (innovation_number, depth, parameter_name,
    input_edges/output_edges, weights, disabled, is_boundary_node, __lt__) that
    genomes.genome.Genome and the existing generic reproduction operators actually rely on, so
    those can be reused unchanged for this node hierarchy too.
    """

    def __init__(self, innovation_number: int, depth: float, d_model: int):
        self.innovation_number = innovation_number
        self.depth = depth
        self.d_model = d_model
        self.parameter_name = f"{type(self).__name__}_{innovation_number}"

        self.input_edges: list = []
        self.output_edges: list = []

        # populated by concrete subclasses with their own already-initialized nn.Parameter
        # tensors (never left as None): existing WeightGenerators only fill None entries, so a
        # node whose weights are always pre-populated is safely skipped by them, while its
        # internal nn.Module (e.g. an EncoderLayer) is initialized using PyTorch's own default
        # init scheme at construction time.
        self.weights: list[torch.Tensor] = []
        self.disabled = False
        self.is_boundary_node = False

        # this node's output from the most recent forward pass, shape (batch, num_tokens,
        # d_model); populated by VisionTransformerBlockGenome.forward, read by downstream edges.
        self.value: torch.Tensor | None = None

    def __lt__(self, other: "BlockNode") -> bool:
        """Orders nodes by depth (then innovation number), matching Node.__lt__, so
        sorted(genome.nodes) evaluates nodes in a valid topological order."""
        if self.depth < other.depth:
            return True
        elif self.depth == other.depth:
            return self.innovation_number < other.innovation_number
        else:
            return False

    def __repr__(self) -> str:
        return (
            f"[block node {type(self).__name__}, innovation: {self.innovation_number}, "
            f"depth: {self.depth}, d_model: {self.d_model}, disabled: {self.disabled}]"
        )

    def add_input_edge(self, edge):
        if edge not in self.input_edges:
            self.input_edges.append(edge)

    def add_output_edge(self, edge):
        if edge not in self.output_edges:
            self.output_edges.append(edge)

    def reset(self):
        """Resets this node's cached output and gradients for the next forward/backward pass."""
        self.value = None
        for weight in self.weights:
            weight.grad = None

    @abstractmethod
    def forward(self, x: torch.Tensor, context: dict) -> torch.Tensor:
        """Args:
            x: the summed input from this node's active input edges, shape
                (batch, num_tokens, d_model).
            context: token-layout metadata for this forward pass, supplied by
                VisionTransformerBlockGenome._forward_graph. Keys: "num_spatial" and
                "num_temporal" (the patch-grid dimensions), "has_cls" (whether token 0 is the CLS
                token), and "region" ("encoder" for the shuffled-visible-token region at
                depth <= 0.5, "decoder" for the restored-full-grid region at depth > 0.5). Most
                node types ignore it; grid-aware nodes (TemporalLSTMBlockNode) use it to recover
                the (parcel x time) structure.

        Returns:
            This node's output, shape (batch, num_tokens, d_model).
        """
        pass
