from __future__ import annotations

from evolution.edge_generator import EdgeGenerator

from genomes.edges.block_edge import BlockEdge
from genomes.genome import Genome

from innovation.innovation_generator import InnovationGenerator


class VisionTransformerBlockEdgeGenerator(EdgeGenerator):
    """Creates BlockEdges for a VisionTransformerBlockGenome. The `recurrent` flag (passed by
    reproduction operators such as AddNode, which always request both a recurrent and a
    non-recurrent edge when wiring a new node) is accepted for interface compatibility but
    ignored: BlockEdge has no time_skip/recurrent concept -- see BlockEdge's docstring for why.
    """

    def __call__(self, target_genome: Genome, input_node, output_node, recurrent: bool) -> BlockEdge:
        return BlockEdge(
            innovation_number=InnovationGenerator.get_innovation_number(),
            input_node=input_node,
            output_node=output_node,
        )
