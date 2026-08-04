class InnovationGenerator:
    innovation_counter: int = 0

    @staticmethod
    def get_innovation_number() -> int:
        """
        Returns:
            The next unique innovation number.
        """
        number = InnovationGenerator.innovation_counter
        InnovationGenerator.innovation_counter += 1
        return number

    @staticmethod
    def advance_past(genome) -> int:
        """Bumps the global counter above every innovation number `genome` already uses, so newly
        generated nodes/edges cannot collide with the genome's existing ones.

        Needed after loading a PICKLED genome into a fresh process: a pretrained-seed pickle carries
        node/edge innovation numbers from the session that built it, but a new evolution session's
        counter starts at 0, so without this the first mutation-added node reuses a seed innovation
        number and trips the uniqueness assertion in Genome.add_node. (The checkpoint-resume path
        restores the counter from saved run state instead; a bare seed pickle has no saved counter,
        so it must be derived from the genome. A no-op when the counter is already ahead.)

        Returns the resulting counter value.
        """
        used = [node.innovation_number for node in genome.nodes]
        used += [edge.innovation_number for edge in genome.edges]
        target = (max(used) + 1) if used else 0
        if target > InnovationGenerator.innovation_counter:
            InnovationGenerator.innovation_counter = target
        return InnovationGenerator.innovation_counter
