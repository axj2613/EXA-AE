from __future__ import annotations

import math
import threading


class ProgressiveDynamicHurdles:
    """Progressive Dynamic Hurdles (PDH), after So et al., "The Evolved Transformer" (2019),
    adapted to this repo's steady-state EXAMM search with Lamarckian weight inheritance.

    THE PROBLEM IT SOLVES. Every genome is trained on a tiny fixed budget so 500 genomes stay
    affordable, but that budget is too small to reveal a good architecture -- and, worse, a
    freshly mutated-in (warm-started, near-identity) block cannot out-train the already-trained
    incumbent in one genome's budget, so the search stalls and never grows depth. A single cheap
    proxy budget also ranks architectures noisily (our measured proxy-vs-full Spearman was ~0.3;
    the Evolved Transformer authors hit the same wall and could find no adequate proxy task).

    THE MECHANISM. Instead of a fixed budget, train each genome for a small increment s[0] and
    evaluate. Genomes whose fitness clears a HURDLE (the mean fitness of the population's
    most-trained members) earn the next increment s[1], are re-evaluated, and so on -- so compute
    is concentrated on promising lineages and a genome that is merely *slightly* better after s[0]
    is given the chance to prove itself with more training. Hurdles are created progressively: after
    every `models_per_hurdle` genomes a new hurdle is appended (the mean fitness of the current
    population members that reached the deepest stage), until len(step_increments) - 1 hurdles
    exist. Because our children inherit their parent's trained weights, escalating a lineage's
    training ACCUMULATES across generations -- exactly the "let a deeper genome earn more steps"
    signal the depth-growth stall needs.

    FITNESS CONVENTION. This repo minimizes validation MSE (lower is better), so the comparisons
    are inverted versus the maximizing formulation in the paper: a genome "clears" a hurdle when
    its fitness is STRICTLY BELOW it, and the sentinel that ends escalation is -inf.

    COMPARABILITY OF DIFFERENTLY-TRAINED GENOMES. Selection here is rank-based (SinglePopulation
    sorts by fitness), and -- assuming fitness improves monotonically with training, which holds in
    practice -- a genome granted more steps was, by construction, already at-or-below the hurdle at
    the smaller step count where a less-trained rival sits, so comparing them at their final
    fitnesses does not advantage the less-trained one (see the paper, Sec. 3.3).

    THREADING. evolve_parallel trains a generation's genomes concurrently; each worker only READS
    the (immutable-within-a-generation) hurdle list. Hurdle CREATION happens in the main thread via
    observe() after insertion. A lock guards the counter/hurdle list so this stays safe even if a
    caller observes concurrently.
    """

    def __init__(self, step_increments: list[int], models_per_hurdle: int):
        """
        Args:
            step_increments: gradient-step budget per stage, e.g. [200, 400, 800]. Stage 0 is
                always run; a genome reaches stage i only by clearing hurdles 0..i-1. The maximum
                total budget a genome can receive is sum(step_increments).
            models_per_hurdle: how many genomes to evaluate before appending the next hurdle
                (the paper's m). At most len(step_increments) - 1 hurdles are ever created.
        """
        if len(step_increments) < 1:
            raise ValueError("step_increments must have at least one stage")
        if models_per_hurdle < 1:
            raise ValueError("models_per_hurdle must be >= 1")
        self.step_increments = list(step_increments)
        self.models_per_hurdle = models_per_hurdle
        self.max_hurdles = len(self.step_increments) - 1
        self.hurdles: list[float] = []          # finite MSE thresholds, one per created hurdle
        self._since_last_hurdle = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ training
    def train(self, genome, dataset, optimizer, batch_size, fitness_batches, use_amp):
        """Trains `genome` with hurdle-escalated budgeting, reusing `optimizer` across stages so
        momentum and the (Lamarckian-inherited) weights accumulate. Sets genome.fitness and stamps
        genome.pdh_stage / genome.pdh_steps for hurdle bookkeeping. The genome's own train() already
        computes fitness and guards fp16 divergence (a diverged genome gets inf and stops here).
        """
        with self._lock:
            hurdles = self.hurdles + [float("-inf")]  # sentinel: nothing is < -inf, so escalation ends

        def train_stage(n_steps: int):
            # iterations=1 => exactly n_steps gradient steps this stage; genome.train recomputes
            # fitness at the end of the call.
            genome.train(
                dataset=dataset, optimizer=optimizer, iterations=1,
                batches_per_iteration=n_steps, batch_size=batch_size,
                fitness_batches=fitness_batches, use_amp=use_amp,
            )

        train_stage(self.step_increments[0])
        stage = 0
        # escalate while below (better than) the hurdle for the current stage; guard the stage index
        # so we never exceed the configured number of increments.
        while (stage < self.max_hurdles
               and math.isfinite(genome.fitness)
               and genome.fitness < hurdles[stage]):
            stage += 1
            train_stage(self.step_increments[stage])

        genome.pdh_stage = stage
        genome.pdh_steps = sum(self.step_increments[: stage + 1])
        return genome

    # ------------------------------------------------------------------ hurdle creation
    def observe(self, population) -> None:
        """Called once per inserted genome (main thread). After models_per_hurdle observations,
        appends a new hurdle equal to the mean fitness of the population members that reached the
        deepest stage seen so far -- the paper's MEAN_FITNESS_OF_MAX. Stops once all hurdles exist.
        """
        with self._lock:
            self._since_last_hurdle += 1
            if len(self.hurdles) >= self.max_hurdles or self._since_last_hurdle < self.models_per_hurdle:
                return
            # Hurdles gate TRAINING BUDGET and must be RECONSTRUCTION-MSE thresholds: escalation
            # happens inside train() where only MSE is known (the Phase-3 clinical probe runs after).
            # So read recon_fitness -- the preserved pure MSE -- not g.fitness, which may already carry
            # the probe's clinical adjustment. Falls back to g.fitness when no probe is in use.
            def recon(g):
                return getattr(g, "recon_fitness", getattr(g, "fitness", None))

            members = [g for g in population
                       if recon(g) is not None and math.isfinite(recon(g))]
            if not members:
                return
            deepest = max(getattr(g, "pdh_stage", 0) for g in members)
            at_deepest = [recon(g) for g in members if getattr(g, "pdh_stage", 0) == deepest]
            if not at_deepest:
                return
            hurdle = sum(at_deepest) / len(at_deepest)
            self.hurdles.append(hurdle)
            self._since_last_hurdle = 0
            print(f"[PDH] created hurdle {len(self.hurdles)}/{self.max_hurdles} = {hurdle:.6f} "
                  f"(mean of {len(at_deepest)} genomes at stage {deepest})")

    # ------------------------------------------------------------------ persistence (checkpoint)
    def state_dict(self) -> dict:
        """Serializable state so PDH survives a Kaggle checkpoint/resume (hurdles and the
        since-last-hurdle counter must persist or the schedule restarts)."""
        with self._lock:
            return {
                "step_increments": list(self.step_increments),
                "models_per_hurdle": self.models_per_hurdle,
                "hurdles": list(self.hurdles),
                "since_last_hurdle": self._since_last_hurdle,
            }

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            self.step_increments = list(state["step_increments"])
            self.models_per_hurdle = state["models_per_hurdle"]
            self.max_hurdles = len(self.step_increments) - 1
            self.hurdles = list(state["hurdles"])
            self._since_last_hurdle = state["since_last_hurdle"]
