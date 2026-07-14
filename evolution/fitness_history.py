from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FitnessHistory:
    """Records the population's best/mean fitness at each checkpoint into a persisted JSON file, so
    the search's progress can be plotted at the end -- watching the best validation MSE descend to
    an asymptote tells you when evolution has plateaued and further generations aren't buying much.

    The history file lives alongside the checkpoint (e.g. /kaggle/working) and is reloaded on
    resume, so a run spanning multiple Kaggle sessions produces one continuous curve. Records are
    keyed by generation and de-duplicated, so re-recording a generation after a resume is safe.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.records: list[dict] = []
        if path and os.path.exists(path):
            with open(path) as history_file:
                self.records = json.load(history_file)

    def record(self, generation: int, population):
        """Appends (and persists) this generation's best/mean fitness and the best genome's size.
        Fitness is validation reconstruction MSE -- lower is better."""
        genomes = population.population
        fitnesses = [genome.fitness for genome in genomes if genome.fitness is not None]
        best = genomes[0] if genomes else None
        best_params = None
        if best is not None and getattr(best, "complexity", None):
            best_params = best.complexity.get("total_active_parameters")

        record = {
            "generation": generation,
            "best_fitness": best.fitness if best is not None else None,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else None,
            "best_active_parameters": best_params,
        }

        # replace any existing record at this generation (idempotent across resume)
        self.records = [r for r in self.records if r["generation"] != generation]
        self.records.append(record)
        self.records.sort(key=lambda r: r["generation"])

        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as history_file:
                json.dump(self.records, history_file, indent=2)
            os.replace(tmp, self.path)

    def plot(self, output_path: str):
        """Plots best (and mean) validation MSE vs generation -- the evolution-progress curve."""
        points = [r for r in self.records if r["best_fitness"] is not None]
        if not points:
            print("no fitness history to plot yet")
            return

        generations = [r["generation"] for r in points]
        best = [r["best_fitness"] for r in points]
        mean = [r["mean_fitness"] for r in points]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(generations, best, "-o", markersize=3, label="best genome")
        if all(m is not None for m in mean):
            axis.plot(generations, mean, "--", alpha=0.6, label="population mean")
        axis.set_xlabel("generation")
        axis.set_ylabel("validation reconstruction MSE")
        axis.set_title("Evolution progress")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        print(f"saved evolution-progress plot to {output_path} "
              f"(best MSE {best[-1]:.5f} at generation {generations[-1]})")
