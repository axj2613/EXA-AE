from __future__ import annotations

import torch

from genomes.transformer_model.vision_transformer_mae import VisionTransformerMAE
from time_series.fmri_patch_dataset import FMRIPatchDataset


DEFAULT_HYPERPARAMETERS = {
    # parcel_patch_size=1 keeps every one of the original parcels as its own token (matches
    # BrainLM's published implementation, which has no multi-parcel-grouping option at all).
    "parcel_patch_size": 1,
    "time_patch_size": 20,
    "d_model": 128,
    "num_heads": 4,
    "encoder_depth": 4,
    "d_ff": 256,
    "decoder_d_model": 64,
    "decoder_num_heads": 4,
    "decoder_depth": 2,
    "decoder_d_ff": 128,
    "mask_ratio": 0.75,
    "dropout": 0.1,
    "use_cls_token": True,
}


class VisionTransformerMAEGenome:
    """A 'genome' wrapping a fixed-topology VisionTransformerMAE model, whose evolvable genes are
    the model's macro-architecture hyperparameters (depth, d_model, patch size, mask ratio, ...)
    rather than a node/edge computational graph.

    This intentionally does not subclass genomes.genome.Genome: that base class, and the
    node/edge reproduction operators built around it, assume a scalar, per-timestep computational
    graph (see genomes/nodes/node.py) -- a fundamentally different unit of evolution than a
    ViT-MAE's batched, patch-token transformer blocks. Instead this class duck-types the subset of
    the Genome interface that population.single_population.SinglePopulation actually relies on
    (fitness ordering via __lt__, viability, parameters()), so the existing population/
    reproduction-selector machinery can be reused as-is for hyperparameter-level evolution -- see
    evolution/vit_mae_reproduction_selector.py and reproduction/mutate_vit_hyperparameters.py.

    True topology evolution (evolving which blocks -- attention, LSTM, GRU, simple neuron --
    compose the encoder/decoder, mixing cell types the way EXAMM does for RNNs) is deliberately
    out of scope here; this genome type is the fixed-topology baseline that phase should build on.
    """

    def __init__(
        self,
        generation_number: int,
        num_parcels: int,
        window_length: int,
        parcel_coordinates: torch.Tensor,
        hyperparameters: dict,
    ):
        self.generation_number = generation_number
        self.num_parcels = num_parcels
        self.window_length = window_length
        self.parcel_coordinates = parcel_coordinates
        self.hyperparameters = dict(hyperparameters)

        self.fitness = None
        self.viable = True
        self.generated_by = "seed"
        self.parent_genome_generation_numbers: list[int] = []

        self.model = VisionTransformerMAE(
            num_parcels=num_parcels,
            window_length=window_length,
            parcel_coordinates=parcel_coordinates,
            **self.hyperparameters,
        )

    def __lt__(self, other: "VisionTransformerMAEGenome") -> bool:
        """Used to sort genomes by fitness (lower masked-reconstruction MSE is better)."""
        if self.fitness is None and other.fitness is None:
            return True
        elif self.fitness is None:
            return False
        elif other.fitness is None:
            return True
        else:
            return self.fitness < other.fitness

    def __repr__(self) -> str:
        return (
            f"VisionTransformerMAEGenome {self.generation_number}: fitness={self.fitness}, "
            f"hyperparameters={self.hyperparameters}"
        )

    def is_valid(self) -> bool:
        """Every hyperparameter combination this genome could be constructed with is valid by
        construction (VisionTransformerMAE.__init__ raises on invalid combinations), so there is
        no separate graph-validity check needed here."""
        return True

    def calculate_reachability(self):
        """No computational graph to walk; a ViT-MAE genome is always viable."""
        self.viable = True

    def plot(self, genome_name: str | None = None):
        """No graph to visualize with graphviz; just report the hyperparameters."""
        name = genome_name or f"vit_mae_genome_{self.generation_number}"
        print(f"{name}: fitness={self.fitness}, hyperparameters={self.hyperparameters}")

    def parameters(self) -> list[torch.Tensor]:
        return list(self.model.parameters())

    def reset(self):
        pass

    def encode_for_analysis(self, x: torch.Tensor, mask_ratio: float = 0.0) -> tuple[torch.Tensor, dict]:
        """Delegates to the underlying VisionTransformerMAE; see that method's docstring. Present
        so interpretability code can treat this genome and VisionTransformerBlockGenome uniformly."""
        return self.model.encode_for_analysis(x, mask_ratio=mask_ratio)

    def train(
        self,
        dataset: FMRIPatchDataset,
        optimizer: torch.optim.Optimizer,
        iterations: int,
        batch_size: int = 8,
        batches_per_iteration: int = 10,
    ):
        """Trains the genome's ViT-MAE using randomly windowed, randomly masked batches sampled
        from the given dataset.

        Args:
            dataset: source of random fMRI windows.
            optimizer: pytorch optimizer over self.parameters().
            iterations: number of training iterations to run (the last iteration only evaluates
                loss, mirroring RecurrentGenome.train's "don't backprop on the last iteration"
                convention so a genome's fitness reflects a clean forward pass).
            batch_size: number of windows per batch.
            batches_per_iteration: number of gradient steps per iteration.
        """
        self.model.train()

        loss = None
        for iteration in range(iterations + 1):
            for _ in range(batches_per_iteration):
                batch = dataset.sample_batch(batch_size=batch_size, window_length=self.window_length)

                loss, _, _, _ = self.model(batch)

                if iteration < iterations:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            print(f"iteration {iteration} loss: {loss.detach().item():.6f}")

        self.fitness = loss.detach().item()

        # clear gradients so the genome's tensors can be cheaply deepcopy'd by the Clone
        # reproduction method
        for parameter in self.model.parameters():
            parameter.grad = None

        print(f"final fitness (masked reconstruction MSE): {self.fitness}")
