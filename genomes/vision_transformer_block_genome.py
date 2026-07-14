from __future__ import annotations

import math

import torch
import torch.nn as nn

from genomes.genome import Genome
from genomes.nodes.block_input_node import BlockInputNode
from genomes.nodes.block_output_node import BlockOutputNode
from genomes.nodes.simple_block_node import SimpleBlockNode
from genomes.nodes.attention_block_node import AttentionBlockNode
from genomes.edges.block_edge import BlockEdge
from genomes.transformer_model.positonal_encoding import PositionalEncoding
from genomes.transformer_model.vision_transformer_mae import (
    PatchEmbed,
    add_positional_embeddings,
    compute_patch_coordinates,
    patchify,
    random_masking,
    reconstruction_loss,
)

from innovation.innovation_generator import InnovationGenerator

from time_series.fmri_patch_dataset import FMRIPatchDataset

from weight_generators.lamarckian_block_weight_generator import LamarckianBlockWeightGenerator
from weight_generators.weight_generator import WeightGenerator


class VisionTransformerBlockGenome(Genome):
    """A masked-autoencoder vision transformer for fMRI parcel time series -- same patchify/
    masking/positional-embedding scheme as
    genomes.transformer_model.vision_transformer_mae.VisionTransformerMAE (see that class's
    docstring for the BrainLM background) -- except the encoder AND decoder internals are an
    EVOLVABLE node/edge graph (of AttentionBlockNode / SequenceLSTMBlockNode /
    TemporalLSTMBlockNode / SimpleBlockNode instances) instead of a fixed stack of encoder
    layers, so EXAMM-style mutation/crossover (see
    evolution/vision_transformer_block_reproduction_selector.py) can evolve which block types
    compose the network and how they're connected -- mirroring how the original EXAMM paper mixes
    LSTM/GRU/simple-neuron cell types within one evolved RNN, but at token-block granularity.

    Fixed (non-evolved) scaffolding: patch embedding, coordinate + temporal positional
    embeddings, random masking, the mask-token-insertion/unshuffle step that transitions from
    encoder token-space to decoder token-space, and the final patch-reconstruction head -- exactly
    the same fixed pieces VisionTransformerMAE has, just wired around an evolvable graph instead
    of a fixed nn.ModuleList. All nodes in the evolvable region (encoder-side and decoder-side
    alike) share a single d_model -- unlike VisionTransformerMAE's asymmetric encoder/decoder
    widths, this avoids needing per-edge dimensionality-changing projections when a mutation
    connects nodes of different depths, at the cost of losing that "lighter decoder" efficiency
    trick. num_heads/d_ff/dropout are likewise fixed genome-level hyperparameters shared by every
    AttentionBlockNode instance, exactly as EXAMM's scalar LSTM cells are always hidden_size=1:
    what evolves is graph TOPOLOGY (which blocks, how many, how connected), not each block's
    internal capacity.

    Seed topology mirrors genomes.autoencoder_genome.AutoencoderGenome: a single input node
    (depth 0.0) connected to a single bottleneck node (depth 0.5) connected to a single output
    node (depth 1.0). Passing autoencoder=True to the reproduction operators (see the driver
    script) is what keeps this bottleneck node structurally protected from Split/Merge/Disable --
    SplitNode/DisableNode explicitly exclude `node.depth == 0.5` and MergeNode's autoencoder-mode
    depth ranges are open intervals that exclude 0.5 too, so this genome relies on that existing,
    unmodified behavior to guarantee there is always exactly one node at depth 0.5, which is what
    VisionTransformerBlockGenome.forward uses to locate the fixed encoder/decoder transition.
    """

    def __init__(
        self,
        generation_number: int,
        num_parcels: int,
        window_length: int,
        parcel_coordinates: torch.Tensor,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        parcel_patch_size: int = 1,
        time_patch_size: int = 20,
        mask_ratio: float = 0.75,
        use_cls_token: bool = True,
        weight_generator: WeightGenerator = LamarckianBlockWeightGenerator(),
    ):
        super().__init__(generation_number)

        if num_parcels % parcel_patch_size != 0:
            raise ValueError(
                f"num_parcels ({num_parcels}) must be divisible by parcel_patch_size ({parcel_patch_size})"
            )
        if window_length % time_patch_size != 0:
            raise ValueError(
                f"window_length ({window_length}) must be divisible by time_patch_size ({time_patch_size})"
            )
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio ({mask_ratio}) must be between 0 and 1, exclusive")
        if parcel_coordinates.shape != (num_parcels, 3):
            raise ValueError(
                f"parcel_coordinates must have shape ({num_parcels}, 3), got {tuple(parcel_coordinates.shape)}"
            )

        self.num_parcels = num_parcels
        self.window_length = window_length
        self.parcel_patch_size = parcel_patch_size
        self.time_patch_size = time_patch_size
        self.num_spatial_patches = num_parcels // parcel_patch_size
        self.num_temporal_patches = window_length // time_patch_size
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches
        self.patch_dim = parcel_patch_size * time_patch_size
        self.mask_ratio = mask_ratio
        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # interpretability capture state, toggled by encode_for_analysis (off during training)
        self.capture_latent = False
        self.latent: torch.Tensor | None = None

        # size/complexity report, populated by train() (see parameter_report)
        self.complexity: dict | None = None

        # device the genome's tensors currently live on; updated by to() and used by forward/train
        # to place freshly-created tensors (masking noise, zero-fills, moved batches) correctly.
        self.device = torch.device("cpu")

        self.patch_coordinates = compute_patch_coordinates(parcel_coordinates, parcel_patch_size)

        # --- fixed (non-evolved) input-side scaffolding ---
        self.patch_embed = PatchEmbed(self.patch_dim, d_model)
        self.spatial_pos_embed = nn.Linear(3, d_model, bias=True)
        self.temporal_pos_embed = PositionalEncoding(d_model=d_model, max_seq_length=self.num_temporal_patches)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # --- fixed (non-evolved) encoder -> decoder transition ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_spatial_pos_embed = nn.Linear(3, d_model, bias=True)
        self.decoder_temporal_pos_embed = PositionalEncoding(d_model=d_model, max_seq_length=self.num_temporal_patches)

        # --- fixed (non-evolved) output-side scaffolding ---
        self.decoder_pred1 = nn.Linear(d_model, d_model // 2)
        self.decoder_pred_nonlinearity = nn.LeakyReLU(0.1)
        self.decoder_pred2 = nn.Linear(d_model // 2, self.patch_dim)

        # --- evolvable graph seed topology: input (0.0) -> bottleneck (0.5) -> output (1.0) ---
        self.encoder_input_node = BlockInputNode(
            innovation_number=InnovationGenerator.get_innovation_number(), depth=0.0, d_model=d_model
        )
        self.add_input_node(self.encoder_input_node)

        self.bottleneck_node = SimpleBlockNode(
            innovation_number=InnovationGenerator.get_innovation_number(), depth=0.5, d_model=d_model
        )
        self.add_node(self.bottleneck_node)

        # A decoder-side self-attention block is seeded at depth 0.75 (bottleneck -> decoder_attn
        # -> output). Masked reconstruction is IMPOSSIBLE without at least one token-mixing block
        # in the decoder region: a masked position enters the decoder as (mask_token + positional
        # embedding) -- identical across samples -- and can only acquire sample-specific content by
        # ATTENDING to the visible tokens (exactly BrainLM's fixed attention decoder, modeling_
        # brainlm.py BrainLMDecoder.decoder_layers). Without it every genome collapses to predicting
        # the per-position mean, so the whole search has no fitness signal to select on. It is seeded
        # into the evolvable graph (not warm-started, so it is functional from step 0) and evolution
        # refines/adds around it.
        self.decoder_attention_node = AttentionBlockNode(
            innovation_number=InnovationGenerator.get_innovation_number(), depth=0.75, d_model=d_model,
            num_heads=num_heads, d_ff=d_ff, dropout=dropout, warm_start=False,
        )
        self.add_node(self.decoder_attention_node)

        self.decoder_output_node = BlockOutputNode(
            innovation_number=InnovationGenerator.get_innovation_number(), depth=1.0, d_model=d_model
        )
        self.add_output_node(self.decoder_output_node)

        self.add_edge(BlockEdge(
            innovation_number=InnovationGenerator.get_innovation_number(),
            input_node=self.encoder_input_node,
            output_node=self.bottleneck_node,
        ))
        self.add_edge(BlockEdge(
            innovation_number=InnovationGenerator.get_innovation_number(),
            input_node=self.bottleneck_node,
            output_node=self.decoder_attention_node,
        ))
        self.add_edge(BlockEdge(
            innovation_number=InnovationGenerator.get_innovation_number(),
            input_node=self.decoder_attention_node,
            output_node=self.decoder_output_node,
        ))

        weight_generator(self)

    def parameters(self) -> list[torch.Tensor]:
        """Extends Genome.parameters() (which only walks node/edge .weights) with the fixed
        scaffolding modules' parameters, which live outside the evolvable node/edge graph."""
        parameters = super().parameters()

        fixed_modules = [
            self.patch_embed, self.spatial_pos_embed, self.temporal_pos_embed,
            self.decoder_spatial_pos_embed, self.decoder_temporal_pos_embed,
            self.decoder_pred1, self.decoder_pred_nonlinearity, self.decoder_pred2,
        ]
        for module in fixed_modules:
            parameters.extend(module.parameters())

        parameters.append(self.mask_token)
        if self.use_cls_token:
            parameters.append(self.cls_token)

        return parameters

    def _fixed_scaffolding_parameters(self) -> int:
        """Trainable parameter count of the FIXED (non-evolved) scaffolding -- constant across all
        genomes (patch embed, spatial-coordinate projections, decoder head, cls/mask tokens). The
        sinusoidal temporal encodings are buffers, so they contribute none."""
        fixed_modules = [
            self.patch_embed, self.spatial_pos_embed,
            self.decoder_spatial_pos_embed, self.decoder_pred1, self.decoder_pred2,
        ]
        total = sum(p.numel() for module in fixed_modules for p in module.parameters())
        total += self.mask_token.numel()
        if self.use_cls_token:
            total += self.cls_token.numel()
        return total

    def parameter_report(self) -> dict:
        """Reports the genome's size/complexity for the EXAMM-style efficiency analysis (the point
        of neuro-evolution is finding compact architectures). Counts only the ACTIVE graph -- the
        nodes/edges actually reachable in the forward pass -- since disabled/unreachable structure
        is dead weight that contributes nothing to the model's function; call
        calculate_reachability() first (the population/reproduction machinery already does).

        Returns a dict with:
            total_active_parameters: fixed scaffolding + active evolved graph (the functional model
                size to compare against, e.g., BrainLM's 111M/650M).
            evolved_active_parameters: just the active block nodes + edges.
            fixed_parameters: the constant scaffolding.
            num_active_hidden_nodes / num_active_edges: EXAMM-style structure counts.
            node_type_counts: active hidden-node count per block type.
        """
        fixed = self._fixed_scaffolding_parameters()

        evolved_active = 0
        num_active_nodes = 0
        node_type_counts: dict[str, int] = {}
        for node in self.nodes:
            if getattr(node, "active", False) and not node.is_boundary_node:
                evolved_active += sum(weight.numel() for weight in node.weights)
                num_active_nodes += 1
                type_name = type(node).__name__
                node_type_counts[type_name] = node_type_counts.get(type_name, 0) + 1

        num_active_edges = 0
        for edge in self.edges:
            if getattr(edge, "active", False):
                evolved_active += sum(weight.numel() for weight in edge.weights if weight is not None)
                num_active_edges += 1

        return {
            "total_active_parameters": fixed + evolved_active,
            "evolved_active_parameters": evolved_active,
            "fixed_parameters": fixed,
            "num_active_hidden_nodes": num_active_nodes,
            "num_active_edges": num_active_edges,
            "node_type_counts": node_type_counts,
        }

    def to(self, device) -> "VisionTransformerBlockGenome":
        """Moves every tensor the genome owns onto `device`: the fixed scaffolding modules and
        their buffers (via _iter_modules), the cls/mask Parameters, the patch_coordinates buffer,
        and each edge's scalar gate. Block-node weights live inside nn.Modules and move with them
        (the Parameter objects keep their identity, so node.weights stays aliased to the modules).

        Call this each generation AFTER the reproduction operators run and BEFORE building the
        optimizer: mutations create new nodes/edges on the CPU, so the child genome is a mix of
        device tensors (deepcopied from the parent) and CPU tensors (freshly generated) until this
        re-homes everything. Moving already-on-device tensors is a cheap no-op.
        """
        device = torch.device(device)
        self.device = device

        for module in self._iter_modules():
            module.to(device)

        self.mask_token.data = self.mask_token.data.to(device)
        if self.use_cls_token:
            self.cls_token.data = self.cls_token.data.to(device)

        self.patch_coordinates = self.patch_coordinates.to(device)

        for edge in self.edges:
            if edge.weights[0] is not None:
                # .to() on a leaf that requires grad produces a non-leaf; detach/re-flag to keep
                # it a trainable leaf parameter.
                edge.weights[0] = edge.weights[0].detach().to(device).requires_grad_(True)

        return self

    def _prepare_decoder_input(self, encoded: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Fixed (non-evolved) transition from encoder token-space (visible tokens only, +cls) to
        decoder token-space (all patches, mask tokens inserted): inserts learned mask tokens at
        the masked positions, un-shuffles back to the original patch order, and adds the
        decoder's own spatial/temporal positional embeddings. Mirrors
        VisionTransformerMAE.forward_decoder's glue logic exactly, just relocated to run at the
        depth=0.5 bottleneck node, between the encoder-region and decoder-region halves of the
        evolvable graph (see forward).
        """
        if self.use_cls_token:
            cls_token = encoded[:, :1, :]
            x = encoded[:, 1:, :]
        else:
            cls_token = None

        batch_size, num_visible, d_model = x.shape
        num_masked = self.num_patches - num_visible
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)

        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d_model))

        x_full = add_positional_embeddings(
            x_full, self.patch_coordinates, self.decoder_spatial_pos_embed, self.decoder_temporal_pos_embed,
            self.num_spatial_patches, self.num_temporal_patches,
        )

        if cls_token is not None:
            x_full = torch.cat([cls_token, x_full], dim=1)

        return x_full

    def _forward_graph(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Walks the evolvable node/edge graph in topological (depth) order once, evaluating
        each active node's forward() on the sum of its active input edges' outputs -- a plain
        feedforward graph evaluation, not a per-timestep interpreter, since every node consumes/
        produces a whole (batch, num_tokens, d_model) tensor per call.

        The bottleneck node is special-cased: immediately after it computes its own output
        (still at "visible token" length), _prepare_decoder_input runs, transitioning the
        sequence into decoder token-space before the walk continues into decoder-side nodes.

        Reproduction operators reused from the scalar graph (AddNode, MergeNode, ...) can create
        "recurrent" edges whose source node is DEEPER than its target, or whose source (via
        chains of Split/MergeNode across generations) ends up on the other side of the depth=0.5
        bottleneck -- in the scalar, per-timestep interpreter backward edges are resolved via
        time_skip (the source's value from an earlier timestep feeds the target at the current
        one); this graph has no such timestep-deferral mechanism (see BlockEdge's docstring), and
        a cross-bottleneck edge's source is at a different token-sequence length entirely (visible
        tokens pre-bottleneck vs. all tokens post-bottleneck). Rather than track this precisely
        through arbitrarily many generations of mutation, each node's input edges are simply
        filtered to those whose source has ALREADY been computed AND is at the token-sequence
        length this node's own depth region expects (< 0.5: visible-token length, matching x;
        > 0.5: full patch [+cls] length, computed analytically from self.num_patches rather than
        read off the bottleneck node, so this doesn't itself depend on the bottleneck having run
        yet); non-matching edges contribute nothing on this pass rather than crashing.
        """
        batch_size = x.shape[0]
        decoder_token_count = self.num_patches + (1 if self.use_cls_token else 0)
        decoder_shape = torch.Size([batch_size, decoder_token_count, self.d_model])

        for node in sorted(self.nodes):
            if not node.active:
                continue

            if node is self.encoder_input_node:
                node.value = x
                continue

            # The bottleneck sits at depth EXACTLY 0.5 and still consumes ENCODER-space visible
            # tokens (its OUTPUT is what _prepare_decoder_input transitions into decoder space).
            # A bare `node.depth < 0.5` excludes it, so its expected_shape became decoder_shape,
            # its (visible-shape) input edge was filtered out by the shape check below, and it
            # received an all-zeros node_input -- starving the ENTIRE decoder of any sample-specific
            # information and collapsing the model to predicting the per-position mean (R^2 ~ 0).
            in_encoder_region = node.depth < 0.5 or node is self.bottleneck_node
            expected_shape = x.shape if in_encoder_region else decoder_shape

            node_input = None
            for edge in node.input_edges:
                if edge.active and edge.input_node.value is not None and edge.input_node.value.shape == expected_shape:
                    edge_value = edge.forward(edge.input_node.value)
                    node_input = edge_value if node_input is None else node_input + edge_value

            if node_input is None:
                node_input = torch.zeros(expected_shape, dtype=x.dtype, device=x.device)

            context = {
                "num_spatial": self.num_spatial_patches,
                "num_temporal": self.num_temporal_patches,
                "has_cls": self.use_cls_token,
                # the bottleneck node (depth 0.5) still operates on encoder-space visible tokens;
                # only depth > 0.5 nodes see the restored full grid (see _prepare_decoder_input).
                "region": "decoder" if node.depth > 0.5 else "encoder",
            }
            node.value = node.forward(node_input, context)

            if node is self.bottleneck_node:
                if self.capture_latent:
                    # the bottleneck's output, in encoder token-space and before the decoder
                    # transition, is the network's most compressed representation; its CLS row
                    # (or the token mean, without a CLS token) is the recording-level summary
                    # embedding -- the block-graph analog of BrainLM's encoder CLS latent, used
                    # for latent-space analysis / clinical-variable prediction.
                    if self.use_cls_token:
                        self.latent = node.value[:, 0, :].detach()
                    else:
                        self.latent = node.value.mean(dim=1).detach()
                node.value = self._prepare_decoder_input(node.value, ids_restore)

        return self.decoder_output_node.value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the full masked-autoencoder forward pass through the evolvable graph.

        Args:
            x: raw fMRI window, shape (batch, num_parcels, window_length).

        Returns:
            loss: scalar masked-patch reconstruction MSE.
            pred: reconstructed patches, shape (batch, num_patches, patch_dim).
            mask: shape (batch, num_patches), 1 for masked tokens.
            patches: the ground-truth patchified input, shape (batch, num_patches, patch_dim).
        """
        patches = patchify(x, self.parcel_patch_size, self.time_patch_size)
        embedded = self.patch_embed(patches)
        embedded = add_positional_embeddings(
            embedded, self.patch_coordinates, self.spatial_pos_embed, self.temporal_pos_embed,
            self.num_spatial_patches, self.num_temporal_patches,
        )

        visible, mask, ids_restore = random_masking(embedded, self.mask_ratio)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(visible.shape[0], -1, -1)
            visible = torch.cat([cls_tokens, visible], dim=1)

        decoder_output = self._forward_graph(visible, ids_restore)

        if self.use_cls_token:
            decoder_output = decoder_output[:, 1:, :]

        pred = self.decoder_pred2(self.decoder_pred_nonlinearity(self.decoder_pred1(decoder_output)))
        loss = reconstruction_loss(pred, patches, mask)

        return loss, pred, mask, patches

    def train(
        self,
        dataset,
        optimizer: torch.optim.Optimizer,
        iterations: int,
        batch_size: int = 8,
        batches_per_iteration: int = 10,
        fitness_batches: int = 4,
        use_amp: bool = False,
    ):
        """Trains this genome on randomly windowed, randomly masked batches from the dataset's
        TRAIN split, then sets fitness to the mean masked-reconstruction loss over the dataset's
        held-out VALIDATION split -- so evolution selects for generalization, not for fitting the
        last training batch (see the module docstring's 3-way subject split rationale).

        Args:
            dataset: provides sample_batch(batch_size, window_length, split); HCPWindowDataset for
                real subject-level splits, FMRIPatchDataset (which ignores split) for small tests.
            optimizer: over self.parameters() (build it AFTER genome.to(device)).
            iterations, batches_per_iteration: gradient-step schedule.
            fitness_batches: number of validation batches averaged for the fitness score.
            use_amp: enable CUDA mixed precision (autocast + GradScaler); ignored on CPU.
        """
        device = self.device
        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

        for module in self._iter_modules():
            module.train()

        diverged = False
        for iteration in range(iterations):
            last_loss = None
            for _ in range(batches_per_iteration):
                # reset every batch (not just once): a node with a self-loop or other
                # "recurrent"-flavored edge (see BlockEdge's docstring) would otherwise still hold
                # its .value from the PREVIOUS batch's already-freed graph, crashing autograd with
                # "trying to backward through the graph a second time".
                self.reset()
                batch = dataset.sample_batch(
                    batch_size=batch_size, window_length=self.window_length, split="train"
                ).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss, _, _, _ = self.forward(batch)
                    scaler.scale(loss).backward()
                    # unscale before clipping so max_norm is measured in true (not fp16-scaled)
                    # gradient units, matching the non-amp branch below.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, _, _, _ = self.forward(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                last_loss = loss.detach().item()

                # evolved graphs have unbounded depth (mutation can stack many blocks), so under
                # fp16 autocast a bad topology/init can still blow up despite clipping. Treat
                # divergence like the OOM case in parallel_training.py: penalize and bail rather
                # than keep training (and eventually validating) on NaN weights, which would give
                # this genome a NaN fitness -- and NaN comparisons are always False, so Genome.__lt__
                # can't reliably sort it out of the population, letting corrupt weights survive into
                # crossover.
                if not math.isfinite(last_loss):
                    diverged = True
                    break
            if diverged:
                break
            print(f"iteration {iteration} train loss: {last_loss:.6f}")

        if diverged:
            print(f"DIVERGED: genome {self.generation_number} produced a non-finite loss "
                  f"({last_loss}) -- penalized with infinite fitness")
            self.fitness = float("inf")
            self.complexity = self.parameter_report()
            self.reset()
            for parameter in self.parameters():
                parameter.grad = None
            return

        self.fitness = self._validation_loss(dataset, batch_size, fitness_batches, amp_enabled)

        # measure size/complexity for the EXAMM-style efficiency analysis and stash it on the
        # genome so it is saved with the pickle (fitness selection itself stays pure val MSE).
        self.complexity = self.parameter_report()

        # clear gradients/values so the genome's tensors can be cheaply deepcopy'd by mutation/Clone
        self.reset()
        for parameter in self.parameters():
            parameter.grad = None

        print(
            f"final fitness (validation MSE): {self.fitness:.6f} | "
            f"active params: {self.complexity['total_active_parameters']:,} "
            f"({self.complexity['evolved_active_parameters']:,} evolved) | "
            f"active hidden nodes: {self.complexity['num_active_hidden_nodes']}, "
            f"edges: {self.complexity['num_active_edges']} | "
            f"types: {self.complexity['node_type_counts']}"
        )

    def _validation_loss(self, dataset, batch_size, fitness_batches, amp_enabled) -> float:
        """Mean masked-reconstruction loss over validation-split batches, in eval mode (dropout
        off) with no gradient -- the genome's fitness signal."""
        for module in self._iter_modules():
            module.eval()

        total = 0.0
        try:
            with torch.no_grad():
                for _ in range(fitness_batches):
                    self.reset()
                    batch = dataset.sample_batch(
                        batch_size=batch_size, window_length=self.window_length, split="val"
                    ).to(self.device, non_blocking=True)
                    if amp_enabled:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss, _, _, _ = self.forward(batch)
                    else:
                        loss, _, _, _ = self.forward(batch)
                    loss_value = loss.item()
                    # a genome can finish TRAINING with a finite last-batch loss (so train()'s own
                    # divergence guard never fires) and still overflow fp16 on a particular
                    # VALIDATION window it never saw during training -- e.g. large-but-finite
                    # weights push an attention logit past fp16's ~65504 range on an outlier input.
                    # One non-finite batch would otherwise silently poison the whole averaged
                    # fitness (a single NaN/inf in the running sum makes the mean NaN/inf too), so
                    # bail immediately rather than average over it.
                    if not math.isfinite(loss_value):
                        print(f"DIVERGED: genome {self.generation_number} produced a non-finite "
                              f"validation loss ({loss_value}) -- penalized with infinite fitness")
                        return float("inf")
                    total += loss_value
        finally:
            for module in self._iter_modules():
                module.train()

        return total / max(1, fitness_batches)

    def _iter_modules(self):
        """Yields every nn.Module in the genome -- each block node's internal module(s) plus the
        fixed input/decoder scaffolding -- so train/eval mode (e.g. for deterministic,
        dropout-free interpretability passes) can be toggled uniformly."""
        for node in self.nodes:
            for value in vars(node).values():
                if isinstance(value, nn.Module):
                    yield value
        for value in vars(self).values():
            if isinstance(value, nn.Module):
                yield value

    def encode_for_analysis(
        self, x: torch.Tensor, mask_ratio: float = 0.0
    ) -> tuple[torch.Tensor, dict]:
        """Runs a deterministic (eval-mode, no-grad) forward pass for interpretability, following
        BrainLM's attention/latent analyses (paper Sections 4.2/4.4/4.5). By default mask_ratio=0
        so every parcel token is visible and therefore gets an attention value and contributes to
        the latent.

        Args:
            x: raw fMRI window, shape (batch, num_parcels, window_length).
            mask_ratio: masking to apply during the analysis pass (0.0 = no masking).

        Returns:
            latent: the recording-level CLS summary embedding, shape (batch, d_model).
            attention_maps: dict keyed by each AttentionBlockNode's innovation number, each value
                a dict with "depth", "region" ("encoder"/"decoder"), and "attention" (the block's
                self-attention probabilities, shape (batch, num_heads, num_query, num_key)).
        """
        from genomes.nodes.attention_block_node import AttentionBlockNode

        saved_mask_ratio = self.mask_ratio
        self.mask_ratio = mask_ratio
        self.capture_latent = True

        attention_nodes = [n for n in self.nodes if isinstance(n, AttentionBlockNode) and n.active]
        for node in attention_nodes:
            node.capture_attention = True
            node.last_attention = None

        for module in self._iter_modules():
            module.eval()

        try:
            with torch.no_grad():
                self.reset()
                self.forward(x)
                latent = self.latent
                attention_maps = {
                    node.innovation_number: {
                        "depth": node.depth,
                        "region": "decoder" if node.depth > 0.5 else "encoder",
                        "attention": node.last_attention,
                    }
                    for node in attention_nodes
                    if node.last_attention is not None
                }
        finally:
            self.mask_ratio = saved_mask_ratio
            self.capture_latent = False
            for node in attention_nodes:
                node.capture_attention = False
            for module in self._iter_modules():
                module.train()
            self.reset()

        return latent, attention_maps
