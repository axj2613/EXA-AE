from __future__ import annotations

import torch
import torch.nn as nn

from genomes.transformer_model.encoder_layer import EncoderLayer
from genomes.transformer_model.positonal_encoding import PositionalEncoding


class PatchEmbed(nn.Module):
    """Projects flattened spatiotemporal fMRI patches into d_model token embeddings."""

    def __init__(self, patch_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


def patchify(x: torch.Tensor, parcel_patch_size: int, time_patch_size: int) -> torch.Tensor:
    """Splits a batch of (parcel x time) fMRI windows into flattened 2D patches, analogous
    to how a vision transformer patchifies an image into (height x width) patches.

    With the default parcel_patch_size=1 (matching BrainLM's public implementation), each patch
    is a single parcel's window of time_patch_size timesteps -- no parcels are combined, so all
    424 (or however many) original parcels remain distinct tokens. parcel_patch_size > 1 is
    supported (grouping multiple parcels' timeseries into one token, trading spatial resolution
    for fewer tokens) but is not part of BrainLM's published model.

    Args:
        x: tensor of shape (batch, num_parcels, window_length).
        parcel_patch_size: number of parcels per patch (spatial axis).
        time_patch_size: number of time steps per patch (temporal axis).

    Returns:
        A tensor of shape (batch, num_spatial_patches * num_temporal_patches, parcel_patch_size *
        time_patch_size). Patches are ordered spatial-major then temporal, i.e. token index =
        spatial_idx * num_temporal_patches + temporal_idx, which must match the ordering used to
        build spatial/temporal positional embeddings.
    """
    batch_size, num_parcels, window_length = x.shape
    assert num_parcels % parcel_patch_size == 0, (
        f"num_parcels ({num_parcels}) must be divisible by parcel_patch_size ({parcel_patch_size})"
    )
    assert window_length % time_patch_size == 0, (
        f"window_length ({window_length}) must be divisible by time_patch_size ({time_patch_size})"
    )

    num_spatial_patches = num_parcels // parcel_patch_size
    num_temporal_patches = window_length // time_patch_size

    x = x.view(batch_size, num_spatial_patches, parcel_patch_size, num_temporal_patches, time_patch_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    x = x.view(batch_size, num_spatial_patches * num_temporal_patches, parcel_patch_size * time_patch_size)
    return x


def unpatchify(
    patches: torch.Tensor, num_parcels: int, window_length: int, parcel_patch_size: int, time_patch_size: int
) -> torch.Tensor:
    """Inverts `patchify`, reassembling flattened 2D patches back into (parcel x time) windows.
    Used for visualizing/evaluating reconstructions rather than during training.
    """
    batch_size = patches.shape[0]
    num_spatial_patches = num_parcels // parcel_patch_size
    num_temporal_patches = window_length // time_patch_size

    x = patches.view(batch_size, num_spatial_patches, num_temporal_patches, parcel_patch_size, time_patch_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    x = x.view(batch_size, num_parcels, window_length)
    return x


def compute_patch_coordinates(parcel_coordinates: torch.Tensor, parcel_patch_size: int) -> torch.Tensor:
    """Reduces per-parcel 3D coordinates to per-spatial-patch coordinates (the centroid of the
    parcels grouped into each patch), matching the contiguous grouping `patchify` uses. With the
    default parcel_patch_size=1 this is the identity (each patch's coordinate is just its
    parcel's coordinate).

    Args:
        parcel_coordinates: shape (num_parcels, 3).
        parcel_patch_size: number of parcels per spatial patch.

    Returns:
        shape (num_parcels // parcel_patch_size, 3).
    """
    num_parcels = parcel_coordinates.shape[0]
    assert num_parcels % parcel_patch_size == 0, (
        f"num_parcels ({num_parcels}) must be divisible by parcel_patch_size ({parcel_patch_size})"
    )
    num_spatial_patches = num_parcels // parcel_patch_size
    grouped = parcel_coordinates.view(num_spatial_patches, parcel_patch_size, 3)
    return grouped.mean(dim=1)


def cls_attention_to_parcels(
    attention: torch.Tensor, num_spatial_patches: int, num_temporal_patches: int, has_cls: bool
) -> torch.Tensor:
    """Reduces a self-attention map to a per-parcel attention vector -- how much the CLS summary
    token attends to each parcel -- following BrainLM's functional-network / attention-map
    analyses (paper Section 4.4/4.5). Averages over heads, takes the CLS query row, drops the CLS
    key column, and averages each parcel's temporal patches.

    Args:
        attention: self-attention probabilities, shape (batch, num_heads, num_query, num_key),
            where token 0 is the CLS token (if has_cls) and the remaining tokens are the
            num_spatial_patches * num_temporal_patches patches in spatial-major order.
        has_cls: whether token 0 is the CLS token.

    Returns:
        Per-parcel attention, shape (batch, num_spatial_patches).
    """
    attention = attention.mean(dim=1)  # average heads -> (batch, num_query, num_key)

    if has_cls:
        cls_row = attention[:, 0, 1:]  # CLS query, drop CLS key -> (batch, num_patches)
    else:
        # no CLS token: use the mean query attention over all tokens instead
        cls_row = attention.mean(dim=1)  # (batch, num_key)

    batch_size = cls_row.shape[0]
    per_patch = cls_row.reshape(batch_size, num_spatial_patches, num_temporal_patches)
    return per_patch.mean(dim=2)  # aggregate temporal patches -> (batch, num_spatial_patches)


def add_positional_embeddings(
    x: torch.Tensor,
    patch_coordinates: torch.Tensor,
    spatial_projection: nn.Linear,
    temporal_encoding: PositionalEncoding,
    num_spatial_patches: int,
    num_temporal_patches: int,
) -> torch.Tensor:
    """Adds a coordinate-derived spatial embedding (shared across a spatial patch's temporal
    patches) and a sinusoidal temporal embedding (shared across spatial patches) to token
    embeddings, mirroring BrainLMEmbeddings.forward. Factored out as a standalone function (rather
    than only a VisionTransformerMAE method) so genomes.vision_transformer_block_genome's
    evolvable-graph genome can reuse the exact same fixed input/output scaffolding logic.

    Args:
        x: token embeddings, shape (batch, num_patches, d_model), in patchify's spatial-major
            token ordering.
        patch_coordinates: shape (num_spatial_patches, 3).
        spatial_projection: linear layer mapping (x, y, z) coordinates to d_model.
        temporal_encoding: sinusoidal positional encoding over the temporal-patch axis.
        num_spatial_patches: number of spatial patches (parcels, if parcel_patch_size=1).
        num_temporal_patches: number of temporal patches per spatial patch.

    Returns:
        x with positional embeddings added, same shape.
    """
    batch_size, num_patches, d_model = x.shape
    x = x.view(batch_size, num_spatial_patches, num_temporal_patches, d_model)

    spatial_embedding = spatial_projection(patch_coordinates)  # (num_spatial_patches, d_model)
    x = x + spatial_embedding.view(1, num_spatial_patches, 1, d_model)

    x = x.reshape(batch_size * num_spatial_patches, num_temporal_patches, d_model)
    x = temporal_encoding(x)

    return x.view(batch_size, num_patches, d_model)


def random_masking(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly drops mask_ratio of the tokens in x, per-sample. Factored out as a standalone
    function for reuse by the evolvable-graph genome as well as VisionTransformerMAE.

    Args:
        x: token embeddings of shape (batch, num_patches, d_model).
        mask_ratio: fraction of tokens to drop, in (0, 1).

    Returns:
        x_visible: the kept tokens, shape (batch, num_visible, d_model).
        mask: binary tensor of shape (batch, num_patches) in the ORIGINAL token order,
            1 for masked/dropped tokens, 0 for visible tokens.
        ids_restore: shape (batch, num_patches), the permutation needed to restore
            shuffled token order back to the original order.
    """
    batch_size, num_patches, d_model = x.shape
    len_keep = max(1, int(round(num_patches * (1 - mask_ratio))))

    noise = torch.rand(batch_size, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d_model))

    mask = torch.ones(batch_size, num_patches, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_visible, mask, ids_restore


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error, averaged only over masked patches (matches MAE/BrainLM: the model is
    never rewarded for copying visible tokens it was already given). Factored out as a standalone
    function for reuse by the evolvable-graph genome as well as VisionTransformerMAE.
    """
    per_patch_loss = ((pred - target) ** 2).mean(dim=-1)
    return (per_patch_loss * mask).sum() / mask.sum().clamp(min=1.0)


class VisionTransformerMAE(nn.Module):
    """A masked-autoencoder vision transformer for fMRI parcel time series, following BrainLM's
    published architecture (github.com/vandijklab/BrainLM, brainlm_mae/modeling_brainlm.py):
    raw (parcel x time) windows are split into per-parcel temporal patches, each patch token is
    the sum of a learned signal-value projection and a projection of the parcel's real 3D (x, y,
    z) atlas coordinate, plus a fixed sinusoidal temporal positional encoding. A random subset of
    patches is masked; an encoder processes only the visible patches (+ a CLS token), and a
    lightweight decoder (with its own, separate coordinate projection) reconstructs every patch
    from the encoded visible tokens plus learned mask tokens.

    BrainLM's public implementation uses Nystromformer (linear-attention) layers in the encoder/
    decoder to stay tractable at ~4240 tokens/window; this uses this repo's existing full
    quadratic-attention EncoderLayer instead, which is simpler and reuses existing code, but will
    be slower at BrainLM's real token counts -- worth revisiting if that becomes a bottleneck.
    """

    def __init__(
        self,
        num_parcels: int,
        window_length: int,
        parcel_coordinates: torch.Tensor,
        parcel_patch_size: int = 1,
        time_patch_size: int = 20,
        d_model: int = 128,
        num_heads: int = 4,
        encoder_depth: int = 4,
        d_ff: int = 256,
        decoder_d_model: int = 64,
        decoder_num_heads: int = 4,
        decoder_depth: int = 2,
        decoder_d_ff: int = 128,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()

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
        if decoder_d_model % decoder_num_heads != 0:
            raise ValueError(
                f"decoder_d_model ({decoder_d_model}) must be divisible by decoder_num_heads ({decoder_num_heads})"
            )
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

        self.register_buffer(
            "patch_coordinates", compute_patch_coordinates(parcel_coordinates, parcel_patch_size), persistent=True
        )

        self.patch_embed = PatchEmbed(self.patch_dim, d_model)
        # a single linear projection of each patch's real (x, y, z) atlas coordinate, matching
        # BrainLM's xyz_embedding_projection -- encodes true anatomical geometry rather than an
        # arbitrary token index, and generalizes across atlases/parcel orderings.
        self.spatial_pos_embed = nn.Linear(3, d_model, bias=True)
        self.temporal_pos_embed = PositionalEncoding(d_model=d_model, max_seq_length=self.num_temporal_patches)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # asymmetric, lighter-weight decoder (mirrors BrainLM/MAE's encoder/decoder split). Note
        # this reuses EncoderLayer (self-attention + FFN) rather than the repo's seq2seq
        # DecoderLayer (causal self-attn + cross-attn): MAE-style decoding has no separate
        # source/target sequence to cross-attend to -- the encoded visible tokens and the mask
        # tokens are reassembled into ONE sequence that self-attends to itself non-causally, so a
        # plain encoder-style block is the structurally correct choice here, exactly as in
        # BrainLM's own BrainLMDecoder.
        self.decoder_embed = nn.Linear(d_model, decoder_d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        # the decoder gets its own, separate coordinate and temporal positional embeddings
        # (BrainLM's BrainLMDecoder does the same, rather than reusing the encoder's)
        self.decoder_spatial_pos_embed = nn.Linear(3, decoder_d_model, bias=True)
        self.decoder_temporal_pos_embed = PositionalEncoding(
            d_model=decoder_d_model, max_seq_length=self.num_temporal_patches
        )
        self.decoder_layers = nn.ModuleList(
            [EncoderLayer(decoder_d_model, decoder_num_heads, decoder_d_ff, dropout) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_d_model)
        # 2-layer prediction head with LeakyReLU, matching BrainLM's decoder_pred1/nonlinearity/pred2
        self.decoder_pred1 = nn.Linear(decoder_d_model, decoder_d_model // 2)
        self.decoder_pred_nonlinearity = nn.LeakyReLU(0.1)
        self.decoder_pred2 = nn.Linear(decoder_d_model // 2, self.patch_dim)

    def _add_positional_embeddings(
        self, x: torch.Tensor, spatial_projection: nn.Linear, temporal_encoding: PositionalEncoding
    ) -> torch.Tensor:
        return add_positional_embeddings(
            x, self.patch_coordinates, spatial_projection, temporal_encoding,
            self.num_spatial_patches, self.num_temporal_patches,
        )

    def random_masking(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return random_masking(x, self.mask_ratio)

    def forward_encoder(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embeds and masks patches, then encodes only the visible tokens.

        Args:
            patches: shape (batch, num_patches, patch_dim).

        Returns:
            encoded: shape (batch, num_visible [+1 if use_cls_token], d_model).
            mask: shape (batch, num_patches), 1 for masked tokens (original order).
            ids_restore: shape (batch, num_patches).
        """
        x = self.patch_embed(patches)
        x = self._add_positional_embeddings(x, self.spatial_pos_embed, self.temporal_pos_embed)

        x, mask, ids_restore = self.random_masking(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        for layer in self.encoder_layers:
            x = layer(x, None)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, encoded: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Reconstructs every patch (visible and masked) from the encoded visible tokens.

        Args:
            encoded: output of forward_encoder, shape (batch, num_visible [+cls], d_model).
            ids_restore: output of forward_encoder / random_masking.

        Returns:
            Reconstructed patches, shape (batch, num_patches, patch_dim).
        """
        x = self.decoder_embed(encoded)

        if self.use_cls_token:
            cls_token = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            cls_token = None

        batch_size, num_visible, d_model = x.shape
        num_patches = ids_restore.shape[1]
        num_masked = num_patches - num_visible
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)

        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d_model))

        x_full = self._add_positional_embeddings(
            x_full, self.decoder_spatial_pos_embed, self.decoder_temporal_pos_embed
        )

        if cls_token is not None:
            x_full = torch.cat([cls_token, x_full], dim=1)

        for layer in self.decoder_layers:
            x_full = layer(x_full, None)
        x_full = self.decoder_norm(x_full)

        pred = self.decoder_pred2(self.decoder_pred_nonlinearity(self.decoder_pred1(x_full)))
        if cls_token is not None:
            pred = pred[:, 1:, :]

        return pred

    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return reconstruction_loss(pred, target, mask)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the full masked-autoencoder forward pass.

        Args:
            x: raw fMRI window, shape (batch, num_parcels, window_length).

        Returns:
            loss: scalar masked-patch reconstruction MSE.
            pred: reconstructed patches, shape (batch, num_patches, patch_dim).
            mask: shape (batch, num_patches), 1 for masked tokens.
            patches: the ground-truth patchified input, shape (batch, num_patches, patch_dim).
        """
        patches = patchify(x, self.parcel_patch_size, self.time_patch_size)
        encoded, mask, ids_restore = self.forward_encoder(patches)
        pred = self.forward_decoder(encoded, ids_restore)
        loss = self.reconstruction_loss(pred, patches, mask)
        return loss, pred, mask, patches

    def encode_for_analysis(self, x: torch.Tensor, mask_ratio: float = 0.0) -> tuple[torch.Tensor, dict]:
        """Deterministic (eval-mode, no-grad) encoder pass for interpretability, mirroring
        VisionTransformerBlockGenome.encode_for_analysis so the same analysis code works on both
        the fixed-topology and evolved models. By default mask_ratio=0 so every parcel token is
        visible and gets an attention value.

        Args:
            x: raw fMRI window, shape (batch, num_parcels, window_length).
            mask_ratio: masking during the analysis pass (0.0 = no masking).

        Returns:
            latent: the CLS summary embedding (encoder output token 0), shape (batch, d_model);
                or the token mean if use_cls_token is False.
            attention_maps: dict keyed by encoder layer index, each value a dict with "depth"
                (fractional layer position), "region" ("encoder"), and "attention" (that layer's
                self-attention probabilities, shape (batch, num_heads, num_query, num_key)).
        """
        saved_mask_ratio = self.mask_ratio
        self.mask_ratio = mask_ratio
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                patches = patchify(x, self.parcel_patch_size, self.time_patch_size)
                h = self.patch_embed(patches)
                h = self._add_positional_embeddings(h, self.spatial_pos_embed, self.temporal_pos_embed)
                h, _, _ = self.random_masking(h)
                if self.use_cls_token:
                    cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)
                    h = torch.cat([cls_tokens, h], dim=1)

                attention_maps = {}
                num_layers = len(self.encoder_layers)
                for i, layer in enumerate(self.encoder_layers):
                    h, attn_probs = layer(h, None, output_attentions=True)
                    attention_maps[i] = {
                        "depth": (i + 1) / num_layers,
                        "region": "encoder",
                        "attention": attn_probs.detach(),
                    }
                h = self.encoder_norm(h)

                latent = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        finally:
            self.mask_ratio = saved_mask_ratio
            self.train(was_training)

        return latent, attention_maps
