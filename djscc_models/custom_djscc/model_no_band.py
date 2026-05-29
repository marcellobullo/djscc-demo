"""DJSCC model variants with per-packet CSI conditioning ("no-band" architecture).

Adds DJSCCDecoderSpatialCSI: a decoder that consumes a per-element CSI map
(same spatial shape as the received latent), enabling spatially-varying
reliability information. The CSI map is built from a per-packet SNR vector
via the packet -> element layout that mirrors codec_no_band.py's
flatten / serialization order.

Encoder is unchanged from the original; we re-export DJSCCEncoder for
convenience so callers only need this module.

Background: horizontal banding in OTA reconstructions comes from raster
mapping (packet index <-> image row) combined with the original decoder
seeing only a scalar SNR. With this decoder, the FiLM still receives a
scalar (mean-SNR global summary) while the head conv also sees a per-element
SNR map -> the decoder learns to spatially down-weight unreliable strips.
"""

from typing import Any, Optional

import torch
from torch import nn

from kaira.constraints.base import BaseConstraint
from kaira.constraints.power import AveragePowerConstraint
from kaira.models.base import BaseModel, ChannelAwareBaseModel

# Re-use the building blocks from the original model module.
# Try both relative and absolute import so this works whether the file is
# imported as `custom_djscc.model_no_band` (package context, e.g. from the
# codec wrappers / TX / RX) or as bare `model_no_band` (script context, e.g.
# when running `python train_no_band.py` from inside this folder).
try:
    from .model import (  # type: ignore
        ConvNeXtBlock,
        Downsample2x,
        FiLMLayer,
        LayerNorm2d,
        Upsample2x,
        DJSCCEncoder,
    )
except ImportError:  # script context
    from model import (  # type: ignore
        ConvNeXtBlock,
        Downsample2x,
        FiLMLayer,
        LayerNorm2d,
        Upsample2x,
        DJSCCEncoder,
    )


__all__ = [
    "packetwise_to_element_map",
    "DJSCCEncoder",
    "DJSCCDecoderSpatialCSI",
    "DJSCCModelNoBand",
]


# ---------------------------------------------------------------------------
# Packet -> element layout helper
# ---------------------------------------------------------------------------

def packetwise_to_element_map(
    per_packet: torch.Tensor,
    M: int,
    H: int,
    W: int,
    pkt_len_complex: int,
) -> torch.Tensor:
    """Expand a per-packet vector to a per-element tensor using the same
    layout that codec_no_band.py uses to serialize the latent.

    Layout
    ------
    The latent at the channel input has shape [B, M, H, W] (M even). The
    runtime codec flattens it in (C, H, W) order, then reads the first
    half as real parts and the second half as imag parts of complex
    symbols. Concretely, complex symbol k packs:

        real part = flat[k]            <-> latent channel  k // (H*W),  hw = k % (H*W)
        imag part = flat[k + half]     <-> latent channel  M/2 + k//(H*W),  same hw

    Packet p covers complex symbols [p*pkt_len_complex, (p+1)*pkt_len_complex).
    Both the real-half element and the imag-half element of every complex
    symbol in packet p therefore share packet p's CSI / noise.

    Args:
      per_packet:        [B, n_pkts] tensor (units depend on caller: dB, sigma, mask, ...)
      M, H, W:           latent channel / spatial dims
      pkt_len_complex:   complex symbols per packet (e.g. 960)

    Returns:
      [B, M, H, W] tensor; element (b, c, h, w) carries the value of the
      packet that transmits the corresponding complex symbol.
    """
    B, n_pkts = per_packet.shape
    if M % 2 != 0:
        raise ValueError(f"M must be even (got {M})")
    half_M = M // 2
    half_size = half_M * H * W

    # Repeat each packet's value pkt_len_complex times -> per-symbol vector.
    expanded = per_packet.repeat_interleave(pkt_len_complex, dim=1)  # [B, n_pkts*pkt_len]

    # Trim / pad to exactly half_size. (Runtime pads the last packet's tail
    # with zeros which never reach the model; here we just clip to data.)
    if expanded.shape[1] >= half_size:
        real_half = expanded[:, :half_size]
    else:
        pad = expanded[:, -1:].expand(B, half_size - expanded.shape[1])
        real_half = torch.cat([expanded, pad], dim=1)

    real_half_map = real_half.reshape(B, half_M, H, W)
    # Imag-half element shares packet (and CSI) with its real-half partner.
    return torch.cat([real_half_map, real_half_map], dim=1)  # [B, M, H, W]


# ---------------------------------------------------------------------------
# Decoder with per-element CSI conditioning
# ---------------------------------------------------------------------------

class DJSCCDecoderSpatialCSI(ChannelAwareBaseModel):
    """CSI-adaptive decoder that consumes a per-element reliability map.

    Differences vs. the original DJSCCDecoder:
      - Head conv accepts (received_latent || csi_map_db) along channels
        -> head input is 2M instead of M.
      - FiLM still uses a *scalar* CSI (per-image mean of the CSI map).
        That gives the decoder both a global "how good was the channel
        overall" summary and a local per-strip reliability map.

    Inputs to forward():
      x:           received latent       [B, M, H, W]
      csi_map_db:  per-element SNR (dB)  [B, M, H, W]

    Build csi_map_db with packetwise_to_element_map() from a per-packet SNR
    vector. For dropped / missing packets, use a sentinel like -20 dB so the
    decoder learns the "very low SNR -> in-paint from neighbors" behavior.
    """

    def __init__(
        self,
        N: int = 256,
        M: int = 16,
        csi_embed_dim: int = 64,
        csi_db_scale: float = 20.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M
        self.csi_embed_dim = csi_embed_dim
        # SNR(dB) is divided by this before the conv input to keep values
        # roughly within ~[-1.5, 1.5] for typical operating ranges.
        self.csi_db_scale = csi_db_scale

        # FiLM gets a scalar (mean of map).
        self.csi_embed = nn.Sequential(
            nn.Linear(1, csi_embed_dim),
            nn.GELU(),
            nn.Linear(csi_embed_dim, csi_embed_dim),
        )

        # Head consumes concat([received, csi_map_norm]) -> 2M channels.
        self.head = nn.Sequential(
            nn.Conv2d(2 * M, N, 1),
            LayerNorm2d(N),
        )

        self.stage1_blocks = nn.ModuleList([ConvNeXtBlock(N), ConvNeXtBlock(N)])
        self.stage1_film = nn.ModuleList(
            [FiLMLayer(N, csi_embed_dim), FiLMLayer(N, csi_embed_dim)]
        )

        self.up1 = Upsample2x(N, N)

        self.stage2_blocks = nn.ModuleList([ConvNeXtBlock(N), ConvNeXtBlock(N)])
        self.stage2_film = nn.ModuleList(
            [FiLMLayer(N, csi_embed_dim), FiLMLayer(N, csi_embed_dim)]
        )

        self.up2 = Upsample2x(N, 3)

    def forward(
        self,
        x: torch.Tensor,
        csi_map_db: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if csi_map_db.dim() != 4:
            raise ValueError(
                "csi_map_db must be 4-D [B, M, H, W], got shape "
                f"{tuple(csi_map_db.shape)}; build it with "
                "packetwise_to_element_map()."
            )
        if csi_map_db.shape != x.shape:
            # Allow broadcasting from [B, 1, H, W] etc.
            csi_map_db = csi_map_db.expand_as(x)

        # Global summary scalar for FiLM (mean over all elements per image).
        csi_scalar = csi_map_db.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        emb = self.csi_embed(csi_scalar)

        # Normalize map for the conv input.
        csi_map_norm = csi_map_db / self.csi_db_scale

        x_in = torch.cat([x, csi_map_norm], dim=1)  # [B, 2M, H, W]
        x = self.head(x_in)

        for block, film in zip(self.stage1_blocks, self.stage1_film):
            x = film(block(x), emb)

        x = self.up1(x)

        for block, film in zip(self.stage2_blocks, self.stage2_film):
            x = film(block(x), emb)

        x = self.up2(x)
        return x


# ---------------------------------------------------------------------------
# End-to-end model
# ---------------------------------------------------------------------------

class DJSCCModelNoBand(BaseModel):
    """End-to-end no-band model: encoder -> constraint -> [channel] -> decoder.

    The channel is supplied by the training loop (it needs per-packet SNR /
    phase / drop info that this class doesn't own), so model.channel is
    None by default. forward() is provided mainly for completeness; train
    uses encoder/constraint/decoder directly.
    """

    def __init__(
        self,
        encoder: DJSCCEncoder,
        decoder: DJSCCDecoderSpatialCSI,
        constraint: Optional[BaseConstraint] = None,
        channel: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.constraint = constraint or AveragePowerConstraint(average_power=1.0)
        self.channel = channel

    def forward(
        self,
        source: torch.Tensor,
        csi_map_db: torch.Tensor,
        *channel_args: Any,
        **channel_kwargs: Any,
    ) -> torch.Tensor:
        encoded = self.encoder(source)
        constrained = self.constraint(encoded)

        if self.channel is not None:
            received = self.channel(constrained, *channel_args, **channel_kwargs)
        else:
            received = constrained

        return self.decoder(received, csi_map_db)
