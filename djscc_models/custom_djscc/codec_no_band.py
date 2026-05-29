"""Inference codec wrappers for the "no-band" DJSCC model.

Same interface as codec.py:
  - Encoder.encode(frame_bytes) -> complex64 ndarray (channel-blind, unchanged)
  - Decoder.decode(chn_in)      -> bytes
  - Decoder.set_snr_db(scalar)        -> sets a uniform per-packet SNR (back-compat)
  - Decoder.set_snr_db_vector(vec)    -> sets per-packet SNR vector

Differences from codec.py:
  - Loads DJSCCDecoderSpatialCSI (from model_no_band) instead of DJSCCDecoder.
  - The decoder consumes a per-element CSI map built from a per-packet
    SNR vector. The vector is expanded with packetwise_to_element_map().
"""

import os

import numpy as np
import torch
import torch.nn as nn

from kaira.constraints.power import AveragePowerConstraint

from .model_no_band import (
    DJSCCDecoderSpatialCSI,
    DJSCCEncoder,
    packetwise_to_element_map,
)


# ---------------------------------------------------------------------------
# Common helpers (mirrors codec.py)
# ---------------------------------------------------------------------------

def _pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _load_weights(module: nn.Module, checkpoint_path: str, prefix: str) -> bool:
    if not os.path.exists(checkpoint_path):
        print(f"codec_no_band: Checkpoint not found: {checkpoint_path}")
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["net"] if "net" in checkpoint else checkpoint
        filtered = {
            k.replace(prefix, ""): v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if not filtered:
            filtered = state
        missing, unexpected = module.load_state_dict(filtered, strict=False)
        if missing:
            print(f"codec_no_band: missing keys (first 5): {list(missing)[:5]}")
        if unexpected:
            print(f"codec_no_band: unexpected keys (first 5): {list(unexpected)[:5]}")
        print(f"codec_no_band: Loaded {len(filtered)} tensors (prefix='{prefix}')")
        return True
    except Exception as e:
        print(f"codec_no_band ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# Encoder (unchanged behavior; channel-blind)
# ---------------------------------------------------------------------------

class Encoder:
    """Image -> complex64 symbols. Channel-blind. Identical to codec.Encoder."""

    def __init__(
        self,
        model_path: str,
        img_width: int = 768,
        img_height: int = 512,
        tcn: int = 16,
        N: int = 256,
        packet_len: int = 960,
        padding_zeros: int = 0,
        quantize_cpu: bool = False,
        device: str = "auto",
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = 3
        self.tcn = tcn
        self.N = N
        self.packet_len = packet_len
        self.padding_zeros = padding_zeros
        self.device = _pick_device(device)

        self.expected_complex_items = (
            tcn * (img_height // 4) * (img_width // 4)
        ) // 2

        print(
            f"Encoder[no_band]: Device={self.device}, "
            f"{img_width}x{img_height}, tcn={tcn}, N={N}, no CSI"
        )

        self.model = DJSCCEncoder(N=N, M=tcn)
        _load_weights(self.model, model_path, "encoder.")
        self.model.eval()

        self.constraint = AveragePowerConstraint(average_power=1.0)

        if self.device.type == "cpu" and quantize_cpu:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                print("Encoder[no_band]: INT8 quantization applied.")
            except Exception as e:
                print(f"Encoder[no_band] WARNING: Quantization failed: {e}")

        self.model.to(self.device)

        dummy_img = torch.randn(
            1, self.img_channels, img_height, img_width,
            dtype=torch.float32, device=self.device,
        )
        try:
            self.model = torch.jit.trace(self.model, (dummy_img,))
            print("Encoder[no_band]: JIT tracing successful.")
        except Exception as e:
            print(f"Encoder[no_band] WARNING: JIT tracing failed: {e}")

        if warmup and self.device.type in ("mps", "cuda"):
            print(f"Encoder[no_band]: Warming up {self.device.type} pipeline...")
            with torch.no_grad():
                for _ in range(5):
                    self.model(dummy_img)
            _sync(self.device)
            print("Encoder[no_band]: Warm-up complete.")

        self._encode_count = 0

    def encode(self, frame_bytes: bytes) -> np.ndarray:
        flat = np.frombuffer(frame_bytes, dtype=np.uint8)
        image_array = flat.reshape(
            (self.img_height, self.img_width, self.img_channels)
        ).copy()
        image_tensor = (
            torch.from_numpy(image_array).float()
            .permute(2, 0, 1).unsqueeze(0)
            .div(255.0).to(self.device)
        )

        with torch.no_grad():
            latent = self.model(image_tensor)
            latent_flat = latent.flatten()
            constrained = self.constraint(latent_flat.unsqueeze(0)).squeeze(0)

        chn_np = constrained.detach().cpu().numpy().astype(np.float32)
        if len(chn_np) % 2 != 0:
            chn_np = np.append(chn_np, 0.0)

        dim_z = len(chn_np) // 2
        chn_in = (chn_np[:dim_z] + 1j * chn_np[dim_z:]).astype(np.complex64)

        if self.padding_zeros > 0:
            chn_in = np.concatenate(
                [chn_in, np.zeros(self.padding_zeros, dtype=np.complex64)]
            )

        remainder = len(chn_in) % self.packet_len
        if remainder != 0:
            auto_pad = self.packet_len - remainder
            chn_in = np.concatenate(
                [chn_in, np.zeros(auto_pad, dtype=np.complex64)]
            )

        self._encode_count += 1
        if self.device.type == "mps" and self._encode_count % 5 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return chn_in


# ---------------------------------------------------------------------------
# Decoder (per-packet CSI vector)
# ---------------------------------------------------------------------------

class Decoder:
    """Complex64 symbols -> image, using a per-packet SNR vector.

    Maintains a stored CSI state via:
      set_snr_db(scalar)       -> uniform across all packets (back-compat)
      set_snr_db_vector(vec)   -> per-packet SNR; length must equal n_pkts
    decode(chn_in) uses the most recently stored CSI to build the per-element
    SNR map that the spatial-CSI decoder expects.
    """

    def __init__(
        self,
        model_path: str,
        img_width: int = 768,
        img_height: int = 512,
        tcn: int = 16,
        N: int = 256,
        snr_db: float = 10.0,
        packet_len: int = 960,
        csi_db_scale: float = 20.0,
        sentinel_drop_db: float = -20.0,
        quantize_cpu: bool = False,
        device: str = "auto",
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = 3
        self.tcn = tcn
        self.N = N
        self.snr_db = float(snr_db)
        self.packet_len = packet_len
        self.csi_db_scale = csi_db_scale
        self.sentinel_drop_db = sentinel_drop_db
        self.device = _pick_device(device)

        # Latent spatial dims.
        self.lat_H = img_height // 4
        self.lat_W = img_width // 4

        self.expected_complex_items = (tcn * self.lat_H * self.lat_W) // 2

        # n_pkts per image = ceil(expected_complex_items / packet_len)
        # which matches the runtime RX layout (last packet may include
        # padding zeros — those positions just get the sentinel SNR).
        self.n_pkts = int(np.ceil(self.expected_complex_items / self.packet_len))

        print(
            f"Decoder[no_band]: Device={self.device}, "
            f"{img_width}x{img_height}, tcn={tcn}, N={N}, "
            f"n_pkts={self.n_pkts}, packet_len={packet_len}, "
            f"expected_symbols={self.expected_complex_items}"
        )

        self.model = DJSCCDecoderSpatialCSI(
            N=N, M=tcn, csi_db_scale=csi_db_scale
        )
        _load_weights(self.model, model_path, "decoder.")
        self.model.eval()

        if self.device.type == "cpu" and quantize_cpu:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.ConvTranspose2d}, dtype=torch.qint8
                )
                print("Decoder[no_band]: INT8 quantization applied.")
            except Exception as e:
                print(f"Decoder[no_band] WARNING: Quantization failed: {e}")

        self.model.to(self.device)

        # Stored CSI vector (per-packet SNR in dB). Default uniform.
        self._snr_per_pkt_db = np.full(
            self.n_pkts, self.snr_db, dtype=np.float32
        )

        dummy_chn = torch.randn(
            1, tcn, self.lat_H, self.lat_W,
            dtype=torch.float32, device=self.device,
        )
        dummy_csi = self._build_csi_map_tensor(self._snr_per_pkt_db)

        # Note: torch.jit.trace can't see the numpy CSI rebuild path, so we
        # only trace the forward; the build_csi_map call stays in Python.
        try:
            self.model = torch.jit.trace(self.model, (dummy_chn, dummy_csi))
            print("Decoder[no_band]: JIT tracing successful.")
        except Exception as e:
            print(f"Decoder[no_band] WARNING: JIT tracing failed: {e}")

        if warmup and self.device.type in ("mps", "cuda"):
            print(f"Decoder[no_band]: Warming up {self.device.type} pipeline...")
            with torch.no_grad():
                for _ in range(5):
                    self.model(dummy_chn, dummy_csi)
            _sync(self.device)
            print("Decoder[no_band]: Warm-up complete.")

        self._decode_count = 0

    # ---- CSI state ---------------------------------------------------------

    def set_snr_db(self, snr_db: float) -> None:
        """Back-compat: set the same SNR for every packet."""
        self.snr_db = float(snr_db)
        self._snr_per_pkt_db = np.full(
            self.n_pkts, float(snr_db), dtype=np.float32
        )

    def set_snr_db_vector(self, snr_db_per_pkt) -> None:
        """Set per-packet SNR. Length must equal n_pkts.

        Any non-finite values (NaN/Inf) and any explicit drops (caller
        marks with -inf or NaN) are replaced with the sentinel SNR
        (default -20 dB) to match the training-time drop encoding.
        """
        vec = np.asarray(snr_db_per_pkt, dtype=np.float32).reshape(-1)
        if vec.size != self.n_pkts:
            raise ValueError(
                f"snr_db_per_pkt has length {vec.size}, expected "
                f"{self.n_pkts}"
            )
        bad = ~np.isfinite(vec)
        if bad.any():
            vec = vec.copy()
            vec[bad] = self.sentinel_drop_db
        self._snr_per_pkt_db = vec

    def get_snr_db_vector(self) -> np.ndarray:
        return self._snr_per_pkt_db.copy()

    # ---- Internals ---------------------------------------------------------

    def _build_csi_map_tensor(self, snr_db_per_pkt_np: np.ndarray) -> torch.Tensor:
        """Build the [1, M, H/4, W/4] CSI map tensor from a numpy
        per-packet SNR vector on the configured device."""
        t = torch.from_numpy(snr_db_per_pkt_np.astype(np.float32))
        t = t.unsqueeze(0).to(self.device)  # [1, n_pkts]
        csi_map = packetwise_to_element_map(
            t, M=self.tcn, H=self.lat_H, W=self.lat_W,
            pkt_len_complex=self.packet_len,
        )  # [1, M, H/4, W/4]
        return csi_map

    # ---- Inference ---------------------------------------------------------

    def decode(self, chn_in: np.ndarray) -> bytes:
        """Decode complex64 symbols back to HWC uint8 image bytes."""
        if len(chn_in) < self.expected_complex_items:
            raise ValueError(
                f"Decoder[no_band]: expected >= {self.expected_complex_items} "
                f"symbols, got {len(chn_in)}"
            )
        chn_in = chn_in[: self.expected_complex_items]

        t = torch.from_numpy(chn_in)
        channel_tensor = torch.cat([t.real, t.imag]).to(
            dtype=torch.float32, device=self.device
        ).reshape(1, self.tcn, self.lat_H, self.lat_W)

        csi_map = self._build_csi_map_tensor(self._snr_per_pkt_db)

        with torch.no_grad():
            decoded = self.model(channel_tensor, csi_map)

        img_byte_tensor = torch.clamp(decoded.squeeze(0) * 255.0, 0, 255).byte()
        img_hwc = img_byte_tensor.permute(1, 2, 0).cpu().numpy()

        self._decode_count += 1
        if self.device.type == "mps" and self._decode_count % 10 == 0:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return img_hwc.tobytes()


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

class Codec:
    """Encoder + Decoder sharing the same checkpoint."""

    def __init__(
        self,
        model_path: str,
        img_width: int = 768,
        img_height: int = 512,
        tcn: int = 16,
        N: int = 256,
        snr_db: float = 10.0,
        packet_len: int = 960,
        padding_zeros: int = 0,
        csi_db_scale: float = 20.0,
        sentinel_drop_db: float = -20.0,
        quantize_cpu: bool = False,
        device: str = "auto",
        warmup: bool = True,
    ):
        self.encoder = Encoder(
            model_path=model_path,
            img_width=img_width, img_height=img_height, tcn=tcn, N=N,
            packet_len=packet_len, padding_zeros=padding_zeros,
            quantize_cpu=quantize_cpu, device=device, warmup=warmup,
        )
        self.decoder = Decoder(
            model_path=model_path,
            img_width=img_width, img_height=img_height, tcn=tcn, N=N,
            snr_db=snr_db, packet_len=packet_len,
            csi_db_scale=csi_db_scale, sentinel_drop_db=sentinel_drop_db,
            quantize_cpu=quantize_cpu, device=device, warmup=warmup,
        )

    def set_snr_db(self, snr_db: float) -> None:
        self.decoder.set_snr_db(snr_db)

    def set_snr_db_vector(self, snr_db_per_pkt) -> None:
        self.decoder.set_snr_db_vector(snr_db_per_pkt)

    def encode(self, frame_bytes: bytes) -> np.ndarray:
        return self.encoder.encode(frame_bytes)

    def decode(self, chn_in: np.ndarray) -> bytes:
        return self.decoder.decode(chn_in)
