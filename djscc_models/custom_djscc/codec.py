"""Inference codec wrappers for the custom DJSCC model.

Provides Encoder/Decoder classes with the same interface as
gr-modules/gr-deepjscc/python/deepjscc/codec.py, but using the
ConvNeXt-based channel-blind encoder and FiLM-conditioned decoder.
"""

import numpy as np
import torch
import torch.nn as nn

from kaira.constraints.power import AveragePowerConstraint

from .model import DJSCCDecoder, DJSCCEncoder


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
    """Load weights from a checkpoint, filtering by prefix."""
    import os
    if not os.path.exists(checkpoint_path):
        print(f"codec: Checkpoint not found: {checkpoint_path}")
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
        module.load_state_dict(filtered, strict=False)
        print(f"codec: Loaded {len(filtered)} tensors (prefix='{prefix}')")
        return True
    except Exception as e:
        print(f"codec ERROR: {e}")
        return False


class Encoder:
    """Wraps DJSCCEncoder for inference: image -> complex64 symbols.

    No SNR/CSI parameter is needed — the encoder is channel-blind.
    """

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

        self.expected_complex_items = (tcn * (img_height // 4) * (img_width // 4)) // 2

        print(f"Encoder: Device={self.device}, {img_width}x{img_height}, "
              f"tcn={tcn}, N={N}, no CSI")

        self.model = DJSCCEncoder(N=N, M=tcn)
        _load_weights(self.model, model_path, "encoder.")
        self.model.eval()

        self.constraint = AveragePowerConstraint(average_power=1.0)

        if self.device.type == "cpu" and quantize_cpu:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
                print("Encoder: INT8 quantization applied.")
            except Exception as e:
                print(f"Encoder WARNING: Quantization failed: {e}")

        self.model.to(self.device)

        dummy_img = torch.randn(
            1, self.img_channels, img_height, img_width,
            dtype=torch.float32, device=self.device)
        try:
            self.model = torch.jit.trace(self.model, (dummy_img,))
            print("Encoder: JIT tracing successful.")
        except Exception as e:
            print(f"Encoder WARNING: JIT tracing failed: {e}")

        if warmup and self.device.type in ("mps", "cuda"):
            print(f"Encoder: Warming up {self.device.type} pipeline...")
            with torch.no_grad():
                for _ in range(5):
                    self.model(dummy_img)
            _sync(self.device)
            print("Encoder: Warm-up complete.")

        self._encode_count = 0

    def encode(self, frame_bytes: bytes) -> np.ndarray:
        """Encode a raw HWC uint8 image to a complex64 array aligned to packet_len."""
        flat = np.frombuffer(frame_bytes, dtype=np.uint8)
        image_array = flat.reshape(
            (self.img_height, self.img_width, self.img_channels)).copy()
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
                [chn_in, np.zeros(self.padding_zeros, dtype=np.complex64)])

        remainder = len(chn_in) % self.packet_len
        if remainder != 0:
            auto_pad = self.packet_len - remainder
            chn_in = np.concatenate(
                [chn_in, np.zeros(auto_pad, dtype=np.complex64)])

        self._encode_count += 1
        if self.device.type == "mps" and self._encode_count % 5 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return chn_in


class Decoder:
    """Wraps DJSCCDecoder for inference: complex64 symbols -> image.

    Requires SNR (or other CSI metric) for channel-adaptive decoding.
    """

    def __init__(
        self,
        model_path: str,
        img_width: int = 768,
        img_height: int = 512,
        tcn: int = 16,
        N: int = 256,
        snr_db: float = 10.0,
        csi_length: int = 1,
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
        self.snr_db = snr_db
        self.csi_length = csi_length
        self.device = _pick_device(device)

        self.expected_complex_items = (tcn * (img_height // 4) * (img_width // 4)) // 2

        print(f"Decoder: Device={self.device}, {img_width}x{img_height}, "
              f"tcn={tcn}, N={N}, SNR={snr_db} dB, "
              f"expected_symbols={self.expected_complex_items}")

        self.model = DJSCCDecoder(N=N, M=tcn, csi_length=csi_length,
                                  csi_embed_dim=64)
        _load_weights(self.model, model_path, "decoder.")
        self.model.eval()

        if self.device.type == "cpu" and quantize_cpu:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.ConvTranspose2d}, dtype=torch.qint8)
                print("Decoder: INT8 quantization applied.")
            except Exception as e:
                print(f"Decoder WARNING: Quantization failed: {e}")

        self.model.to(self.device)

        dummy_chn = torch.randn(
            1, tcn, img_height // 4, img_width // 4,
            dtype=torch.float32, device=self.device)
        dummy_csi = torch.tensor([[self.snr_db]], dtype=torch.float32,
                                 device=self.device)
        try:
            self.model = torch.jit.trace(self.model, (dummy_chn, dummy_csi))
            print("Decoder: JIT tracing successful.")
        except Exception as e:
            print(f"Decoder WARNING: JIT tracing failed: {e}")

        if warmup and self.device.type in ("mps", "cuda"):
            print(f"Decoder: Warming up {self.device.type} pipeline...")
            with torch.no_grad():
                for _ in range(5):
                    self.model(dummy_chn, dummy_csi)
            _sync(self.device)
            print("Decoder: Warm-up complete.")

        self._decode_count = 0

    def set_snr_db(self, snr_db: float) -> None:
        self.snr_db = snr_db

    def decode(self, chn_in: np.ndarray) -> bytes:
        """Decode complex64 symbols back to HWC uint8 image bytes."""
        if len(chn_in) < self.expected_complex_items:
            raise ValueError(
                f"Decoder: expected >= {self.expected_complex_items} symbols, "
                f"got {len(chn_in)}")
        chn_in = chn_in[:self.expected_complex_items]

        t = torch.from_numpy(chn_in)
        channel_tensor = torch.cat([t.real, t.imag]).to(
            dtype=torch.float32, device=self.device
        ).reshape(1, self.tcn, self.img_height // 4, self.img_width // 4)

        csi = torch.tensor(
            [[self.snr_db]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            decoded = self.model(channel_tensor, csi)

        img_byte_tensor = torch.clamp(decoded.squeeze(0) * 255.0, 0, 255).byte()
        img_hwc = img_byte_tensor.permute(1, 2, 0).cpu().numpy()

        self._decode_count += 1
        if self.device.type == "mps" and self._decode_count % 10 == 0:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return img_hwc.tobytes()


class Codec:
    """Bundles Encoder + Decoder sharing the same checkpoint."""

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
            snr_db=snr_db, quantize_cpu=quantize_cpu, device=device,
            warmup=warmup,
        )

    def set_snr_db(self, snr_db: float) -> None:
        self.decoder.set_snr_db(snr_db)

    def encode(self, frame_bytes: bytes) -> np.ndarray:
        return self.encoder.encode(frame_bytes)

    def decode(self, chn_in: np.ndarray) -> bytes:
        return self.decoder.decode(chn_in)
