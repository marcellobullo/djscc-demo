#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# DeepJSCC Codec — high-level Encoder / Decoder classes wrapping the
# Attention_Encoder / Attention_Decoder networks.
#
# Usage:
#     from deepjscc.codec import Encoder, Decoder, Codec
#
#     enc = Encoder(model_path="checkpoint.pth.tar",
#                   img_width=768, img_height=512, tcn=16,
#                   snr_db=10.0, packet_len=960)
#     chn_in = enc.encode(frame_bytes)              # np.complex64[]
#
#     dec = Decoder(model_path="checkpoint.pth.tar",
#                   img_width=768, img_height=512, tcn=16,
#                   snr_db=10.0)
#     img_bytes = dec.decode(chn_in)                # uint8 HWC bytes
#
# Both classes handle device selection, weight loading, JIT tracing,
# warm-up, and (optional) INT8 dynamic quantization on CPU.

import numpy as np
import torch
import torch.nn as nn

from .nn_model import (
    Attention_Encoder,
    Attention_Decoder,
    Args,
    powerConstraint,
)


def _pick_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


# =====================================================================
# ENCODER
# =====================================================================
class Encoder:
    """Wraps Attention_Encoder with the full frame->complex64 pipeline:
    reshape -> normalize -> NN forward -> power constraint -> complex pairing
    -> optional padding -> packet-length alignment.
    """
    def __init__(self,
                 model_path,
                 img_width=768,
                 img_height=512,
                 tcn=16,
                 snr_db=10.0,
                 packet_len=960,
                 padding_zeros=0,
                 quantize_cpu=False,
                 device="auto",
                 warmup=True):
        self.model_path = model_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = 3
        self.tcn = tcn
        self.snr_db = snr_db
        self.packet_len = packet_len
        self.padding_zeros = padding_zeros
        self.quantize_cpu = quantize_cpu
        self.device = _pick_device(device)

        print(f"Encoder: Device={self.device}, {img_width}x{img_height}, tcn={tcn}")

        self.model = Attention_Encoder(Args(tcn=tcn))
        self.model.load_pretrained_weights(model_path)
        self.model.eval()

        if self.device.type == 'cpu' and self.quantize_cpu:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
                print("Encoder: INT8 quantization applied.")
            except Exception as e:
                print(f"Encoder WARNING: Quantization failed: {e}")

        self.model.to(self.device)

        # JIT trace — big speedup on MPS/CUDA, harmless on CPU.
        dummy_img = torch.randn(1, self.img_channels, img_height, img_width,
                                dtype=torch.float32, device=self.device)
        dummy_attn = torch.tensor([[self.snr_db]], dtype=torch.float32, device=self.device)
        try:
            self.model = torch.jit.trace(self.model, (dummy_img, dummy_attn))
            print("Encoder: JIT tracing successful.")
        except Exception as e:
            print(f"Encoder WARNING: JIT tracing failed: {e}")

        if warmup and self.device.type in ('mps', 'cuda'):
            print(f"Encoder: Warming up {self.device.type} pipeline...")
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_img, dummy_attn)
            _sync(self.device)
            print("Encoder: Warm-up complete.")

        self._encode_count = 0

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db

    def encode(self, frame_bytes):
        """Encode a raw HWC uint8 image frame (length = H*W*3 bytes) to
        a np.complex64 array aligned to `packet_len`."""
        flat = np.frombuffer(frame_bytes, dtype=np.uint8)
        image_array = flat.reshape((self.img_height, self.img_width, self.img_channels)).copy()
        image_tensor = (torch.from_numpy(image_array).float()
                        .permute(2, 0, 1).unsqueeze(0)
                        .div(255.0).to(self.device))
        attention = torch.tensor([[self.snr_db]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            chn_in_tensor = self.model(image_tensor, attention)
            chn_in_norm = powerConstraint(chn_in_tensor.flatten(), P=1)

        chn_in_np = chn_in_norm.detach().cpu().numpy().astype(np.float32)
        if len(chn_in_np) % 2 != 0:
            chn_in_np = np.append(chn_in_np, 0.0)

        dim_z = len(chn_in_np) // 2
        chn_in = (chn_in_np[:dim_z] + 1j * chn_in_np[dim_z:]).astype(np.complex64)

        if self.padding_zeros > 0:
            chn_in = np.concatenate([chn_in, np.zeros(self.padding_zeros, dtype=np.complex64)])

        remainder = len(chn_in) % self.packet_len
        if remainder != 0:
            auto_pad = self.packet_len - remainder
            chn_in = np.concatenate([chn_in, np.zeros(auto_pad, dtype=np.complex64)])

        self._encode_count += 1
        if self.device.type == 'mps' and self._encode_count % 5 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return chn_in


# =====================================================================
# DECODER
# =====================================================================
class Decoder:
    """Wraps Attention_Decoder with the full complex64->frame pipeline:
    real/imag unpack -> NN forward -> clamp+scale -> HWC uint8 bytes.
    """
    def __init__(self,
                 model_path,
                 img_width=768,
                 img_height=512,
                 tcn=16,
                 snr_db=10.0,
                 quantize_cpu=False,
                 device="auto",
                 warmup=True):
        self.model_path = model_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = 3
        self.tcn = tcn
        self.snr_db = snr_db
        self.quantize_cpu = quantize_cpu
        self.device = _pick_device(device)

        # Expected number of complex symbols per frame — matches encoder output.
        self.expected_complex_items = (tcn * (img_height // 4) * (img_width // 4)) // 2

        print(f"Decoder: Device={self.device}, {img_width}x{img_height}, tcn={tcn}, "
              f"expected_symbols={self.expected_complex_items}")

        self.model = Attention_Decoder(Args(tcn=tcn))
        self.model.load_pretrained_weights(model_path)
        self.model.eval()

        if self.device.type == 'cpu' and self.quantize_cpu:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.ConvTranspose2d}, dtype=torch.qint8)
                print("Decoder: INT8 quantization applied.")
            except Exception as e:
                print(f"Decoder WARNING: Quantization failed: {e}")

        self.model.to(self.device)

        dummy_chn = torch.randn(1, tcn, img_height // 4, img_width // 4,
                                dtype=torch.float32, device=self.device)
        dummy_attn = torch.tensor([[self.snr_db]], dtype=torch.float32, device=self.device)
        try:
            self.model = torch.jit.trace(self.model, (dummy_chn, dummy_attn))
            print("Decoder: JIT tracing successful.")
        except Exception as e:
            print(f"Decoder WARNING: JIT tracing failed: {e}")

        if warmup and self.device.type in ('mps', 'cuda'):
            print(f"Decoder: Warming up {self.device.type} pipeline...")
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_chn, dummy_attn)
            _sync(self.device)
            print("Decoder: Warm-up complete.")

        self._decode_count = 0

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db

    def decode(self, chn_in):
        """Decode a np.complex64 array (length = expected_complex_items) back
        to HWC uint8 image bytes."""
        if len(chn_in) < self.expected_complex_items:
            raise ValueError(
                f"Decoder: expected >= {self.expected_complex_items} complex symbols, "
                f"got {len(chn_in)}")
        chn_in = chn_in[:self.expected_complex_items]

        t = torch.from_numpy(chn_in)
        channel_tensor = torch.cat([t.real, t.imag]).to(
            dtype=torch.float32, device=self.device
        ).reshape(1, self.tcn, self.img_height // 4, self.img_width // 4)
        attention = torch.tensor([[self.snr_db]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            decoded = self.model(channel_tensor, attention)

        img_byte_tensor = torch.clamp(decoded.squeeze(0) * 255.0, 0, 255).byte()
        img_hwc = img_byte_tensor.permute(1, 2, 0).cpu().numpy()

        self._decode_count += 1
        if self.device.type == 'mps' and self._decode_count % 10 == 0:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return img_hwc.tobytes()


# =====================================================================
# CODEC — convenience wrapper that owns both halves
# =====================================================================
class Codec:
    """Bundles an Encoder and Decoder that share image size / tcn / SNR.
    Model weights are loaded once per half from the same checkpoint."""
    def __init__(self,
                 model_path,
                 img_width=768,
                 img_height=512,
                 tcn=16,
                 snr_db=10.0,
                 packet_len=960,
                 padding_zeros=0,
                 quantize_cpu=False,
                 device="auto",
                 warmup=True):
        self.encoder = Encoder(
            model_path=model_path,
            img_width=img_width, img_height=img_height, tcn=tcn,
            snr_db=snr_db, packet_len=packet_len, padding_zeros=padding_zeros,
            quantize_cpu=quantize_cpu, device=device, warmup=warmup,
        )
        self.decoder = Decoder(
            model_path=model_path,
            img_width=img_width, img_height=img_height, tcn=tcn,
            snr_db=snr_db, quantize_cpu=quantize_cpu, device=device, warmup=warmup,
        )

    def set_snr_db(self, snr_db):
        self.encoder.set_snr_db(snr_db)
        self.decoder.set_snr_db(snr_db)

    def encode(self, frame_bytes):
        return self.encoder.encode(frame_bytes)

    def decode(self, chn_in):
        return self.decoder.decode(chn_in)
