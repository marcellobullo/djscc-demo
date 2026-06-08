"""Per-OFDM-symbol pilot-based phase de-rotation (no DFE).

Sits between digital_ofdm_frame_equalizer_vcvc_1 (which uses the static
payload equalizer, applying the preamble-derived channel estimate only) and
digital_ofdm_serializer_vcc_payload. The static equalizer does NOT track
residual CFO/SCO/LO drift across the OFDM symbols of a frame, so any
per-symbol phase rotation propagates downstream and corrupts the spatial
structure of the DJSCC feature map (manifests as fine horizontal banding
in the reconstructed image).

For each input OFDM vector (vlen=fft_len), this block:
  1. Extracts the pilot subcarriers.
  2. Computes est = mean(pilot_received * conj(pilot_expected)),
     whose phase is the residual rotation theta for this OFDM symbol.
  3. Multiplies the whole OFDM vector by conj(est)/|est|, derotating it
     so that pilots match the expected [1, 1, 1, -1].

This is NOT decision-feedback equalization: continuous-valued DJSCC
symbols pass through with unchanged modulus, only a per-OFDM-symbol scalar
phase correction is applied. Stream tags propagate via the default
TPP_ALL_TO_ALL policy.
"""

import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):
    def __init__(self, fft_len=64,
                 pilot_positions=[11, 25, 39, 53],
                 pilot_symbols=[1.0, 1.0, 1.0, -1.0]):
        gr.sync_block.__init__(
            self,
            name='pilot_phase_track',
            in_sig=[(np.complex64, fft_len)],
            out_sig=[(np.complex64, fft_len)],
        )
        self.pilot_positions = np.array(pilot_positions, dtype=np.int64)
        self.pilot_symbols = np.array(pilot_symbols, dtype=np.complex64)
        self.pilot_conj = np.conj(self.pilot_symbols).astype(np.complex64)

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        n = len(in0)
        if n == 0:
            return 0
        pilots = in0[:, self.pilot_positions]
        cross = pilots * self.pilot_conj[None, :]
        est = cross.mean(axis=1)
        mag = np.maximum(np.abs(est), 1e-9)
        correction = (est.conj() / mag).astype(np.complex64)
        out0[:] = (in0 * correction[:, None]).astype(np.complex64)
        return n
