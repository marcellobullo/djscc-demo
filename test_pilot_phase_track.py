#!/usr/bin/env python3
"""Hardware-free verification of the pilot-phase-tracking GR block.

We can't run the OTA OFDM chain without a USRP, but we can exactly simulate
the failure mode we believe is causing the horizontal banding: a per-OFDM-
symbol residual phase rotation that the static payload equalizer doesn't
correct. This script:

  1. Builds synthetic post-equalizer OFDM vectors (vlen=64) with known
     random data on the 48 data subcarriers and the standard pilots
     [1, 1, 1, -1] at positions [11, 25, 39, 53].
  2. Applies a known per-OFDM-symbol phase ramp theta_k = k * delta + offset
     (mimicking residual CFO/SCO drift across a packet).
  3. Runs the rotated vectors through `pilot_phase_track_0.blk.work(...)`.
  4. Asserts the pilots are restored to [1, 1, 1, -1] within float roundoff
     AND the data subcarriers match what was originally placed (i.e. the
     correction is exactly the inverse rotation, not just a partial one).

No GNU Radio runtime is needed — we instantiate the block directly and
invoke `work([rotated], [out])` like the scheduler would.

If this passes, the block math is correct; what remains to verify on
hardware is whether the *real* residual rotation is well-approximated by
"one phase scalar per OFDM symbol" (it should be, for typical CFO).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GR_DIR = os.path.join(_HERE, "receiver", "gnu_radio")
if _GR_DIR not in sys.path:
    sys.path.insert(0, _GR_DIR)

import djscc_rx_pilot_phase_track_0 as ppt  # noqa: E402


FFT_LEN = 64
PILOT_POSITIONS = np.array([11, 25, 39, 53], dtype=np.int64)
PILOT_SYMBOLS = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex64)
NULL_POSITIONS = np.array([0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63], dtype=np.int64)
DATA_POSITIONS = np.array(
    sorted(set(range(FFT_LEN)) - set(PILOT_POSITIONS.tolist())
           - set(NULL_POSITIONS.tolist())),
    dtype=np.int64,
)
assert DATA_POSITIONS.size == 48, f"expected 48 data carriers, got {DATA_POSITIONS.size}"


def build_ofdm_frame(n_ofdm: int, seed: int = 0) -> np.ndarray:
    """One packet: n_ofdm OFDM vectors of vlen=64.

    Data carriers get random unit-variance complex symbols (DJSCC-like).
    Pilots are placed at the standard positions; nulls stay zero.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((n_ofdm, FFT_LEN), dtype=np.complex64)
    data = (rng.standard_normal((n_ofdm, DATA_POSITIONS.size))
            + 1j * rng.standard_normal((n_ofdm, DATA_POSITIONS.size))) / np.sqrt(2)
    frame[:, DATA_POSITIONS] = data.astype(np.complex64)
    frame[:, PILOT_POSITIONS] = PILOT_SYMBOLS[None, :]
    return frame


def apply_phase_drift(frame: np.ndarray, theta_per_symbol: float,
                      theta_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate per-OFDM-symbol residual phase rotation.

    theta_k = theta_offset + k * theta_per_symbol  (radians)
    Returns (rotated_frame, theta_array).
    """
    n_ofdm = frame.shape[0]
    thetas = theta_offset + np.arange(n_ofdm) * theta_per_symbol
    rot = np.exp(1j * thetas).astype(np.complex64)
    return (frame * rot[:, None]).astype(np.complex64), thetas


def run_block(rotated: np.ndarray) -> np.ndarray:
    """Invoke the new block's work() directly (no GR scheduler needed)."""
    block = ppt.blk(
        fft_len=FFT_LEN,
        pilot_positions=PILOT_POSITIONS.tolist(),
        pilot_symbols=PILOT_SYMBOLS.tolist(),
    )
    out = np.zeros_like(rotated)
    n = block.work([rotated], [out])
    assert n == len(rotated), f"block returned {n}, expected {len(rotated)}"
    return out


def assert_close(name: str, a: np.ndarray, b: np.ndarray, atol: float) -> None:
    diff = np.abs(a - b)
    mx = float(diff.max())
    print(f"  {name}: max|err| = {mx:.3e}  (tol = {atol:.0e})")
    assert mx <= atol, f"{name} exceeded tolerance: {mx} > {atol}"


def test_static_phase_offset() -> None:
    """Constant phase rotation (theta_per_symbol = 0): trivial DC offset case."""
    print("\n[test] constant phase offset (theta = 1.3 rad)")
    frame = build_ofdm_frame(n_ofdm=20, seed=1)
    rotated, _ = apply_phase_drift(frame, theta_per_symbol=0.0, theta_offset=1.3)
    out = run_block(rotated)
    assert_close("pilots", out[:, PILOT_POSITIONS],
                 np.broadcast_to(PILOT_SYMBOLS, (20, 4)), atol=1e-5)
    assert_close("data", out[:, DATA_POSITIONS], frame[:, DATA_POSITIONS], atol=1e-5)


def test_linear_phase_drift() -> None:
    """Per-OFDM-symbol phase ramp (mimics residual CFO)."""
    print("\n[test] linear phase drift (theta_per_symbol = 0.15 rad)")
    frame = build_ofdm_frame(n_ofdm=20, seed=2)
    rotated, thetas = apply_phase_drift(
        frame, theta_per_symbol=0.15, theta_offset=-0.4)
    out = run_block(rotated)
    # Sanity: the block should have computed thetas close to truth (modulo 2pi).
    est = np.angle((rotated[:, PILOT_POSITIONS]
                    * np.conj(PILOT_SYMBOLS)[None, :]).mean(axis=1))
    err = np.angle(np.exp(1j * (est - thetas)))
    print(f"  estimated theta error (rad): max={float(np.abs(err).max()):.3e}")
    assert_close("pilots", out[:, PILOT_POSITIONS],
                 np.broadcast_to(PILOT_SYMBOLS, (20, 4)), atol=1e-5)
    assert_close("data", out[:, DATA_POSITIONS], frame[:, DATA_POSITIONS], atol=1e-5)


def test_drift_with_noise() -> None:
    """Linear drift + AWGN on the rotated frame; verify pilots/data are
    restored on average (not exactly — noise is on by construction)."""
    print("\n[test] linear drift + AWGN (theta_per_symbol = 0.05, sigma = 0.05)")
    n_ofdm = 200
    frame = build_ofdm_frame(n_ofdm=n_ofdm, seed=3)
    rotated, _ = apply_phase_drift(frame, theta_per_symbol=0.05, theta_offset=0.7)
    rng = np.random.default_rng(4)
    sigma = 0.05
    noise = ((rng.standard_normal(rotated.shape)
              + 1j * rng.standard_normal(rotated.shape)) * sigma / np.sqrt(2)
             ).astype(np.complex64)
    # Don't add noise on nulls — they stay 0 in our model.
    noise[:, NULL_POSITIONS] = 0
    out = run_block(rotated + noise)
    # Pilots: deviation should be on the order of sigma / sqrt(4 pilots)
    pilot_err = np.abs(out[:, PILOT_POSITIONS]
                       - PILOT_SYMBOLS[None, :]).mean()
    print(f"  mean |pilot - expected| = {pilot_err:.3e}  "
          f"(rough bound ~ sigma = {sigma})")
    assert pilot_err < 3 * sigma, "pilots far from expected — block broken"
    # Data symbols: bias due to phase estimation error should be << sigma
    data_bias = np.abs(out[:, DATA_POSITIONS]
                       - frame[:, DATA_POSITIONS]).mean()
    print(f"  mean |data_out - data_true| = {data_bias:.3e}")
    assert data_bias < 3 * sigma, "data deviates much more than channel noise"


def test_coherence_gate_strict_passthrough() -> None:
    """coh_thresh=2.0 is impossible for unit-magnitude pilots — block must
    be a strict pass-through regardless of input."""
    print("\n[test] coh_thresh=2.0 → strict pass-through")
    frame = build_ofdm_frame(n_ofdm=20, seed=1)
    rotated, _ = apply_phase_drift(frame, theta_per_symbol=0.15, theta_offset=-0.4)
    block = ppt.blk(
        fft_len=FFT_LEN,
        pilot_positions=PILOT_POSITIONS.tolist(),
        pilot_symbols=PILOT_SYMBOLS.tolist(),
        coh_thresh=2.0,
    )
    out = np.zeros_like(rotated)
    block.work([rotated], [out])
    assert_close("strict pass-through", out, rotated, atol=1e-6)


def test_coherence_gate_rejects_noise_pilots() -> None:
    """With pilots replaced by pure unit-variance noise, the gate should reject
    the vast majority of OFDM symbols (rather than apply wild rotations)."""
    print("\n[test] noise pilots → gate rejects most symbols")
    n_ofdm = 400
    frame = build_ofdm_frame(n_ofdm=n_ofdm, seed=11)
    rng = np.random.default_rng(12)
    frame[:, PILOT_POSITIONS] = (
        (rng.standard_normal((n_ofdm, 4)) + 1j * rng.standard_normal((n_ofdm, 4)))
        / np.sqrt(2)
    ).astype(np.complex64)
    block = ppt.blk(
        fft_len=FFT_LEN,
        pilot_positions=PILOT_POSITIONS.tolist(),
        pilot_symbols=PILOT_SYMBOLS.tolist(),
        coh_thresh=0.95,
    )
    out = np.zeros_like(frame)
    block.work([frame], [out])
    frac = block._n_corrected / max(block._n_seen, 1)
    print(f"  corrected fraction = {frac:.1%} (expect << 100%)")
    # With 4 iid CN(0,1) → |mean| Rayleigh, sigma_R ≈ 0.354.
    # P(|est| > 0.95) ≈ exp(-(0.95/0.354)^2 / 2) ≈ 2.7% → assert < 10%.
    assert frac < 0.10, f"gate let too many noise symbols through: {frac:.1%}"


def test_zero_input_no_crash() -> None:
    """Block should handle a 0-row chunk gracefully (GR may call work
    with empty inputs during initialization)."""
    print("\n[test] zero-length input")
    block = ppt.blk(fft_len=FFT_LEN,
                    pilot_positions=PILOT_POSITIONS.tolist(),
                    pilot_symbols=PILOT_SYMBOLS.tolist())
    empty = np.zeros((0, FFT_LEN), dtype=np.complex64)
    out = np.zeros_like(empty)
    n = block.work([empty], [out])
    assert n == 0, f"expected 0, got {n}"
    print("  handled cleanly")


if __name__ == "__main__":
    test_static_phase_offset()
    test_linear_phase_drift()
    test_drift_with_noise()
    test_coherence_gate_strict_passthrough()
    test_coherence_gate_rejects_noise_pilots()
    test_zero_input_no_crash()
    print("\nAll tests passed.")
