#!/usr/bin/env python3
"""
Adaptive DJSCC Image Receiver — no-band variant.

Differences vs. socket_adaptive_djscc_rx.py:

  * Imports the Decoder from custom_djscc.codec_no_band, which wraps the
    spatial-CSI decoder (DJSCCDecoderSpatialCSI) trained with
    train_no_band.py.

  * Inside ``_flush_and_decode`` it builds a *per-slot SNR vector* of length
    PKT_PER_IMG instead of collapsing everything into one scalar:

      - For each slot with valid (h, sigma2_raw): SNR_slot = 10*log10(1 / (sigma2 / |H_data|^2).mean())
      - For each slot that was not received OR has no CSI ingredients: use sentinel SNR (default -20 dB).
      - For each slot that has data but no CSI feed: use cfg.snr_db as fallback.

    The vector is fed to ``Decoder.set_snr_db_vector(...)`` before ``decode()``,
    and the decoder turns it into a per-element CSI map for spatial conditioning.

Everything else (PDU ingestion, slot bookkeeping, OFDM SNR / erasure helpers,
duplicate-detection display loop, forensic dumps, --drop-slots policy, etc.)
is unchanged from the original RX.

Run with a *no-band* checkpoint:

  python socket_adaptive_djscc_rx_no_band.py \
      --model <no_band_ckpt.pth> \
      --packet-len 960 --comp-ratio 12 \
      --use-live-snr --snr-port 5560 \
      --sentinel-drop-db -20.0
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pmt
import zmq

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "djscc_models"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from custom_djscc.codec_no_band import Decoder  # noqa: E402


_OCCUPIED_CARRIERS = (
    list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0))
    + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27))
)
_PILOT_CARRIERS = (-21, -7, 7, 21)
_DATA_CARRIERS = [k for k in _OCCUPIED_CARRIERS if k not in _PILOT_CARRIERS]
_DATA_IDX = np.array([k + 32 for k in _DATA_CARRIERS], dtype=np.int64)


# ---------------------------------------------------------------------------
# Symbol interleaver (must match socket_adaptive_djscc_tx.py)
# ---------------------------------------------------------------------------

def _make_interleaver(num_items: int, num_slots: int):
    """Return (perm_fwd, perm_inv) for a block interleaver over num_items.

      perm_fwd[src_pos] = tx_pos   — apply at RX: latent = rx[perm_fwd]
      perm_inv[tx_pos]  = src_pos  — apply at TX: tx    = latent[perm_inv]

    Identical construction to the TX so the inverse is exact.
    """
    bps = num_items // num_slots
    p = np.arange(num_items, dtype=np.int64)
    perm_fwd = (p % num_slots) * bps + (p // num_slots)
    perm_inv = np.empty(num_items, dtype=np.int64)
    perm_inv[perm_fwd] = p
    return perm_fwd, perm_inv


@dataclass
class RxConfig:
    model_path: str
    width: int
    height: int
    channel: int
    chn_in_len: int
    comp_ratio: float
    tcn: int
    N: int
    snr_db: float
    device: str
    quantize_cpu: bool
    port: str
    connect_host: str
    output_dir: str
    save: bool
    count: int
    duplicate_threshold: float
    timeout: float
    packet_len: int
    use_live_snr: bool = False
    snr_port: str = "5560"
    renorm: bool = False
    renorm_target: float = 2.0
    clip_mag: float = 5.0
    erase_snr_db: float | None = None
    debug_dump: str = ""
    drop_spec: str = ""
    drop_seed: int = 0
    sentinel_drop_db: float = -20.0
    csi_db_scale: float = 20.0
    interleave: bool = False
    # Controlled-experiment mode: one PNG per transmitted image, named by
    # transmission-order index, no SSIM dedup. tx_gain/rx_gain only tag the
    # output folder (the RX does not set the radio gains).
    exp_id_mode: bool = False
    tx_gain: str | None = None
    rx_gain: str | None = None


def build_decoder(cfg: RxConfig) -> Decoder:
    print(f"[*] Loading no-band CSI-adaptive decoder from {cfg.model_path}")
    t0 = time.time()
    dec = Decoder(
        model_path=cfg.model_path,
        img_width=cfg.width,
        img_height=cfg.height,
        tcn=cfg.tcn,
        N=cfg.N,
        snr_db=cfg.snr_db,
        packet_len=cfg.packet_len,
        csi_db_scale=cfg.csi_db_scale,
        sentinel_drop_db=cfg.sentinel_drop_db,
        quantize_cpu=cfg.quantize_cpu,
        device=cfg.device,
        warmup=True,
    )
    print(
        f"[*] Decoder ready in {time.time() - t0:.2f}s "
        f"(device={dec.device}, tcn={cfg.tcn}, n_pkts={dec.n_pkts}, "
        f"fallback_SNR={cfg.snr_db} dB, sentinel={cfg.sentinel_drop_db} dB, "
        f"expected_symbols={dec.expected_complex_items})"
    )
    return dec


def compute_ssim_fast(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        return 0.0
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    a_mean, b_mean = a.mean(), b.mean()
    a_std, b_std = a.std(), b.std()
    if a_std < 1.0 or b_std < 1.0:
        return 0.0
    return float(np.mean((a - a_mean) * (b - b_mean)) / (a_std * b_std))


def image_quality_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class _DropPolicy:
    """Same artificial-drop policy as the original RX (random / list / range)."""

    def __init__(self, spec: str, seed: int) -> None:
        self.mode = "off"
        self.rate: float = 0.0
        self.targets: set[int] = set()
        self._rng = np.random.default_rng(seed if seed else None)
        if not spec:
            return
        kind, _, body = spec.partition(":")
        if kind == "random":
            self.mode, self.rate = "random", float(body)
        elif kind == "list":
            self.mode = "list"
            self.targets = {int(x) for x in body.split(",") if x.strip()}
        elif kind == "range":
            lo, hi = (int(x) for x in body.split(":"))
            self.mode = "range"
            self.targets = set(range(lo, hi))
        else:
            raise ValueError(
                f"--drop-slots: unknown mode '{kind}' (expected random/list/range)"
            )

    def should_drop(self, slot: int) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "random":
            return bool(self._rng.random() < self.rate)
        return slot in self.targets

    def __repr__(self) -> str:
        if self.mode == "off":
            return "DropPolicy(off)"
        if self.mode == "random":
            return f"DropPolicy(random rate={self.rate})"
        head = sorted(self.targets)[:16]
        more = "..." if len(self.targets) > 16 else ""
        return f"DropPolicy({self.mode}, slots={head}{more})"


# ---------------------------------------------------------------------------
# Per-slot CSI vector builder (the heart of the no-band fix)
# ---------------------------------------------------------------------------

def _build_per_slot_snr_db(
    pkt_per_img: int,
    pn_log: list[tuple[int, int]],
    h_by_pn: dict[int, np.ndarray],
    raw_by_pn: dict[int, float],
    seen: list[bool],
    fallback_db: float,
    sentinel_db: float,
    snr_feed_enabled: bool,
) -> tuple[np.ndarray, int, int]:
    """Construct the per-slot SNR(dB) vector that the spatial-CSI decoder needs.

    Rules:
      * Missing slot (seen[slot] == False) -> sentinel_db (training-time
        "drop" encoding).
      * Slot has data but no CSI ingredients (either feed disabled or PDU
        not received) -> fallback_db (cfg.snr_db).
      * Slot has data + valid (h, sigma2) -> SNR computed from the same
        Nulls+Taps formula the original RX averaged across the image:

            sigma2_eff_per_subc = sigma2_raw / |H_data|^2
            SNR_lin              = 1 / mean(sigma2_eff_per_subc)
            SNR_db               = 10 * log10(SNR_lin)

    Returns:
      (snr_db_per_slot, n_csi_filled, n_missing)
    """
    snr_db_per_slot = np.full(pkt_per_img, fallback_db, dtype=np.float32)
    n_csi_filled = 0

    if snr_feed_enabled:
        for pn, slot in pn_log:
            if slot < 0 or slot >= pkt_per_img:
                continue
            h = h_by_pn.get(pn)
            s2 = raw_by_pn.get(pn)
            if h is None or s2 is None or s2 <= 0:
                continue
            h_pow = np.maximum(np.abs(h[_DATA_IDX]) ** 2, 1e-12)
            lin_noise = float((s2 / h_pow).mean())
            if lin_noise > 0:
                snr_db_per_slot[slot] = 10.0 * np.log10(1.0 / lin_noise)
                n_csi_filled += 1

    n_missing = 0
    for slot, ok in enumerate(seen):
        if not ok:
            snr_db_per_slot[slot] = sentinel_db
            n_missing += 1

    # Clamp to a sane range so the per-element map stays well-conditioned.
    snr_db_per_slot = np.clip(snr_db_per_slot, sentinel_db, 40.0)
    return snr_db_per_slot.astype(np.float32), n_csi_filled, n_missing


# ---------------------------------------------------------------------------
# Main RX loop
# ---------------------------------------------------------------------------

def receive_loop(
    decoder: Decoder,
    socket: zmq.Socket,
    cfg: RxConfig,
    drop_policy: _DropPolicy,
    snr_socket: zmq.Socket | None,
) -> None:
    expected = decoder.expected_complex_items
    PKT_LEN = cfg.packet_len
    PKT_PER_IMG = math.ceil(expected / PKT_LEN)
    samples_per_image = PKT_PER_IMG * PKT_LEN

    # De-interleave permutation (latent = rx[perm_fwd]) over the full padded
    # stream. Built once; matches the TX block interleaver exactly.
    perm_fwd = None
    if cfg.interleave:
        perm_fwd, _ = _make_interleaver(samples_per_image, PKT_PER_IMG)
        print(f"[*] Interleave ON: de-interleaving {samples_per_image} symbols "
              f"({PKT_PER_IMG} slots x {PKT_LEN}) before decode.")

    if decoder.n_pkts != PKT_PER_IMG:
        print(
            f"[!] WARNING: decoder.n_pkts={decoder.n_pkts} but PKT_PER_IMG="
            f"{PKT_PER_IMG}. Check --tcn / --packet-len / --width / --height "
            "match between TX, RX, and the trained checkpoint."
        )

    slot_buf = np.zeros(samples_per_image, dtype=np.complex64)
    seen = [False] * PKT_PER_IMG
    pn_log: list = []

    h_by_pn: dict[int, np.ndarray] = {}
    raw_by_pn: dict[int, float] = {}
    pilots_by_pn: dict[int, float] = {}

    anchor_pn = None
    current_image_idx = None
    n_images = 0

    window_title = "Adaptive DJSCC Received (no-band, per-slot CSI)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, cfg.width, cfg.height)

    if cfg.save:
        os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.debug_dump:
        os.makedirs(cfg.debug_dump, exist_ok=True)
        print(f"[*] Forensic dumps -> {os.path.abspath(cfg.debug_dump)}")

    unique_images = []
    exp_manifest: list[dict] = []
    last_saved_img = None
    total_frames = 0
    last_frame_time = time.time()
    unique_count = 0
    last_valid_pdu_time = time.time()

    print(
        f"[*] Lock-step RX: tcn={decoder.tcn}, "
        f"{PKT_PER_IMG} packets/image, {PKT_LEN} syms/packet, "
        f"expected={expected}, samples_per_image={samples_per_image}"
    )
    print("[*] Waiting for a valid transmission burst...")

    def _flush_and_decode():
        nonlocal n_images, total_frames, last_frame_time, unique_count
        nonlocal last_saved_img

        n_missed = sum(1 for ok in seen if not ok)
        if n_missed == PKT_PER_IMG:
            return

        if cfg.debug_dump:
            try:
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1e3)%1000:03d}"
                dump_path = os.path.join(
                    cfg.debug_dump,
                    f"image_{n_images+1:04d}_{stamp}.npz")
                pn_arr = np.array([pn for pn, _ in pn_log], dtype=np.int64)
                slt_arr = np.array([slt for _, slt in pn_log], dtype=np.int32)
                np.savez(
                    dump_path,
                    slot_buf=slot_buf,
                    seen=np.array(seen, dtype=bool),
                    packet_nums=pn_arr,
                    slots=slt_arr,
                    anchor_pn=np.int64(anchor_pn if anchor_pn is not None else -1),
                    expected=np.int64(expected),
                    pkt_len=np.int32(PKT_LEN),
                    pkt_per_img=np.int32(PKT_PER_IMG),
                    tcn=np.int32(decoder.tcn),
                    snr_db=np.float32(cfg.snr_db),
                    width=np.int32(cfg.width),
                    height=np.int32(cfg.height),
                )
                print(
                    f"  [*] forensic dump -> {os.path.basename(dump_path)} "
                    f"({len(pn_arr)} packets, {n_missed} missing)"
                )
            except Exception as e:
                print(f"  [!] forensic dump failed: {e}")

        if n_missed:
            print(
                f"  [*] image {n_images+1}: "
                f"{n_missed}/{PKT_PER_IMG} slots missing, zero-filled"
            )
            for i, ok in enumerate(seen):
                if not ok:
                    slot_buf[i * PKT_LEN:(i + 1) * PKT_LEN] = 0

        # With interleaving the latent is scattered across all packets, so the
        # padding tail can't simply be dropped yet — keep the full padded
        # stream (still in TX/channel order) through the per-packet processing
        # below, then de-interleave + truncate right before decode.
        if cfg.interleave:
            symbols = slot_buf.copy()
        else:
            symbols = slot_buf[:expected].copy()

        # ---- optional subcarrier-level erasure (unchanged from original) ----
        if snr_socket is not None and cfg.erase_snr_db is not None:
            n_data_per_ofdm = len(_DATA_IDX)
            n_ofdm_per_pkt = PKT_LEN // n_data_per_ofdm
            n_erased_total = 0
            n_pkts_with_erasure = 0
            for pn, slot in pn_log:
                h = h_by_pn.get(pn)
                s2 = raw_by_pn.get(pn)
                if h is None or s2 is None or s2 <= 0:
                    continue
                h_pow = np.abs(h[_DATA_IDX]) ** 2
                snr_per_sub_db = 10.0 * np.log10(np.maximum(h_pow / s2, 1e-12))
                bad = snr_per_sub_db < cfg.erase_snr_db
                if not bad.any():
                    continue
                slot_start = slot * PKT_LEN
                for ofdm_idx in range(n_ofdm_per_pkt):
                    base = slot_start + ofdm_idx * n_data_per_ofdm
                    if base + n_data_per_ofdm > len(symbols):
                        break
                    symbols[base:base + n_data_per_ofdm][bad] = 0
                n_erased_total += int(bad.sum()) * n_ofdm_per_pkt
                n_pkts_with_erasure += 1
            if n_erased_total > 0:
                print(
                    f"  [erase] {n_erased_total} symbols from weak subcarriers "
                    f"(< {cfg.erase_snr_db:.1f} dB) zeroed across "
                    f"{n_pkts_with_erasure} packets"
                )

        # ---- magnitude clip / renorm (unchanged from original) --------------
        pwr = float(np.mean(np.abs(symbols) ** 2))
        mag_max_before = float(np.max(np.abs(symbols)))

        if cfg.clip_mag > 0:
            mags = np.abs(symbols)
            clip_mask = mags > cfg.clip_mag
            num_clipped = int(np.sum(clip_mask))
            if num_clipped > 0:
                symbols[clip_mask] = (symbols[clip_mask] / mags[clip_mask]) * cfg.clip_mag
                print(
                    f"  [clip] {num_clipped} symbols above |x|>{cfg.clip_mag} "
                    f"pulled to magnitude {cfg.clip_mag}"
                )

        if cfg.renorm:
            pwr_pre = float(np.mean(np.abs(symbols) ** 2))
            if pwr_pre > 0:
                symbols *= np.complex64(np.sqrt(cfg.renorm_target / pwr_pre))
            pwr_after = float(np.mean(np.abs(symbols) ** 2))
            print(
                f"  [renorm] avg |x|^2: "
                f"raw={pwr:.3f}  pre={pwr_pre:.3f}  target={cfg.renorm_target:.2f}  "
                f"post={pwr_after:.3f}  (max |x| now={np.max(np.abs(symbols)):.2f})"
            )
        else:
            print(
                f"  [pwr ] avg |x|^2={pwr:.3f}  max |x|={mag_max_before:.2f}  "
                f"post-clip avg |x|^2={np.mean(np.abs(symbols) ** 2):.3f}"
            )

        total_frames += 1
        last_frame_time = time.time()

        # ---- PER-SLOT CSI vector (the no-band change) ----------------------
        snr_feed_enabled = snr_socket is not None
        snr_db_per_slot, n_csi_filled, n_missing = _build_per_slot_snr_db(
            pkt_per_img=PKT_PER_IMG,
            pn_log=pn_log,
            h_by_pn=h_by_pn,
            raw_by_pn=raw_by_pn,
            seen=seen,
            fallback_db=cfg.snr_db,
            sentinel_db=cfg.sentinel_drop_db,
            snr_feed_enabled=snr_feed_enabled,
        )
        if cfg.interleave:
            # Scatter the per-slot SNR the same way the signal was scattered:
            # expand to per-symbol (TX/channel order), de-interleave to latent
            # order, drop the padding tail, then feed as a per-element map so
            # the decoder is told which *scattered* elements are unreliable.
            pos_snr = np.repeat(snr_db_per_slot, PKT_LEN)        # [samples_per_image]
            elem_snr = pos_snr[perm_fwd][:expected]              # latent order
            decoder.set_element_snr_db(elem_snr)
        else:
            decoder.set_snr_db_vector(snr_db_per_slot)

        finite_mask = snr_db_per_slot > (cfg.sentinel_drop_db + 0.5)
        if finite_mask.any():
            mean_finite_db = float(np.mean(snr_db_per_slot[finite_mask]))
            min_finite_db = float(np.min(snr_db_per_slot[finite_mask]))
            max_finite_db = float(np.max(snr_db_per_slot[finite_mask]))
        else:
            mean_finite_db = min_finite_db = max_finite_db = float("nan")

        extra = ""
        if snr_feed_enabled:
            pilot_noise_samples = [pilots_by_pn[pn] for pn in pilots_by_pn if pilots_by_pn.get(pn, 0) > 0]
            if pilot_noise_samples:
                snr_p = 10.0 * np.log10(1.0 / float(np.mean(pilot_noise_samples)))
                extra = f" | pilots={snr_p:+.2f} dB"

        print(
            f"  [snr-vec] CSI={n_csi_filled}/{PKT_PER_IMG} filled, "
            f"missing={n_missing}, "
            f"finite SNR mean/min/max={mean_finite_db:+.2f}/"
            f"{min_finite_db:+.2f}/{max_finite_db:+.2f} dB"
            f"{extra}"
        )

        # Drain CSI feed state for packets we've already consumed.
        if snr_feed_enabled:
            for pn, _ in pn_log:
                h_by_pn.pop(pn, None)
                raw_by_pn.pop(pn, None)
                pilots_by_pn.pop(pn, None)

        # ---- de-interleave back to latent order, then decode ----------------
        if cfg.interleave:
            symbols = symbols[perm_fwd][:expected]

        t0 = time.time()
        try:
            img_bytes = decoder.decode(symbols)
        except Exception as e:
            print(f"  [!] decode failed: {e}")
            return

        dec_ms = (time.time() - t0) * 1000.0
        rgb = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
            (cfg.height, cfg.width, 3))
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        n_images += 1

        # -------- controlled-experiment save (order-keyed, no dedup) --------
        if cfg.exp_id_mode:
            # n_images is incremented once per flushed image just above, in
            # transmission order. Do NOT use current_image_idx: the anchor is
            # reset between images, so it is not a global running index.
            img_id = n_images - 1
            filename = f"image_{img_id:03d}.png"
            save_path = os.path.join(cfg.output_dir, filename)
            if cfg.save:
                cv2.imwrite(save_path, frame_bgr)
            exp_manifest.append({
                "image_id": img_id,
                "filename": filename,
                "frame": total_frames,
                "slots_seen": PKT_PER_IMG - n_missed,
                "slots_missing": n_missed,
                "mean_snr_db": round(mean_finite_db, 3),
                "decode_ok": 1,
            })
            print(
                f"  [*] Frame {total_frames}: image_id={img_id} "
                f"{'saved -> ' + filename if cfg.save else '(not saved)'} "
                f"({PKT_PER_IMG - n_missed}/{PKT_PER_IMG} slots, "
                f"decode={dec_ms:.1f} ms)"
            )
            display = frame_bgr.copy()
            cv2.putText(
                display, f"id {img_id} | frame {total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow(window_title, display)
            return

        is_duplicate = False
        if last_saved_img is not None:
            sim = compute_ssim_fast(frame_bgr, last_saved_img)
            if sim > cfg.duplicate_threshold:
                is_duplicate = True
                new_q = image_quality_score(frame_bgr)
                if unique_images and new_q > unique_images[-1][1]:
                    old_path = unique_images[-1][2]
                    unique_images[-1] = (frame_bgr.copy(), new_q, old_path)
                    if cfg.save:
                        cv2.imwrite(old_path, frame_bgr)
                    last_saved_img = frame_bgr.copy()
                    print(
                        f"  [*] Frame {total_frames}: duplicate "
                        f"(better quality, updated {os.path.basename(old_path)}, "
                        f"decode={dec_ms:.1f} ms)"
                    )
                else:
                    print(
                        f"  [*] Frame {total_frames}: duplicate "
                        f"(skipped, sim={sim:.3f}, decode={dec_ms:.1f} ms)"
                    )

        if not is_duplicate:
            unique_count += 1
            quality = image_quality_score(frame_bgr)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{unique_count:03d}_{stamp}.png"
            save_path = os.path.join(cfg.output_dir, filename)
            if cfg.save:
                cv2.imwrite(save_path, frame_bgr)
            unique_images.append((frame_bgr.copy(), quality, save_path))
            last_saved_img = frame_bgr.copy()
            print(
                f"  [*] Frame {total_frames}: NEW image #{unique_count} "
                f"{'saved -> ' + filename if cfg.save else '(not saved)'} "
                f"(q={quality:.1f}, decode={dec_ms:.1f} ms)"
            )

        display = frame_bgr.copy()
        label = f"#{unique_count} | frame {total_frames}"
        if is_duplicate:
            label += " [DUP]"
        cv2.putText(
            display, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imshow(window_title, display)

    # -----------------------------------------------------------------------
    # Main poll loop (unchanged from original)
    # -----------------------------------------------------------------------
    try:
        while True:
            if cfg.timeout > 0 and total_frames > 0 \
                    and time.time() - last_frame_time > cfg.timeout:
                print(f"\n[*] No frames for {cfg.timeout}s. Exiting.")
                break

            if current_image_idx is not None \
                    and time.time() - last_valid_pdu_time > 0.3:
                n_seen = sum(1 for ok in seen if ok)
                print(
                    f"\n[*] 0.3s idle gap detected. Flushing partial image "
                    f"({n_seen}/{PKT_PER_IMG} slots received)."
                )
                _flush_and_decode()
                seen = [False] * PKT_PER_IMG
                slot_buf.fill(0)
                pn_log = []
                current_image_idx = None
                anchor_pn = None

            if snr_socket is not None:
                while True:
                    try:
                        snr_raw = snr_socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        snr_pdu = pmt.deserialize_str(snr_raw)
                        snr_meta = pmt.car(snr_pdu)
                        pn = pmt.to_long(pmt.dict_ref(
                            snr_meta, pmt.intern('packet_num'), pmt.from_long(-1)))
                        if pn < 0:
                            continue
                        kind = pmt.symbol_to_string(pmt.dict_ref(
                            snr_meta, pmt.intern('kind'), pmt.intern('?')))
                        if kind == 'h':
                            h_by_pn[pn] = np.array(
                                pmt.c32vector_elements(pmt.cdr(snr_pdu)),
                                dtype=np.complex64,
                            )
                        elif kind == 'raw':
                            raw_by_pn[pn] = pmt.to_double(pmt.dict_ref(
                                snr_meta, pmt.intern('sigma2'),
                                pmt.from_double(float('nan'))))
                        elif kind == 'pilots':
                            pilots_by_pn[pn] = pmt.to_double(pmt.dict_ref(
                                snr_meta, pmt.intern('sigma2'),
                                pmt.from_double(float('nan'))))
                    except Exception as e:
                        print(f"[!] SNR PDU parse failed: {e}")
                if anchor_pn is not None:
                    cutoff = anchor_pn - 4 * PKT_PER_IMG
                    for d in (h_by_pn, raw_by_pn, pilots_by_pn):
                        stale = [k for k in d if k < cutoff]
                        for k in stale:
                            d.pop(k, None)

            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                if (cv2.waitKey(10) & 0xFF) == ord('q'):
                    break
                continue

            try:
                pdu = pmt.deserialize_str(raw)
                metadata = pmt.car(pdu)
                payload = pmt.cdr(pdu)

                key_pn = pmt.intern("packet_num")
                packet_num = pmt.to_long(
                    pmt.dict_ref(metadata, key_pn, pmt.from_long(-1)))
                assert packet_num >= 0

                iq_pkt = np.array(
                    pmt.c32vector_elements(payload), dtype=np.complex64)

            except Exception as e:
                print(f"[!] PDU parse failed: {e}")
                continue

            if len(iq_pkt) != PKT_LEN:
                print(f"[!] Anomalous PDU: got {len(iq_pkt)} cf32, "
                      f"expected {PKT_LEN}")
            if len(iq_pkt) < PKT_LEN:
                continue
            iq_pkt = iq_pkt[:PKT_LEN]

            last_valid_pdu_time = time.time()

            if anchor_pn is None:
                anchor_pn = packet_num

            delta = packet_num - anchor_pn
            slot = delta % PKT_PER_IMG
            image_idx_local = delta // PKT_PER_IMG
            print(
                f"[*] PDU Metadata: packet_num={packet_num} "
                f"anchor={anchor_pn} delta={delta} "
                f"image_idx={image_idx_local} slot={slot}"
            )

            if drop_policy.should_drop(slot):
                print(f"[drop] slot {slot:>4d} (pn={packet_num}) artificially dropped")
                continue

            if current_image_idx is None:
                current_image_idx = image_idx_local
            elif image_idx_local != current_image_idx:
                _flush_and_decode()
                seen = [False] * PKT_PER_IMG
                slot_buf.fill(0)
                pn_log = []
                current_image_idx = image_idx_local

            slot_buf[slot * PKT_LEN:(slot + 1) * PKT_LEN] = iq_pkt
            seen[slot] = True
            pn_log.append((packet_num, slot))

            if slot == PKT_PER_IMG - 1:
                _flush_and_decode()
                seen = [False] * PKT_PER_IMG
                slot_buf.fill(0)
                pn_log = []
                current_image_idx = None

            done_n = n_images if cfg.exp_id_mode else unique_count
            if cfg.count > 0 and done_n >= cfg.count:
                print(f"\n[*] Received {unique_count} unique image(s). Done.")
                cv2.waitKey(2000)
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[*] Receiver stopped.")
    finally:
        cv2.destroyAllWindows()
        if cfg.exp_id_mode and exp_manifest:
            try:
                os.makedirs(cfg.output_dir, exist_ok=True)
                man_path = os.path.join(cfg.output_dir, "manifest.csv")
                with open(man_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(exp_manifest[0].keys()))
                    w.writeheader()
                    w.writerows(exp_manifest)
                print(f"  Manifest written:       {man_path} "
                      f"({len(exp_manifest)} rows)")
            except Exception as e:
                print(f"  [!] manifest write failed: {e}")
        print(f"\n{'=' * 50}")
        print(f"  Total frames received:  {total_frames}")
        print(f"  Unique images:          {unique_count}")
        if cfg.save:
            print(f"  Output directory:       {os.path.abspath(cfg.output_dir)}")
            for _, q, path in unique_images:
                print(f"    - {os.path.basename(path)} (q={q:.1f})")
        print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="No-band Adaptive DJSCC Receiver (spatial-CSI decoder)"
    )

    p.add_argument("--model", type=str, required=True,
                   help="Path to *no-band* model checkpoint (.pth)")
    p.add_argument("--snr-db", type=float, default=19.0,
                   help="Fallback SNR (dB) used for slots that received data "
                        "but no CSI ingredients (default: 19.0)")
    p.add_argument("--sentinel-drop-db", type=float, default=-20.0,
                   help="CSI sentinel SNR (dB) for missing / dropped slots. "
                        "Match the value used during training (default: -20.0)")
    p.add_argument("--csi-db-scale", type=float, default=20.0,
                   help="Divisor for the CSI map fed to the decoder. Match "
                        "the trained value (default: 20.0)")
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--comp-ratio", type=int, default=12)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--quantize-cpu", action="store_true")

    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--channel", type=int, default=3)

    p.add_argument("--port", type=str, default="5558")
    p.add_argument("--connect-host", type=str, default="127.0.0.1")

    p.add_argument("--output-dir", type=str, default="./received_images")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--duplicate-threshold", type=float, default=0.92)
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--debug-dump", type=str, default="")
    p.add_argument("--drop-slots", type=str, default="")
    p.add_argument("--drop-seed", type=int, default=0)

    p.add_argument("--use-live-snr", action="store_true",
                   help="Subscribe to per-packet SNR ingredients on --snr-port "
                        "and build a per-slot SNR vector for the decoder. "
                        "Falls back to --snr-db per slot if no PDUs arrive.")
    p.add_argument("--snr-port", type=str, default="5560")
    p.add_argument("--renorm", action="store_true")
    p.add_argument("--renorm-target", type=float, default=2.0)
    p.add_argument("--clip-mag", type=float, default=5.0)
    p.add_argument("--erase-snr-db", type=float, default=None)
    p.add_argument("--interleave", action="store_true",
                   help="De-interleave the received symbols (and scatter the "
                        "per-slot CSI to a per-element map) before decoding. "
                        "Must match --interleave on the TX.")

    p.add_argument("--exp-id-mode", action="store_true",
                   help="Controlled-experiment mode: save exactly one PNG per "
                        "transmitted image, named image_<order>.png by "
                        "transmission-order index (no SSIM dedup), plus a "
                        "manifest.csv. Run the TX with --no-warmup and start "
                        "this RX first so order index 0 == first original.")
    p.add_argument("--tx-gain", type=str, default=None,
                   help="TX USRP gain for this run (folder tag only; the RX "
                        "does not set the radio gain). If both --tx-gain and "
                        "--rx-gain are given, output dir becomes "
                        "received_images/tx-<tx>_rx-<rx>.")
    p.add_argument("--rx-gain", type=str, default=None,
                   help="RX USRP gain for this run (folder tag only).")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> RxConfig:
    args.comp_ratio = 1 / args.comp_ratio
    tcn = int(args.comp_ratio * 4 * 4 * 2 * 3)
    chn_in_len = (tcn * (args.height // 4) * (args.width // 4)) // 2
    padding_zeros = (args.packet_len - chn_in_len % args.packet_len) % args.packet_len

    total_per_image = chn_in_len + padding_zeros
    total_per_image += (args.packet_len - total_per_image % args.packet_len) % args.packet_len

    print(
        f"[*] tcn={tcn}, chn_in_len={chn_in_len} cf32, "
        f"padding_zeros={padding_zeros}, "
        f"packets/image={total_per_image // args.packet_len}"
    )

    if tcn <= 0:
        raise ValueError(
            f"--comp-ratio={int(round(1 / args.comp_ratio))} produces tcn={tcn}; "
            f"valid options yield positive integer tcn."
        )

    output_dir = args.output_dir
    if args.tx_gain is not None and args.rx_gain is not None:
        output_dir = os.path.join(
            "received_images", f"tx-{args.tx_gain}_rx-{args.rx_gain}")

    return RxConfig(
        model_path=args.model,
        width=args.width,
        height=args.height,
        channel=args.channel,
        chn_in_len=chn_in_len,
        comp_ratio=args.comp_ratio,
        tcn=tcn,
        N=args.N,
        snr_db=args.snr_db,
        device=args.device,
        quantize_cpu=args.quantize_cpu,
        port=args.port,
        connect_host=args.connect_host,
        output_dir=output_dir,
        save=not args.no_save,
        count=args.count,
        duplicate_threshold=args.duplicate_threshold,
        timeout=args.timeout,
        packet_len=args.packet_len,
        use_live_snr=args.use_live_snr,
        snr_port=args.snr_port,
        renorm=args.renorm,
        renorm_target=args.renorm_target,
        clip_mag=args.clip_mag,
        erase_snr_db=args.erase_snr_db,
        debug_dump=args.debug_dump,
        drop_spec=args.drop_slots,
        drop_seed=args.drop_seed,
        sentinel_drop_db=args.sentinel_drop_db,
        csi_db_scale=args.csi_db_scale,
        interleave=args.interleave,
        exp_id_mode=args.exp_id_mode,
        tx_gain=args.tx_gain,
        rx_gain=args.rx_gain,
    )


def main() -> int:
    args = parse_arguments()

    if not os.path.isfile(args.model):
        print(f"[!] Model checkpoint not found: {args.model}")
        return 2

    cfg = build_config(args)
    decoder = build_decoder(cfg)

    zmq_address = f"tcp://{cfg.connect_host}:{cfg.port}"
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 5000)
    socket.connect(zmq_address)

    print(f"[*] ZMQ PULL connected to {zmq_address} (PDU stream)")
    print(
        f"[*] Resolution: {cfg.width}x{cfg.height} | "
        f"tcn={cfg.tcn} | fallback SNR={cfg.snr_db} dB | "
        f"sentinel={cfg.sentinel_drop_db} dB (no-band, per-slot CSI)"
    )

    snr_socket = None
    if cfg.use_live_snr:
        snr_address = f"tcp://{cfg.connect_host}:{cfg.snr_port}"
        snr_socket = ctx.socket(zmq.PULL)
        snr_socket.setsockopt(zmq.RCVHWM, 5000)
        snr_socket.connect(snr_address)
        print(f"[*] ZMQ PULL connected to {snr_address} (live SNR feed)")
    else:
        print(
            "[*] Live SNR feed disabled (use --use-live-snr to enable). "
            "All received slots will use the fallback SNR; missing slots "
            "use the sentinel."
        )

    if cfg.save:
        print(f"[*] Saving to: {os.path.abspath(cfg.output_dir)}")
    else:
        print("[*] Save disabled (--no-save)")

    drop_policy = _DropPolicy(cfg.drop_spec, cfg.drop_seed)
    print(
        f"[*] Drop policy: {drop_policy} "
        f"(seed={cfg.drop_seed or 'nondeterministic'})"
    )

    try:
        receive_loop(decoder, socket, cfg, drop_policy, snr_socket)
        return 0
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
        return 130
    finally:
        try:
            socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass
        socket.close()
        if snr_socket is not None:
            try:
                snr_socket.setsockopt(zmq.LINGER, 0)
            except Exception:
                pass
            snr_socket.close()
        ctx.term()
        print("[*] Resources released. Exiting.")


if __name__ == "__main__":
    sys.exit(main())
