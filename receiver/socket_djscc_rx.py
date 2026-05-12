#!/usr/bin/env python3
"""
DeepJSCC Image Receiver with PDU-based inline neural decoding.

Subscribes to a single ZMQ PUB socket that streams GNU Radio PDUs
(containing BOTH header metadata and complex64 payload symbols) to perfectly
synchronize the OFDM stream for the DeepJSCC decoder in-process, and displays /
saves the reconstructed images with OpenCV.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import zmq
import pmt
import math

# Make the in-tree `deepjscc` package importable when running from video_test/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(
    os.path.join(_HERE, "..", "gr-modules", "gr-deepjscc", "python"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from deepjscc.codec import Decoder

@dataclass
class RxConfig:
    model_path: str
    width: int
    height: int
    channel: int
    chn_in_len: int
    comp_ratio: float
    tcn: int
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
    debug_dump: str = ""

def build_decoder(cfg: RxConfig) -> Decoder:
    print(f"[*] Loading decoder from {cfg.model_path}")
    t0 = time.time()
    dec = Decoder(
        model_path=cfg.model_path,
        img_width=cfg.width,
        img_height=cfg.height,
        tcn=cfg.tcn,
        snr_db=cfg.snr_db,
        quantize_cpu=cfg.quantize_cpu,
        device=cfg.device,
        warmup=True,
    )
    print(f"[*] Decoder ready in {time.time() - t0:.2f}s "
          f"(device={dec.device}, tcn={cfg.tcn}, SNR={cfg.snr_db} dB, "
          f"expected_symbols={dec.expected_complex_items})")
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

def receive_loop(decoder: Decoder, socket: zmq.Socket, cfg: RxConfig) -> None:

    # Expected number of complex symbols per frame.
    expected = decoder.expected_complex_items

    # Complex symbols per packet.
    PKT_LEN = cfg.packet_len

    # Number of `PKT_LEN`-sized packets needed to carry the full compressed
    # image of `expected` symbols. The last packet may be partially filled
    # with valid symbols and zero-padded.
    PKT_PER_IMG = math.ceil(expected / PKT_LEN)
    samples_per_image = PKT_PER_IMG * PKT_LEN

    # Pre-allocated buffer for one image's worth of complex payload.
    slot_buf = np.zeros(samples_per_image, dtype=np.complex64)

    # Boolean checklist of which slots have arrived for the current image.
    seen = [False] * PKT_PER_IMG

    # Per-image forensic log: list of (packet_num, slot) tuples actually
    # placed in slot_buf. Reset alongside `seen` and `slot_buf`. Used for
    # `--debug-dump` so the user can replay decodes offline.
    pn_log: list = []

    # The TX-side `packet_num` counter is free-running and not aligned with
    # image boundaries (the TX flowgraph doesn't know when a new image
    # starts). So we cannot use `packet_num // PKT_PER_IMG` directly to find
    # image boundaries — that would split one image's symbols across two
    # image_idx buckets whenever a 205-packet burst happens to cross such a
    # boundary, producing colored-noise decodes.
    #
    # Instead, we anchor the slot grid on the first packet received after an
    # idle gap (which marks "user pressed send"). For each subsequent packet:
    #   delta = packet_num - anchor_pn
    #   slot           = delta %  PKT_PER_IMG
    #   image_idx_local = delta // PKT_PER_IMG  (0 for the first image after
    #                                            anchor, 1 for the next, …)
    # This is drift-free for the whole 24-bit packet_num range — the wide
    # field guarantees `delta` never wraps within one TX session.
    anchor_pn = None
    current_image_idx = None

    n_images = 0

    window_title = "DJSCC Received (PDU lock-step)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, cfg.width, cfg.height)

    if cfg.save:
        os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.debug_dump:
        os.makedirs(cfg.debug_dump, exist_ok=True)
        print(f"[*] Forensic dumps -> {os.path.abspath(cfg.debug_dump)}")

    unique_images = []
    last_saved_img = None
    total_frames = 0
    last_frame_time = time.time()
    unique_count = 0

    last_valid_pdu_time = time.time()

    print(f"[*] Lock-step RX: tcn={decoder.tcn}, "
          f"{PKT_PER_IMG} packets/image, {PKT_LEN} syms/packet, "
          f"expected={expected}, samples_per_image={samples_per_image}")
    print("[*] Waiting for a valid transmission burst...")

    def _flush_and_decode():
        """Decode whatever is currently in slot_buf, update display/save state.
        Caller is responsible for zeroing slot_buf and seen afterwards."""
        nonlocal n_images, total_frames, last_frame_time, unique_count
        nonlocal last_saved_img

        n_missed = sum(1 for ok in seen if not ok)
        if n_missed == PKT_PER_IMG:
            return

        # Forensic dump *before* any mutation of slot_buf. The user can
        # replay-decode offline by loading slot_buf, applying any candidate
        # normalization, slicing slot_buf[:expected], and feeding it to the
        # decoder. Saves the raw on-the-wire data exactly as received.
        if cfg.debug_dump:
            try:
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1e3)%1000:03d}"
                dump_path = os.path.join(
                    cfg.debug_dump,
                    f"image_{n_images+1:04d}_{stamp}.npz")
                pn_arr  = np.array([pn  for pn, _ in pn_log], dtype=np.int64)
                slt_arr = np.array([slt for _, slt in pn_log], dtype=np.int32)
                np.savez(dump_path,
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
                         height=np.int32(cfg.height))
                print(f"  [*] forensic dump -> {os.path.basename(dump_path)} "
                      f"({len(pn_arr)} packets, {n_missed} missing)")
            except Exception as e:
                print(f"  [!] forensic dump failed: {e}")

        if n_missed:
            print(f"  [*] image {n_images+1}: "
                  f"{n_missed}/{PKT_PER_IMG} slots missing, zero-filled")
            for i, ok in enumerate(seen):
                if not ok:
                    slot_buf[i*PKT_LEN:(i+1)*PKT_LEN] = 0

        symbols = slot_buf[:expected].copy()

        pwr = float(np.mean(np.abs(symbols)**2))
        mags = np.abs(symbols)
        clip_mask = mags > 2.0
        num_clipped = int(np.sum(clip_mask))
        if num_clipped > 0:
            print(f"  [!] WARNING: {num_clipped} symbols exploded. Clipping their magnitudes to 2.0.")
            symbols[clip_mask] = (symbols[clip_mask] / mags[clip_mask]) * 2.0

        print(f"[!] MAX POWER: {np.max(np.abs(symbols))}")
        print(f"[!] MIN POWER: {np.min(np.abs(symbols))}")
        print(f"[!] AVERAGE POWER BEFORE CLIPPING: {pwr:.2f}, AFTER: {np.mean(np.abs(symbols)**2):.2f}")

        total_frames += 1
        last_frame_time = time.time()

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
                    print(f"  [*] Frame {total_frames}: duplicate "
                          f"(better quality, updated {os.path.basename(old_path)}, "
                          f"decode={dec_ms:.1f} ms)")
                else:
                    print(f"  [*] Frame {total_frames}: duplicate "
                          f"(skipped, sim={sim:.3f}, decode={dec_ms:.1f} ms)")

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
            print(f"  [*] Frame {total_frames}: NEW image #{unique_count} "
                  f"{'saved -> ' + filename if cfg.save else '(not saved)'} "
                  f"(q={quality:.1f}, decode={dec_ms:.1f} ms)")

        display = frame_bgr.copy()
        label = f"#{unique_count} | frame {total_frames}"
        if is_duplicate:
            label += " [DUP]"
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_title, display)

    try:
        while True:
            if cfg.timeout > 0 and total_frames > 0 \
                    and time.time() - last_frame_time > cfg.timeout:
                print(f"\n[*] No frames for {cfg.timeout}s. Exiting.")
                break

            # Idle-gap flush. Runs every loop iteration (including the
            # zmq.Again path) so it fires even when TX has stopped sending.
            # Decodes whatever partial image we've accumulated and clears the
            # anchor so the next packet of the next "send" re-anchors the
            # slot grid.
            if current_image_idx is not None \
                    and time.time() - last_valid_pdu_time > 0.3:
                n_seen = sum(1 for ok in seen if ok)
                print(f"\n[*] 0.3s idle gap detected. Flushing partial image "
                      f"({n_seen}/{PKT_PER_IMG} slots received).")
                _flush_and_decode()
                seen = [False] * PKT_PER_IMG
                slot_buf.fill(0)
                pn_log = []
                current_image_idx = None
                anchor_pn = None

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
                packet_num = pmt.to_long(pmt.dict_ref(metadata, key_pn, pmt.from_long(-1)))
                assert packet_num >= 0, f"Invalid packet_num in metadata: {packet_num}"

                iq_pkt = np.array(pmt.c32vector_elements(payload), dtype=np.complex64)

            except Exception as e:
                print(f"[!] PDU parse failed: {e}")
                continue

            if len(iq_pkt) != PKT_LEN:
                print(f"[!] Anomalous PDU: got {len(iq_pkt)} cf32, expected {PKT_LEN} ")
            if len(iq_pkt) < PKT_LEN:
                continue
            iq_pkt = iq_pkt[:PKT_LEN]

            last_valid_pdu_time = time.time()

            # Anchor the slot grid on the first packet of each "send" burst.
            # Anchor is cleared on every idle gap, so each user-driven
            # transmission starts at slot 0 of a fresh image regardless of
            # the absolute packet_num.
            if anchor_pn is None:
                anchor_pn = packet_num

            delta = packet_num - anchor_pn
            slot = delta % PKT_PER_IMG
            image_idx_local = delta // PKT_PER_IMG
            print(f"[*] PDU Metadata: packet_num={packet_num} "
                  f"anchor={anchor_pn} delta={delta} "
                  f"image_idx={image_idx_local} slot={slot}")

            # Image-boundary detection: when image_idx_local advances, the
            # current image is done and the next 205 packets belong to the
            # next image in this same burst (back-to-back send case).
            if current_image_idx is None:
                current_image_idx = image_idx_local
            elif image_idx_local != current_image_idx:
                _flush_and_decode()
                seen = [False] * PKT_PER_IMG
                slot_buf.fill(0)
                pn_log = []
                current_image_idx = image_idx_local

            slot_buf[slot*PKT_LEN:(slot+1)*PKT_LEN] = iq_pkt
            seen[slot] = True
            pn_log.append((packet_num, slot))

            # Fast path: when slot 204 arrives the image is complete on-air.
            # Decode whatever we have (missing slots will be zero-filled by
            # _flush_and_decode). Reset image-state but keep anchor_pn so a
            # back-to-back next image still aligns correctly.
            if slot == PKT_PER_IMG - 1:
                _flush_and_decode()
                seen = [False] * PKT_PER_IMG
                slot_buf.fill(0)
                pn_log = []
                current_image_idx = None

            if cfg.count > 0 and unique_count >= cfg.count:
                print(f"\n[*] Received {unique_count} unique image(s). Done.")
                cv2.waitKey(2000)
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[*] Receiver stopped.")
    finally:
        cv2.destroyAllWindows()
        print(f"\n{'='*50}")
        print(f"  Total frames received:  {total_frames}")
        print(f"  Unique images:          {unique_count}")
        if cfg.save:
            print(f"  Output directory:       {os.path.abspath(cfg.output_dir)}")
            for _, q, path in unique_images:
                print(f"    - {os.path.basename(path)} (q={q:.1f})")
        print(f"{'='*50}")

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepJSCC Image Receiver with PDU-based inline neural decoding")
    
     # Model / encoder
    p.add_argument("--model", type=str, required=True,
                   help="Path to DeepJSCC checkpoint (.pth.tar)")
    p.add_argument("--snr-db", type=float, default=19.0,
                   help="SNR in dB passed to the attention network (default: 10.0)")
    p.add_argument("--packet-len", type=int, default=960,
                   help="OFDM packet length in complex symbols (default: 960)")
    p.add_argument("--comp-ratio", type=int, default=12,
                   help="Inverse compression ratio (default: 12)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"],
                   help="Torch device for encoding (default: auto)")
    p.add_argument("--quantize-cpu", action="store_true",
                   help="Apply INT8 dynamic quantization on CPU for speed")

    # Source
    p.add_argument("--source", type=str, default="camera",
                   choices=["camera", "file", "folder"],
                   help="Input source (default: camera)")
    p.add_argument("--path", type=str, default="0",
                   help="Camera index, file path, or folder path (default: 0)")
    p.add_argument("--shots", type=int, default=0,
                   help="Auto-capture N shots (0 = interactive, default: 0)")
    p.add_argument("--interval", type=float, default=3.0,
                   help="Seconds between auto shots (default: 3.0)")

    # Image size
    p.add_argument("--width", type=int, default=768,
                   help="Image width (default: 768)")
    p.add_argument("--height", type=int, default=512,
                   help="Image height (default: 512)")
    p.add_argument("--channel", type=int, default=3,
                   help="Image channel (default: 3)")

    # Transmission
    p.add_argument("--repeat", type=int, default=1,
                   help="Times to re-publish each encoded frame (default: 1 )")
    p.add_argument("--repeat-interval", type=float, default=0.5,
                   help="Seconds between repeat transmissions (default: 0.5)")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip warmup frames at startup")
    p.add_argument("--warmup-frames", type=int, default=3,
                   help="Number of warmup frames to send (default: 3)")
    p.add_argument("--warmup-interval", type=float, default=0.5,
                   help="Seconds between warmup frames (default: 0.5)")

    # ZMQ
    p.add_argument("--port", type=str, default="5558",
                   help="ZMQ PUB port (default: 5558")
    p.add_argument("--topic", type=str, default="",
                   help="Optional ZMQ topic prefix (default: none)")
    p.add_argument("--connect-host", type=str, default="127.0.0.1",
                   help="ZMQ connect host (default: 127.0.0.1)")

    # To remove
    p.add_argument("--output-dir", type=str, default="./received_images",
                   help="Directory to save images (default: ./received_images)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing PNGs to disk; display only")
    p.add_argument("--count", type=int, default=0,
                   help="Exit after N unique images (0 = unlimited)")
    p.add_argument("--duplicate-threshold", type=float, default=0.92,
                   help="NCC threshold to treat two frames as duplicates (default: 0.92)")
    p.add_argument("--timeout", type=float, default=0,
                   help="Seconds of idle before auto-exit (0 = never)")
    p.add_argument("--debug-dump", type=str, default="",
                   help="Directory to dump per-image .npz forensic files "
                        "(slot_buf, seen, packet_nums, anchor) for offline "
                        "decode replay; disabled if empty")

    return p.parse_args()

def build_config(args: argparse.Namespace) -> RxConfig:

    args.comp_ratio = 1/args.comp_ratio
    tcn = int(args.comp_ratio * 4 * 4 * 2 * 3)
    chn_in_len = (tcn * (args.height//4) * (args.width//4) )//2
    padding_zeros = (args.packet_len - chn_in_len%args.packet_len) % args.packet_len
    
    total_per_image = chn_in_len + padding_zeros
    total_per_image += (args.packet_len - total_per_image % args.packet_len) % args.packet_len
    
    print(f"[*] tcn={tcn}, chn_in_len={chn_in_len} cf32, "
        f"padding_zeros={padding_zeros}, "
        f"packets/image={total_per_image // args.packet_len}")
    
    if tcn <= 0:
        raise ValueError(
            f"--comp-ratio={int(round(1/args.comp_ratio))} produces tcn={tcn}; "
            f"valid options yield positive integer tcn (e.g. comp-ratio 6→16, 12→8).")
    
    return RxConfig(
        model_path=args.model,
        width=args.width,
        height=args.height,
        channel=args.channel,
        chn_in_len = chn_in_len,
        comp_ratio = args.comp_ratio,
        tcn=tcn,
        packet_len=args.packet_len,
        snr_db=args.snr_db,
        device=args.device,
        quantize_cpu=args.quantize_cpu,
        port=args.port,
        connect_host=args.connect_host,
        output_dir=args.output_dir,
        save=not args.no_save,
        count=args.count,
        duplicate_threshold=args.duplicate_threshold,
        timeout=args.timeout,
        debug_dump=args.debug_dump,
    )

def main() -> int:
    args = parse_arguments()

    if not os.path.isfile(args.model):
        print(f"[!] Model checkpoint not found: {args.model}")
        return 2

    # Construct config
    cfg = build_config(args)

    # Build decoder
    decoder = build_decoder(cfg)

    # Set up ZMQ SUB socket to receive PDUs from the GR flowgraph publisher
    zmq_address = f"tcp://{cfg.connect_host}:{cfg.port}"
    ctx = zmq.Context()
    #socket = ctx.socket(zmq.SUB)
    socket = ctx.socket(zmq.PULL)

    socket.setsockopt(zmq.RCVHWM, 5000)
    socket.connect(zmq_address) # this code is a client, therefore it is connecting and not binding
    #socket.setsockopt(zmq.SUBSCRIBE, b"")

    #print(f"[*] ZMQ SUB connected to {zmq_address} (PDU stream)")
    print(f"[*] ZMQ PULL connected to {zmq_address} (PDU stream)")
    print(f"[*] Resolution: {cfg.width}x{cfg.height} | "
          f"tcn={cfg.tcn} | SNR={cfg.snr_db} dB")
    
    # Save directory info
    if cfg.save:
        print(f"[*] Saving to: {os.path.abspath(cfg.output_dir)}")
    else:
        print("[*] Save disabled (--no-save)")

    # Receive loop with graceful shutdown on Ctrl+C
    try:
        receive_loop(decoder, socket, cfg)
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
        ctx.term()
        print("[*] Resources released. Exiting.")

if __name__ == "__main__":
    sys.exit(main())