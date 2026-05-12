#!/usr/bin/env python3
"""
DeepJSCC Image Transmitter with inline neural encoding.

Same capture UX as ``image_tx.py`` (camera / file / folder, interactive
or batch), but the DeepJSCC encoder runs **in-process** here.  Instead of
publishing raw RGB frames to a GNU Radio encoder block, this tool publishes
the resulting ``complex64`` channel symbols to a slim GR flowgraph that only
has to run OFDM + USRP.

Benefits:
  * The GR flowgraph has no heavy Python block on the scheduler thread —
    one less place for the USRP to underflow.
  * Iterating on the encoder (weights, SNR, device) is a Python-only loop;
    no GR rebuild.
  * Encoding happens once per image even when ``--repeat > 1``: we re-publish
    the cached complex samples.

Examples:
  # Interactive camera preview with CAPTURE button
  python image_tx_encoded.py --model /path/to/checkpoint.pth.tar

  # 3 automatic shots, 2s apart, 5 repeats each
  python image_tx_encoded.py --model checkpoint.pth.tar \\
      --shots 3 --interval 2 --repeat 5

  # Single file, MPS device, SNR=10 dB
  python image_tx_encoded.py --model checkpoint.pth.tar \\
      --source file --path photo.jpg --device mps --snr-db 10

  # All images in a folder
  python image_tx_encoded.py --model checkpoint.pth.tar \\
      --source folder --path ./photos/ --interval 3 --repeat 3
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import zmq

# Make the in-tree `deepjscc` package importable when running from video_test/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(
    os.path.join(_HERE, "..", "gr-modules", "gr-deepjscc", "python"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from deepjscc.codec import Encoder  # noqa: E402


# ── UI constants ────────────────────────────────────────────────────────────
BTN_COLOR_IDLE    = (34, 139, 34)    # green
BTN_COLOR_HOVER   = (0, 200, 0)      # bright green
BTN_COLOR_ACTIVE  = (0, 80, 200)     # blue — "sending"
BTN_COLOR_DONE    = (0, 180, 180)    # cyan — brief "sent!" flash
OVERLAY_ALPHA     = 0.55
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
# Config + encoder factory
# =====================================================================
@dataclass
class TxConfig:
    model_path: str
    width: int
    height: int
    channel: int
    chn_in_len: int
    comp_ratio: int
    tcn: int
    packet_len: int
    snr_db: float
    padding_zeros: int
    device: str
    quantize_cpu: bool
    repeat: int
    repeat_interval: float
    warmup_frames: int
    warmup_interval: float
    port: str
    topic: bytes


def build_encoder(cfg: TxConfig) -> Encoder:
    """Instantiate the DeepJSCC encoder with timings printed for visibility."""
    print(f"[*] Loading encoder from {cfg.model_path}")
    t0 = time.time()
    enc = Encoder(
        model_path=cfg.model_path,
        img_width=cfg.width,
        img_height=cfg.height,
        tcn=cfg.tcn,
        snr_db=cfg.snr_db,
        packet_len=cfg.packet_len,
        padding_zeros=cfg.padding_zeros,
        quantize_cpu=cfg.quantize_cpu,
        device=cfg.device,
        warmup=True,
    )
    print(f"[*] Encoder ready in {time.time() - t0:.2f}s "
          f"(device={enc.device}, tcn={cfg.tcn}, SNR={cfg.snr_db} dB)")
    return enc


# =====================================================================
# Capture helpers
# =====================================================================
def prepare_frame(image_bgr: np.ndarray, width: int, height: int) -> bytes:
    """Resize, convert BGR -> RGB, return raw HWC uint8 bytes."""
    resized = cv2.resize(image_bgr, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.tobytes()


# =====================================================================
# Encode + publish
# =====================================================================
def publish_symbols(socket: zmq.Socket, topic: bytes,
                    symbols: np.ndarray, label: str = "") -> None:
    """Publish a complex64 symbol array as a single ZMQ message."""
    assert symbols.dtype == np.complex64, \
        f"expected complex64, got {symbols.dtype}"
    payload = symbols.tobytes()
    if topic:
        socket.send_multipart([topic, payload])
    else:
        socket.send(payload)
    tag = f" [{label}]" if label else ""
    print(f"  TX publish {len(symbols)} cf32 ({len(payload)/1024:.1f} KiB){tag}")


def encode_and_send(encoder: Encoder, socket: zmq.Socket, cfg: TxConfig,
                    frame_bytes: bytes, label: str = "") -> None:
    """Encode once, then publish ``cfg.repeat`` copies of the symbols."""
    t0 = time.time()
    symbols = encoder.encode(frame_bytes)
    enc_ms = (time.time() - t0) * 1000.0
    print(f"  NN encode {len(symbols)} symbols in {enc_ms:.1f} ms"
          + (f" [{label}]" if label else ""))

    for r in range(cfg.repeat):
        publish_symbols(socket, cfg.topic, symbols,
                        label=f"{label} {r+1}/{cfg.repeat}" if label
                              else f"{r+1}/{cfg.repeat}")
        if r < cfg.repeat - 1:
            time.sleep(cfg.repeat_interval)


def send_warmup(encoder: Encoder, socket: zmq.Socket, cfg: TxConfig) -> None:
    """Publish encoded black frames so the GR/OFDM/USRP chain can acquire
    sync before the first real image arrives."""
    if cfg.warmup_frames <= 0:
        print("[*] Warmup skipped.")
        return
    print(f"[*] Sending {cfg.warmup_frames} warmup frames for OFDM sync...")
    dummy = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8).tobytes()
    symbols = encoder.encode(dummy)
    for i in range(cfg.warmup_frames):
        publish_symbols(socket, cfg.topic, symbols, label=f"warmup {i+1}")
        time.sleep(cfg.warmup_interval)
    print("[*] Warmup complete.")


# =====================================================================
# Interactive camera GUI (button / HUD)
# =====================================================================
def _btn_rect(w: int, h: int) -> tuple[int, int, int, int]:
    bw, bh = 260, 54
    x1 = (w - bw) // 2
    y1 = h - bh - 20
    return x1, y1, x1 + bw, y1 + bh


def _draw_button(canvas: np.ndarray, w: int, h: int,
                 color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = _btn_rect(w, h)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, canvas, 1 - OVERLAY_ALPHA, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    tw, th = cv2.getTextSize(text, FONT, 0.72, 2)[0]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, shot_count: int, status: str) -> None:
    cv2.putText(canvas, f"Shots sent: {shot_count}", (12, 34),
                FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Shots sent: {shot_count}", (12, 34),
                FONT, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
    if status:
        cv2.putText(canvas, status, (12, 68),
                    FONT, 0.6, (0, 220, 220), 2, cv2.LINE_AA)


def _point_in_rect(px: int, py: int, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


class _SendState:
    """Tiny thread-safe state machine shared between GUI and sender thread."""
    IDLE    = "idle"
    SENDING = "sending"
    DONE    = "done"

    def __init__(self) -> None:
        self.state = self.IDLE
        self.shot_count = 0
        self._lock = threading.Lock()

    def start_send(self) -> None:
        with self._lock:
            self.state = self.SENDING

    def finish_send(self) -> None:
        with self._lock:
            self.state = self.DONE
            self.shot_count += 1

    def ack_done(self) -> None:
        with self._lock:
            if self.state == self.DONE:
                self.state = self.IDLE

    @property
    def is_sending(self) -> bool:
        with self._lock:
            return self.state == self.SENDING

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self.state, self.shot_count


def _mouse_cb(event, x, y, flags, param) -> None:
    state, w, h, trigger, mouse_pos = param
    mouse_pos[0], mouse_pos[1] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        if _point_in_rect(x, y, _btn_rect(w, h)) and not state.is_sending:
            trigger[0] = True


def camera_interactive(encoder: Encoder, socket: zmq.Socket,
                       cfg: TxConfig, cam_path: str) -> None:
    cam_idx = int(cam_path) if cam_path.isdigit() else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_idx}")
        return

    time.sleep(0.8)
    for _ in range(5):
        cap.read()

    win = "DeepJSCC TX (inline encode)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cfg.width, cfg.height)
    cv2.imshow(win, np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8))
    cv2.waitKey(1)

    state = _SendState()
    trigger = [False]
    mouse_pos = [-1, -1]
    captured_frame: list[Optional[np.ndarray]] = [None]
    done_flash_end = [0.0]
    # ###
    # last_tx_time = [time.time()] 
    # ###

    cv2.setMouseCallback(win, _mouse_cb,
                         param=(state, cfg.width, cfg.height, trigger, mouse_pos))

    send_warmup(encoder, socket, cfg)

    print(f"\n[*] Camera {cam_idx} live. Click CAPTURE & SEND or press Space.")
    print("[*] Press 'q' or Esc to quit.\n")

    def _send_worker(frame_bgr: np.ndarray) -> None:
        state.start_send()
        try:
            # ###
            # if time.time() - last_tx_time[0] > 2.00 and cfg.warmup_frames > 0:
            #     print("[*] Idle gap > 2s detected. Firing warmup burst to stabilize RX AGC...")
            #     old_frames, old_int = cfg.warmup_frames, cfg.warmup_interval
            #     send_warmup(encoder, socket, cfg)
            #     cfg.warmup_frames, cfg.warmup_interval = old_frames, old_int
            # ###
            frame_bytes = prepare_frame(frame_bgr, cfg.width, cfg.height)
            encode_and_send(encoder, socket, cfg, frame_bytes,
                            label=f"shot #{state.shot_count + 1}")
        finally:
            state.finish_send()
            done_flash_end[0] = time.time() + 1.2
            # ###
            # last_tx_time[0] = time.time()
            # ###

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                print("[!] Failed to read camera frame.")
                break

            display = cv2.resize(raw, (cfg.width, cfg.height))
            now = time.time()
            cur_state, shot_count = state.snapshot()

            if cur_state == _SendState.SENDING:
                if captured_frame[0] is not None:
                    display = cv2.resize(captured_frame[0], (cfg.width, cfg.height))
                color = BTN_COLOR_ACTIVE
                text = f"  Encoding + Sending ({cfg.repeat}x)..."
                status = "Transmitting over SDR..."
            elif cur_state == _SendState.DONE or now < done_flash_end[0]:
                color = BTN_COLOR_DONE
                text = "  Sent!"
                status = f"Shot #{shot_count} transmitted ({cfg.repeat}x)"
                state.ack_done()
            else:
                hover = _point_in_rect(mouse_pos[0], mouse_pos[1],
                                       _btn_rect(cfg.width, cfg.height))
                color = BTN_COLOR_HOVER if hover else BTN_COLOR_IDLE
                text = "[ CAPTURE & SEND ]"
                status = ("Space or click to capture" if shot_count == 0
                          else f"Last: shot #{shot_count}")

            _draw_hud(display, shot_count, status)
            _draw_button(display, cfg.width, cfg.height, color, text)
            cv2.imshow(win, display)

            if trigger[0] and not state.is_sending:
                trigger[0] = False
                captured_frame[0] = raw.copy()
                threading.Thread(target=_send_worker, args=(raw.copy(),),
                                 daemon=True).start()
                print(f"[*] Shot #{shot_count + 1} captured & queued for encode+TX")

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and not state.is_sending:
                trigger[0] = True
            elif key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[*] Session ended. Total shots sent: {state.shot_count}")


# =====================================================================
# Non-interactive modes
# =====================================================================
def camera_auto(encoder: Encoder, socket: zmq.Socket,
                cfg: TxConfig, cam_path: str,
                shots: int, interval: float) -> None:
    cam_idx = int(cam_path) if cam_path.isdigit() else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_idx}")
        return

    print(f"[*] Camera {cam_idx} opened. Taking {shots} shot(s)...")
    time.sleep(1.0)
    for _ in range(5):
        cap.read()

    send_warmup(encoder, socket, cfg)

    try:
        for i in range(shots):
            if i > 0:
                print(f"[*] Waiting {interval}s before next shot...")
                time.sleep(interval)
            ret, frame = cap.read()
            if not ret:
                print(f"[!] Failed to capture shot {i+1}")
                continue
            print(f"\n[*] Shot {i+1}/{shots} captured")
            frame_bytes = prepare_frame(frame, cfg.width, cfg.height)
            encode_and_send(encoder, socket, cfg, frame_bytes,
                            label=f"shot {i+1}/{shots}")
    finally:
        cap.release()


def send_from_file(encoder: Encoder, socket: zmq.Socket,
                   cfg: TxConfig, path: str) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[!] Error: Could not read '{path}'")
        return
    print(f"[*] Loaded image: {path}")
    send_warmup(encoder, socket, cfg)
    frame_bytes = prepare_frame(frame, cfg.width, cfg.height)
    encode_and_send(encoder, socket, cfg, frame_bytes,
                    label=os.path.basename(path))


def send_from_folder(encoder: Encoder, socket: zmq.Socket,
                     cfg: TxConfig, path: str, interval: float) -> None:
    if not os.path.isdir(path):
        print(f"[!] Error: Directory '{path}' not found")
        return
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    all_files = sorted(glob.glob(os.path.join(path, "*.*")))
    images = [f for f in all_files
              if os.path.splitext(f)[1].lower() in valid_exts]
    if not images:
        print(f"[!] No valid images in '{path}'")
        return

    print(f"[*] Found {len(images)} images in '{path}'")
    send_warmup(encoder, socket, cfg)
    for i, img_path in enumerate(images):
        if i > 0:
            print(f"\n[*] Waiting {interval}s...")
            time.sleep(interval)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        name = os.path.basename(img_path)
        print(f"\n[*] Image {i+1}/{len(images)}: {name}")
        frame_bytes = prepare_frame(frame, cfg.width, cfg.height)
        encode_and_send(encoder, socket, cfg, frame_bytes, label=name)


# =====================================================================
# Argument parsing + main
# =====================================================================
def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepJSCC Image Transmitter with inline neural encoding")

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
    p.add_argument("--port", type=str, default="5556",
                   help="ZMQ PUB port (default: 5556 — distinct from raw-frame "
                        "port 5555 used by image_tx.py)")
    p.add_argument("--topic", type=str, default="",
                   help="Optional ZMQ topic prefix (default: none)")
    p.add_argument("--bind-host", type=str, default="127.0.0.1",
                   help="ZMQ bind host (default: 127.0.0.1)")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> TxConfig:

    args.comp_ratio = 1/args.comp_ratio
    tcn = int(args.comp_ratio * 4 * 4 * 2 * 3)
    chn_in_len = (tcn * (args.height//4) * (args.width//4) )//2
    padding_zeros = (args.packet_len - chn_in_len % args.packet_len) % args.packet_len

    total_per_image = chn_in_len + padding_zeros
    total_per_image += (args.packet_len - total_per_image % args.packet_len) % args.packet_len
    print(
        f"[*] tcn={tcn}, chn_in_len={chn_in_len} cf32,", f"padding_zeros={padding_zeros},", f"packets/image={total_per_image // args.packet_len}")

    if tcn <= 0:
        raise ValueError(
            f"--comp-ratio={int(round(1/args.comp_ratio))} produces tcn={tcn}; "
            f"valid options yield positive integer tcn (e.g. comp-ratio 6→16, 12→8)."
        )

    
    return TxConfig(
        model_path=args.model,
        width=args.width,
        height=args.height,
        channel=args.channel,
        chn_in_len = chn_in_len,
        comp_ratio = args.comp_ratio,
        tcn=tcn,
        packet_len=args.packet_len,
        snr_db=args.snr_db,
        padding_zeros=padding_zeros,
        device=args.device,
        quantize_cpu=args.quantize_cpu,
        repeat=args.repeat,
        repeat_interval=args.repeat_interval,
        warmup_frames=0 if args.no_warmup else args.warmup_frames,
        warmup_interval=args.warmup_interval,
        port=args.port,
        topic=args.topic.encode("utf-8") if args.topic else b"",
    )


def main() -> int:
    args = parse_arguments()

    if not os.path.isfile(args.model):
        print(f"[!] Model checkpoint not found: {args.model}")
        return 2

    cfg = build_config(args)
    encoder = build_encoder(cfg)

    zmq_address = f"tcp://{args.bind_host}:{cfg.port}"
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    #socket = ctx.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 5000)
    socket.bind(zmq_address)

    print(f"[*] ZMQ PUB bound to {zmq_address}"
          + (f" (topic='{cfg.topic.decode()}')" if cfg.topic else ""))
    print(f"[*] Resolution: {cfg.width}x{cfg.height} | "
          f"tcn={cfg.tcn} | SNR={cfg.snr_db} dB | packet_len={cfg.packet_len}")
    print(f"[*] Repeat: {cfg.repeat}x per image, {cfg.repeat_interval}s apart")
    print("[*] Waiting 4s for ZMQ subscribers to connect...")
    time.sleep(4.0)

    try:
        if args.source == "camera":
            if args.shots == 0:
                camera_interactive(encoder, socket, cfg, args.path)
            else:
                camera_auto(encoder, socket, cfg, args.path,
                            args.shots, args.interval)
        elif args.source == "file":
            send_from_file(encoder, socket, cfg, args.path)
        elif args.source == "folder":
            send_from_folder(encoder, socket, cfg, args.path, args.interval)

        print("\n[*] All transmissions complete.")
        return 0

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
        return 130
    finally:
        socket.close()
        ctx.term()
        print("[*] Resources released. Exiting.")


if __name__ == "__main__":
    sys.exit(main())