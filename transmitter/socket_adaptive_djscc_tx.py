#!/usr/bin/env python3
"""
Adaptive DJSCC Image Transmitter with channel-blind encoder.

Same capture UX as ``socket_djscc_tx.py`` (camera / file / folder, interactive
or batch), but uses the custom DJSCC encoder that does NOT require channel
state information (SNR). The encoder produces a robust representation that
works across all channel conditions.

The encoder runs in-process and publishes complex64 channel symbols to a
slim GNU Radio flowgraph that only handles OFDM + USRP.

Examples:
  # Interactive camera preview
  python socket_adaptive_djscc_tx.py --model /path/to/checkpoint.pth

  # 3 automatic shots, 2s apart, 5 repeats each
  python socket_adaptive_djscc_tx.py --model checkpoint.pth \\
      --shots 3 --interval 2 --repeat 5

  # Single file, MPS device
  python socket_adaptive_djscc_tx.py --model checkpoint.pth \\
      --source file --path photo.jpg --device mps
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
import pmt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "djscc_models"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from custom_djscc.codec import Encoder  # noqa: E402


# ── UI constants ────────────────────────────────────────────────────────────
BTN_COLOR_IDLE    = (34, 139, 34)
BTN_COLOR_HOVER   = (0, 200, 0)
BTN_COLOR_ACTIVE  = (0, 80, 200)
BTN_COLOR_DONE    = (0, 180, 180)
OVERLAY_ALPHA     = 0.55
FONT              = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class TxConfig:
    model_path: str
    width: int
    height: int
    channel: int
    chn_in_len: int
    comp_ratio: float
    tcn: int
    N: int
    packet_len: int
    padding_zeros: int
    device: str
    quantize_cpu: bool
    repeat: int
    repeat_interval: float
    warmup_frames: int
    warmup_interval: float
    port: str
    topic: bytes
    direct_zmq: bool
    interleave: bool = False


# ── Symbol interleaver ──────────────────────────────────────────────────────
# Block interleaver over the padded complex-symbol stream — same construction
# as socket_conventional_tx.py's bit interleaver, applied to symbols instead.
# Symbols are laid out as a (num_slots x packet_len) matrix filled row-by-row
# and read column-by-column, so each OFDM packet ends up carrying one symbol
# from every original packet-strip of the latent.  A lost / weak packet then
# scatters its damage uniformly across the whole latent (diffuse speckle)
# instead of wiping one contiguous strip (a visible band).  The RX applies the
# inverse permutation before handing the latent to the decoder.  Must match
# --interleave on the RX.
def _make_interleaver(num_items: int, num_slots: int):
    """Return (perm_fwd, perm_inv) for a block interleaver over num_items.

      perm_fwd[src_pos] = tx_pos   — apply at RX: latent = rx[perm_fwd]
      perm_inv[tx_pos]  = src_pos  — apply at TX: tx    = latent[perm_inv]
    """
    bps = num_items // num_slots  # == packet_len when num_items = n_pkts*pkt_len
    p = np.arange(num_items, dtype=np.int64)
    perm_fwd = (p % num_slots) * bps + (p // num_slots)
    perm_inv = np.empty(num_items, dtype=np.int64)
    perm_inv[perm_fwd] = p
    return perm_fwd, perm_inv


_INTERLEAVE_CACHE: dict = {}


def _interleave_symbols(symbols: np.ndarray, packet_len: int) -> np.ndarray:
    """Apply the TX-side (forward) interleave to a padded symbol stream."""
    n = len(symbols)
    num_slots = n // packet_len
    key = (n, num_slots)
    perm_inv = _INTERLEAVE_CACHE.get(key)
    if perm_inv is None:
        _, perm_inv = _make_interleaver(n, num_slots)
        _INTERLEAVE_CACHE[key] = perm_inv
    return symbols[perm_inv]


def build_encoder(cfg: TxConfig) -> Encoder:
    print(f"[*] Loading channel-blind encoder from {cfg.model_path}")
    t0 = time.time()
    enc = Encoder(
        model_path=cfg.model_path,
        img_width=cfg.width,
        img_height=cfg.height,
        tcn=cfg.tcn,
        N=cfg.N,
        packet_len=cfg.packet_len,
        padding_zeros=cfg.padding_zeros,
        quantize_cpu=cfg.quantize_cpu,
        device=cfg.device,
        warmup=True,
    )
    print(f"[*] Encoder ready in {time.time() - t0:.2f}s "
          f"(device={enc.device}, tcn={cfg.tcn}, NO CSI)")
    return enc


def prepare_frame(image_bgr: np.ndarray, width: int, height: int) -> bytes:
    resized = cv2.resize(image_bgr, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.tobytes()


def publish_symbols(socket: zmq.Socket, topic: bytes,
                    symbols: np.ndarray, label: str = "") -> None:
    assert symbols.dtype == np.complex64
    payload = symbols.tobytes()
    if topic:
        socket.send_multipart([topic, payload])
    else:
        socket.send(payload)
    tag = f" [{label}]" if label else ""
    print(f"  TX publish {len(symbols)} cf32 ({len(payload)/1024:.1f} KiB){tag}")


_DIRECT_PACKET_COUNTER = [0]

def publish_symbols_pdu(socket: zmq.Socket, topic: bytes,
                        symbols: np.ndarray, cfg: TxConfig,
                        label: str = "") -> None:
    assert symbols.dtype == np.complex64, f"expected complex64, got {symbols.dtype}"
    pkt_len = cfg.packet_len
    if len(symbols) % pkt_len:
        raise ValueError(
            f"payload {len(symbols)} symbols is not a multiple of "
            f"packet_len = {pkt_len}")
    n_pkts = len(symbols) // pkt_len
    pn_start = _DIRECT_PACKET_COUNTER[0]

    serialized = []
    for i in range(n_pkts):
        chunk = symbols[i * pkt_len:(i + 1) * pkt_len]
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern("packet_num"),
                            pmt.from_long(_DIRECT_PACKET_COUNTER[0]))
        vec = pmt.init_c32vector(len(chunk), chunk.tolist())
        pdu = pmt.cons(meta, vec)
        serialized.append(pmt.serialize_str(pdu))
        _DIRECT_PACKET_COUNTER[0] += 1

    for s in serialized:
        socket.send(s)
    tag = f" [{label}]" if label else ""
    print(f"  TX direct-PDU push {n_pkts} pkts × {pkt_len} syms "
          f"(pn {pn_start}..{_DIRECT_PACKET_COUNTER[0] - 1}, "
          f"{len(symbols) * 8 / 1024:.1f} KiB){tag}")


def encode_and_send(encoder: Encoder, socket: zmq.Socket, cfg: TxConfig,
                    frame_bytes: bytes, label: str = "") -> None:
    t0 = time.time()
    symbols = encoder.encode(frame_bytes)
    if cfg.interleave:
        symbols = _interleave_symbols(symbols, cfg.packet_len)
    enc_ms = (time.time() - t0) * 1000.0
    print(f"  NN encode {len(symbols)} symbols in {enc_ms:.1f} ms"
          + ("  [interleaved]" if cfg.interleave else "")
          + (f" [{label}]" if label else ""))

    for r in range(cfg.repeat):
        rep_label = (f"{label} {r+1}/{cfg.repeat}" if label
                     else f"{r+1}/{cfg.repeat}")
        if cfg.direct_zmq:
            publish_symbols_pdu(socket, cfg.topic, symbols, cfg, label=rep_label)
        else:
            publish_symbols(socket, cfg.topic, symbols, label=rep_label)
        if r < cfg.repeat - 1:
            time.sleep(cfg.repeat_interval)


def send_warmup(encoder: Encoder, socket: zmq.Socket, cfg: TxConfig) -> None:
    if cfg.warmup_frames <= 0:
        print("[*] Warmup skipped.")
        return
    print(f"[*] Sending {cfg.warmup_frames} warmup frames for OFDM sync...")
    dummy = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8).tobytes()
    symbols = encoder.encode(dummy)
    if cfg.interleave:
        symbols = _interleave_symbols(symbols, cfg.packet_len)
    for i in range(cfg.warmup_frames):
        if cfg.direct_zmq:
            publish_symbols_pdu(socket, cfg.topic, symbols, cfg, label=f"warmup {i+1}")
        else:
            publish_symbols(socket, cfg.topic, symbols, label=f"warmup {i+1}")
        time.sleep(cfg.warmup_interval)
    print("[*] Warmup complete.")


# ── Interactive camera GUI ──────────────────────────────────────────────────
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

    win = "Adaptive DJSCC TX (channel-blind encoder)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cfg.width, cfg.height)
    cv2.imshow(win, np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8))
    cv2.waitKey(1)

    state = _SendState()
    trigger = [False]
    mouse_pos = [-1, -1]
    captured_frame: list[Optional[np.ndarray]] = [None]
    done_flash_end = [0.0]

    cv2.setMouseCallback(win, _mouse_cb,
                         param=(state, cfg.width, cfg.height, trigger, mouse_pos))

    send_warmup(encoder, socket, cfg)

    print(f"\n[*] Camera {cam_idx} live. Click CAPTURE & SEND or press Space.")
    print("[*] Press 'q' or Esc to quit.\n")

    def _send_worker(frame_bgr: np.ndarray) -> None:
        state.start_send()
        try:
            frame_bytes = prepare_frame(frame_bgr, cfg.width, cfg.height)
            encode_and_send(encoder, socket, cfg, frame_bytes,
                            label=f"shot #{state.shot_count + 1}")
        finally:
            state.finish_send()
            done_flash_end[0] = time.time() + 1.2

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


# ── Non-interactive modes ───────────────────────────────────────────────────
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


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive DJSCC Transmitter (channel-blind encoder)")

    p.add_argument("--model", type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--packet-len", type=int, default=960,
                   help="OFDM packet length in complex symbols (default: 960)")
    p.add_argument("--comp-ratio", type=int, default=12,
                   help="Inverse compression ratio (default: 12)")
    p.add_argument("--N", type=int, default=256,
                   help="Intermediate channel width (default: 256)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"],
                   help="Torch device (default: auto)")
    p.add_argument("--quantize-cpu", action="store_true",
                   help="Apply INT8 dynamic quantization on CPU")

    p.add_argument("--source", type=str, default="camera",
                   choices=["camera", "file", "folder"],
                   help="Input source (default: camera)")
    p.add_argument("--path", type=str, default="0",
                   help="Camera index, file path, or folder path (default: 0)")
    p.add_argument("--shots", type=int, default=0,
                   help="Auto-capture N shots (0 = interactive)")
    p.add_argument("--interval", type=float, default=3.0,
                   help="Seconds between auto shots (default: 3.0)")

    p.add_argument("--width", type=int, default=768,
                   help="Image width (default: 768)")
    p.add_argument("--height", type=int, default=512,
                   help="Image height (default: 512)")
    p.add_argument("--channel", type=int, default=3,
                   help="Image channels (default: 3)")

    p.add_argument("--repeat", type=int, default=1,
                   help="Times to re-publish each encoded frame (default: 1)")
    p.add_argument("--repeat-interval", type=float, default=0.5,
                   help="Seconds between repeats (default: 0.5)")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip warmup frames")
    p.add_argument("--warmup-frames", type=int, default=3,
                   help="Number of warmup frames (default: 3)")
    p.add_argument("--warmup-interval", type=float, default=0.5,
                   help="Seconds between warmup frames (default: 0.5)")

    p.add_argument("--port", type=str, default="5556",
                   help="ZMQ PUB port (default: 5556)")
    p.add_argument("--topic", type=str, default="",
                   help="Optional ZMQ topic prefix")
    p.add_argument("--bind-host", type=str, default="127.0.0.1",
                   help="ZMQ bind host (default: 127.0.0.1)")
    p.add_argument("--direct-zmq", action="store_true",
                   help="Skip GNU Radio: PUSH PMT PDUs (with packet_num "
                        "metadata) directly to the RX's PULL socket on port "
                        "5559. Pair with `--port 5559` on the RX.")
    p.add_argument("--interleave", action="store_true",
                   help="Block-interleave the encoded symbols across OFDM "
                        "packets before transmit, so a lost/weak packet "
                        "spreads as diffuse speckle over the whole image "
                        "instead of wiping one contiguous band. Must match "
                        "--interleave on the RX.")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> TxConfig:
    args.comp_ratio = 1 / args.comp_ratio
    tcn = int(args.comp_ratio * 4 * 4 * 2 * 3)
    chn_in_len = (tcn * (args.height // 4) * (args.width // 4)) // 2
    padding_zeros = (args.packet_len - chn_in_len % args.packet_len) % args.packet_len

    total_per_image = chn_in_len + padding_zeros
    total_per_image += (args.packet_len - total_per_image % args.packet_len) % args.packet_len
    print(f"[*] tcn={tcn}, chn_in_len={chn_in_len} cf32, "
          f"padding_zeros={padding_zeros}, "
          f"packets/image={total_per_image // args.packet_len}")

    if tcn <= 0:
        raise ValueError(
            f"--comp-ratio={int(round(1/args.comp_ratio))} produces tcn={tcn}; "
            f"valid options yield positive integer tcn.")

    return TxConfig(
        model_path=args.model,
        width=args.width,
        height=args.height,
        channel=args.channel,
        chn_in_len=chn_in_len,
        comp_ratio=args.comp_ratio,
        tcn=tcn,
        N=args.N,
        packet_len=args.packet_len,
        padding_zeros=padding_zeros,
        device=args.device,
        quantize_cpu=args.quantize_cpu,
        repeat=args.repeat,
        repeat_interval=args.repeat_interval,
        warmup_frames=0 if args.no_warmup else args.warmup_frames,
        warmup_interval=args.warmup_interval,
        port=args.port,
        topic=args.topic.encode("utf-8") if args.topic else b"",
        direct_zmq=args.direct_zmq,
        interleave=args.interleave,
    )


def main() -> int:
    args = parse_arguments()

    if not os.path.isfile(args.model):
        print(f"[!] Model checkpoint not found: {args.model}")
        return 2

    cfg = build_config(args)
    encoder = build_encoder(cfg)

    if cfg.direct_zmq:
        # Direct-to-RX debug: PUSH PDUs into the RX's PULL on 5559.
        zmq_address = f"tcp://{args.bind_host}:5559"
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 5000)
        socket.bind(zmq_address)
        print(f"[*] [DEBUG] direct-ZMQ mode: PUSH bound to {zmq_address} "
              "(skipping GR — RX must use `--port 5559`)")
    else:
        zmq_address = f"tcp://{args.bind_host}:{cfg.port}"
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 5000)
        socket.bind(zmq_address)
        print(f"[*] ZMQ PUB bound to {zmq_address}"
              + (f" (topic='{cfg.topic.decode()}')" if cfg.topic else ""))

    print(f"[*] Resolution: {cfg.width}x{cfg.height} | "
          f"tcn={cfg.tcn} | NO CSI | packet_len={cfg.packet_len}")
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
