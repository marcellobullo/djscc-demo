#!/usr/bin/env python3
"""
DeepJSCC TX dashboard — v2.

Streamlit UI that:
  * shows a live camera preview (browser webcam via streamlit-webrtc),
  * starts/stops the GNU Radio TX flowgraph (burst_tx.py) as a subprocess,
  * runs the DeepJSCC encoder in-process, and
  * on "Capture & Send" encodes the latest camera frame and publishes the
    resulting complex64 symbols to ZMQ port 5556 (what burst_tx.py subscribes
    to).

Run:
  pip install streamlit streamlit-webrtc
  streamlit run video_test/tx_dashboard.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import yaml

import av  # noqa: F401  (pulled in by streamlit-webrtc; import here to fail fast)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import zmq
from streamlit_webrtc import webrtc_streamer, WebRtcMode


LOG_MAX_LINES = 500

# Persistent on-disk state so we survive browser refresh / session reset.
PID_FILE = Path("/tmp/djscc_tx_gr.pid")
LOG_FILE = Path("/tmp/djscc_tx_gr.log")
TMP_CONFIG_PATH = Path("/tmp/tmp_config.yaml")

# Where the optional Save button writes captured frames.
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"


# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
GR_DIR    = REPO_ROOT / "transmitter" / "gnu_radio"
DEEPJSCC_PKG = REPO_ROOT / "gr-modules" / "gr-deepjscc" / "python"
CONFIG_PATH = REPO_ROOT / "config.yaml"

if str(DEEPJSCC_PKG) not in sys.path:
    sys.path.insert(0, str(DEEPJSCC_PKG))

from deepjscc.codec import Encoder  # noqa: E402

def load_base_config():
    if TMP_CONFIG_PATH.exists():
        with open(TMP_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

base_config = load_base_config()

def init_session_state(d, prefix=""):
    for k, v in d.items():
        key = f"cfg_{prefix}{k}"
        if isinstance(v, dict):
            init_session_state(v, prefix + k + "_")
        else:
            if key not in st.session_state:
                st.session_state[key] = v

init_session_state(base_config)

def get_current_config(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"cfg_{prefix}{k}"
        if isinstance(v, dict):
            out[k] = get_current_config(v, prefix + k + "_")
        else:
            out[k] = st.session_state.get(key, v)
    return out

# ── Encoder + ZMQ (cached so they persist across reruns) ──────────────────
@st.cache_resource(show_spinner="Loading DeepJSCC encoder…")
def get_encoder(model_path: str, width: int, height: int, tcn: int,
                snr_db: float, packet_len: int, device: str) -> Encoder:
    return Encoder(
        model_path=model_path,
        img_width=width,
        img_height=height,
        tcn=tcn,
        snr_db=snr_db,
        packet_len=packet_len,
        padding_zeros=0,
        quantize_cpu=False,
        device=device,
        warmup=True,
    )


@st.cache_resource(show_spinner="Loading Conventional encoder…")
def get_conventional_encoder(width: int, height: int, codec: str, codec_quality: int, 
                             ldpc_n: int, ldpc_k: int, bits_per_symbol: int, target_complex_symbols: int, 
                             packet_len: int, interleave: bool):
    from socket_conventional_tx import ConventionalEncoder, TxConventionalConfig
    cfg = TxConventionalConfig(
        width=width, height=height, channel=3,
        codec=codec, codec_quality=codec_quality, target_bytes=0, fit_to_budget=True,
        ldpc_n=ldpc_n, ldpc_k=ldpc_k, bits_per_symbol=bits_per_symbol,
        target_complex_symbols=target_complex_symbols, packet_len=packet_len,
        repeat=1, repeat_interval=0, warmup_frames=0, warmup_interval=0,
        port="", bind_host="", topic=b"", direct_zmq=False, interleave=interleave
    )
    return ConventionalEncoder(cfg)

@st.cache_resource
def get_zmq_socket(port: int) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 50)
    sock.bind(f"tcp://127.0.0.1:{port}")
    # Small delay so late SUB connects can catch up before the first send.
    time.sleep(0.3)
    return sock

@st.cache_resource
def get_plot_socket(port: int = 5560) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.CONFLATE, 1)  # Only keep the absolute latest message, drop stale queued data
    sock.connect(f"tcp://127.0.0.1:{port}")
    return sock

def get_all_new_data(sock: zmq.Socket) -> list[bytes]:
    chunks = []
    while True:
        try:
            chunks.append(sock.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            break
    return chunks


def prepare_frame(image_bgr: np.ndarray, width: int, height: int) -> bytes:
    """Resize + BGR->RGB + raw HWC uint8 bytes (matches image_tx_encoded.py)."""
    resized = cv2.resize(image_bgr, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.tobytes()


def save_capture_to_disk(bgr: np.ndarray) -> Path:
    """Write a BGR frame to CAPTURES_DIR with a timestamped filename."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = CAPTURES_DIR / f"tx_{ts}.png"
    cv2.imwrite(str(path), bgr)
    return path


# ── Process helpers (pidfile-based so we survive browser refresh) ────────
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def read_gr_pid() -> int | None:
    """Return the GR child's PID if it's still running, else None."""
    try:
        pid = int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if not _pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return None
    return pid


def read_log_tail(max_lines: int = LOG_MAX_LINES) -> str:
    if not LOG_FILE.exists():
        return ""
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, 128 * 1024)  # last 128 KiB is plenty
            f.seek(size - read_size)
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def start_gr() -> None:
    if read_gr_pid() is not None:
        return

    mode = st.session_state.get("tx_mode", "DJSCC")
    gr_script_name = "djscc_tx.py" if mode == "DJSCC" else "conventional_tx.py"
    gr_script = GR_DIR / gr_script_name
    if not gr_script.is_file():
        st.error(f"GR script not found: {gr_script}")
        return

    # Save the config to tmp_config.yaml
    current_cfg = get_current_config(load_base_config())
    TMP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TMP_CONFIG_PATH, "w") as f:
        yaml.safe_dump(current_cfg, f, sort_keys=False)

    # Truncate the log so each run starts fresh.
    LOG_FILE.write_bytes(b"")
    logf = open(LOG_FILE, "ab", buffering=0)
    
    gr_cfg = current_cfg.get("gnuradio", {})
    device_address = gr_cfg.get("tx", {}).get("device_address", "192.168.1.68")
    samp_rate = gr_cfg.get("samp_rate", 1000000.0)
    carrier_freq = gr_cfg.get("carrier_freq", 2450000000.0)
    band = gr_cfg.get("band", 5000000.0)
    mod_order = current_cfg.get("gnuradio", {}).get("mod_order", 2)

    cmd_args = [
        sys.executable, "-u", str(gr_script), 
        "--device-address", str(device_address),
        "--samp-rate", str(samp_rate),
        "--carrier-freq", str(carrier_freq),
        "--band", str(band)
    ]
    if mode != "DJSCC":
        cmd_args.extend(["--mod-order", str(mod_order)])
            
    proc = subprocess.Popen(
        cmd_args,
        cwd=str(GR_DIR),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own pgroup, so we can kill the whole tree
    )
    PID_FILE.write_text(str(proc.pid))
    st.session_state["gr_started_at"] = time.time()


def stop_gr() -> None:
    pid = read_gr_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return
    # Give it up to 5s to exit gracefully, then SIGKILL.
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        # macOS returns EPERM (not ESRCH) when the process is already gone
        # and its PID has been recycled — treat both as "already dead".
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    PID_FILE.unlink(missing_ok=True)


# --- previous Popen-handle based version (kept for reference) ---
# def _proc_alive(proc): return proc is not None and proc.poll() is None
# def _drain_stdout(proc, buffer, lock): ...   # stdin→deque reader thread
# def start_gr():  # Popen → session_state["gr_proc"], reader thread
# def stop_gr():   # kills via session_state handle (lost on refresh)


# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DeepJSCC TX", layout="wide")
st.title("DeepJSCC Transmitter")

tab_demo, tab_config = st.tabs(["Demo", "Configuration"])

def draw_config_ui(d, prefix=""):
    for k, v in d.items():
        key = f"cfg_{prefix}{k}"
        if isinstance(v, dict):
            with st.expander(k.capitalize(), expanded=True):
                draw_config_ui(v, prefix + k + "_")
        else:
            if key == "cfg_common_torch_device":
                options = ["auto", "cpu", "mps", "cuda"]
                st.selectbox(k, options, key=key)
            elif key == "cfg_conventional_codec":
                options = ["jpeg", "jpeg2000"]
                st.selectbox(k, options, key=key)
            elif key == "cfg_conventional_modulation_demap":
                options = ["soft", "hard"]
                st.selectbox(k, options, key=key)
            elif isinstance(v, bool):
                st.checkbox(k, key=key)
            elif isinstance(v, int):
                st.number_input(k, step=1, key=key)
            elif isinstance(v, float):
                st.number_input(k, step=0.1, format="%.2f", key=key)
            else:
                st.text_input(k, key=key)

with tab_config:
    st.header("Configuration Parameters")
    draw_config_ui(base_config)

class CameraProcessor:
    """Class-based WebRTC video processor.

    The instance lives inside the streamlit-webrtc component as long as the
    stream is open, so ``ctx.video_processor.get_latest()`` survives reruns —
    unlike a plain ``video_frame_callback`` closure, which gets re-created on
    every rerun and loses its capture.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._frame = img
        return frame

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# --- previous callback-based approach (lost frames after rerun) ---
# if "frame_slot" not in st.session_state:
#     st.session_state["frame_slot"] = {"bgr": None, "lock": threading.Lock()}
# frame_slot = st.session_state["frame_slot"]
# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     with frame_slot["lock"]:
#         frame_slot["bgr"] = img
#     return frame


gr_pid = read_gr_pid()
running = gr_pid is not None

with st.sidebar:
    st.header("Demo control")

    tx_mode = st.selectbox("Transmission Mode", ["DJSCC", "Conventional (SSCC)"], key="tx_mode", disabled=running)
    gr_script_name = "djscc_tx.py" if tx_mode == "DJSCC" else "conventional_tx.py"
    gr_script = GR_DIR / gr_script_name

    col_a, col_b = st.columns(2)
    col_a.button("▶ Start demo", type="primary", disabled=running,
                 width="stretch", on_click=start_gr)
    col_b.button("⏹ Stop demo", disabled=not running,
                 width="stretch", on_click=stop_gr)

    if running:
        started_at = st.session_state.get("gr_started_at")
        if started_at is not None:
            uptime_txt = f" · {time.time() - started_at:0.0f}s"
        else:
            uptime_txt = ""  # started before this browser session — no uptime
        st.success(f"Running — PID {gr_pid}{uptime_txt}")
    else:
        st.info("Stopped")

    st.caption(f"GR script: `{gr_script.relative_to(REPO_ROOT)}`")

    st.divider()
    with st.expander("GR output", expanded=False):
        # Manual refresh — avoids a 1 s auto-rerun that was remounting the
        # webrtc camera component and making the preview "disappear".
        st.button("↻ Refresh log", width="stretch")
        tail = read_log_tail()
        st.code(tail or "(no output yet)", language="text")
        st.caption(f"log: `{LOG_FILE}`")

with tab_demo:
    # Main area: camera on the left, capture + thumbnail on the right so
    # everything is visible without scrolling.
    col_cam, col_side = st.columns([3, 2])
    
    with col_cam:
        st.subheader("Camera preview")
        # NOTE: SENDRECV (not SENDONLY) is required for the local preview to
        # render. In SENDONLY mode browsers only show the <video> during device
        # selection and clear it once the peer connection is up. With SENDRECV
        # the server echoes the frames back (our processor is a passthrough),
        # and the <video> element renders the returned stream.
        ctx = webrtc_streamer(
            key="tx-camera",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False,
            },
            # Safari will not render a local preview unless all three of
            # muted / autoPlay / playsInline are set. Without these the
            # capture still works server-side but the <video> stays blank.
            video_html_attrs={
                "muted": True,
                "autoPlay": True,
                "playsInline": True,
                "controls": False,
                "style": {"width": "100%", "height": "auto", "border-radius": "8px"},
            },
            video_processor_factory=CameraProcessor,
            async_processing=True,
        )
        st.caption("Click **START** above to allow your browser's webcam. "
                   "The preview appears after you grant permission.")
    
    with col_side:
        st.subheader("Capture")

        tx_mode = st.session_state.get("tx_mode", "DJSCC")
        
        btn_disabled = False
        if tx_mode == "DJSCC":
            model_path_str = st.session_state["cfg_djscc_model_path"]
            model_path_full = Path(model_path_str)
            if not model_path_full.is_absolute():
                model_path_full = REPO_ROOT / model_path_full
            if not model_path_full.is_file():
                btn_disabled = True
                st.caption("⚠ Model checkpoint path does not exist.")

        if st.button("📸 Capture & Send", type="primary", width="stretch",
                     disabled=btn_disabled):
            latest = None
            if ctx.video_processor is not None:
                latest = ctx.video_processor.get_latest()
            if latest is None:
                st.warning("No camera frame yet — click START above the preview and "
                           "allow browser webcam access, then wait a moment.")
            else:
                width = st.session_state["cfg_common_image_width"]
                height = st.session_state["cfg_common_image_height"]
                packet_len = st.session_state["cfg_common_packet_len"]

                try:
                    t0 = time.time()
                    if tx_mode == "DJSCC":
                        comp_ratio = st.session_state["cfg_djscc_comp_ratio"]
                        snr_db = st.session_state["cfg_djscc_snr_db"]
                        device = st.session_state["cfg_common_torch_device"]
                        zmq_port = st.session_state["cfg_zmq_djscc_tx_port"]
                        
                        tcn = int((1.0 / comp_ratio) * 4 * 4 * 2 * 3)

                        encoder = get_encoder(str(model_path_full), int(width), int(height),
                                              int(tcn), float(snr_db), int(packet_len), device)
                        socket  = get_zmq_socket(zmq_port)
                        
                        rgb_bytes = prepare_frame(latest, int(width), int(height))
                        symbols = encoder.encode(rgb_bytes)
                        payload = symbols.tobytes()
                        symbol_info = f"{len(symbols)} symbols"
                    else:
                        comp_ratio = st.session_state["cfg_conventional_comp_ratio_equivalent"]
                        tcn = int((1.0 / comp_ratio) * 4 * 4 * 2 * 3)
                        target_symbols = (tcn * (height // 4) * (width // 4)) // 2
                        codec = st.session_state["cfg_conventional_codec"]
                        codec_quality = st.session_state["cfg_conventional_codec_quality"]
                        ldpc_n = st.session_state["cfg_conventional_ldpc_n"]
                        ldpc_k = st.session_state["cfg_conventional_ldpc_k"]
                        bps = st.session_state["cfg_conventional_modulation_bits_per_symbol"]
                        interleave = st.session_state["cfg_conventional_interleave"]
                        zmq_port = st.session_state["cfg_zmq_conventional_tx_port"]
                        
                        encoder = get_conventional_encoder(int(width), int(height), codec, int(codec_quality),
                                                           int(ldpc_n), int(ldpc_k), int(bps), int(target_symbols),
                                                           int(packet_len), interleave)
                        socket = get_zmq_socket(zmq_port)
                        
                        symbols = encoder.encode(latest)
                        payload = symbols.tobytes()
                        symbol_info = f"{len(symbols)} bytes"

                    socket.send(payload)
                    dt = (time.time() - t0) * 1000
                    st.session_state["last_sent_bgr"] = latest
                    st.session_state["shots_sent"] = st.session_state.get("shots_sent", 0) + 1
                    st.success(f"Sent {symbol_info} in {dt:.0f} ms "
                               f"(total shots: {st.session_state['shots_sent']})")
                    if not running:
                        st.info("GR is not running — symbols were published but nothing "
                                "is transmitting over the air.")
                except Exception as e:
                    st.error(f"Encoder / ZMQ failed: {e}")
    
        last = st.session_state.get("last_sent_bgr")
        if last is not None:
            # Header row: caption on the left, a close (X) button in the top-right.
            # Streamlit doesn't support absolute positioning inside a component,
            # so "top-right of the preview" is realised as a right-aligned column
            # just above the image.
            hdr_l, hdr_r = st.columns([6, 1])
            hdr_l.caption(f"Last sent (#{st.session_state.get('shots_sent', 0)})")
            if hdr_r.button("✕", key="close_preview", help="Clear the preview"):
                st.session_state.pop("last_sent_bgr", None)
                st.rerun()
    
            st.image(cv2.cvtColor(last, cv2.COLOR_BGR2RGB), width="stretch")
    
            if st.button("💾 Save image", key="save_preview",
                         help=f"Save to {CAPTURES_DIR}"):
                path = save_capture_to_disk(last)
                st.toast(f"Saved {path.name}", icon="💾")
                st.caption(f"Saved to `{path}`")

    st.divider()
    st.subheader("Live Constellation (Port 5560)")
    
    plot_sock = get_plot_socket(5560)
    
    col_plt, col_plt_ctl = st.columns([4, 1])
    with col_plt_ctl:
        st.caption("Requires a **ZMQ PUB Sink** on port `5560` in your GR flowgraph publishing `complex64` vectors.")
        st.caption("*(Tip: place the PUB sink **before** OFDM carrier allocation to see the pure symbols!)*")
        if st.button("↻ Capture Data", use_container_width=True):
            pass # Rerun naturally fetches the latest data
        if st.button("✕ Clear Plot", use_container_width=True):
            st.session_state["constellation_history"] = np.array([], dtype=np.complex64)
            st.rerun()
            
    with col_plt:
        chunks = get_all_new_data(plot_sock)
        if chunks:
            if "constellation_history" not in st.session_state:
                st.session_state["constellation_history"] = np.array([], dtype=np.complex64)
            try:
                new_symbols = np.concatenate([np.frombuffer(c, dtype=np.complex64) for c in chunks])
                # Filter out pure zeros (often used as idle padding between bursts)
                #new_symbols = new_symbols[np.abs(new_symbols) > 1e-6]
                new_symbols = new_symbols[np.abs(new_symbols) > 1e-4]
                
                hist = np.concatenate((st.session_state["constellation_history"], new_symbols))
                if len(hist) > 1024:
                    hist = hist[-1024:]
                st.session_state["constellation_history"] = hist
            except Exception as e:
                st.error(f"Error parsing plot data: {e}")
                
        hist = st.session_state.get("constellation_history", np.array([], dtype=np.complex64))
        if len(hist) > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            # ax.scatter(np.real(hist), np.imag(hist), alpha=0.3, s=5, c='blue', edgecolors='none')
            # ax.scatter(np.real(hist), np.imag(hist), alpha=0.6, s=10, c='blue', edgecolors='none')
            ax.plot(np.real(hist), np.imag(hist), marker='o', linestyle='', color='#0044ff', markersize=3, alpha=0.6)
            ax.axhline(0, color='black', lw=0.5)
            ax.axvline(0, color='black', lw=0.5)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel("In-Phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            st.pyplot(fig)
        else:
            st.info("No active symbols received on tcp://127.0.0.1:5560 yet (idle zeros are ignored). Ensure GR is running and publishing data to this port.")

# --- previous 1 s auto-rerun (remounted webrtc, caused preview flicker) ---
# if running:
#     time.sleep(1.0)
#     st.rerun()
