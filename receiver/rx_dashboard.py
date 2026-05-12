#!/usr/bin/env python3
"""
DeepJSCC RX dashboard.

Streamlit UI that:
  * starts/stops the GNU Radio RX flowgraph (e.g., djscc_rx.py) as a subprocess,
  * starts/stops the Python DeepJSCC/Conventional decoder as a subprocess,
  * displays the latest received image from the output directory.

Run:
  pip install streamlit
  streamlit run receiver/rx_dashboard.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
import streamlit as st

LOG_MAX_LINES = 500

GR_PID_FILE = Path("/tmp/djscc_rx_gr.pid")
GR_LOG_FILE = Path("/tmp/djscc_rx_gr.log")
PY_PID_FILE = Path("/tmp/djscc_rx_py.pid")
PY_LOG_FILE = Path("/tmp/djscc_rx_py.log")
TMP_CONFIG_PATH = Path("/tmp/tmp_rx_config.yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
GR_DIR = REPO_ROOT / "receiver" / "gnu_radio"
RX_DIR = REPO_ROOT / "receiver"
CONFIG_PATH = REPO_ROOT / "config.yaml"

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

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def read_pid(pid_file: Path) -> int | None:
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if not _pid_alive(pid):
        pid_file.unlink(missing_ok=True)
        return None
    return pid

def read_log_tail(log_file: Path, max_lines: int = LOG_MAX_LINES) -> str:
    if not log_file.exists():
        return ""
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, 128 * 1024)
            f.seek(size - read_size)
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])

def kill_process(pid_file: Path):
    pid = read_pid(pid_file)
    if pid is None:
        pid_file.unlink(missing_ok=True)
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    pid_file.unlink(missing_ok=True)

def start_rx() -> None:
    if read_pid(GR_PID_FILE) is not None or read_pid(PY_PID_FILE) is not None:
        return

    mode = st.session_state.get("rx_mode", "DJSCC")
    gr_script_name = "djscc_rx.py" if mode == "DJSCC" else "conventional_rx.py"
    gr_script = GR_DIR / gr_script_name

    current_cfg = get_current_config(load_base_config())
    TMP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TMP_CONFIG_PATH, "w") as f:
        yaml.safe_dump(current_cfg, f, sort_keys=False)

    if gr_script.is_file():
        GR_LOG_FILE.write_bytes(b"")
        gr_logf = open(GR_LOG_FILE, "ab", buffering=0)
        gr_proc = subprocess.Popen(
            [sys.executable, "-u", str(gr_script)],
            cwd=str(GR_DIR),
            stdout=gr_logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        GR_PID_FILE.write_text(str(gr_proc.pid))
    else:
        st.error(f"GR script not found: {gr_script}")

    PY_LOG_FILE.write_bytes(b"")
    py_logf = open(PY_LOG_FILE, "ab", buffering=0)

    if mode == "DJSCC":
        py_script = RX_DIR / "socket_djscc_rx.py"
        model_path_str = current_cfg["djscc"]["model_path"]
        model_path_full = Path(model_path_str)
        if not model_path_full.is_absolute():
            model_path_full = REPO_ROOT / model_path_full

        output_dir = REPO_ROOT / current_cfg["djscc"]["rx"].get("output_dir", "./received_images_djscc")
        
        cmd = [
            sys.executable, "-u", str(py_script),
            "--model", str(model_path_full),
            "--snr-db", str(current_cfg["djscc"]["snr_db"]),
            "--packet-len", str(current_cfg["common"]["packet_len"]),
            "--comp-ratio", str(current_cfg["djscc"]["comp_ratio"]),
            "--device", str(current_cfg["common"]["torch_device"]),
            "--width", str(current_cfg["common"]["image"]["width"]),
            "--height", str(current_cfg["common"]["image"]["height"]),
            "--channel", str(current_cfg["common"]["image"]["channel"]),
            "--port", str(current_cfg["zmq"]["djscc_rx_port"]),
            "--connect-host", str(current_cfg["zmq"]["host"]),
            "--output-dir", str(output_dir)
        ]
        if current_cfg["djscc"].get("quantize_cpu", False):
            cmd.append("--quantize-cpu")
    else:
        py_script = RX_DIR / "socket_conventional_rx.py"
        output_dir = REPO_ROOT / current_cfg["conventional"].get("output_dir", "./received_images_conventional")
        
        cmd = [
            sys.executable, "-u", str(py_script),
            "--width", str(current_cfg["common"]["image"]["width"]),
            "--height", str(current_cfg["common"]["image"]["height"]),
            "--channel", str(current_cfg["common"]["image"]["channel"]),
            "--codec", str(current_cfg["conventional"]["codec"]),
            "--ldpc-n", str(current_cfg["conventional"]["ldpc"]["n"]),
            "--ldpc-k", str(current_cfg["conventional"]["ldpc"]["k"]),
            "--bp-iters", str(current_cfg["conventional"]["ldpc"]["bp_iters"]),
            "--bits-per-symbol", str(current_cfg["conventional"]["modulation"]["bits_per_symbol"]),
            "--comp-ratio", str(current_cfg["conventional"]["comp_ratio_equivalent"]),
            "--packet-len", str(current_cfg["common"]["packet_len"]),
            "--demap", str(current_cfg["conventional"]["modulation"]["demap"]),
            "--device", str(current_cfg["common"]["torch_device"]),
            "--port", str(current_cfg["zmq"]["conventional_rx_port"]),
            "--connect-host", str(current_cfg["zmq"]["host"]),
            "--output-dir", str(output_dir)
        ]
        if current_cfg["conventional"].get("interleave", False):
            cmd.append("--interleave")

    if py_script.is_file():
        py_proc = subprocess.Popen(
            cmd,
            cwd=str(RX_DIR),
            stdout=py_logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        PY_PID_FILE.write_text(str(py_proc.pid))
    else:
        st.error(f"Python RX script not found: {py_script}")

    st.session_state["rx_started_at"] = time.time()

def stop_rx() -> None:
    kill_process(PY_PID_FILE)
    kill_process(GR_PID_FILE)

def get_latest_image(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    if not images:
        return None
    return max(images, key=lambda p: p.stat().st_mtime)

st.set_page_config(page_title="DeepJSCC RX", layout="wide")
st.title("DeepJSCC Receiver")

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

gr_pid = read_pid(GR_PID_FILE)
py_pid = read_pid(PY_PID_FILE)
running = gr_pid is not None or py_pid is not None

with st.sidebar:
    st.header("Demo control")

    rx_mode = st.selectbox("Reception Mode", ["DJSCC", "Conventional (SSCC)"], key="rx_mode", disabled=running)
    gr_script_name = "djscc_rx.py" if rx_mode == "DJSCC" else "conventional_rx.py"
    gr_script = GR_DIR / gr_script_name

    col_a, col_b = st.columns(2)
    col_a.button("▶ Start demo", type="primary", disabled=running,
                 use_container_width=True, on_click=start_rx)
    col_b.button("⏹ Stop demo", disabled=not running,
                 use_container_width=True, on_click=stop_rx)

    if running:
        started_at = st.session_state.get("rx_started_at")
        uptime_txt = f" · {time.time() - started_at:0.0f}s" if started_at else ""
        st.success(f"Running — GR PID {gr_pid or 'N/A'}, PY PID {py_pid or 'N/A'}{uptime_txt}")
    else:
        st.info("Stopped")

    st.caption(f"GR script: `{gr_script.relative_to(REPO_ROOT) if gr_script.is_absolute() else gr_script}`")

    st.divider()
    with st.expander("Python RX output", expanded=False):
        st.button("↻ Refresh Py Log", use_container_width=True, key="ref_py")
        tail_py = read_log_tail(PY_LOG_FILE)
        st.code(tail_py or "(no output yet)", language="text")
        st.caption(f"log: `{PY_LOG_FILE}`")

    with st.expander("GR RX output", expanded=False):
        st.button("↻ Refresh GR Log", use_container_width=True, key="ref_gr")
        tail_gr = read_log_tail(GR_LOG_FILE)
        st.code(tail_gr or "(no output yet)", language="text")
        st.caption(f"log: `{GR_LOG_FILE}`")

with tab_demo:
    current_cfg = get_current_config(load_base_config())
    rx_mode_val = st.session_state.get("rx_mode", "DJSCC")
    
    if rx_mode_val == "DJSCC":
        out_dir = REPO_ROOT / current_cfg["djscc"]["rx"].get("output_dir", "./received_images_djscc")
    else:
        out_dir = REPO_ROOT / current_cfg["conventional"].get("output_dir", "./received_images_conventional")

    col_img, col_side = st.columns([3, 2])
    
    with col_img:
        st.subheader("Received Image")
        latest_img_path = get_latest_image(out_dir)
        
        if latest_img_path:
            try:
                img_bytes = latest_img_path.read_bytes()
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.info(f"No images received yet in `{out_dir}`. Waiting for transmission...")
            
    with col_side:
        st.subheader("Status")
        st.button("↻ Refresh Image", use_container_width=True)
        if latest_img_path:
            st.success("Image received!")
            st.caption(f"**File:** `{latest_img_path.name}`")
            st.caption(f"**Time:** {time.strftime('%H:%M:%S', time.localtime(latest_img_path.stat().st_mtime))}")
            st.caption(f"**Directory:** `{out_dir}`")
        
        st.divider()
        st.markdown("**Note:**")
        st.markdown("The dashboard automatically polls the output directory for new images when the demo is running.")

if running:
    time.sleep(1.0)
    st.rerun()