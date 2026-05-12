#!/usr/bin/env python3
"""
Standalone Python ZMQ client to read raw OFDM symbols and compute SNR.
"""

import argparse
import time
import numpy as np
import zmq

def main():
    parser = argparse.ArgumentParser(description="Standalone Python SNR Monitor")
    parser.add_argument("--port", type=str, default="5561", help="ZMQ PULL port (default: 5561)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ZMQ host (default: 127.0.0.1)")
    args = parser.parse_args()

    zmq_address = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 5000)
    socket.connect(zmq_address)

    print(f"[*] Python SNR Monitor connected to {zmq_address}")
    print("[*] Waiting for raw OFDM symbols...")

    # OFDM setup
    fft_len = 64
    occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) +
                         list(range(-6, 0)) + list(range(1, 7)) +
                         list(range(8, 21)) + list(range(22, 27)))
    pilot_carriers = [-21, -7, 7, 21]
    pilot_symbols = np.array([1, 1, 1, -1], dtype=np.complex64)

    # Shifted indices (since shift=True in fft_vcc, DC is at index 32)
    pilot_idx = [k + 32 for k in pilot_carriers]
    data_idx = [k + 32 for k in occupied_carriers if k not in pilot_carriers]

    burst_es_acc = 0.0
    burst_noise_acc = 0.0
    burst_sym_count = 0

    try:
        while True:
            # Poll with 300ms timeout to detect the end of a burst
            if socket.poll(300):
                raw = socket.recv()
                # ZMQ push sink from GR with vlen=64 sends 64 complex numbers per array
                symbols = np.frombuffer(raw, dtype=np.complex64).reshape(-1, fft_len)
                
                for sym in symbols:
                    # Skip if it's dead air / zero power
                    total_pwr = np.mean(np.abs(sym[data_idx])**2)
                    if total_pwr < 1e-12:
                        continue
                        
                    # GR has already equalized the symbol. Just measure the residual error on pilots.
                    err = sym[pilot_idx] - pilot_symbols
                    burst_noise_acc += np.mean(np.abs(err)**2)
                    
                    # Accumulate total power from data carriers
                    burst_es_acc += total_pwr
                    burst_sym_count += 1
            else:
                # 300ms elapsed without data -> End of burst
                if burst_sym_count > 0:
                    avg_total = burst_es_acc / burst_sym_count
                    avg_noise = burst_noise_acc / burst_sym_count
                    
                    # True Signal Energy (E_s) = Total Power - Noise Power
                    true_es = avg_total - avg_noise
                    
                    if avg_noise > 1e-12 and true_es > 1e-12:
                        snr_db = 10 * np.log10(true_es / avg_noise)
                        print(f"[*] Burst completed. Mean SNR: {snr_db:6.2f} dB (averaged over {burst_sym_count} symbols)")
                    else:
                        print(f"[*] Burst completed. (Too noisy or weak to estimate SNR, {burst_sym_count} symbols)")
                        
                    # Reset for the next burst
                    burst_es_acc = 0.0
                    burst_noise_acc = 0.0
                    burst_sym_count = 0

    except KeyboardInterrupt:
        print("\n[*] Exiting SNR Monitor.")
    finally:
        socket.close()
        ctx.term()

if __name__ == "__main__":
    main()