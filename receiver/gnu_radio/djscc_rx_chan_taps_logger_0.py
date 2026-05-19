"""Header-gated channel-tap logger with packet_len validation.

Sits between digital_ofdm_chanest_vcvc and digital_ofdm_frame_equalizer_vcvc.
Passes vectors through unchanged. For every input item, reads any
ofdm_sync_chan_taps stream tags and appends the complex vector to a
pending buffer. On every header_valid PDU, parses the header's
packet_len field; only if it exactly equals the expected packet_len
(e.g. 960) AND packet_num is in a sane 24-bit non-negative range does
it commit the most-recent pending h (and its packet_num) to permanent
storage. Otherwise the message is treated as a spurious sync.

Output: .npz with two aligned arrays:
  - packet_num: int64 (N,)
  - h:          complex64 (N, fft_len)
Use packet_num to merge with pilot_snr_logger output offline.

Wire the 'header_data' output of digital_packet_headerparser_b to the
'header_valid' input of this block (parallel to its existing
connection to digital_header_payload_demux_0).
"""

import threading
import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):
    def __init__(self, fft_len=64, packet_len=960,
                 out_path='/tmp/h_dataset.npz', flush_every=200):
        gr.sync_block.__init__(
            self,
            name='chan_taps_logger',
            in_sig=[(np.complex64, fft_len)],
            out_sig=[(np.complex64, fft_len)],
        )
        self.fft_len = int(fft_len)
        self.packet_len = int(packet_len)
        self.out_path = str(out_path)
        self.flush_every = int(flush_every)
        self.tag_key = pmt.intern('ofdm_sync_chan_taps')
        self.pkt_num_key = pmt.intern('packet_num')
        self.pkt_len_key = pmt.intern('packet_len')
        self.pending = []
        self.max_pending = 8
        self.pn_list = []
        self.h_list = []
        self.n_rejected = 0
        self._lock = threading.Lock()
        self.msg_port = pmt.intern('header_valid')
        self.message_port_register_in(self.msg_port)
        self.set_msg_handler(self.msg_port, self._on_header_valid)

    def _extract_int(self, msg, key):
        """Robust int extraction. A PMT dict is internally a list of pairs, so
        is_pair() can't reliably distinguish dict vs (meta, payload) pair."""
        sentinel = pmt.from_long(-(1 << 30))
        candidates = [msg]
        try:
            if pmt.is_pair(msg):
                candidates.append(pmt.car(msg))
        except Exception:
            pass
        for cand in candidates:
            try:
                val = pmt.dict_ref(cand, key, sentinel)
                if not pmt.equal(val, sentinel):
                    return pmt.to_long(val)
            except Exception:
                continue
        return None

    def _on_header_valid(self, msg):
        pl = self._extract_int(msg, self.pkt_len_key)
        pn = self._extract_int(msg, self.pkt_num_key)
        if pl != self.packet_len or pn is None or pn < 0 or pn >= (1 << 24):
            self.n_rejected += 1
            return
        with self._lock:
            if not self.pending:
                return
            self.h_list.append(self.pending[-1])
            self.pn_list.append(int(pn))
            self.pending.clear()
            if len(self.pn_list) % self.flush_every == 0:
                self._flush()

    def _flush(self):
        if not self.pn_list:
            return
        np.savez(self.out_path,
                 packet_num=np.array(self.pn_list, dtype=np.int64),
                 h=np.array(self.h_list, dtype=np.complex64))

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        out0[:] = in0
        n = len(in0)
        for tag in self.get_tags_in_window(0, 0, n, self.tag_key):
            taps = np.array(pmt.c32vector_elements(tag.value), dtype=np.complex64)
            with self._lock:
                self.pending.append(taps)
                if len(self.pending) > self.max_pending:
                    self.pending = self.pending[-self.max_pending:]
        return n

    def stop(self):
        with self._lock:
            self._flush()
        print('[chan_taps_logger] committed=%d rejected=%d -> %s'
              % (len(self.pn_list), self.n_rejected, self.out_path))
        return True
