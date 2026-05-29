#!/usr/bin/env python3
"""
Adaptive DJSCC Image Transmitter (no-band variant).

The encoder in the no-band model is *identical* to the channel-blind encoder
in the original socket_adaptive_djscc_tx.py — it produces a robust
representation with no CSI input. The only thing that needs to change at TX
time is which module the Encoder class is loaded from.

Rather than duplicating ~600 lines, this file is a thin wrapper that:

  1. Pre-registers ``custom_djscc.codec_no_band`` under the module name
     ``custom_djscc.codec`` in ``sys.modules`` BEFORE importing the original
     TX script.
  2. Imports and runs the original ``socket_adaptive_djscc_tx.main()``.

Effect: the line ``from custom_djscc.codec import Encoder`` inside the
original TX resolves to the no-band Encoder class, which has an identical
public interface.

Pass ``--model`` pointing at a *no-band* checkpoint (i.e. trained with
train_no_band.py). All other CLI args behave exactly as in the original TX.

  python socket_adaptive_djscc_tx_no_band.py --model <no_band_ckpt> ...
"""

from __future__ import annotations

import importlib
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "djscc_models"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _install_no_band_codec_alias() -> None:
    """Make ``custom_djscc.codec`` resolve to ``custom_djscc.codec_no_band``
    for any subsequent imports in this process."""
    codec_nb = importlib.import_module("custom_djscc.codec_no_band")
    sys.modules["custom_djscc.codec"] = codec_nb
    # Also alias on the package so attribute access works
    # (e.g. ``custom_djscc.codec.Encoder``).
    import custom_djscc  # noqa: WPS433  (local import is intentional)
    custom_djscc.codec = codec_nb  # type: ignore[attr-defined]


def main() -> int:
    _install_no_band_codec_alias()

    # Importing here guarantees the alias is already in place when the
    # original TX module's top-level ``from custom_djscc.codec import
    # Encoder`` statement runs.
    tx = importlib.import_module("socket_adaptive_djscc_tx")

    # Quick sanity check: the loaded Encoder class should come from
    # codec_no_band, not from the original codec.
    enc_module = getattr(tx.Encoder, "__module__", "")
    if "codec_no_band" not in enc_module:
        print(
            f"[!] WARNING: TX imported Encoder from '{enc_module}' "
            "(expected '...codec_no_band'). The no-band redirection may "
            "not have taken effect; double-check sys.path and module load "
            "order."
        )
    else:
        print(f"[*] TX using Encoder from {enc_module}")

    return tx.main()


if __name__ == "__main__":
    sys.exit(main())
