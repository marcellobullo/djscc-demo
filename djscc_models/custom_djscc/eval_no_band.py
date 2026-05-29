"""Evaluation script for the no-band DJSCC model.

Reports PSNR (and optionally MS-SSIM / LPIPS) across SNR values, with
optional band-corruption stress tests:

  --drop-prob   p   simulate per-packet drops at probability p
  --strip-only      drop a contiguous range of packets (worst case for raster
                    mapping) to verify the model handles strip-erasures

Compared to eval.py, this version:
  - Builds the per-element CSI map from a per-packet SNR vector.
  - Uses the same PacketwiseChannel as training.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_no_band import (
    DJSCCDecoderSpatialCSI,
    DJSCCEncoder,
    DJSCCModelNoBand,
    packetwise_to_element_map,
)
from train_no_band import PacketwiseChannel
from dataset import build_val_loader
from kaira.constraints.power import AveragePowerConstraint

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from pytorch_msssim import ssim, ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def evaluate(
    model: DJSCCModelNoBand,
    channel: PacketwiseChannel,
    loader: DataLoader,
    device: torch.device,
    n_pkts: int,
    snr_list: list[float],
    drop_prob: float,
    strip_pkt_range: tuple[int, int] | None,
    sentinel_drop_db: float,
    lpips_fn=None,
):
    model.eval()
    results = {}

    for snr_val in snr_list:
        psnr_all = []
        lpips_all = []
        ssim_all = []
        ms_ssim_all = []

        for images, _ in loader:
            images = images.to(device)
            B, C, H, W = images.shape

            encoded = model.encoder(images)
            constrained = model.constraint(encoded)

            snr_db_per_pkt = torch.full(
                (B, n_pkts), float(snr_val),
                dtype=torch.float32, device=device,
            )

            drop_mask = None
            csi_snr = snr_db_per_pkt
            if strip_pkt_range is not None:
                lo, hi = strip_pkt_range
                drop_mask = torch.zeros(B, n_pkts, dtype=torch.bool, device=device)
                drop_mask[:, lo:hi] = True
            elif drop_prob > 0:
                drop_mask = torch.rand(B, n_pkts, device=device) < drop_prob

            if drop_mask is not None:
                csi_snr = torch.where(
                    drop_mask,
                    torch.full_like(snr_db_per_pkt, sentinel_drop_db),
                    snr_db_per_pkt,
                )

            received = channel(constrained, snr_db_per_pkt, None, drop_mask)
            Mch, Hlat, Wlat = constrained.shape[1], constrained.shape[2], constrained.shape[3]
            csi_map = packetwise_to_element_map(
                csi_snr, M=Mch, H=Hlat, W=Wlat,
                pkt_len_complex=channel.pkt_len,
            )
            decoded = model.decoder(received, csi_map)

            mse_per = F.mse_loss(
                decoded.view(B, -1), images.view(B, -1), reduction="none"
            ).mean(dim=1)
            psnr_per = 10.0 * torch.log10(1.0 / mse_per)
            psnr_all.extend(psnr_per.cpu().numpy())

            if HAS_MSSSIM and min(H, W) >= 160:
                s = ssim(decoded, images, data_range=1.0, size_average=True)
                ms = ms_ssim(decoded, images, data_range=1.0, size_average=True)
                ms_db = -10.0 * torch.log10(1.0 - ms)
                ssim_all.append(s.item())
                ms_ssim_all.append(ms_db.item())

            if HAS_LPIPS and lpips_fn is not None:
                lp = lpips_fn(decoded, images).mean().item()
                lpips_all.append(lp)

        row = {"psnr": float(np.mean(psnr_all))}
        if ms_ssim_all:
            row["ms_ssim_db"] = float(np.mean(ms_ssim_all))
            row["ssim"] = float(np.mean(ssim_all))
        if lpips_all:
            row["lpips"] = float(np.mean(lpips_all))
        results[snr_val] = row

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate no-band DJSCC model")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--val-data-dir", type=str, nargs="+", default=["./data/kodak"])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--tcn", type=int, default=16)
    p.add_argument("--csi-embed-dim", type=int, default=64)
    p.add_argument("--csi-db-scale", type=float, default=20.0)
    p.add_argument("--img-width", type=int, default=768)
    p.add_argument("--img-height", type=int, default=512)
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--snr-list", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--drop-prob", type=float, default=0.0)
    p.add_argument("--strip-range", type=str, default="",
                   help="Force-drop a contiguous packet range, e.g. '40:50' "
                        "(slots 40..49). Overrides --drop-prob if set.")
    p.add_argument("--sentinel-drop-db", type=float, default=-20.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-lpips", action="store_true")

    args = p.parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    encoder = DJSCCEncoder(N=args.N, M=args.tcn)
    decoder = DJSCCDecoderSpatialCSI(
        N=args.N, M=args.tcn,
        csi_embed_dim=args.csi_embed_dim,
        csi_db_scale=args.csi_db_scale,
    )
    constraint = AveragePowerConstraint(average_power=1.0)
    model = DJSCCModelNoBand(encoder=encoder, decoder=decoder, constraint=constraint)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["net"] if "net" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARN: missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"WARN: unexpected keys (first 5): {list(unexpected)[:5]}")
    model.to(device)
    model.eval()

    print(f"Loaded: epoch={ckpt.get('epoch', '?')}, "
          f"train_avg_PSNR={ckpt.get('Ave_PSNR', '?')}")

    channel = PacketwiseChannel(pkt_len_complex=args.packet_len)
    complex_per_image = (args.tcn * (args.img_height // 4) * (args.img_width // 4)) // 2
    n_pkts = int(np.ceil(complex_per_image / args.packet_len))

    strip_range = None
    if args.strip_range:
        lo_s, hi_s = args.strip_range.split(":")
        strip_range = (int(lo_s), int(hi_s))
        print(f"Forced strip drop: packets [{strip_range[0]}, {strip_range[1]})")

    lpips_fn = None
    if HAS_LPIPS and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)

    img_size = (args.img_width, args.img_height)
    val_loader = build_val_loader(args.val_data_dir, img_size, 1, args.num_workers)
    print(f"Val images: {len(val_loader.dataset)}, n_pkts={n_pkts}")
    print("-" * 70)

    results = evaluate(
        model, channel, val_loader, device, n_pkts,
        args.snr_list, args.drop_prob, strip_range,
        args.sentinel_drop_db, lpips_fn,
    )

    header = f"{'SNR':>6s}  {'PSNR':>8s}"
    has_ssim = any("ms_ssim_db" in v for v in results.values())
    has_lp = any("lpips" in v for v in results.values())
    if has_ssim:
        header += f"  {'MS-SSIM(dB)':>11s}  {'SSIM':>6s}"
    if has_lp:
        header += f"  {'LPIPS':>7s}"
    print(header)
    print("-" * len(header))

    for snr in sorted(results.keys()):
        r = results[snr]
        line = f"{snr:6.1f}  {r['psnr']:8.3f}"
        if has_ssim:
            line += f"  {r.get('ms_ssim_db', 0):11.3f}  {r.get('ssim', 0):6.4f}"
        if has_lp:
            line += f"  {r.get('lpips', 0):7.5f}"
        print(line)

    psnr_list = [results[s]["psnr"] for s in sorted(results.keys())]
    print(f"\nAvg PSNR: {np.mean(psnr_list):.3f} dB")


if __name__ == "__main__":
    main()
