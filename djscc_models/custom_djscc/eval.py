"""Evaluation script for the custom ConvNeXt + FiLM DJSCC model.

Reports PSNR, MS-SSIM (dB), LPIPS, E1, E10 across SNR values — same
metrics as the ADJSCC eval.py for direct comparison.

Usage:
  python eval.py --ckpt ckpts/AWGN_rate_16_ConvNeXt_FiLM_SNR_random_EP_49.pth \
                 --val-data-dir ./data/kodak --tcn 16
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DJSCCEncoder, DJSCCDecoder, DJSCCModel
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


class AWGNChannel(nn.Module):
    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        noise_std = torch.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
        while noise_std.dim() < x.dim():
            noise_std = noise_std.unsqueeze(-1)
        return x + torch.randn_like(x) * noise_std


class RayleighFadingChannel(nn.Module):
    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h_real = torch.randn(B, 1, 1, 1, device=x.device) * (0.5 ** 0.5)
        h_imag = torch.randn(B, 1, 1, 1, device=x.device) * (0.5 ** 0.5)
        h_mag = (h_real ** 2 + h_imag ** 2).sqrt()
        noise_std = torch.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
        while noise_std.dim() < x.dim():
            noise_std = noise_std.unsqueeze(-1)
        return h_mag * x + torch.randn_like(x) * noise_std


@torch.no_grad()
def evaluate(
    model: DJSCCModel,
    channel: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_list: list[float],
    lpips_fn=None,
):
    model.eval()
    results = {}

    for snr_val in snr_list:
        psnr_all = []
        lpips_all = []
        ssim_all = []
        ms_ssim_all = []
        e1_all = []
        e10_all = []

        for images, _ in loader:
            images = images.to(device)
            B, C, H, W = images.shape
            snr_db = torch.full((B, 1), snr_val, dtype=torch.float32, device=device)

            encoded = model.encoder(images)
            constrained = model.constraint(encoded)
            received = channel(constrained, snr_db)
            decoded = model.decoder(received, snr_db)

            mse_per = F.mse_loss(
                decoded.view(B, -1), images.view(B, -1), reduction="none"
            ).mean(dim=1)
            psnr_per = 10.0 * torch.log10(1.0 / mse_per)
            psnr_all.extend(psnr_per.cpu().numpy())

            if HAS_MSSSIM and min(H, W) >= 160:
                ssim_val = ssim(decoded, images, data_range=1.0, size_average=True)
                ms_ssim_val = ms_ssim(decoded, images, data_range=1.0, size_average=True)
                ms_ssim_db = -10.0 * torch.log10(1.0 - ms_ssim_val)
                ssim_all.append(ssim_val.item())
                ms_ssim_all.append(ms_ssim_db.item())
                e1_all.append(((ssim_val + ms_ssim_val) / 2.0).item())
                e10_all.append(((ssim_val * ms_ssim_val) ** 0.5).item())

            if HAS_LPIPS and lpips_fn is not None:
                lp = lpips_fn(decoded, images).mean().item()
                lpips_all.append(lp)

        row = {"psnr": float(np.mean(psnr_all))}
        if ms_ssim_all:
            row["ms_ssim_db"] = float(np.mean(ms_ssim_all))
            row["ssim"] = float(np.mean(ssim_all))
            row["e1"] = float(np.mean(e1_all))
            row["e10"] = float(np.mean(e10_all))
        if lpips_all:
            row["lpips"] = float(np.mean(lpips_all))
        results[snr_val] = row

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate custom DJSCC model")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--val-data-dir", type=str, nargs="+",
                        default=["./data/kodak"])
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--tcn", type=int, default=16)
    parser.add_argument("--csi-length", type=int, default=1)
    parser.add_argument("--csi-embed-dim", type=int, default=64)
    parser.add_argument("--img-width", type=int, default=768, help="Image width")
    parser.add_argument("--img-height", type=int, default=512, help="Image height")
    parser.add_argument("--snr-list", type=float, nargs="+",
                        default=[0, 5, 10, 15, 20])
    parser.add_argument("--channel", type=str, default="awgn",
                        choices=["awgn", "rayleigh"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-lpips", action="store_true",
                        help="Skip LPIPS (saves GPU memory)")

    args = parser.parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # --- Model ---
    encoder = DJSCCEncoder(N=args.N, M=args.tcn)
    decoder = DJSCCDecoder(
        N=args.N, M=args.tcn,
        csi_length=args.csi_length,
        csi_embed_dim=args.csi_embed_dim,
    )
    constraint = AveragePowerConstraint(average_power=1.0)
    model = DJSCCModel(encoder=encoder, decoder=decoder, constraint=constraint)

    # --- Load checkpoint ---
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["net"] if "net" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    trained_snr = ckpt.get("SNR", "?")
    trained_psnr = ckpt.get("Ave_PSNR", "?")
    trained_epoch = ckpt.get("epoch", "?")
    print(f"Loaded: epoch={trained_epoch}, trained_SNR={trained_snr}, "
          f"train_avg_PSNR={trained_psnr}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: N={args.N}, M={args.tcn}, params={n_params:,}")

    # --- Channel ---
    channel = RayleighFadingChannel() if args.channel == "rayleigh" else AWGNChannel()

    # --- LPIPS ---
    lpips_fn = None
    if HAS_LPIPS and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # --- Data ---
    img_size = (args.img_width, args.img_height)
    val_loader = build_val_loader(args.val_data_dir, img_size, 1, args.num_workers)
    print(f"Val images: {len(val_loader.dataset)}")
    print(f"Channel: {args.channel}")
    print("-" * 70)

    results = evaluate(model, channel, val_loader, device, args.snr_list, lpips_fn)

    header = f"{'SNR':>6s}  {'PSNR':>8s}"
    has_ssim = any("ms_ssim_db" in v for v in results.values())
    has_lp = any("lpips" in v for v in results.values())
    if has_ssim:
        header += f"  {'MS-SSIM(dB)':>11s}  {'SSIM':>6s}  {'E1':>8s}  {'E10':>8s}"
    if has_lp:
        header += f"  {'LPIPS':>7s}"

    print(header)
    print("-" * len(header))

    for snr in sorted(results.keys()):
        r = results[snr]
        line = f"{snr:6.1f}  {r['psnr']:8.3f}"
        if has_ssim:
            line += (f"  {r.get('ms_ssim_db', 0):11.3f}"
                     f"  {r.get('ssim', 0):6.4f}"
                     f"  {r.get('e1', 0):8.5f}"
                     f"  {r.get('e10', 0):8.5f}")
        if has_lp:
            line += f"  {r.get('lpips', 0):7.5f}"
        print(line)

    psnr_list = [results[s]["psnr"] for s in sorted(results.keys())]
    print(f"\nAvg PSNR: {np.mean(psnr_list):.3f} dB")


if __name__ == "__main__":
    main()
