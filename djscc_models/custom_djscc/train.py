"""Training script for the custom ConvNeXt + FiLM DJSCC model.

Trains encoder (channel-blind) + decoder (CSI-adaptive) end-to-end
through a differentiable AWGN or Rayleigh fading channel.

Accepts one or more training directories (flat image dirs like DIV2K
or ImageFolder layouts). They are combined into a single dataset.

Memory-efficient: supports mixed precision (AMP), gradient accumulation,
and gradient checkpointing for training at full 768x512 resolution.

Usage:
  # Train on DIV2K at 768x512 (fits 25 GB GPU with defaults):
  python train.py --train-data-dir ~/Downloads/DIV2K_train_HR \
                  --val-data-dir ~/Downloads/DIV2K_valid_HR \
                  --epochs 50 --batch-size 4 --accum-steps 8 --tcn 16

  # Multiple directories combined:
  python train.py --train-data-dir ~/Downloads/DIV2K_train_HR \
                                   ~/Downloads/Flickr2K \
                  --val-data-dir ~/Downloads/DIV2K_valid_HR

  # Disable AMP (e.g. for MPS which has limited fp16 support):
  python train.py --no-amp ...

  # Resume from checkpoint:
  python train.py --resume ckpts/best.pth ...
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model import DJSCCEncoder, DJSCCDecoder, DJSCCModel
from dataset import build_train_loader, build_val_loader
from kaira.constraints.power import AveragePowerConstraint

SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Channel simulation (differentiable)
# ---------------------------------------------------------------------------

class AWGNChannel(nn.Module):
    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        noise_std = torch.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
        while noise_std.dim() < x.dim():
            noise_std = noise_std.unsqueeze(-1)
        noise = torch.randn_like(x) * noise_std
        return x + noise


class RayleighFadingChannel(nn.Module):
    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h_real = torch.randn(B, 1, 1, 1, device=x.device) * (0.5 ** 0.5)
        h_imag = torch.randn(B, 1, 1, 1, device=x.device) * (0.5 ** 0.5)
        h_mag = (h_real ** 2 + h_imag ** 2).sqrt()

        noise_std = torch.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
        while noise_std.dim() < x.dim():
            noise_std = noise_std.unsqueeze(-1)
        noise = torch.randn_like(x) * noise_std

        return h_mag * x + noise


# ---------------------------------------------------------------------------
# Gradient-checkpointed forward pass
# ---------------------------------------------------------------------------

def simulate_packet_drop(received: torch.Tensor, drop_prob: float, pkt_len: int = 960) -> torch.Tensor:
    """Simulates OFDM packet drops by zeroing out contiguous blocks of symbols."""
    if drop_prob <= 0.0:
        return received
        
    B, C, H, W = received.shape
    flat = received.reshape(B, -1)
    
    # The codec splits the flattened tensor into a real half and an imaginary half
    dim_z = flat.shape[1] // 2
    
    # Calculate how many packets are needed for this image
    n_pkts = (dim_z + pkt_len - 1) // pkt_len
    
    # Generate a mask for packets: 1 = keep, 0 = drop
    pkt_mask = (torch.rand(B, n_pkts, device=received.device) > drop_prob).to(received.dtype)
    
    # Expand mask from packet-level to symbol-level and trim any excess
    sym_mask = pkt_mask.repeat_interleave(pkt_len, dim=1)[:, :dim_z]
    
    # Apply the exact same mask to both the real and imaginary halves
    full_mask = torch.cat([sym_mask, sym_mask], dim=1)
    
    return (flat * full_mask).reshape(B, C, H, W)

def forward_pass(model, channel, images, snr_db, use_grad_ckpt=False, drop_prob=0.0, packet_len=960):
    """Run encoder → constraint → channel → decoder, optionally with
    gradient checkpointing on encoder and decoder to save memory."""
    if use_grad_ckpt:
        encoded = grad_checkpoint(model.encoder, images, use_reentrant=False)
        constrained = model.constraint(encoded)
        received = channel(constrained, snr_db)
        if model.training:
            received = simulate_packet_drop(received, drop_prob, packet_len)
        decoded = grad_checkpoint(
            model.decoder, received, snr_db, use_reentrant=False)
    else:
        encoded = model.encoder(images)
        constrained = model.constraint(encoded)
        received = channel(constrained, snr_db)
        if model.training:
            received = simulate_packet_drop(received, drop_prob, packet_len)
        decoded = model.decoder(received, snr_db)
    return decoded


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DJSCCModel,
    channel: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    snr_min: float,
    snr_max: float,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    accum_steps: int = 1,
    use_grad_ckpt: bool = False,
    log_interval: int = 50,
    drop_prob: float = 0.0,
    packet_len: int = 960,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    n_batches = 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    optimizer.zero_grad()

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        B = images.shape[0]
        snr_db = torch.rand(B, 1, device=device) * (snr_max - snr_min) + snr_min

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            decoded = forward_pass(
                model, channel, images, snr_db, use_grad_ckpt, drop_prob, packet_len)
            loss = criterion(decoded, images) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = running_loss / n_batches
            psnr = -10.0 * np.log10(avg + 1e-10)
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}]  "
                  f"loss={avg:.6f}  psnr~{psnr:.2f} dB")

    # Flush remaining gradients
    if len(loader) % accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: DJSCCModel,
    channel: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_list: list[float],
    use_amp: bool = False,
) -> dict[float, float]:
    model.eval()
    results = {}
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for snr_val in snr_list:
        psnr_all = []
        for images, _ in loader:
            images = images.to(device)
            B = images.shape[0]
            snr_db = torch.full((B, 1), snr_val, dtype=torch.float32, device=device)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                decoded = forward_pass(model, channel, images, snr_db)

            mse_per_image = F.mse_loss(
                decoded.float().view(B, -1),
                images.float().view(B, -1),
                reduction="none",
            ).mean(dim=1)
            psnr_per_image = 10.0 * torch.log10(1.0 / mse_per_image)
            psnr_all.extend(psnr_per_image.cpu().numpy())

        results[snr_val] = float(np.mean(psnr_all))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train custom DJSCC (ConvNeXt + FiLM)")

    parser.add_argument("--train-data-dir", type=str, nargs="+", required=True,
                        help="One or more training image directories")
    parser.add_argument("--val-data-dir", type=str, nargs="+",
                        default=["./data/kodak"],
                        help="One or more validation image directories")
    parser.add_argument("--ckpt-dir", type=str, default="./ckpts",
                        help="Directory to save checkpoints")

    parser.add_argument("--N", type=int, default=256, help="Intermediate channels")
    parser.add_argument("--tcn", type=int, default=16, help="Latent channels (M)")
    parser.add_argument("--csi-length", type=int, default=1, help="CSI vector length")
    parser.add_argument("--csi-embed-dim", type=int, default=64, help="CSI embedding dim")

    parser.add_argument("--img-width", type=int, default=768, help="Image width")
    parser.add_argument("--img-height", type=int, default=512, help="Image height")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Micro-batch size per GPU step")
    parser.add_argument("--accum-steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = "
                             "batch-size * accum-steps, default 4*8=32)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (AMP)")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing")

    parser.add_argument("--snr-min", type=float, default=0.0, help="Min training SNR (dB)")
    parser.add_argument("--snr-max", type=float, default=20.0, help="Max training SNR (dB)")
    parser.add_argument("--val-snr-list", type=float, nargs="+",
                        default=[0, 5, 10, 15, 20],
                        help="SNR values for validation")

    parser.add_argument("--drop-prob", type=float, default=0.0,
                        help="Maximum probability of dropping an OFDM packet (sampled randomly in [0, p])")
    parser.add_argument("--packet-len", type=int, default=960,
                        help="Number of complex symbols per OFDM packet")

    parser.add_argument("--channel", type=str, default="awgn",
                        choices=["awgn", "rayleigh"],
                        help="Channel model for training")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Print every N batches")

    args = parser.parse_args()
    device = get_device(args.device)
    use_amp = (not args.no_amp) and (device.type == "cuda")
    use_grad_ckpt = not args.no_grad_ckpt
    print(f"Device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # --- Model ---
    encoder = DJSCCEncoder(N=args.N, M=args.tcn)
    decoder = DJSCCDecoder(
        N=args.N, M=args.tcn,
        csi_length=args.csi_length,
        csi_embed_dim=args.csi_embed_dim,
    )
    constraint = AveragePowerConstraint(average_power=1.0)
    model = DJSCCModel(encoder=encoder, decoder=decoder, constraint=constraint)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    eff_batch = args.batch_size * args.accum_steps
    print(f"Model: N={args.N}, M={args.tcn}, params={n_params:,}")
    print(f"Batch: {args.batch_size} x {args.accum_steps} accum = {eff_batch} effective")
    print(f"AMP: {use_amp}, Grad checkpoint: {use_grad_ckpt}")

    # --- Channel ---
    if args.channel == "rayleigh":
        channel = RayleighFadingChannel()
    else:
        channel = AWGNChannel()

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_psnr = 0.0

    # --- Resume ---
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["net"])
        if "op" in ckpt:
            optimizer.load_state_dict(ckpt["op"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("Ave_PSNR", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    model.to(device)

    # --- Data ---
    img_size = (args.img_width, args.img_height)
    train_loader = build_train_loader(
        args.train_data_dir, img_size, args.batch_size, args.num_workers)
    val_loader = build_val_loader(args.val_data_dir, img_size, 1, args.num_workers)

    print(f"Train: {len(train_loader.dataset)} images, "
          f"Val: {len(val_loader.dataset)} images")
    print(f"Image size: {args.img_width}x{args.img_height}")
    print(f"Channel: {args.channel}, SNR range: [{args.snr_min}, {args.snr_max}] dB")
    print(f"Packet drop prob: random in [0, {args.drop_prob}] (packet_len={args.packet_len})")
    print(f"Training for {args.epochs} epochs, lr={args.lr}")
    print("-" * 60)

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, channel, train_loader, optimizer, device,
            args.snr_min, args.snr_max, epoch,
            scaler=scaler, use_amp=use_amp,
            accum_steps=args.accum_steps,
            use_grad_ckpt=use_grad_ckpt,
            log_interval=args.log_interval,
            drop_prob=args.drop_prob,
            packet_len=args.packet_len,
        )
        dt = time.time() - t0

        psnr_results = validate(
            model, channel, val_loader, device, args.val_snr_list,
            use_amp=use_amp,
        )

        avg_psnr = np.mean(list(psnr_results.values()))
        lr_now = optimizer.param_groups[0]["lr"]

        snr_str = "  ".join(
            f"{s:.0f}dB:{p:.2f}" for s, p in sorted(psnr_results.items()))
        print(f"Epoch {epoch:3d}  loss={train_loss:.6f}  "
              f"avg_psnr={avg_psnr:.2f}  lr={lr_now:.2e}  "
              f"time={dt:.0f}s")
        print(f"  Val PSNR: {snr_str}")

        scheduler.step()

        # -- Build checkpoint dict (reused for both saves) --
        save_dict = {
            "model_name": "ConvNeXt_FiLM_DJSCC",
            "net": model.state_dict(),
            "op": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "SNR": "random",
            "Ave_PSNR": avg_psnr,
            "args": vars(args),
        }
        if scaler is not None:
            save_dict["scaler"] = scaler.state_dict()

        # -- Always save latest (overwritten each epoch) --
        latest_path = os.path.join(args.ckpt_dir, "latest.pth")
        torch.save(save_dict, latest_path)

        # -- Save best --
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_path = os.path.join(
                args.ckpt_dir,
                f"{args.channel.upper()}_rate_{args.tcn}_ConvNeXt_FiLM_"
                f"SNR_random_EP_{epoch}.pth",
            )
            torch.save(save_dict, best_path)
            print(f"  ** New best: {avg_psnr:.2f} dB -> {best_path}")

    print(f"\nTraining complete. Best avg PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
