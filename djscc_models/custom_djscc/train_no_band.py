"""Training script for the no-band DJSCC model.

Per-packet channel simulation (matches OTA failure modes):
  - Per-packet SNR  = base SNR (per image)  +  per-packet jitter (dB).
  - Per-packet phase rotation (residual CFO / phase noise).
  - Optional PA soft clipping at a random IBO (transmitter PA model).
  - Optional packet drops: each image's drop probability is sampled
    uniformly in [0, --drop-prob] (so the batch sees a mix of clean,
    light-loss, and heavy-loss conditions, matching the original training
    script's semantics). Dropped packets get sentinel SNR (default -20 dB)
    AND their elements are zeroed (matching the runtime RX zero-fill).
  - The same per-packet SNR vector is fed to the decoder as a per-element
    CSI map (built with packetwise_to_element_map), so the decoder knows
    which strips are unreliable.

Usage:
  python train_no_band.py \\
      --train-data-dir <data_dir>... \\
      --val-data-dir <val_dir>... \\
      --epochs 50 --batch-size 4 --accum-steps 8 \\
      --tcn 16 --packet-len 960 \\
      --snr-min 0 --snr-max 20 --jitter-snr-db 3.0 \\
      --phase-noise-rad 0.15 \\
      --clip-ibo-min 4 --clip-ibo-max 9 \\
      --drop-prob 0.30   # actual rate ~ U[0, 0.30] per image

Set any of (--jitter-snr-db, --phase-noise-rad, --drop-prob) to 0 (and pass
--clip-ibo-max <= --clip-ibo-min with large IBO) to disable that impairment
during a curriculum / ablation.
"""

import argparse
import os
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model_no_band import (
    DJSCCDecoderSpatialCSI,
    DJSCCEncoder,
    DJSCCModelNoBand,
    packetwise_to_element_map,
)
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
# Per-packet impairment helpers
# ---------------------------------------------------------------------------

def soft_clip_complex(x: torch.Tensor, ibo_db: torch.Tensor) -> torch.Tensor:
    """Soft clip per-symbol magnitudes with a tanh PA model.

    With kaira AveragePowerConstraint(power=1.0) the average per-real-dim
    power is 1, so the average |c|^2 of a complex symbol is 2. IBO is
    expressed in dB relative to that average power, so the clip ceiling is
    A_max = sqrt(2 * 10^(ibo_db/10)).

    Args:
      x: [B, M, H, W] real layout; first M/2 channels are real parts,
         last M/2 channels are imag parts (matches codec serialization).
      ibo_db: [B] or [B, 1] PA back-off in dB (larger = less clipping).
    """
    B, M, H, W = x.shape
    half_M = M // 2
    re = x[:, :half_M]
    im = x[:, half_M:]
    mag = torch.sqrt(re * re + im * im + 1e-12)

    A_max = torch.sqrt(2.0 * 10.0 ** (ibo_db / 10.0))
    while A_max.dim() < mag.dim():
        A_max = A_max.unsqueeze(-1)

    soft_mag = A_max * torch.tanh(mag / A_max)
    scale = soft_mag / (mag + 1e-12)
    return torch.cat([re * scale, im * scale], dim=1)


def sample_per_packet_snr(
    B: int, n_pkts: int, snr_min: float, snr_max: float,
    jitter_std_db: float, device: torch.device,
) -> torch.Tensor:
    """Sample [B, n_pkts] per-packet SNR (dB).

    base ~ U[snr_min, snr_max] per image, then per-packet jitter ~ N(0, jitter_std_db).
    """
    base = torch.rand(B, 1, device=device) * (snr_max - snr_min) + snr_min
    if jitter_std_db > 0:
        jitter = torch.randn(B, n_pkts, device=device) * jitter_std_db
    else:
        jitter = torch.zeros(B, n_pkts, device=device)
    return base + jitter


class PacketwiseChannel(nn.Module):
    """Channel with per-packet AWGN + phase noise + drops.

    Args (forward):
      x:                    [B, M, H, W] (real layout of complex symbols)
      snr_db_per_pkt:       [B, n_pkts]
      phi_per_pkt:          optional [B, n_pkts] phase (radians)
      drop_mask:            optional [B, n_pkts] bool, True = drop (zero-fill)
    """

    def __init__(self, pkt_len_complex: int = 960) -> None:
        super().__init__()
        self.pkt_len = pkt_len_complex

    def forward(
        self,
        x: torch.Tensor,
        snr_db_per_pkt: torch.Tensor,
        phi_per_pkt: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, M, H, W = x.shape
        half_M = M // 2

        # ---- per-packet phase rotation (on complex pairs) -----------------
        if phi_per_pkt is not None:
            # Build phase map for the real-half channels only; same map is
            # the rotation angle for the (re, im) complex pair.
            # packetwise_to_element_map produces a tensor with shape
            # [B, M, H, W] but the two halves carry the same per-packet
            # value -> taking the first M/2 channels gives the same map
            # for each real channel.
            phi_map_full = packetwise_to_element_map(
                phi_per_pkt, M=M, H=H, W=W, pkt_len_complex=self.pkt_len,
            )  # [B, M, H, W]
            phi = phi_map_full[:, :half_M]  # [B, M/2, H, W]
            re = x[:, :half_M]
            im = x[:, half_M:]
            cos_phi = phi.cos()
            sin_phi = phi.sin()
            re2 = re * cos_phi - im * sin_phi
            im2 = re * sin_phi + im * cos_phi
            x = torch.cat([re2, im2], dim=1)

        # ---- per-packet AWGN ---------------------------------------------
        sigma_per_pkt = torch.sqrt(10.0 ** (-snr_db_per_pkt / 10.0) / 2.0)
        sigma_map = packetwise_to_element_map(
            sigma_per_pkt, M=M, H=H, W=W, pkt_len_complex=self.pkt_len,
        )  # [B, M, H, W]
        noise = torch.randn_like(x) * sigma_map
        y = x + noise

        # ---- packet drops (zero-fill) ------------------------------------
        if drop_mask is not None:
            drop_map = packetwise_to_element_map(
                drop_mask.float(), M=M, H=H, W=W,
                pkt_len_complex=self.pkt_len,
            )
            y = y * (1.0 - drop_map)

        return y


# ---------------------------------------------------------------------------
# Forward pass (with optional gradient checkpointing)
# ---------------------------------------------------------------------------

def forward_pass(
    model: DJSCCModelNoBand,
    channel: PacketwiseChannel,
    images: torch.Tensor,
    n_pkts: int,
    snr_min: float,
    snr_max: float,
    jitter_snr_db: float,
    phase_noise_rad: float,
    clip_ibo_min: float,
    clip_ibo_max: float,
    drop_prob_max: float,
    sentinel_drop_db: float,
    use_grad_ckpt: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder -> constraint -> [clip] -> channel -> decoder.

    Per-image randomization (matches the original training's semantics for
    drop probability — every image gets its own drop rate sampled from
    [0, drop_prob_max], so the decoder sees the full range from clean to
    heavy-drop within a single batch):

      drop_p_per_image  ~ Uniform[0, drop_prob_max]
      drop_mask[b, k]   ~ Bernoulli(drop_p_per_image[b])

    Returns (decoded, snr_db_per_pkt_for_csi).
    """
    B = images.shape[0]
    device = images.device

    # --- encoder + constraint ---
    if use_grad_ckpt:
        encoded = grad_checkpoint(model.encoder, images, use_reentrant=False)
    else:
        encoded = model.encoder(images)
    constrained = model.constraint(encoded)

    # --- PA soft clipping (per-image random IBO) ---
    if clip_ibo_max > clip_ibo_min:
        ibo_db = torch.empty(B, device=device).uniform_(clip_ibo_min, clip_ibo_max)
        constrained = soft_clip_complex(constrained, ibo_db)

    # --- per-packet impairment sampling ---
    snr_db_per_pkt = sample_per_packet_snr(
        B, n_pkts, snr_min, snr_max, jitter_snr_db, device,
    )
    snr_db_per_pkt = snr_db_per_pkt.clamp(min=snr_min - 10.0, max=snr_max + 5.0)

    phi_per_pkt = None
    if phase_noise_rad > 0:
        phi_per_pkt = torch.randn(B, n_pkts, device=device) * phase_noise_rad

    drop_mask = None
    csi_snr_for_decoder = snr_db_per_pkt
    if drop_prob_max > 0:
        # Per-image drop probability sampled uniformly in [0, drop_prob_max].
        # Some images in the batch will see zero drops (clean), some will
        # see heavy drops — matches the original train.py semantics.
        drop_p_per_image = torch.rand(B, 1, device=device) * drop_prob_max  # [B, 1]
        drop_mask = torch.rand(B, n_pkts, device=device) < drop_p_per_image
        csi_snr_for_decoder = torch.where(
            drop_mask,
            torch.full_like(snr_db_per_pkt, sentinel_drop_db),
            snr_db_per_pkt,
        )

    # --- channel ---
    received = channel(constrained, snr_db_per_pkt, phi_per_pkt, drop_mask)

    # --- CSI map for decoder ---
    M, H, W = constrained.shape[1], constrained.shape[2], constrained.shape[3]
    csi_map_db = packetwise_to_element_map(
        csi_snr_for_decoder, M=M, H=H, W=W,
        pkt_len_complex=channel.pkt_len,
    )

    if use_grad_ckpt:
        decoded = grad_checkpoint(
            model.decoder, received, csi_map_db, use_reentrant=False,
        )
    else:
        decoded = model.decoder(received, csi_map_db)

    return decoded, csi_snr_for_decoder


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DJSCCModelNoBand,
    channel: PacketwiseChannel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_pkts: int,
    args: argparse.Namespace,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    log_interval: int = 50,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    n_batches = 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    optimizer.zero_grad()

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            decoded, _csi = forward_pass(
                model, channel, images, n_pkts,
                snr_min=args.snr_min, snr_max=args.snr_max,
                jitter_snr_db=args.jitter_snr_db,
                phase_noise_rad=args.phase_noise_rad,
                clip_ibo_min=args.clip_ibo_min,
                clip_ibo_max=args.clip_ibo_max,
                drop_prob_max=args.drop_prob,
                sentinel_drop_db=args.sentinel_drop_db,
                use_grad_ckpt=not args.no_grad_ckpt,
            )
            loss = criterion(decoded, images) / args.accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % args.accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * args.accum_steps
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = running_loss / n_batches
            psnr = -10.0 * np.log10(avg + 1e-10)
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}]  "
                f"loss={avg:.6f}  psnr~{psnr:.2f} dB"
            )

    if len(loader) % args.accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: DJSCCModelNoBand,
    channel: PacketwiseChannel,
    loader: DataLoader,
    device: torch.device,
    n_pkts: int,
    snr_list: list[float],
    args: argparse.Namespace,
    use_amp: bool = False,
    val_drop_prob: float = 0.0,
) -> dict[float, float]:
    """Validate at multiple fixed SNRs (uniform per-packet, no drops by default)."""
    model.eval()
    results = {}
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for snr_val in snr_list:
        psnr_all = []
        for images, _ in loader:
            images = images.to(device)
            B = images.shape[0]

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                encoded = model.encoder(images)
                constrained = model.constraint(encoded)

                snr_db_per_pkt = torch.full(
                    (B, n_pkts), float(snr_val),
                    dtype=torch.float32, device=device,
                )
                drop_mask = None
                csi_snr = snr_db_per_pkt
                if val_drop_prob > 0:
                    drop_mask = torch.rand(B, n_pkts, device=device) < val_drop_prob
                    csi_snr = torch.where(
                        drop_mask,
                        torch.full_like(snr_db_per_pkt, args.sentinel_drop_db),
                        snr_db_per_pkt,
                    )

                received = channel(constrained, snr_db_per_pkt, None, drop_mask)
                M, H, W = constrained.shape[1], constrained.shape[2], constrained.shape[3]
                csi_map = packetwise_to_element_map(
                    csi_snr, M=M, H=H, W=W,
                    pkt_len_complex=channel.pkt_len,
                )
                decoded = model.decoder(received, csi_map)

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
    parser = argparse.ArgumentParser(
        description="Train no-band DJSCC (ConvNeXt + FiLM + spatial CSI)"
    )

    parser.add_argument("--train-data-dir", type=str, nargs="+", required=True)
    parser.add_argument("--val-data-dir", type=str, nargs="+",
                        default=["./data/kodak"])
    parser.add_argument("--ckpt-dir", type=str, default="./ckpts_no_band")

    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--tcn", type=int, default=16)
    parser.add_argument("--csi-embed-dim", type=int, default=64)
    parser.add_argument("--csi-db-scale", type=float, default=20.0)

    parser.add_argument("--img-width", type=int, default=768)
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-grad-ckpt", action="store_true")

    parser.add_argument("--packet-len", type=int, default=960,
                        help="Complex symbols per OFDM packet")

    # ---- per-packet impairment knobs ----
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--jitter-snr-db", type=float, default=3.0,
                        help="Std. dev. of per-packet SNR jitter around the "
                             "per-image base SNR. Set 0 to disable.")
    parser.add_argument("--phase-noise-rad", type=float, default=0.15,
                        help="Std. dev. of per-packet phase rotation (radians)")
    parser.add_argument("--clip-ibo-min", type=float, default=4.0,
                        help="Min PA IBO (dB). Set min>=max to disable clipping.")
    parser.add_argument("--clip-ibo-max", type=float, default=9.0,
                        help="Max PA IBO (dB)")
    parser.add_argument("--drop-prob", type=float, default=0.30,
                        help="Maximum per-packet drop probability. Each image "
                             "gets an actual drop probability sampled uniformly "
                             "in [0, drop-prob] (matches the original training "
                             "script's semantics). Set to 0 to disable drops.")
    parser.add_argument("--sentinel-drop-db", type=float, default=-20.0,
                        help="CSI sentinel SNR (dB) for dropped/missing packets")

    parser.add_argument("--val-snr-list", type=float, nargs="+",
                        default=[0, 5, 10, 15, 20])
    parser.add_argument("--val-drop-prob", type=float, default=0.0,
                        help="Optional drop prob during validation "
                             "(reports robustness)")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=50)

    args = parser.parse_args()
    device = get_device(args.device)
    use_amp = (not args.no_amp) and (device.type == "cuda")
    use_grad_ckpt = not args.no_grad_ckpt
    print(f"Device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # --- Model ---
    encoder = DJSCCEncoder(N=args.N, M=args.tcn)
    decoder = DJSCCDecoderSpatialCSI(
        N=args.N, M=args.tcn,
        csi_embed_dim=args.csi_embed_dim,
        csi_db_scale=args.csi_db_scale,
    )
    constraint = AveragePowerConstraint(average_power=1.0)
    model = DJSCCModelNoBand(
        encoder=encoder, decoder=decoder, constraint=constraint,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    eff_batch = args.batch_size * args.accum_steps
    print(f"Model: N={args.N}, M={args.tcn}, params={n_params:,}")
    print(f"Batch: {args.batch_size} x {args.accum_steps} accum = {eff_batch} effective")
    print(f"AMP: {use_amp}, Grad checkpoint: {use_grad_ckpt}")

    # --- Channel ---
    channel = PacketwiseChannel(pkt_len_complex=args.packet_len)

    # Compute n_pkts per image to match codec_no_band layout.
    complex_per_image = (args.tcn * (args.img_height // 4) * (args.img_width // 4)) // 2
    n_pkts = int(math.ceil(complex_per_image / args.packet_len))
    print(f"Packets/image: {n_pkts}  (complex_per_image={complex_per_image}, "
          f"pkt_len={args.packet_len})")

    # --- Optimizer / Scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
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
        args.train_data_dir, img_size, args.batch_size, args.num_workers,
    )
    val_loader = build_val_loader(
        args.val_data_dir, img_size, 1, args.num_workers,
    )

    print(f"Train: {len(train_loader.dataset)} images, "
          f"Val: {len(val_loader.dataset)} images")
    print(f"Image size: {args.img_width}x{args.img_height}")
    print(f"Per-packet impairments:")
    print(f"  SNR base:           U[{args.snr_min}, {args.snr_max}] dB")
    print(f"  SNR jitter std:     {args.jitter_snr_db} dB")
    print(f"  Phase noise std:    {args.phase_noise_rad} rad")
    print(f"  Drop prob:          U[0, {args.drop_prob}] per image  "
          f"(sentinel SNR = {args.sentinel_drop_db} dB)")
    print(f"  PA IBO range:       [{args.clip_ibo_min}, {args.clip_ibo_max}] dB")
    print(f"Training for {args.epochs} epochs, lr={args.lr}")
    print("-" * 60)

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, channel, train_loader, optimizer, device,
            n_pkts, args, epoch,
            scaler=scaler, use_amp=use_amp,
            log_interval=args.log_interval,
        )
        dt = time.time() - t0

        psnr_results = validate(
            model, channel, val_loader, device, n_pkts,
            args.val_snr_list, args,
            use_amp=use_amp,
            val_drop_prob=args.val_drop_prob,
        )

        avg_psnr = float(np.mean(list(psnr_results.values())))
        lr_now = optimizer.param_groups[0]["lr"]
        snr_str = "  ".join(
            f"{s:.0f}dB:{p:.2f}" for s, p in sorted(psnr_results.items())
        )

        print(f"Epoch {epoch:3d}  loss={train_loss:.6f}  "
              f"avg_psnr={avg_psnr:.2f}  lr={lr_now:.2e}  "
              f"time={dt:.0f}s")
        print(f"  Val PSNR: {snr_str}")

        scheduler.step()

        save_dict = {
            "model_name": "ConvNeXt_FiLM_DJSCC_NoBand",
            "net": model.state_dict(),
            "op": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "SNR": "random",
            "Ave_PSNR": avg_psnr,
            "args": vars(args),
            "n_pkts": n_pkts,
        }
        if scaler is not None:
            save_dict["scaler"] = scaler.state_dict()

        # Always save latest.
        latest_path = os.path.join(args.ckpt_dir, "latest_no_band.pth")
        torch.save(save_dict, latest_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_path = os.path.join(
                args.ckpt_dir,
                f"NoBand_rate_{args.tcn}_ConvNeXt_FiLM_"
                f"SNR_random_EP_{epoch}.pth",
            )
            torch.save(save_dict, best_path)
            print(f"  ** New best: {avg_psnr:.2f} dB -> {best_path}")

    print(f"\nTraining complete. Best avg PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
