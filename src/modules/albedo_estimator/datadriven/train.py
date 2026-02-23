"""Training script for the data-driven albedo model.

Usage::

    python -m clouds_decoded.albedo_estimator.datadriven.train \\
        --dataset scratch/data/albedo_training_dataset.parquet \\
        --epochs 100 --batch_size 4096 --lr 1e-3
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import OUTPUT_BANDS
from .dataset import AlbedoDataset
from .model import AlbedoNet, NormalizationWrapper

logger = logging.getLogger(__name__)


def validate(
    model: NormalizationWrapper,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    plot_dir: str,
    band_names: list,
) -> tuple:
    """Run validation and save scatter plots.

    Returns:
        (mean_mae, mean_mse, preds, targets, per_band_mae)
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for features, targets in val_loader:
            preds = model(features.to(device))
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    per_band_mae = np.abs(preds - targets).mean(axis=0)
    mean_mae = float(per_band_mae.mean())
    mean_mse = float(((preds - targets) ** 2).mean())

    # Scatter plots: 2 rows x 7 cols (13 bands + 1 empty)
    n_bands = len(band_names)
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    fig.suptitle(f"Epoch {epoch} — Val MAE={mean_mae:.5f}  MSE={mean_mse:.6f}", fontsize=16)

    rng = np.random.default_rng(42)
    for i in range(14):
        ax = axes.flat[i]
        if i >= n_bands:
            ax.axis("off")
            continue

        x, y = targets[:, i], preds[:, i]
        if len(x) > 10_000:
            idx = rng.choice(len(x), 10_000, replace=False)
            x, y = x[idx], y[idx]

        ax.scatter(x, y, s=1, alpha=0.3, c="steelblue", rasterized=True)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{band_names[i]}  MAE={per_band_mae[i]:.5f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(plot_dir) / f"val_epoch_{epoch:03d}.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    return mean_mae, mean_mse, preds, targets, per_band_mae


def main():
    parser = argparse.ArgumentParser(description="Train data-driven albedo model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to albedo_training_dataset.parquet")
    parser.add_argument("--output", type=str, default=None,
                        help="Checkpoint save path (default: models/albedo_model.pth)")
    parser.add_argument("--plot-dir", type=str, default="plots/albedo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr-decay-step", type=int, default=30,
                        help="Halve LR every N epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None,
                        help="Device ('cuda', 'cpu', or None=auto)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ──
    bands = list(OUTPUT_BANDS)
    train_ds = AlbedoDataset(args.dataset, split="train", bands=bands)
    val_ds = AlbedoDataset(args.dataset, split="val", bands=bands)
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── Model ──
    core = AlbedoNet(
        input_size=train_ds.features.shape[1],
        output_size=train_ds.targets.shape[1],
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    )
    model = NormalizationWrapper(
        core,
        input_stats=train_ds.input_stats,
        output_stats=train_ds.output_stats,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimiser & loss ──
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    # ── Training loop ──
    output_path = args.output or str(
        Path(__file__).parent / "models" / "albedo_model.pth"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Step LR decay
        lr = args.lr * (0.5 ** ((epoch - 1) // args.lr_decay_step))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_mse = 0.0
        epoch_mae = 0.0
        n_batches = 0

        for features, targets in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100,
        ):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_mse += loss.item()
            with torch.no_grad():
                epoch_mae += torch.abs(preds - targets).mean().item()
            n_batches += 1

        train_mse = epoch_mse / n_batches
        train_mae = epoch_mae / n_batches

        # Validation
        val_mae, val_mse, _, _, _ = validate(
            model, val_loader, device, epoch, args.plot_dir, bands,
        )

        logger.info(
            f"Epoch {epoch} | LR={lr:.2e} | "
            f"Train MSE={train_mse:.6f} MAE={train_mae:.6f} | "
            f"Val MSE={val_mse:.6f} MAE={val_mae:.6f}"
        )

        # Early stopping + checkpointing
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            logger.info(f"  Saved best model (MAE={best_val_mae:.6f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best val MAE: {best_val_mae:.6f}")
    logger.info(f"Checkpoint: {output_path}")


if __name__ == "__main__":
    main()
