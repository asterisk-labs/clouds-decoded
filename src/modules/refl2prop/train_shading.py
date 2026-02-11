"""
Training script for shading-aware cloud property inversion model.

This script trains a model that can:
1. Predict per-pixel tau_effective_shading from bags of shaded reflectances
2. Predict global physical properties (tau, ice_liq_ratio, r_eff_liq, r_eff_ice)

Unlike train.py, this approach uses shading modulation instead of noise augmentation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dataset_shading import (
    ShadingBagDataset,
    InMemoryShadingBagDataset,
    shading_collate_fn,
)
from .model_shading import ShadingAwareInversionNet, ShadingNormalizationWrapper
from .loss_shading import combined_shading_loss
from .config import Refl2PropConfig


def validate_and_plot_shading(
    model,
    iterator,
    device,
    epoch,
    output_dir,
    val_batches: int = 20
):
    """
    Validation for shading-aware model.

    Creates a 5-panel plot: shading prediction + 4 global properties.
    """
    model.eval()
    print(f"\n--- Running Validation (Epoch {epoch}) ---")

    all_shading_preds = []
    all_shading_targets = []
    all_global_preds = []
    all_global_targets = []

    with torch.no_grad():
        for _ in range(val_batches):
            try:
                inputs, shading_targets, global_targets = next(iterator)
            except StopIteration:
                break

            inputs = inputs.to(device)
            shading_pred, global_pred = model(inputs, return_denormalized=True)

            all_shading_preds.append(shading_pred.cpu())
            all_shading_targets.append(shading_targets)
            all_global_preds.append(global_pred.cpu())
            all_global_targets.append(global_targets)

    if not all_shading_preds:
        return

    # Concatenate results
    shading_preds = torch.cat(all_shading_preds).numpy().flatten()
    shading_targets = torch.cat(all_shading_targets).numpy().flatten()
    global_preds = torch.cat(all_global_preds).numpy()
    global_targets = torch.cat(all_global_targets).numpy()

    # Print summary statistics
    shading_mae = np.abs(shading_preds - shading_targets).mean()
    print(f"  Shading MAE: {shading_mae:.4f}")

    # Create 2x3 plot: shading + 4 global properties
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Validation - Epoch {epoch}', fontsize=16)

    # Panel 1: Shading prediction (log-log)
    ax = axes[0, 0]
    # Filter out very small values for log scale
    valid_mask = (shading_targets > 0.01) & (shading_preds > 0.01)
    if valid_mask.sum() > 0:
        ax.scatter(shading_targets[valid_mask], shading_preds[valid_mask],
                   s=1, alpha=0.3, c='blue')
        ax.set_xscale('log')
        ax.set_yscale('log')
        lims = [0.01, max(shading_targets.max(), shading_preds.max()) * 1.1]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xlabel('True tau_shading')
    ax.set_ylabel('Predicted tau_shading')
    ax.set_title(f'Shading (N={valid_mask.sum()})')
    ax.grid(True, alpha=0.3)

    # Panels 2-5: Global properties
    param_names = ["Optical Thickness", "Ice/Liq Ratio", "Reff Liquid", "Reff Ice"]
    axes_flat = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    # Extract columns for masking
    tau_true = global_targets[:, 0]
    ice_liq_ratio = global_targets[:, 1]

    # Masks
    is_ice = ice_liq_ratio > 0.75
    is_liquid = ice_liq_ratio < 0.25
    valid_tau = tau_true > 1.0

    cmap = plt.get_cmap('viridis')
    colors = cmap(ice_liq_ratio)

    for i, (ax, name) in enumerate(zip(axes_flat, param_names)):
        true_vals = global_targets[:, i]
        pred_vals = global_preds[:, i]

        current_mask = np.ones_like(true_vals, dtype=bool)

        if i == 0:  # Tau
            pass
        elif i == 1:  # Ice/Liq Ratio
            current_mask = valid_tau
        elif i == 2:  # Reff Liquid
            current_mask = is_liquid & valid_tau
        elif i == 3:  # Reff Ice
            current_mask = is_ice & valid_tau

        x = true_vals[current_mask]
        y = pred_vals[current_mask]
        c = colors[current_mask]

        if len(x) == 0:
            ax.text(0.5, 0.5, "No Valid Samples", ha='center', transform=ax.transAxes)
            continue

        ax.scatter(x, y, s=5, c=c, alpha=0.5)

        # 1:1 Line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

        ax.set_title(f"{name} (N={len(x)})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

        if i != 1:
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)

    # Empty panel
    axes[1, 2].axis('off')

    plt.tight_layout()
    plot_path = Path(output_dir) / f"val_epoch_{epoch:03d}.png"
    plt.savefig(plot_path, dpi=100)
    print(f"  Saved validation plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train shading-aware cloud property inversion model'
    )
    # Data paths
    parser.add_argument('--lut', type=str, required=True,
                        help='Path to main LUT file (NetCDF/Zarr)')
    parser.add_argument('--transmission_dir', type=str, required=True,
                        help='Path to directory with intermediate parquet files for transmission')
    parser.add_argument('--output', type=str, default='model_shading.pth',
                        help='Path to save model checkpoint')
    parser.add_argument('--plot_dir', type=str, default='plots_shading',
                        help='Directory to save validation plots')

    # Band selection
    parser.add_argument('--bands', type=str, nargs='+', default=None,
                        help='Bands to use (e.g., --bands B01 B02 B04). Default: all 11 bands')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of bags per batch')
    parser.add_argument('--bag_size', type=int, default=128,
                        help='Number of pixels per bag')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--in_memory', action='store_true',
                        help='Load base LUT into RAM for speed')
    parser.add_argument('--num_workers', type=int, default=4)

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for model')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_attention_layers', type=int, default=2,
                        help='Number of self-attention layers')

    # Loss weights
    parser.add_argument('--shading_weight', type=float, default=1.0,
                        help='Weight for shading loss')
    parser.add_argument('--global_weight', type=float, default=1.0,
                        help='Weight for global physics loss')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading Dataset from {args.lut}")
    print(f"Transmission LUTs from {args.transmission_dir}")

    DatasetClass = InMemoryShadingBagDataset if args.in_memory else ShadingBagDataset
    dataset = DatasetClass(
        lut_path=args.lut,
        transmission_lut_dir=args.transmission_dir,
        bag_size=args.bag_size,
        selected_bands=args.bands,
    )

    if args.bands:
        print(f"Using bands: {args.bands}")
    else:
        print(f"Using all {dataset.num_bands} default bands")

    # 2. Setup DataLoader with custom collate function
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=shading_collate_fn,
        persistent_workers=True,
    )

    # 3. Create config (for input_size calculation)
    config = Refl2PropConfig(
        bands=dataset.selected_bands,
        model_path=args.output,
    )
    print(f"Model: input_size={config.input_size}, bag_size={args.bag_size}, "
          f"hidden_dim={args.hidden_dim}, n_heads={args.n_heads}")

    # 4. Initialize Model
    core_model = ShadingAwareInversionNet(
        input_size=config.input_size,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_attention_layers=args.n_attention_layers,
        output_size=config.output_size,
    ).to(device)

    model = ShadingNormalizationWrapper(
        core_model,
        input_stats=dataset.input_stats,
        output_stats=dataset.output_stats,
        shading_stats=dataset.shading_stats,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # 5. Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create infinite iterator
    train_iterator = iter(loader)

    # 6. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_shading_loss = 0
        total_global_loss = 0

        # LR Decay: reduce by 0.2 every 30 epochs
        lr = args.lr * (0.2 ** (epoch // 30))
        optimizer.param_groups[0]['lr'] = lr
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {lr:.2e}")

        # Training steps per epoch
        epoch_batches = 1000

        for i in tqdm(range(epoch_batches), desc=f"Epoch {epoch+1}",
                      smoothing=0.05, ncols=100):
            try:
                inputs, shading_targets, global_targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(loader)
                inputs, shading_targets, global_targets = next(train_iterator)

            inputs = inputs.to(device)
            shading_targets = shading_targets.to(device)
            global_targets = global_targets.to(device)

            optimizer.zero_grad()

            # Forward pass (returns log-space shading and normalized global)
            shading_pred_log, global_pred_norm = model(inputs, return_denormalized=False)

            # Convert targets to same space for loss computation
            # shading_targets are physical tau values -> convert to log-space
            shading_targets_log = model.normalize_shading(shading_targets)
            global_targets_norm = model.normalize_global(global_targets)

            # Compute combined loss (both shading values in log-space)
            loss, loss_dict = combined_shading_loss(
                pred_log_shading=shading_pred_log,
                target_log_shading=shading_targets_log,
                pred_global_norm=global_pred_norm,
                target_global_norm=global_targets_norm,
                target_global_denorm=global_targets,
                shading_weight=args.shading_weight,
                global_weight=args.global_weight,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss_dict['total']
            total_shading_loss += loss_dict['shading']
            total_global_loss += loss_dict['global']

        avg_loss = total_loss / epoch_batches
        avg_shading = total_shading_loss / epoch_batches
        avg_global = total_global_loss / epoch_batches
        print(f"  Avg Loss: {avg_loss:.6f} (Shading: {avg_shading:.6f}, Global: {avg_global:.6f})")

        # Validation
        validate_and_plot_shading(
            model, train_iterator, device, epoch + 1, args.plot_dir, val_batches=20
        )

        # Save checkpoint
        torch.save(model.state_dict(), args.output)

    print(f"\nTraining complete. Final model saved to {args.output}")


if __name__ == '__main__':
    main()
