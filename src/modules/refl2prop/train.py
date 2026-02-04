import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dataset import InMemoryRefl2PropDataset, Refl2PropDataset, collate_fn
from .model import InversionNet, NormalizationWrapper, NoiseGenerator
from .loss import physics_masked_loss, combined_loss


def validate_and_plot(model, iterator, device, epoch, output_dir, val_batches=20):
    """
    Runs inference using the EXISTING iterator.
    No new processes spawned, no dataset re-initialization.
    """
    model.eval()
    print(f"\n--- Running Validation (Epoch {epoch}) ---")

    all_preds = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for _ in range(val_batches):
            try:
                # Consume batches from the already-live worker pool
                inputs, targets = next(iterator)
            except StopIteration:
                # Should not happen with infinite dataset, but safety first
                break

            inputs = inputs.to(device)
            # model() returns PHYSICAL values (denormalized) and uncertainty
            preds, uncertainty = model(inputs, return_uncertainty=True)

            all_preds.append(preds.cpu())
            all_targets.append(targets)
            all_uncertainties.append(uncertainty.cpu())

    if not all_preds:
        return

    # Cat into large tensors
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    uncertainties = torch.cat(all_uncertainties, dim=0).numpy()

    # Print uncertainty statistics
    print(f"  Uncertainty stats: mean={uncertainties.mean():.4f}, std={uncertainties.std():.4f}, "
          f"min={uncertainties.min():.4f}, max={uncertainties.max():.4f}")

    # --- Plotting Configuration ---
    param_names = ["Optical Thickness", "Ice/Liq Ratio", "Reff Liquid", "Reff Ice"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Validation - Epoch {epoch}', fontsize=16)
    
    # Extract columns
    tau_true = targets[:, 0]
    ice_liq_ratio = targets[:, 1]
    
    # Masks
    is_ice = ice_liq_ratio > 0.75
    is_liquid = ice_liq_ratio < 0.25
    valid_tau = tau_true > 1.0

    cmap = plt.get_cmap('viridis')
    colors = cmap(ice_liq_ratio) 

    for i, ax in enumerate(axes.flat):
        name = param_names[i]
        true_vals = targets[:, i]
        pred_vals = preds[:, i]
        
        current_mask = np.ones_like(true_vals, dtype=bool)
        
        if i == 0: # Tau
            pass
        elif i == 1: # Ice/Liq Ratio
            current_mask = valid_tau
        elif i == 2: # Reff Liquid
            current_mask = is_liquid & valid_tau
        elif i == 3: # Reff Ice
            current_mask = is_ice & valid_tau
            
        x = true_vals[current_mask]
        y = pred_vals[current_mask]
        c = colors[current_mask]
        
        if len(x) == 0:
            ax.text(0.5, 0.5, "No Valid Samples", ha='center')
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

    plt.tight_layout()
    plot_path = Path(output_dir) / f"val_epoch_{epoch:03d}.png"
    plt.savefig(plot_path)
    print(f"Saved validation plot to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lut', type=str, required=True, help='Path to .nc/.zip LUT file')
    parser.add_argument('--output', type=str, default='model.pth', help='Path to save model')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save validation plots')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--in_memory', action='store_true', help='Load dataset into RAM')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--noise_min', type=float, default=0.001, help='Min noise magnitude (fraction of range)')
    parser.add_argument('--noise_max', type=float, default=0.02, help='Max noise magnitude (fraction of range)')
    parser.add_argument('--physics_weight', type=float, default=1.0, help='Weight for physics loss')
    parser.add_argument('--noise_weight', type=float, default=1.0, help='Weight for noise reconstruction loss')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    DatasetClass = InMemoryRefl2PropDataset if args.in_memory else Refl2PropDataset
    print(f"Loading Dataset: {args.lut}")
    dataset = DatasetClass(args.lut, n_dims_interp=4, spectral_mode='variable')
    
    # 2. Setup Persistent Loader
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn,
        persistent_workers=True 
    )

    # 3. Init Model
    input_size = len(dataset.input_stats['min'])
    core_model = InversionNet(input_size=input_size, output_size=4, noise_output_size=6).to(device)
    model = NormalizationWrapper(core_model, dataset.input_stats, dataset.output_stats).to(device)

    # 4. Init Noise Generator
    noise_generator = NoiseGenerator(dataset.input_stats, noise_min=args.noise_min, noise_max=args.noise_max)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create the infinite iterator ONCE
    train_iterator = iter(loader)

    # 5. Train Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_physics_loss = 0
        total_noise_loss = 0

        # LR Decay
        lr = args.lr * (0.2 ** (epoch // 30))
        optimizer.param_groups[0]['lr'] = lr
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {lr:.2e}")

        # Training Steps
        epoch_batches = 1000
        for i in tqdm(range(epoch_batches), desc=f"Epoch {epoch+1}", smoothing=0.05, ncols=100):
            try:
                inputs, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(loader)
                inputs, targets = next(train_iterator)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Generate and add noise
            noise_to_add, target_noise = noise_generator.generate(inputs.size(0), device)
            noisy_inputs = inputs + noise_to_add

            # Forward pass with noisy inputs
            norm_in = model.normalize_input(noisy_inputs)
            norm_tgt = (targets - model.out_min) / (model.out_max - model.out_min) * 2 - 1
            physics_pred, noise_pred = model.model(norm_in)

            # Combined loss
            loss, loss_dict = combined_loss(
                physics_pred, norm_tgt, targets,
                noise_pred, target_noise,
                physics_weight=args.physics_weight, noise_weight=args.noise_weight
            )

            loss.backward()
            optimizer.step()

            total_loss += loss_dict['total']
            total_physics_loss += loss_dict['physics']
            total_noise_loss += loss_dict['noise']

        avg_loss = total_loss / epoch_batches
        avg_physics = total_physics_loss / epoch_batches
        avg_noise = total_noise_loss / epoch_batches
        print(f"  Avg Loss: {avg_loss:.6f} (Physics: {avg_physics:.6f}, Noise: {avg_noise:.6f})")

        # Validation Steps (Reuse same iterator!)
        # Pass 20 batches (approx 2500 samples) for validation
        validate_and_plot(model, train_iterator, device, epoch + 1, args.plot_dir, val_batches=20)

        # Save checkpoint
        torch.save(model.state_dict(), args.output)

    print(f"Final model saved to {args.output}")

if __name__ == '__main__':
    main()