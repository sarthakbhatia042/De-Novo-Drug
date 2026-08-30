"""
Train the graph denoising diffusion model.

Usage:
    python train.py --input path/to/zinc_subset.csv --epochs 50 --batch_size 64

With no --input, trains on the tiny bundled demo set (10 molecules) — useful
only to sanity-check that the pipeline runs end-to-end and the loss decreases
on an overfit test, per the manual verification plan.
"""

from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

from data.dataset import get_dataloader
from model.gnn import DenoisingGNN
from model.diffusion import GraphDiscreteDiffusion
from utils.chemistry import NUM_ATOM_TYPES, NUM_BOND_TYPES, MAX_ATOMS


def build_node_mask(X: torch.Tensor) -> torch.Tensor:
    """A node is 'real' (not padding) if its atom-type index is nonzero."""
    return X != 0


def main():
    parser = argparse.ArgumentParser(description="Train the graph diffusion de novo drug design model.")
    parser.add_argument("--input", type=str, default=None, help="Path to SMILES CSV/TXT (ZINC/ChEMBL subset).")
    parser.add_argument("--max_atoms", type=int, default=MAX_ATOMS)
    parser.add_argument("--limit", type=int, default=100_000, help="Max molecules after filtering (50k-100k recommended).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=500, help="Number of diffusion timesteps.")
    parser.add_argument("--node_dim", type=int, default=128)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device
    print(f"Using device: {device}")

    loader, kept_smiles = get_dataloader(
        input_path=args.input,
        max_atoms=args.max_atoms,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    model = DenoisingGNN(
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        n_layers=args.n_layers,
    ).to(device)

    diffusion = GraphDiscreteDiffusion(
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        T=args.T,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters. Training on {len(kept_smiles)} molecules.")

    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_node_loss, epoch_edge_loss, n_batches = 0.0, 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            X0 = batch["X"].to(device)
            E0 = batch["E"].to(device)
            node_mask = build_node_mask(X0)

            optimizer.zero_grad()
            loss, logs = diffusion.training_loss(model, X0, E0, node_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_node_loss += logs["node_loss"]
            epoch_edge_loss += logs["edge_loss"]
            n_batches += 1
            pbar.set_postfix(loss=loss.item(), node=logs["node_loss"], edge=logs["edge_loss"])

        print(
            f"Epoch {epoch}: avg_loss={epoch_loss / n_batches:.4f}  "
            f"node_loss={epoch_node_loss / n_batches:.4f}  edge_loss={epoch_edge_loss / n_batches:.4f}"
        )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
