"""
Sample molecules from a trained checkpoint and run the RDKit validation
(Validity, Uniqueness, Novelty) described in the verification plan.

Usage:
    python generate.py --checkpoint checkpoints/model_epoch50.pt --n_samples 1000 \
        --training_smiles path/to/training_subset.csv
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm

from data.dataset import load_smiles_file, DEMO_SMILES
from model.gnn import DenoisingGNN
from model.diffusion import GraphDiscreteDiffusion
from utils.chemistry import (
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    MAX_ATOMS,
    graph_to_smiles,
    evaluate_generated_set,
    correct_valence_graph,
)


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt["args"]
    model = DenoisingGNN(
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        node_dim=train_args["node_dim"],
        edge_dim=train_args["edge_dim"],
        n_layers=train_args["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, train_args


def main():
    parser = argparse.ArgumentParser(description="Sample and evaluate molecules from a trained diffusion model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_nodes", type=int, default=MAX_ATOMS, help="Graph size to sample at.")
    parser.add_argument("--batch_size", type=int, default=100, help="Samples per reverse-diffusion batch.")
    parser.add_argument("--training_smiles", type=str, default=None,
                         help="Path to the SMILES file used for training, for Novelty calc. "
                              "Defaults to the bundled demo set if omitted.")
    parser.add_argument("--output", type=str, default="outputs/generated_smiles.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = args.device

    model, train_args = load_model(args.checkpoint, device)
    diffusion = GraphDiscreteDiffusion(
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        T=train_args["T"],
        device=device,
    )

    node_mask = torch.ones((args.batch_size, args.n_nodes), dtype=torch.bool, device=device)

    all_graphs = []
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    for _ in tqdm(range(n_batches), desc="Sampling"):
        X, E = diffusion.sample(model, args.batch_size, args.n_nodes, node_mask, device=device)
        for i in range(X.shape[0]):
            all_graphs.append((X[i], E[i]))
    all_graphs = all_graphs[: args.n_samples]

    if args.training_smiles is not None:
        training_smiles_set = set(load_smiles_file(args.training_smiles))
    else:
        print("No --training_smiles given: using the bundled demo set for Novelty (not meaningful "
              "for a real trained model - pass the actual training subset).")
        training_smiles_set = set(DEMO_SMILES)

    # Apply post-hoc valence correction to every sampled graph before
    # converting to RDKit molecules.  This removes excess bonds from
    # over-valenced atoms (the dominant source of sanitization failures)
    # without touching the diffusion process itself.
    repaired_graphs = []
    n_repaired = 0
    for X, E in all_graphs:
        X_fixed, E_fixed = correct_valence_graph(X, E)
        if not (X_fixed == X).all() or not (E_fixed == E).all():
            n_repaired += 1
        repaired_graphs.append((X_fixed, E_fixed))
    print(f"Valence repair: {n_repaired}/{len(all_graphs)} graphs had bonds removed.")

    metrics = evaluate_generated_set(repaired_graphs, training_smiles_set)
    print(json.dumps(metrics, indent=2))

    valid_smiles = [graph_to_smiles(X, E) for X, E in repaired_graphs]
    valid_smiles = [s for s in valid_smiles if s]
    with open(args.output, "w") as f:
        f.write("\n".join(valid_smiles))
    print(f"Wrote {len(valid_smiles)} valid SMILES to {args.output}")

    metrics_path = args.output.replace(".txt", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
