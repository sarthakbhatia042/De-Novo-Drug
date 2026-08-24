"""
Shape/sanity unit tests (no RDKit/chemistry required beyond the fixed vocab
sizes), per the automated tests item in the verification plan:
"Tensor shape validation to ensure the GNN handles batched node matrices
(B x N x d_X) and edge tensors (B x N x N x d_E) correctly."

Run with: pytest tests/test_shapes.py -v
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gnn import DenoisingGNN
from model.diffusion import GraphDiscreteDiffusion

NUM_ATOM_TYPES = 10  # 9 elements + padding
NUM_BOND_TYPES = 5   # none/single/double/triple/aromatic
B, N = 4, 12          # small batch/graph size for fast tests


def _make_model():
    return DenoisingGNN(
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        node_dim=32,
        edge_dim=16,
        n_layers=2,
        n_heads=4,
        time_emb_dim=32,
    )


def test_gnn_output_shapes():
    model = _make_model()
    X_t = torch.randint(0, NUM_ATOM_TYPES, (B, N))
    E_t = torch.randint(0, NUM_BOND_TYPES, (B, N, N))
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + E_t.transpose(1, 2)
    t = torch.randint(0, 500, (B,))
    node_mask = torch.ones(B, N, dtype=torch.bool)

    X_logits, E_logits = model(X_t, E_t, t, node_mask)

    assert X_logits.shape == (B, N, NUM_ATOM_TYPES), f"got {X_logits.shape}"
    assert E_logits.shape == (B, N, N, NUM_BOND_TYPES), f"got {E_logits.shape}"


def test_edge_logits_symmetric():
    model = _make_model()
    X_t = torch.randint(0, NUM_ATOM_TYPES, (B, N))
    E_t = torch.randint(0, NUM_BOND_TYPES, (B, N, N))
    t = torch.randint(0, 500, (B,))
    node_mask = torch.ones(B, N, dtype=torch.bool)

    _, E_logits = model(X_t, E_t, t, node_mask)
    assert torch.allclose(E_logits, E_logits.transpose(1, 2), atol=1e-5)


def test_padding_mask_handled():
    """A fully-padded row of nodes should not produce NaNs in the output."""
    model = _make_model()
    X_t = torch.randint(0, NUM_ATOM_TYPES, (B, N))
    E_t = torch.randint(0, NUM_BOND_TYPES, (B, N, N))
    t = torch.randint(0, 500, (B,))
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[:, N // 2 :] = False  # pad out the second half of each graph

    X_logits, E_logits = model(X_t, E_t, t, node_mask)
    assert not torch.isnan(X_logits).any()
    assert not torch.isnan(E_logits).any()


def test_diffusion_q_sample_shapes():
    diffusion = GraphDiscreteDiffusion(NUM_ATOM_TYPES, NUM_BOND_TYPES, T=100, device="cpu")
    X0 = torch.randint(1, NUM_ATOM_TYPES, (B, N))
    E0 = torch.randint(0, NUM_BOND_TYPES, (B, N, N))
    E0 = torch.triu(E0, diagonal=1)
    E0 = E0 + E0.transpose(1, 2)
    t = torch.randint(0, 100, (B,))

    X_t, E_t = diffusion.q_sample(X0, E0, t)
    assert X_t.shape == (B, N)
    assert E_t.shape == (B, N, N)
    assert torch.equal(E_t, E_t.transpose(1, 2))


def test_diffusion_training_loss_runs_and_is_finite():
    model = _make_model()
    diffusion = GraphDiscreteDiffusion(NUM_ATOM_TYPES, NUM_BOND_TYPES, T=100, device="cpu")
    X0 = torch.randint(1, NUM_ATOM_TYPES, (B, N))
    E0 = torch.randint(0, NUM_BOND_TYPES, (B, N, N))
    E0 = torch.triu(E0, diagonal=1)
    E0 = E0 + E0.transpose(1, 2)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    loss, logs = diffusion.training_loss(model, X0, E0, node_mask)
    assert torch.isfinite(loss)
    assert "node_loss" in logs and "edge_loss" in logs
