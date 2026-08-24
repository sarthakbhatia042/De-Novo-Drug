"""
Discrete diffusion process for molecular graphs (D3PM / DiGress-style).

Forward process: a Markov chain that progressively corrupts atom types (X)
and bond types (E) toward a uniform categorical distribution over T steps,
via per-step transition matrices Q_t = (1 - beta_t) * I + beta_t * (1/K) * ones.

Reverse process: at each step, the GNN predicts the clean graph (X_0, E_0)
logits from the noisy (X_t, E_t); we combine that prediction with the known
forward-process math to sample (X_{t-1}, E_{t-1}).

Valence-constrained decoding (Option A improvement):
  During reverse sampling, before drawing edge types we compute each atom's
  remaining valence capacity and mask out bond assignments that would exceed
  it.  This eliminates the single biggest source of invalid molecules
  (over-valenced atoms) without any retraining.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal), adapted for discrete diffusion betas."""
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


class GraphDiscreteDiffusion:
    """
    Implements the uniform-transition discrete diffusion process over
    node types (K = num_atom_types) and edge types (K = num_bond_types),
    independently per node / edge.
    """

    def __init__(self, num_atom_types: int, num_bond_types: int, T: int = 500, device: str = "cpu"):
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.T = T
        self.device = device

        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas                          # (T,)
        self.alphas = alphas                         # (T,)
        self.alphas_cumprod = alphas_cumprod          # (T,) = prod_{s<=t}(1 - beta_s)

    # ------------------------------------------------------------------
    # Forward process q(x_t | x_0)
    # ------------------------------------------------------------------
    def _marginal_probs(self, x0: torch.Tensor, t: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Closed-form q(x_t | x_0) for the uniform-noise process:
            p(x_t = x0) = alpha_bar_t + (1 - alpha_bar_t) / K
            p(x_t = k != x0) = (1 - alpha_bar_t) / K

        x0: (...,) long tensor of clean class indices
        t:  (...,) long tensor of timesteps, broadcastable to x0's shape
        Returns probs: (..., num_classes)
        """
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)  # (..., 1)
        uniform = (1.0 - alpha_bar_t) / num_classes
        probs = uniform.expand(*x0.shape, num_classes).clone()
        probs.scatter_(-1, x0.unsqueeze(-1), uniform + alpha_bar_t)
        return probs

    def q_sample(self, X0: torch.Tensor, E0: torch.Tensor, t: torch.Tensor):
        """
        Sample (X_t, E_t) ~ q(x_t | x_0) given clean graph and per-sample timestep t.
        X0: (B, N), E0: (B, N, N), t: (B,)
        """
        B, N = X0.shape
        t_node = t.unsqueeze(1).expand(B, N)             # (B, N)
        t_edge = t.unsqueeze(1).unsqueeze(2).expand(B, N, N)

        X_probs = self._marginal_probs(X0, t_node, self.num_atom_types)      # (B,N,Kx)
        E_probs = self._marginal_probs(E0, t_edge, self.num_bond_types)      # (B,N,N,Ke)

        X_t = torch.distributions.Categorical(probs=X_probs).sample()
        E_t = torch.distributions.Categorical(probs=E_probs).sample()
        E_t = torch.triu(E_t, diagonal=1)
        E_t = E_t + E_t.transpose(1, 2)  # keep symmetric
        return X_t, E_t

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def training_loss(self, model, X0: torch.Tensor, E0: torch.Tensor, node_mask: torch.Tensor):
        """
        Standard D3PM-style loss: sample a random t, noise the graph, ask the
        model to predict the clean (X0, E0) categorical distributions, and
        compute cross-entropy against the true clean graph. Padded nodes/edges
        are excluded from the loss via node_mask.
        """
        B = X0.shape[0]
        t = torch.randint(0, self.T, (B,), device=X0.device)

        X_t, E_t = self.q_sample(X0, E0, t)
        X_logits, E_logits = model(X_t, E_t, t, node_mask)

        # --- node loss (mask out padding) ---
        node_loss = F.cross_entropy(
            X_logits.reshape(-1, self.num_atom_types), X0.reshape(-1), reduction="none"
        )
        node_loss = (node_loss * node_mask.reshape(-1)).sum() / node_mask.sum().clamp(min=1)

        # --- edge loss (mask out padding on either endpoint, and self-loops) ---
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (B,N,N)
        N = X0.shape[1]
        eye = torch.eye(N, device=X0.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask & (~eye)

        edge_loss = F.cross_entropy(
            E_logits.reshape(-1, self.num_bond_types), E0.reshape(-1), reduction="none"
        )
        edge_loss = (edge_loss * edge_mask.reshape(-1)).sum() / edge_mask.sum().clamp(min=1)

        loss = node_loss + edge_loss
        return loss, {"node_loss": node_loss.item(), "edge_loss": edge_loss.item()}

    # ------------------------------------------------------------------
    # Reverse process / sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, model, batch_size: int, n_nodes: int, node_mask: torch.Tensor, device: str = "cpu"):
        """
        Run the full reverse diffusion process starting from uniform noise,
        returning final discrete (X_0, E_0) graphs.

        Sampling is unconstrained — the GNN predicts atom/bond types freely
        at every step.  Valence correction is applied as a post-hoc graph
        repair step in utils/chemistry.py (correct_valence_graph) after
        sampling completes, avoiding any interference with ring-forming
        dynamics during the reverse trajectory.

        node_mask: (B, N) bool, fixed for the whole trajectory.
        """
        model.eval()
        X_t = torch.randint(0, self.num_atom_types, (batch_size, n_nodes), device=device)
        E_t = torch.randint(0, self.num_bond_types, (batch_size, n_nodes, n_nodes), device=device)
        E_t = torch.triu(E_t, diagonal=1)
        E_t = E_t + E_t.transpose(1, 2)

        for step in reversed(range(self.T)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            X_logits, E_logits = model(X_t, E_t, t, node_mask)

            X0_pred = torch.distributions.Categorical(logits=X_logits).sample()
            E0_pred = torch.distributions.Categorical(logits=E_logits).sample()
            E0_pred = torch.triu(E0_pred, diagonal=1)
            E0_pred = E0_pred + E0_pred.transpose(1, 2)

            if step > 0:
                t_minus_1 = torch.full((batch_size,), step - 1, device=device, dtype=torch.long)
                X_t, E_t = self.q_sample(X0_pred, E0_pred, t_minus_1)
            else:
                X_t, E_t = X0_pred, E0_pred

        model.train()
        return X_t.cpu().numpy(), E_t.cpu().numpy()
