"""
Denoising network for graph diffusion (DiGress-style).

Takes a noisy graph (X_t, E_t) and timestep t, and predicts the clean
graph's categorical distributions (X_0 logits, E_0 logits). Implemented
as a Graph Transformer: dense self-attention over nodes, modulated by
edge features, with FiLM-style timestep conditioning.

We deliberately use a *dense* attention formulation (not sparse PyG
message passing) because our graphs are small (<= MAX_ATOMS ~ 38 nodes)
and fully padded/batched as dense tensors - this keeps the implementation
simple and matches how the diffusion process (dense X_t, E_t) is defined.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long tensor of timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class FiLM(nn.Module):
    """Feature-wise linear modulation for injecting timestep info into node/edge features."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # feat: (B, ..., feat_dim), cond: (B, cond_dim)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        while gamma.dim() < feat.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return feat * (1 + gamma) + beta


class GraphTransformerLayer(nn.Module):
    """
    One layer of edge-conditioned dense self-attention over nodes, plus an
    edge-update MLP. Roughly follows the DiGress graph transformer block.
    """

    def __init__(self, node_dim: int, edge_dim: int, n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert node_dim % n_heads == 0, "node_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        # Edge features bias the attention logits, one scalar bias per head.
        self.edge_to_bias = nn.Linear(edge_dim, n_heads)
        self.out_proj = nn.Linear(node_dim, node_dim)

        self.node_norm1 = nn.LayerNorm(node_dim)
        self.node_norm2 = nn.LayerNorm(node_dim)
        self.node_ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * ff_mult),
            nn.GELU(),
            nn.Linear(node_dim * ff_mult, node_dim),
        )

        # Update edge features from the pair of incident node features.
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_update = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim * ff_mult),
            nn.GELU(),
            nn.Linear(edge_dim * ff_mult, edge_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, E: torch.Tensor, node_mask: torch.Tensor):
        """
        X: (B, N, node_dim), E: (B, N, N, edge_dim), node_mask: (B, N) bool (True = real node)
        """
        B, N, _ = X.shape
        residual = X
        Xn = self.node_norm1(X)

        q = self.q_proj(Xn).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,N,d)
        k = self.k_proj(Xn).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(Xn).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        edge_bias = self.edge_to_bias(E).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn_logits = attn_logits + edge_bias

        # Mask out padded nodes (both as queries producing garbage and as keys)
        pad_mask = (~node_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,N) - True where key is padding
        attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn)  # rows that are fully padded -> softmax of all -inf -> nan
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # (B,H,N,d)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)

        X = residual + self.dropout(out)
        X = X + self.dropout(self.node_ff(self.node_norm2(X)))

        # Edge update: concat the two endpoint node features with current edge feature
        Xi = X.unsqueeze(2).expand(B, N, N, X.shape[-1])
        Xj = X.unsqueeze(1).expand(B, N, N, X.shape[-1])
        edge_in = torch.cat([Xi, Xj, self.edge_norm(E)], dim=-1)
        E = E + self.dropout(self.edge_update(edge_in))
        E = 0.5 * (E + E.transpose(1, 2))  # keep symmetric (undirected graph)

        return X, E


class DenoisingGNN(nn.Module):
    """
    Full denoising network: embeds (X_t, E_t, t) -> stack of GraphTransformerLayers
    -> per-node atom-type logits and per-edge bond-type logits (predicting X_0, E_0).
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        node_dim: int = 128,
        edge_dim: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        time_emb_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types

        self.atom_embed = nn.Embedding(num_atom_types, node_dim)
        self.bond_embed = nn.Embedding(num_bond_types, edge_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.node_film = FiLM(time_emb_dim, node_dim)
        self.edge_film = FiLM(time_emb_dim, edge_dim)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(node_dim, edge_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.node_out = nn.Linear(node_dim, num_atom_types)
        self.edge_out = nn.Linear(edge_dim, num_bond_types)

    def forward(self, X_t: torch.Tensor, E_t: torch.Tensor, t: torch.Tensor, node_mask: torch.Tensor):
        """
        X_t: (B, N) long - noisy atom type indices
        E_t: (B, N, N) long - noisy bond type indices
        t:   (B,) long - diffusion timestep per sample
        node_mask: (B, N) bool - True where the node is real (not padding)

        Returns: X_logits (B, N, num_atom_types), E_logits (B, N, N, num_bond_types)
        """
        X = self.atom_embed(X_t)  # (B, N, node_dim)
        E = self.bond_embed(E_t)  # (B, N, N, edge_dim)

        cond = self.time_mlp(t)  # (B, time_emb_dim)
        X = self.node_film(X, cond)
        E = self.edge_film(E, cond)

        for layer in self.layers:
            X, E = layer(X, E, node_mask)

        X_logits = self.node_out(X)
        E_logits = self.edge_out(E)
        # Zero out logits for padded nodes/edges contribution is handled in the loss via masking,
        # but we still symmetrize edge logits for cleanliness.
        E_logits = 0.5 * (E_logits + E_logits.transpose(1, 2))
        return X_logits, E_logits
