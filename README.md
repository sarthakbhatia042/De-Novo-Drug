# De Novo Drug Design — Graph Diffusion Model

Generates novel, chemically valid, drug-like molecular structures using a
discrete graph diffusion model (DiGress / D3PM-style), trained on 100k SMILES
from the ZINC250k dataset and represented natively as molecular graphs (atoms =
nodes, bonds = edges) rather than as SMILES text.

## Results (Trained Model — 100 epochs, 100k molecules)

| Metric | Value | Notes |
|---|---|---|
| **Validity** | **31.7%** | Post-hoc valence repair applied |
| **Uniqueness** | **100%** | Zero mode collapse |
| **Novelty** | **100%** | No training set memorisation |
| **Drug-like rate** | **9.5%** (95 / 1000) | Lipinski RO5 compliant |
| Mean LogP | 2.43 | Approved drug avg: 2.5 ✅ |
| Mean QED | 0.611 | Approved drug avg: 0.67 ✅ |
| Mean MW | 409 Da | Within Lipinski space ✅ |
| Scaffold diversity | 0.989 / 1.0 | 94 unique scaffolds from 95 hits |
| Veber oral bioavailability | 96% of hits | TPSA ≤ 140, RotBonds ≤ 10 |
| Egan permeability | 100% of hits | TPSA ≤ 131.6, LogP ≤ 5.88 |

## How it works

1. **Data** (`data/dataset.py`) — loads a SMILES CSV, filters to molecules
   with ≤ `MAX_ATOMS` (38) heavy atoms, converts each to a padded `(X, E)`
   graph pair (`X`: atom types, `E`: dense bond-type adjacency), and wraps
   it in a PyTorch `Dataset` / `DataLoader`.

2. **Model** (`model/gnn.py`) — a Graph Transformer that takes a noisy graph
   `(X_t, E_t)` plus timestep `t` and predicts the clean graph's atom/bond-
   type distributions. Dense self-attention over nodes, edge-conditioned
   attention bias, FiLM timestep conditioning.

3. **Diffusion** (`model/diffusion.py`) — uniform-transition discrete diffusion
   (D3PM-style) with a cosine noise schedule. Forward process corrupts
   atom/bond types toward uniform noise over T=500 steps; reverse process
   iteratively denoises using the trained GNN.

4. **Valence repair** (`utils/chemistry.py` → `correct_valence_graph`) —
   after sampling, over-valenced atoms have their weakest bonds removed
   deterministically before RDKit conversion. This is a post-hoc correction
   that does not interfere with the diffusion trajectory.

5. **Training** (`train.py`) — samples random timesteps, noises the batch,
   computes cross-entropy loss between predicted and true clean graphs
   (masked to exclude padding nodes).

6. **Generation & evaluation** (`generate.py`) — runs the full T-step reverse
   diffusion from random noise, applies valence repair, converts graphs to
   SMILES via RDKit, and reports **Validity**, **Uniqueness**, **Novelty**,
   **Drug-like rate**, mean LogP / QED / MW.

7. **Analysis** (`analyze_hits.py`) — computes 14 physicochemical properties
   per molecule (MW, LogP, HBD, HBA, TPSA, RotBonds, rings, QED, Lipinski /
   Veber / Egan compliance, Murcko scaffold), generates property distribution
   plots, a radar chart vs approved drugs, functional group census, scaffold
   diversity analysis, and a ranked top-16 molecule grid.

## Setup

```bash
# Clone and install
cd molgen_diffusion
python3 -m venv .venv
source .venv/bin/activate

# Analysis only (no GPU needed)
pip install rdkit matplotlib pandas numpy

# Full training (GPU recommended)
pip install -r requirements.txt
```

## Usage

```bash
# 1. Smoke-test the data pipeline
python data/dataset.py

# 2. Train on a real ZINC/ChEMBL subset (100k molecules, GPU recommended)
python train.py --input path/to/zinc_250k.csv \
    --epochs 100 --batch_size 64 --n_workers 4

# 3. Generate and evaluate (uses existing checkpoint, no retraining)
python generate.py \
    --checkpoint checkpoints/model_epoch100.pt \
    --n_samples 1000 \
    --batch_size 100 \
    --training_smiles path/to/zinc_250k.csv \
    --output outputs/generated_smiles.txt

# 4. Run property analysis on the generated hits
python analyze_hits.py

# 5. Run unit tests
pytest tests/test_shapes.py -v
```

`--input` expects a CSV (with a `smiles` column) or a plain-text file with one
SMILES string per line. Point it at a local ZINC250k or ChEMBL export.

## Architecture

```
Input: noisy graph (X_t, E_t) at timestep t
  ↓  timestep embedding (sinusoidal → linear → FiLM scale/shift)
  ↓  node embedding  [N × node_dim]
  ↓  edge embedding  [N × N × edge_dim]
  ↓  Graph Transformer layers (L=6):
       dense self-attention with edge-conditioned bias
       FiLM conditioning from timestep embedding
       residual + layer norm
  ↓  node head  → logits [N × num_atom_types]
  ↓  edge head  → logits [N × N × num_bond_types]
Output: predicted clean graph (X_0, E_0) distributions
```

- **Atom types (10):** padding, C, N, O, F, P, S, Cl, Br, I
- **Bond types (5):** none, single, double, triple, aromatic
- **Max graph size:** 38 heavy atoms (covers ~95% of ZINC drug-like space)
- **Diffusion steps:** T = 500 (cosine schedule)
- **Training loss:** cross-entropy on X and E, masked for padding nodes

## Design decisions

- **Graph diffusion over SMILES diffusion** — discrete diffusion on SMILES
  strings is less mature; graph-based diffusion (DiGress) is the current
  state-of-the-art for molecular generation.
- **Dense adjacency tensors** — small fixed-size graphs (≤38 nodes) make
  dense `N×N` attention tractable and simpler than sparse message passing.
- **Post-hoc valence repair** — applying valence constraints inside the
  diffusion loop (at every step) causes feedback collapse to linear chains;
  applying it once on the final committed graph correctly fixes valence
  violations without affecting ring-forming dynamics.

## Verification status

| Check | Status |
|---|---|
| Unit tests (`tests/test_shapes.py`) | ✅ All passing |
| End-to-end pipeline (demo set) | ✅ No errors |
| Training on ZINC250k (100k mol, 100 ep) | ✅ Completed on Colab A100 |
| Generated molecule analysis | ✅ 95 drug-like hits characterised |

## Known limitations / next steps

- **Validity ceiling (~32%)** — further training (150+ epochs) expected to
  push toward 40–50%. Property-conditional generation (targeting specific
  LogP / QED) not yet implemented (would require classifier-free guidance).
- **Fixed generation size** — molecules are generated at a fixed `--n_nodes`
  size; a learned size distribution from the training set would give more
  realistic diversity.
- **No 3D geometry** — this model generates 2D connectivity graphs only; a
  subsequent conformer generation step (RDKit `EmbedMolecule` / ETKDGv3)
  would be needed for docking or MD simulation.
- **Sampling speed** — O(T) sequential steps (~60 s / 100 molecules on GPU);
  a DDIM-style shortcut sampler could reduce this 10×.
