# De Novo Drug Design — Graph Diffusion Model

Generates novel, chemically valid, drug-like molecular structures using a
discrete graph diffusion model (DiGress / D3PM-style), trained on 100k SMILES
from the ZINC250k dataset and represented natively as molecular graphs (atoms =
nodes, bonds = edges) rather than as SMILES text.


<img width="1520" height="1200" alt="image" src="https://github.com/user-attachments/assets/61955c2b-d62e-4762-ad65-d0e0c3ad205c" />



## Results (Trained Model — 100 epochs, 100k molecules)

| Metric | Value | Notes |
|---|---|---|
| **Validity** | **31.7%** | Post-hoc valence repair applied |
| **Uniqueness** | **100%** | Zero mode collapse |
| **Novelty** | **100%** | No training set memorisation |
| **Drug-like rate** | **9.5%** (95 / 1000) | Lipinski RO5 compliant |
| Mean LogP | 2.43 | Approved drug avg: 2.5 |
| Mean QED | 0.611 | Approved drug avg: 0.67 |
| Mean MW | 409 Da | Within Lipinski space |
| Scaffold diversity | 0.989 / 1.0 | 94 unique scaffolds from 95 hits |
| Veber oral bioavailability | 96% of hits | TPSA ≤ 140, RotBonds ≤ 10 |
| Egan permeability | 100% of hits | TPSA ≤ 131.6, LogP ≤ 5.88 |


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
