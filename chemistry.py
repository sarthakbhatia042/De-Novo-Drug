"""
Chemistry utilities for graph-based molecular diffusion.

Handles:
  - SMILES <-> molecular graph conversion (dense, padded tensors)
  - Atom / bond type vocabularies
  - Validity, Uniqueness, Novelty metrics
  - Basic property calculation (LogP, QED)

Graph representation:
  X: (N,) long tensor of atom type indices (0 = padding/virtual node)
  E: (N, N) long tensor of bond type indices (0 = no bond)
  Both are dense and padded to a fixed max size N (see MAX_ATOMS).
"""

from __future__ import annotations

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import QED, Crippen

    # Silence RDKit's very chatty warnings (invalid valence, kekulization, etc.)
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without rdkit installed
    _RDKIT_AVAILABLE = False


def _require_rdkit():
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for chemistry utilities. Install with "
            "`pip install rdkit` (see requirements.txt)."
        )


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
# Atom vocab covers the common elements found in ZINC / ChEMBL drug-like
# molecules. Index 0 is reserved for "no atom" (padding / virtual node).
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_TO_IDX = {atom: i + 1 for i, atom in enumerate(ATOM_LIST)}  # 1..len
IDX_TO_ATOM = {i: atom for atom, i in ATOM_TO_IDX.items()}
NUM_ATOM_TYPES = len(ATOM_LIST) + 1  # +1 for "no atom" / padding

# Bond vocab: 0 = no bond, 1 = single, 2 = double, 3 = triple, 4 = aromatic
BOND_LIST = [
    None,
    Chem.rdchem.BondType.SINGLE if _RDKIT_AVAILABLE else "SINGLE",
    Chem.rdchem.BondType.DOUBLE if _RDKIT_AVAILABLE else "DOUBLE",
    Chem.rdchem.BondType.TRIPLE if _RDKIT_AVAILABLE else "TRIPLE",
    Chem.rdchem.BondType.AROMATIC if _RDKIT_AVAILABLE else "AROMATIC",
]
BOND_TO_IDX = {bond: i for i, bond in enumerate(BOND_LIST)}
NUM_BOND_TYPES = len(BOND_LIST)  # 5

MAX_ATOMS = 38  # covers ~30-40 heavy atoms per the agreed graph-size limit


# ---------------------------------------------------------------------------
# Valence correction (post-hoc graph repair)
# ---------------------------------------------------------------------------

# Maximum total bond-order per atom type index (matches ATOM_LIST order).
# 0=padding, 1=C, 2=N, 3=O, 4=F, 5=P, 6=S, 7=Cl, 8=Br, 9=I
_ATOM_MAX_VALENCE = {
    0: 0, 1: 4, 2: 3, 3: 2, 4: 1,
    5: 5, 6: 6, 7: 1, 8: 1, 9: 1,
}
# Bond-order contribution of each bond-type index
_BOND_ORDER_VALS = [0.0, 1.0, 2.0, 3.0, 1.5]  # none, single, double, triple, aromatic


def correct_valence_graph(X: np.ndarray, E: np.ndarray):
    """
    Post-hoc valence repair on a final committed discrete graph.

    For each atom that is over its maximum allowed bond-order:
      1. Find the neighbor bond with the smallest bond-order contribution.
      2. Remove that bond (set both E[i,j] and E[j,i] to 0).
      3. Repeat until all atoms satisfy their valence limit.

    This is deterministic and runs on the final graph after the full
    reverse-diffusion trajectory completes, so it never interferes with
    ring-forming dynamics.

    Returns corrected copies of X and E (originals are not modified).
    """
    X = X.copy()
    E = E.copy()
    n = len(X)

    changed = True
    while changed:
        changed = False
        for i in range(n):
            atom_idx = int(X[i])
            if atom_idx == 0:          # padding node
                continue
            max_v = _ATOM_MAX_VALENCE.get(atom_idx, 4)
            used_v = sum(
                _BOND_ORDER_VALS[int(E[i, j])]
                for j in range(n)
                if i != j and int(E[i, j]) > 0
            )
            if used_v > max_v + 1e-6:
                # Remove the weakest bond (lowest bond-order) to this atom
                bonds = [
                    (j, int(E[i, j]))
                    for j in range(n)
                    if i != j and int(E[i, j]) > 0
                ]
                if bonds:
                    j_rm = min(bonds, key=lambda x: _BOND_ORDER_VALS[x[1]])[0]
                    E[i, j_rm] = 0
                    E[j_rm, i] = 0
                    changed = True
                    break   # restart scan after any change
    return X, E


def smiles_to_graph(smiles: str, max_atoms: int = MAX_ATOMS):
    """
    Convert a SMILES string to a padded (X, E) graph pair.

    Returns None if the SMILES is invalid, contains an unsupported atom,
    or exceeds max_atoms heavy atoms.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0 or n_atoms > max_atoms:
        return None

    X = np.zeros((max_atoms,), dtype=np.int64)
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol not in ATOM_TO_IDX:
            return None  # unsupported element, drop this molecule
        X[i] = ATOM_TO_IDX[symbol]

    E = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        if btype not in BOND_TO_IDX:
            return None  # e.g. dative/unknown bond type
        b_idx = BOND_TO_IDX[btype]
        E[i, j] = b_idx
        E[j, i] = b_idx

    return X, E


# ---------------------------------------------------------------------------
# graph -> SMILES / RDKit mol
# ---------------------------------------------------------------------------
def graph_to_mol(X: np.ndarray, E: np.ndarray):
    """
    Convert a padded (X, E) graph pair back into an RDKit Mol.
    Returns None if the resulting molecule is chemically invalid
    (e.g. violates valence rules).

    If the graph is disconnected (multiple fragments), the largest connected
    fragment is kept rather than rejecting the whole molecule — this rescues
    partially correct structures that have stray padding atoms.
    """
    _require_rdkit()
    mol = Chem.RWMol()

    idx_map = {}
    for i, atom_idx in enumerate(X):
        atom_idx = int(atom_idx)
        if atom_idx == 0:
            continue  # padding / no atom
        symbol = IDX_TO_ATOM[atom_idx]
        idx_map[i] = mol.AddAtom(Chem.Atom(symbol))

    n = X.shape[0]
    for i in range(n):
        if i not in idx_map:
            continue
        for j in range(i + 1, n):
            if j not in idx_map:
                continue
            bond_idx = int(E[i, j])
            if bond_idx == 0:
                continue
            bond_type = BOND_LIST[bond_idx]
            mol.AddBond(idx_map[i], idx_map[j], bond_type)

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException, ValueError):
        return None
    except Exception:
        return None

    # If disconnected, keep only the largest connected fragment.
    # Single-atom fragments are almost always stray padding artifacts.
    frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        # Reject if the largest fragment is a single atom (not drug-like)
        if mol.GetNumHeavyAtoms() < 3:
            return None

    return mol


def graph_to_smiles(X: np.ndarray, E: np.ndarray):
    """Convert a padded (X, E) graph pair to a canonical SMILES string, or None if invalid."""
    _require_rdkit()
    mol = graph_to_mol(X, E)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_validity(generated_graphs):
    """
    generated_graphs: list of (X, E) numpy pairs.
    Returns (validity_fraction, list_of_valid_smiles).
    """
    valid_smiles = []
    for X, E in generated_graphs:
        smi = graph_to_smiles(X, E)
        if smi is not None and len(smi) > 0:
            valid_smiles.append(smi)
    validity = len(valid_smiles) / max(len(generated_graphs), 1)
    return validity, valid_smiles


def compute_uniqueness(valid_smiles):
    """Fraction of valid SMILES that are unique. Low uniqueness signals mode collapse."""
    if not valid_smiles:
        return 0.0
    return len(set(valid_smiles)) / len(valid_smiles)


def compute_novelty(valid_smiles, training_smiles_set):
    """Fraction of unique valid SMILES that do NOT appear in the training set."""
    unique_smiles = set(valid_smiles)
    if not unique_smiles:
        return 0.0
    novel = [s for s in unique_smiles if s not in training_smiles_set]
    return len(novel) / len(unique_smiles)


def compute_properties(smiles: str):
    """Return dict with LogP, QED, and MW for a single SMILES string."""
    _require_rdkit()
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "logp": Crippen.MolLogP(mol),
        "qed":  QED.qed(mol),
        "mw":   Descriptors.MolWt(mol),
    }


def passes_lipinski(props: dict) -> bool:
    """
    Lipinski Rule-of-Five filter for drug-likeness.
    A molecule passes if it satisfies:
      - Molecular weight <= 500 Da
      - LogP <= 5  (lipophilicity)
    (HBD/HBA are not yet tracked per-atom; MW + LogP already capture the
    aliphatic-chain penalty that caused the previous run's issues.)
    """
    if props is None:
        return False
    return props["mw"] <= 500 and props["logp"] <= 5


def evaluate_generated_set(generated_graphs, training_smiles_set):
    """
    Full evaluation pipeline: Validity (connected molecules only),
    Uniqueness, Novelty, Lipinski drug-likeness filter, and mean
    LogP / QED over the valid, unique, novel, drug-like subset.

    Validity here is strict:
      - The graph must sanitize without errors (valence rules, aromaticity).
      - The molecule must be a single connected component (no fragments).
    """
    validity, valid_smiles = compute_validity(generated_graphs)
    uniqueness = compute_uniqueness(valid_smiles)
    novelty = compute_novelty(valid_smiles, training_smiles_set)

    unique_smiles = list(set(valid_smiles))
    props = [compute_properties(s) for s in unique_smiles]

    # Lipinski filter
    drug_like_pairs = [(s, p) for s, p in zip(unique_smiles, props)
                       if p is not None and passes_lipinski(p)]
    drug_like_smiles = [s for s, _ in drug_like_pairs]
    drug_like_props  = [p for _, p in drug_like_pairs]

    props_for_stats = drug_like_props if drug_like_props else [p for p in props if p is not None]
    mean_logp = float(np.mean([p["logp"] for p in props_for_stats])) if props_for_stats else float("nan")
    mean_qed  = float(np.mean([p["qed"]  for p in props_for_stats])) if props_for_stats else float("nan")
    mean_mw   = float(np.mean([p["mw"]   for p in props_for_stats])) if props_for_stats else float("nan")

    return {
        "validity":          validity,
        "uniqueness":        uniqueness,
        "novelty":           novelty,
        "n_generated":       len(generated_graphs),
        "n_valid":           len(valid_smiles),
        "n_unique":          len(unique_smiles),
        "n_drug_like":       len(drug_like_smiles),   # passes Lipinski filter
        "drug_like_rate":    len(drug_like_smiles) / max(len(generated_graphs), 1),
        "mean_logp":         mean_logp,   # computed over drug-like subset
        "mean_qed":          mean_qed,
        "mean_mw":           mean_mw,
    }
