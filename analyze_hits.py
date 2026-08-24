"""
Property Analysis of Drug-Like Hits from Graph Diffusion De Novo Design
=======================================================================
Run locally:  .venv/bin/python analyze_hits.py
Run in Colab: exec(open('analyze_hits.py').read())

Requires: rdkit, matplotlib, pandas, numpy
Input   : outputs/generated_smiles_v4.txt
Outputs : outputs/analysis/
            drug_like_hits.csv
            property_distributions.png
            radar_chart.png
            functional_groups.png
            scaffold_diversity.csv
            top16_by_qed.png
            property_analysis_summary.txt
"""

import os, math, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — works in terminal and Colab
import matplotlib.pyplot as plt
from collections import Counter

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, QED, Crippen, rdMolDescriptors, Draw
)
from rdkit.Chem.Scaffolds import MurckoScaffold

SMILES_FILE = "outputs/generated_smiles_v4.txt"
OUT_DIR     = "outputs/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load SMILES and compute full property table
# ──────────────────────────────────────────────────────────────────────────────
print("Loading SMILES and computing properties...")

with open(SMILES_FILE) as f:
    all_smiles = [s.strip() for s in f if s.strip()]

records = []
for smi in all_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    mw       = Descriptors.MolWt(mol)
    logp     = Crippen.MolLogP(mol)
    hbd      = rdMolDescriptors.CalcNumHBD(mol)
    hba      = rdMolDescriptors.CalcNumHBA(mol)
    tpsa     = rdMolDescriptors.CalcTPSA(mol)
    rotb     = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings    = rdMolDescriptors.CalcNumRings(mol)
    ar_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    stereo   = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    qed_val  = QED.qed(mol)

    ro5   = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    veber = (tpsa <= 140 and rotb <= 10)
    egan  = (tpsa <= 131.6 and logp <= 5.88)

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""

    records.append({
        "smiles": smi, "mw": round(mw, 2), "logp": round(logp, 3),
        "hbd": hbd, "hba": hba, "tpsa": round(tpsa, 2),
        "rot_bonds": rotb, "rings": rings, "arom_rings": ar_rings,
        "stereocenters": stereo, "qed": round(qed_val, 4),
        "lipinski": ro5, "veber": veber, "egan": egan, "scaffold": scaffold,
    })

df_all = pd.DataFrame(records)
df     = df_all[df_all["lipinski"]].sort_values("qed", ascending=False).reset_index(drop=True)

print(f"Total valid SMILES    : {len(all_smiles)}")
print(f"Lipinski-passing      : {len(df)}")
print(f"Also pass Veber rules : {df['veber'].sum()}")
print(f"Also pass Egan rules  : {df['egan'].sum()}")

df.to_csv(f"{OUT_DIR}/drug_like_hits.csv", index=False)

props = ["mw", "logp", "hbd", "hba", "tpsa", "rot_bonds", "rings", "qed"]
print("\n── Property Statistics (Drug-like Hits) ────────────────────────")
print(df[props].describe().round(3).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 2. Property distribution plots (8-panel)
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating property distribution plots...")

DRUG_REF = {"mw": 340, "logp": 2.5, "hbd": 1.8, "hba": 5.1,
            "tpsa": 76, "rot_bonds": 5.6, "rings": 2.8, "qed": 0.67}
LABELS   = {"mw": "Molecular Weight (Da)", "logp": "LogP",
            "hbd": "H-Bond Donors", "hba": "H-Bond Acceptors",
            "tpsa": "TPSA (Å²)", "rot_bonds": "Rotatable Bonds",
            "rings": "Ring Count", "qed": "QED Score"}
LIMITS   = {"mw": (0,550), "logp": (-2,6), "hbd": (0,6), "hba": (0,12),
            "tpsa": (0,160), "rot_bonds": (0,12), "rings": (0,8), "qed": (0,1)}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Physicochemical Property Distributions — Drug-like Generated Molecules",
             fontsize=14, fontweight="bold")

for ax, prop in zip(axes.flat, props):
    vals = df[prop].values
    ax.hist(vals, bins=20, color="#6c63ff", edgecolor="white", alpha=0.85, density=True)
    ax.axvline(vals.mean(), color="#00d4ff", linewidth=2,
               label=f"Generated: {vals.mean():.2f}")
    ax.axvline(DRUG_REF[prop], color="#ff6b6b", linewidth=2, linestyle="--",
               label=f"Drugs avg: {DRUG_REF[prop]}")
    ax.set_xlabel(LABELS[prop], fontsize=10)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(LIMITS[prop])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/property_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Radar chart: generated hits vs approved oral drugs
# ──────────────────────────────────────────────────────────────────────────────
print("Generating radar chart...")

radar_props  = ["mw", "logp", "hbd", "hba", "tpsa", "rot_bonds", "qed"]
radar_labels = ["MW/500", "LogP/5", "HBD/5", "HBA/10", "TPSA/140", "RotB/10", "QED"]
norms        = {"mw": 500, "logp": 5, "hbd": 5, "hba": 10,
                "tpsa": 140, "rot_bonds": 10, "qed": 1}

gen_vals  = [min(df[p].mean() / norms[p], 1.2) for p in radar_props]
drug_vals = [DRUG_REF[p] / norms[p] for p in radar_props]

N      = len(radar_props)
angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]
gv     = gen_vals  + gen_vals[:1]
dv     = drug_vals + drug_vals[:1]

fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax2.plot(angles, gv, color="#6c63ff", linewidth=2.5, label="Generated (mean)")
ax2.fill(angles, gv, color="#6c63ff", alpha=0.25)
ax2.plot(angles, dv, color="#ff6b6b", linewidth=2.5, linestyle="--", label="Approved drugs (avg)")
ax2.fill(angles, dv, color="#ff6b6b", alpha=0.15)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(radar_labels, fontsize=11)
ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
ax2.set_title("Property Radar: Generated Hits vs Approved Oral Drugs",
              fontsize=12, pad=20, fontweight="bold")
ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/radar_chart.png", dpi=150, bbox_inches="tight")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Functional group census
# ──────────────────────────────────────────────────────────────────────────────
print("Functional group analysis...")

FG_SMARTS = {
    "Amide (C=O-N)":          "[CX3](=O)[NX3]",
    "Amine (1°/2°/3°)":       "[NX3;H2,H1,H0;!$(NC=O)]",
    "Aromatic ring":           "c1ccccc1",
    "Ether (C-O-C)":          "[OD2]([#6])[#6]",
    "Hydroxyl (-OH)":          "[OX2H]",
    "Ketone/Aldehyde (C=O)":  "[CX3](=O)[#6]",
    "Cyclopropane":            "C1CC1",
    "Oxetane":                 "C1CCO1",
    "Pyrrolidine/lactam":      "C1CCNC1",
    "Piperidine":              "C1CCNCC1",
    "Halogen (F/Cl/Br/I)":    "[F,Cl,Br,I]",
    "Nitrile (C≡N)":          "[CX2]#[NX1]",
    "Sulfonamide":             "[SX4](=O)(=O)[NX3]",
    "Sulfide/Thioether":       "[SX2]([#6])[#6]",
}

fg_counts = {}
for name, smarts in FG_SMARTS.items():
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        continue
    count = 0
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol and mol.HasSubstructMatch(patt):
            count += 1
    fg_counts[name] = count

fg_df = (pd.DataFrame(fg_counts.items(), columns=["Functional Group", "Count"])
           .assign(**{"Fraction (%)": lambda d: (d["Count"] / len(df) * 100).round(1)})
           .sort_values("Count", ascending=False)
           .reset_index(drop=True))

print("\n── Functional Group Census ──────────────────────────────────────")
print(fg_df.to_string(index=False))

fig3, ax3 = plt.subplots(figsize=(12, 6))
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(fg_df)))
bars = ax3.barh(fg_df["Functional Group"], fg_df["Fraction (%)"],
                color=colors, edgecolor="white")
for bar, val in zip(bars, fg_df["Fraction (%)"]):
    ax3.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}%", va="center", fontsize=9)
ax3.set_xlabel("Fraction of Drug-like Hits (%)", fontsize=11)
ax3.set_title("Functional Group Census — Drug-like Generated Molecules",
              fontsize=12, fontweight="bold")
ax3.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/functional_groups.png", dpi=150, bbox_inches="tight")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Scaffold diversity
# ──────────────────────────────────────────────────────────────────────────────
scaffold_counts   = Counter(s for s in df["scaffold"] if s)
n_unique_sc       = len(scaffold_counts)
singleton_sc      = sum(1 for c in scaffold_counts.values() if c == 1)
diversity_index   = n_unique_sc / len(df)

print(f"\n── Scaffold Diversity ────────────────────────────────────────────")
print(f"Unique Murcko scaffolds : {n_unique_sc} / {len(df)} hits")
print(f"Singleton scaffolds     : {singleton_sc}  ({singleton_sc/n_unique_sc*100:.0f}% of scaffolds)")
print(f"Diversity index         : {diversity_index:.3f}  (1.0 = all unique)")

sc_df = pd.DataFrame(scaffold_counts.most_common(20),
                     columns=["Scaffold SMILES", "Count"])
sc_df["Fraction (%)"] = (sc_df["Count"] / len(df) * 100).round(1)
sc_df.to_csv(f"{OUT_DIR}/scaffold_diversity.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Top-16 by QED — clean grid with property labels
# ──────────────────────────────────────────────────────────────────────────────
print("Generating top-16 QED molecule grid...")

top16  = df.head(16)
mols16 = [Chem.MolFromSmiles(s) for s in top16["smiles"]]
legends = [
    f"QED={row.qed:.3f}  MW={row.mw:.0f}\nLogP={row.logp:.2f}  TPSA={row.tpsa:.0f}"
    for _, row in top16.iterrows()
]
img = Draw.MolsToGridImage(mols16, molsPerRow=4, subImgSize=(380, 300), legends=legends)
img.save(f"{OUT_DIR}/top16_by_qed.png")
print(f"Saved: {OUT_DIR}/top16_by_qed.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Written summary report
# ──────────────────────────────────────────────────────────────────────────────
report_lines = [
    "De Novo Drug Design — Property Analysis Report",
    "=" * 55,
    f"Generated molecules : 1000",
    f"Valid (RDKit)        : {len(all_smiles)}  ({len(all_smiles)/1000*100:.1f}%)",
    f"Lipinski RO5 hits   : {len(df)}  ({len(df)/1000*100:.1f}%)",
    f"Also pass Veber     : {df['veber'].sum()}  ({df['veber'].sum()/len(df)*100:.0f}% of hits)",
    f"Also pass Egan      : {df['egan'].sum()}  ({df['egan'].sum()/len(df)*100:.0f}% of hits)",
    "",
    "── Mean Properties vs Approved Oral Drugs ─────────────────────────────",
    f"{'Property':<22} {'Generated':>12} {'Approved':>12}",
    f"{'Mol. Weight (Da)':<22} {df['mw'].mean():>12.1f} {DRUG_REF['mw']:>12}",
    f"{'LogP':<22} {df['logp'].mean():>12.3f} {DRUG_REF['logp']:>12}",
    f"{'H-Bond Donors':<22} {df['hbd'].mean():>12.2f} {DRUG_REF['hbd']:>12}",
    f"{'H-Bond Acceptors':<22} {df['hba'].mean():>12.2f} {DRUG_REF['hba']:>12}",
    f"{'TPSA (Å²)':<22} {df['tpsa'].mean():>12.1f} {DRUG_REF['tpsa']:>12}",
    f"{'Rotatable Bonds':<22} {df['rot_bonds'].mean():>12.2f} {DRUG_REF['rot_bonds']:>12}",
    f"{'Ring Count':<22} {df['rings'].mean():>12.2f} {DRUG_REF['rings']:>12}",
    f"{'QED Score':<22} {df['qed'].mean():>12.4f} {DRUG_REF['qed']:>12}",
    "",
    "── Scaffold Diversity ──────────────────────────────────────────────────",
    f"Unique Murcko scaffolds : {n_unique_sc} / {len(df)}",
    f"Singleton scaffolds     : {singleton_sc} ({singleton_sc/n_unique_sc*100:.0f}%)",
    f"Diversity index         : {diversity_index:.3f}",
    "",
    "── Top Functional Groups ───────────────────────────────────────────────",
]
for _, row in fg_df.head(6).iterrows():
    report_lines.append(f"  {row['Functional Group']:<30} {row['Count']:>3} hits ({row['Fraction (%)']:.0f}%)")

report_lines += [
    "",
    "── Top 5 Molecules by QED ──────────────────────────────────────────────",
]
for i, (_, row) in enumerate(df.head(5).iterrows()):
    report_lines.append(
        f"  Rank {i+1}: QED={row.qed:.3f}  MW={row.mw:.0f}  LogP={row.logp:.2f}"
        f"  SMILES={row.smiles[:55]}..."
    )

report_text = "\n".join(report_lines)
with open(f"{OUT_DIR}/property_analysis_summary.txt", "w") as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\n✓ All outputs saved to {OUT_DIR}/")
