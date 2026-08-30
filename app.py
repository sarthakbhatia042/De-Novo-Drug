"""
De Novo Drug Design — Streamlit Demo
=====================================
Interactive demo for the Graph Diffusion molecular generation model.
Serves pre-generated drug-like molecules instantly (Option A — no GPU needed).

Author : Sarthak Bhatia
GitHub : https://github.com/sarthakbhatia042/De-Novo-Drug
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="De Novo Drug Design · Sarthak Bhatia",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a0a3a 0%, #0a1a3a 45%, #0a2a2a 100%);
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 36px 48px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 25% 50%, rgba(124,111,255,0.18) 0%, transparent 55%),
                radial-gradient(ellipse at 75% 50%, rgba(0,212,255,0.12) 0%, transparent 55%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero .subtitle { font-size: 1.05rem; color: #9999bb; margin: 10px 0 4px; }
.hero .stats    { font-size: 0.88rem; color: #6666aa; margin: 0; }

/* Metric cards */
.metric-row { display: flex; gap: 14px; margin: 16px 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: linear-gradient(135deg, #1a1a35, #1a2035);
    border: 1px solid #2a2a45;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.metric-card .val {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .lbl { font-size: 0.75rem; color: #8888aa; margin-top: 4px; }

/* Info box */
.info-box {
    background: rgba(124,111,255,0.08);
    border: 1px solid rgba(124,111,255,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #b0b0cc;
    margin-top: 12px;
}

/* Section header */
.section-header {
    font-size: 1.15rem; font-weight: 700;
    color: #a78bfa;
    border-bottom: 1px solid #2a2a45;
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* SMILES tag */
.smiles-tag {
    font-family: monospace;
    background: #1a1a30;
    border: 1px solid #2a2a45;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #7c6fff;
    word-break: break-all;
    display: block;
    margin: 4px 0;
}

/* Pass / fail badges */
.badge-pass { background:#14532d; color:#86efac; border-radius:5px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-fail { background:#450a0a; color:#fca5a5; border-radius:5px; padding:2px 8px; font-size:0.78rem; font-weight:600; }

/* GitHub link button */
.gh-btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 9px 20px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50px;
    color: #ccccee !important;
    text-decoration: none !important;
    font-size: 0.87rem;
    transition: all 0.2s;
}
.gh-btn:hover { background: rgba(124,111,255,0.2); border-color: rgba(124,111,255,0.4); }

/* Streamlit widget tweaks */
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600; font-size: 0.92rem;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #7c6fff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & data
# ─────────────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
ASSETS = BASE / "assets"

@st.cache_data
def load_pool() -> pd.DataFrame:
    df = pd.read_csv(ASSETS / "drug_like_hits.csv")
    return df[df["smiles"].notna()].reset_index(drop=True)

POOL = load_pool()
POOL_SIZE = len(POOL)

DRUG_REF = {"mw": 340, "logp": 2.5, "hbd": 1.8, "hba": 5.1,
            "tpsa": 76, "rot_bonds": 5.6, "rings": 2.8, "qed": 0.67}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_molecules(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    n = min(n, POOL_SIZE)
    return POOL.iloc[rng.sample(range(POOL_SIZE), n)].reset_index(drop=True)


def mol_grid_image(smiles_list: list[str], mols_per_row: int = 4) -> Image.Image:
    mols, legends = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            row = POOL[POOL["smiles"] == smi]
            if not row.empty:
                r = row.iloc[0]
                legends.append(f"QED={r['qed']:.3f}  MW={r['mw']:.0f}\nLogP={r['logp']:.2f}")
            else:
                legends.append("")
            mols.append(mol)
    if not mols:
        return Image.new("RGB", (400, 200), (17, 17, 39))
    return Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row,
        subImgSize=(320, 256), legends=legends, returnPNG=False,
    )


def radar_chart(df: pd.DataFrame) -> Image.Image:
    props  = ["mw", "logp", "hbd", "hba", "tpsa", "rot_bonds", "qed"]
    labels = ["MW/500", "LogP/5", "HBD/5", "HBA/10", "TPSA/140", "RotB/10", "QED"]
    norms  = {"mw":500,"logp":5,"hbd":5,"hba":10,"tpsa":140,"rot_bonds":10,"qed":1}
    N = len(props)
    angles = [n/N*2*math.pi for n in range(N)] + [0]
    gv = [min(df[p].mean()/norms[p], 1.2) for p in props] + [0]
    dv = [DRUG_REF[p]/norms[p] for p in props] + [0]
    gv[-1], dv[-1] = gv[0], dv[0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.plot(angles, gv, color="#7c6fff", linewidth=2.5, label="Generated (mean)")
    ax.fill(angles, gv, color="#7c6fff", alpha=0.28)
    ax.plot(angles, dv, color="#ff6b6b", linewidth=2.5, linestyle="--", label="Approved drugs")
    ax.fill(angles, dv, color="#ff6b6b", alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="white")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], fontsize=0)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#333355")
    ax.grid(color="#333355", linewidth=0.7)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", loc="upper right", bbox_to_anchor=(1.5, 1.15))
    ax.set_title("Drug-likeness Radar", fontsize=11, color="white", pad=15, fontweight="bold")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def analyze_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None, None
    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings= rdMolDescriptors.CalcNumRings(mol)
    qed  = QED.qed(mol)
    ro5  = mw<=500 and logp<=5 and hbd<=5 and hba<=10
    veber= tpsa<=140 and rotb<=10
    egan = tpsa<=131.6 and logp<=5.88
    img  = Draw.MolToImage(mol, size=(420, 320))
    props = dict(mw=mw, logp=logp, hbd=hbd, hba=hba, tpsa=tpsa,
                 rotb=rotb, rings=rings, qed=qed, ro5=ro5, veber=veber, egan=egan)
    return img, props

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="font-size:52px;margin-bottom:10px;">🧬</div>
  <h1>De Novo Drug Design</h1>
  <p class="subtitle">Graph Diffusion Model · Trained on 100k ZINC250k Molecules</p>
  <p class="stats">31.7% validity &nbsp;·&nbsp; 100% uniqueness &nbsp;·&nbsp; 100% novelty &nbsp;·&nbsp; 9.5% drug-like rate</p>
  <div style="margin-top:18px;">
    <a href="https://github.com/sarthakbhatia042/De-Novo-Drug" target="_blank" class="gh-btn">
      <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57
                 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695
                 -.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99
                 .105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225
                 -.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405
                 c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0
                 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0
                 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
      </svg>
      View Source on GitHub
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Global metrics strip
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Validity",      "31.7%",  help="Molecules that pass RDKit sanitisation")
m2.metric("Uniqueness",    "100%",   help="Zero mode collapse")
m3.metric("Novelty",       "100%",   help="Not seen in the 100k training set")
m4.metric("Drug-like Rate","9.5%",   help="Lipinski Rule-of-Five compliant")
m5.metric("Mean QED",      "0.611",  help="Approved drug avg: 0.67")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_gen, tab_analyze, tab_analysis, tab_about = st.tabs([
    "🧬  Generate Molecules",
    "🔍  Analyze a SMILES",
    "📊  Full Analysis",
    "📖  About the Model",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ══════════════════════════════════════════════════════════════════════════════
with tab_gen:
    st.markdown(
        "Sample from the **95 drug-like molecules** produced by the trained 100-epoch "
        "Graph Diffusion model. Every molecule passes Lipinski Rule-of-Five, is 100% novel "
        "(not in ZINC250k), and has QED > 0.5."
    )

    col_ctrl, col_main = st.columns([1, 3], gap="large")

    with col_ctrl:
        n_mols = st.slider("Molecules to sample", 4, 16, 8, step=4)
        seed   = st.number_input("Random seed", value=42, step=1,
                                  help="Change for a different random selection")
        generate_btn = st.button("⚗️ Generate", type="primary", use_container_width=True)

        st.markdown("""
<div class="info-box">
<strong>ℹ️ Demo Mode</strong><br><br>
Samples are drawn instantly from the model's pre-generated output pool.<br><br>
Full reverse diffusion (500 steps) runs in ~60s/100 molecules on an A100 GPU.<br><br>
<a href="https://github.com/sarthakbhatia042/De-Novo-Drug" target="_blank"
   style="color:#a78bfa;">→ Run locally with GPU</a>
</div>
""", unsafe_allow_html=True)

    with col_main:
        if generate_btn or "gen_df" not in st.session_state:
            st.session_state.gen_df = sample_molecules(n_mols, int(seed))

        df = st.session_state.gen_df
        n  = len(df)

        # ── Batch metrics ──────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sampled", n)
        c2.metric("Mean QED",  f"{df['qed'].mean():.3f}")
        c3.metric("Mean LogP", f"{df['logp'].mean():.3f}")
        c4.metric("Mean MW",   f"{df['mw'].mean():.0f} Da")
        veber_pct = int(df["veber"].sum() / n * 100) if n else 0
        c5.metric("Veber Pass", f"{veber_pct}%")

        # ── Molecule grid ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">2D Molecule Grid</div>', unsafe_allow_html=True)
        mols_per_row = min(4, n)
        grid_img = mol_grid_image(df["smiles"].tolist(), mols_per_row=mols_per_row)
        st.image(grid_img, use_container_width=True)

        # ── Two-column: radar + property table ────────────────────────────
        r_col, t_col = st.columns([1, 1], gap="medium")
        with r_col:
            st.markdown('<div class="section-header">Drug-likeness Radar</div>',
                        unsafe_allow_html=True)
            st.image(radar_chart(df), use_container_width=True)

        with t_col:
            st.markdown('<div class="section-header">Property Table</div>',
                        unsafe_allow_html=True)
            display_df = df[["smiles","qed","mw","logp","hbd","hba","tpsa","veber","egan"]].copy()
            display_df["smiles"] = display_df["smiles"].apply(lambda s: s[:30]+"…" if len(s)>30 else s)
            display_df.columns = ["SMILES","QED","MW","LogP","HBD","HBA","TPSA","Veber","Egan"]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "QED":  st.column_config.ProgressColumn("QED", min_value=0, max_value=1, format="%.3f"),
                    "MW":   st.column_config.NumberColumn("MW (Da)", format="%.0f"),
                    "LogP": st.column_config.NumberColumn("LogP", format="%.2f"),
                },
            )

        # ── SMILES expander ────────────────────────────────────────────────
        with st.expander("📋 Show SMILES strings"):
            for smi in df["smiles"].tolist():
                st.markdown(f'<span class="smiles-tag">{smi}</span>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download SMILES as .txt",
                data="\n".join(df["smiles"].tolist()),
                file_name="generated_molecules.txt",
                mime="text/plain",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analyze a SMILES
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown("Paste any SMILES string to compute its **physicochemical properties** and "
                "drug-likeness filter results.")

    EXAMPLES = [
        "Cc1ccccc2cc(cc1C(=O)NCCN)N(CC1CCO1)C2=O",
        "CCNC(=O)C(C1=CC=CC=CC=C1)N(C)CC",
        "CC1CC(=O)NCC(=O)N2CCC3=CC=CC1=CC=CC=CC2=C3",
        "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
    ]

    example_labels = [
        "Hit #1 (QED=0.851)",
        "Hit #2 (QED=0.805)",
        "Hit #3 (QED=0.745)",
        "Aspirin",
        "Caffeine",
    ]

    selected = st.selectbox("Try an example", ["(paste your own below)"] + example_labels)
    if selected != "(paste your own below)":
        default_smi = EXAMPLES[example_labels.index(selected)]
    else:
        default_smi = ""

    user_smi = st.text_input("SMILES string", value=default_smi,
                              placeholder="e.g. Cc1ccccc2cc(cc1C(=O)NCCN)N(CC1CCO1)C2=O")
    analyze_btn = st.button("🔬 Analyze", type="primary")

    if analyze_btn and user_smi.strip():
        img, props = analyze_smiles(user_smi)
        if img is None:
            st.error("❌ Invalid SMILES string — RDKit could not parse it. Please check.")
        else:
            st.success("✅ Molecule parsed successfully")
            img_col, prop_col = st.columns([1, 1.4], gap="large")

            with img_col:
                st.markdown('<div class="section-header">2D Structure</div>',
                            unsafe_allow_html=True)
                st.image(img, use_container_width=True)

            with prop_col:
                st.markdown('<div class="section-header">Properties</div>',
                            unsafe_allow_html=True)

                p = props
                prop_data = {
                    "Property": ["Molecular Weight", "LogP", "H-Bond Donors",
                                 "H-Bond Acceptors", "TPSA", "Rotatable Bonds",
                                 "Ring Count", "QED Score"],
                    "Value":    [f"{p['mw']:.2f} Da", f"{p['logp']:.3f}", str(p['hbd']),
                                 str(p['hba']), f"{p['tpsa']:.1f} Å²", str(p['rotb']),
                                 str(p['rings']), f"{p['qed']:.4f}"],
                    "Limit":    ["≤500 Da","≤5","≤5","≤10","≤140 Å²","≤10","—","—"],
                    "Status":   [
                        "✅" if p['mw']  <=500 else "❌",
                        "✅" if p['logp']<=5   else "❌",
                        "✅" if p['hbd'] <=5   else "❌",
                        "✅" if p['hba'] <=10  else "❌",
                        "✅" if p['tpsa']<=140 else "❌",
                        "✅" if p['rotb']<=10  else "❌",
                        "—", "—",
                    ],
                }
                st.dataframe(pd.DataFrame(prop_data), hide_index=True, use_container_width=True)

                st.markdown("**Drug-likeness Filters**")
                f1, f2, f3 = st.columns(3)
                f1.metric("Lipinski RO5",  "PASS ✅" if p['ro5']   else "FAIL ❌")
                f2.metric("Veber Oral",    "PASS ✅" if p['veber'] else "FAIL ❌")
                f3.metric("Egan Perm.",    "PASS ✅" if p['egan']  else "FAIL ❌")

                qed_color = "#22c55e" if p['qed']>0.6 else "#f59e0b" if p['qed']>0.4 else "#ef4444"
                qed_label = "Good" if p['qed']>0.6 else "Fair" if p['qed']>0.4 else "Low"
                st.markdown(f"""
<div style="margin-top:14px; padding:14px 18px; background:#1a1a30;
            border-left: 4px solid {qed_color}; border-radius:8px;">
  <span style="color:#8888aa; font-size:0.82rem;">QED Score</span><br>
  <span style="font-size:2rem; font-weight:800; color:{qed_color};">{p['qed']:.4f}</span>
  <span style="color:{qed_color}; font-size:0.85rem; margin-left:8px;">● {qed_label}</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Full Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown("Pre-computed analysis of all **95 drug-like hits** from 1000 generated molecules.")

    st.markdown('<div class="section-header">Physicochemical Property Distributions</div>',
                unsafe_allow_html=True)
    st.image(str(ASSETS / "property_distributions.png"), use_container_width=True)

    c_radar, c_fg = st.columns(2, gap="large")
    with c_radar:
        st.markdown('<div class="section-header">Radar: Generated vs Approved Drugs</div>',
                    unsafe_allow_html=True)
        st.image(str(ASSETS / "radar_chart.png"), use_container_width=True)
    with c_fg:
        st.markdown('<div class="section-header">Functional Group Census</div>',
                    unsafe_allow_html=True)
        st.image(str(ASSETS / "functional_groups.png"), use_container_width=True)

    st.markdown('<div class="section-header">Top 16 Molecules by QED Score</div>',
                unsafe_allow_html=True)
    st.image(str(ASSETS / "top16_by_qed.png"), use_container_width=True)

    st.markdown("**Summary Statistics — Drug-like Hits**")
    df_stats = POOL[["mw","logp","hbd","hba","tpsa","rot_bonds","rings","qed"]]
    st.dataframe(
        df_stats.describe().round(3),
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in df_stats.columns}
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## 🔬 How the Model Works

Generates drug-like molecules from pure random noise using **discrete graph diffusion** —
representing molecules as graphs (atoms = nodes, bonds = edges) rather than SMILES strings.

### Architecture
```
Input: noisy graph (X_t, E_t) at timestep t
  ↓  sinusoidal timestep embedding → FiLM conditioning
  ↓  node embedding  [N × 128]
  ↓  edge embedding  [N × N × 64]
  ↓  6 × Graph Transformer layers:
       dense self-attention with edge-conditioned bias
       FiLM conditioning from timestep embedding
       residual + LayerNorm
  ↓  node head  → atom-type logits [N × 10]
  ↓  edge head  → bond-type logits [N × N × 5]
Output: predicted clean graph (X_0, E_0)
```
""")

    col_arch, col_res = st.columns(2, gap="large")

    with col_arch:
        st.markdown("**Model Parameters**")
        arch_df = pd.DataFrame({
            "Parameter": ["Atom types","Bond types","Max atoms","Diffusion steps",
                           "Transformer layers","Node dim","Edge dim","Training"],
            "Value":     ["10 (C,N,O,F,P,S,Cl,Br,I + pad)","5 (none,single,double,triple,aromatic)",
                           "38 heavy atoms","T = 500 (cosine schedule)",
                           "6","128","64","100 epochs · 100k ZINC250k · A100 GPU"],
        })
        st.dataframe(arch_df, hide_index=True, use_container_width=True)

    with col_res:
        st.markdown("**Results**")
        res_df = pd.DataFrame({
            "Metric":  ["Validity","Uniqueness","Novelty","Drug-like rate",
                         "Mean LogP","Mean QED","Mean MW","Scaffold diversity",
                         "Veber oral bioavail.","Egan permeability"],
            "Value":   ["31.7%","100%","100%","9.5% (95/1000)",
                         "2.43","0.611","409 Da","0.989 / 1.0",
                         "96% of hits","100% of hits"],
            "Notes":   ["Post-hoc valence repair","Zero mode collapse","No training-set memorisation",
                         "Lipinski RO5 compliant","Approved drug avg: 2.5 ✅",
                         "Approved drug avg: 0.67 ✅","Within Lipinski space ✅",
                         "94 unique scaffolds from 95 hits",
                         "TPSA ≤ 140, RotBonds ≤ 10","TPSA ≤ 131.6, LogP ≤ 5.88"],
        })
        st.dataframe(res_df, hide_index=True, use_container_width=True)

    st.markdown("""
### Key Design Decisions

- **Graph diffusion over SMILES** — discrete diffusion on SMILES strings is less mature;
  graph-based diffusion (DiGress) is current state-of-the-art for molecular generation.
- **Dense adjacency tensors** — small fixed-size graphs (≤38 nodes) make dense N×N
  attention tractable without sparse message-passing complexity.
- **Post-hoc valence repair** — applying valence constraints *during* the diffusion loop
  causes collapse to linear chains; post-hoc repair correctly fixes violations without
  affecting ring-forming dynamics.

---

📂 **[Full source code & training instructions → GitHub](https://github.com/sarthakbhatia042/De-Novo-Drug)**

> *This demo samples from 95 drug-like hits generated by the trained model
> (1 000 molecules sampled, 317 valid, 95 pass Lipinski RO5).
> All molecules are 100% novel — none appear in the ZINC250k training set.*
""")
