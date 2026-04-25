# app/ui.py

import streamlit as st
import pandas as pd
from pathlib import Path
from rag_pipeline import generate_for_target

# ===============================
# PATHS
# ===============================
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
EMB = ROOT / "embeddings"

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="GenAI Drug Discovery", layout="wide")
st.title("🧬 GenAI Drug Discovery (RAG + SELFIES-LSTM)")

st.markdown("""
Generate **novel drug-like molecules** using:

- Retrieval-Augmented Generation (PubChem)
- Trained SELFIES-LSTM
- RDKit evaluation metrics
""")

# ===============================
# LOAD PUBCHEM SMILES
# ===============================
@st.cache_data
def load_pubchem_smiles():
    df = pd.read_parquet(EMB / "pubchem_metadata.parquet")
    return set(df["SMILES"].dropna().astype(str))

PUBCHEM_SMILES = load_pubchem_smiles()

# ===============================
# INPUTS
# ===============================
target = st.text_input("🎯 Target", value="EGFR inhibitor")

num_generate = st.slider(
    "🔢 Number of molecules to generate",
    min_value=5,
    max_value=100,
    value=20
)

# ===============================
# GENERATION
# ===============================
if st.button("🚀 Generate Molecules"):
    with st.spinner("Generating molecules..."):
        df = generate_for_target(target, num_generate)

    st.success("✅ Generation complete!")

    total = len(df)
    valid_df = df[df.valid]

    valid_pct = len(valid_df) / total * 100 if total else 0
    unique_pct = df.SMILES.nunique() / total * 100 if total else 0
    lipinski_pct = valid_df.lipinski.mean() * 100 if len(valid_df) else 0
    novelty_pct = (
        (~valid_df.SMILES.isin(PUBCHEM_SMILES)).mean() * 100
        if len(valid_df) else 0
    )
    avg_qed = valid_df.QED.mean() if len(valid_df) else 0

    # ===============================
    # METRICS
    # ===============================
    st.subheader("📊 Evaluation Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Validity (%)", f"{valid_pct:.2f}")
    c2.metric("Uniqueness (%)", f"{unique_pct:.2f}")
    c3.metric("Lipinski pass (%)", f"{lipinski_pct:.2f}")
    c4.metric("Novelty (%)", f"{novelty_pct:.2f}")
    c5.metric("Avg QED", f"{avg_qed:.3f}")

    # ===============================
    # TABLE
    # ===============================
    st.subheader("🧪 Generated Molecules")
    st.dataframe(df, use_container_width=True)

    # ===============================
    # DOWNLOAD
    # ===============================
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "generated_molecules.csv",
        "text/csv"
    )
