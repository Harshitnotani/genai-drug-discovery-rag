# # app/rag_pipeline.py

# import faiss
# import pandas as pd
# import numpy as np
# import torch
# import random
# from pathlib import Path
# from rdkit import Chem
# from rdkit.Chem import QED, Descriptors

# # ===============================
# # PATH HANDLING (IMPORTANT)
# # ===============================
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# EMB = PROJECT_ROOT / "embeddings"
# DATA = PROJECT_ROOT / "data"

# # ===============================
# # LOAD FAISS + METADATA
# # ===============================
# FAISS_PATH = EMB / "pubchem_index.faiss"
# META_PATH = EMB / "pubchem_metadata.parquet"

# if not FAISS_PATH.exists():
#     raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")

# if not META_PATH.exists():
#     raise FileNotFoundError(f"Metadata not found: {META_PATH}")

# pubchem_index = faiss.read_index(str(FAISS_PATH))
# pubchem_meta = pd.read_parquet(META_PATH)

# # ===============================
# # SIMPLE RETRIEVAL
# # ===============================
# def retrieve_similar_smiles(k=5):
#     """Random retrieval fallback (FAISS embedding optional)"""
#     return pubchem_meta.sample(k)["SMILES"].astype(str).tolist()

# # ===============================
# # PLACEHOLDER GENERATOR
# # (Replace with LSTM / SELFIES / Transformer)
# # ===============================
# def generate_smiles(seed="CCO", max_len=60):
#     """
#     Dummy generator.
#     Replace this with:
#     - generate_lstm(...)
#     - generate_selfies(...)
#     - generate_transformer(...)
#     """
#     return seed + "C" * random.randint(3, 10)

# # ===============================
# # VALIDATION
# # ===============================
# def validate_smiles(smiles_list):
#     rows = []
#     for sm in smiles_list:
#         mol = Chem.MolFromSmiles(sm)
#         if mol:
#             rows.append({
#                 "SMILES": sm,
#                 "valid": True,
#                 "MolWt": Descriptors.MolWt(mol),
#                 "QED": QED.qed(mol),
#                 "LogP": Descriptors.MolLogP(mol)
#             })
#         else:
#             rows.append({
#                 "SMILES": sm,
#                 "valid": False,
#                 "MolWt": None,
#                 "QED": None,
#                 "LogP": None
#             })
#     return pd.DataFrame(rows)

# # ===============================
# # MAIN PIPELINE FUNCTION
# # ===============================
# def generate_for_target(
#     target: str,
#     num_generate: int = 20,
#     retrieve_k: int = 3
# ) -> pd.DataFrame:

#     retrieved = retrieve_similar_smiles(k=retrieve_k)

#     generated = [
#         generate_smiles(seed=random.choice(retrieved)[:5])
#         for _ in range(num_generate)
#     ]

#     df = validate_smiles(generated)
#     df["target_query"] = target
#     df["retrieved_examples"] = "|".join(retrieved)

#     return df




# # app/rag_pipeline.py

# import faiss
# import pandas as pd
# import numpy as np
# import torch
# import random
# from pathlib import Path
# from rdkit import Chem
# from rdkit.Chem import QED, Descriptors
# import selfies as sf

# # ===============================
# # PATH HANDLING
# # ===============================
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# EMB = PROJECT_ROOT / "embeddings"

# FAISS_PATH = EMB / "pubchem_index.faiss"
# META_PATH = EMB / "pubchem_metadata.parquet"

# # ===============================
# # LOAD FAISS + METADATA
# # ===============================
# if not FAISS_PATH.exists():
#     raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")

# if not META_PATH.exists():
#     raise FileNotFoundError(f"Metadata not found: {META_PATH}")

# pubchem_index = faiss.read_index(str(FAISS_PATH))
# pubchem_meta = pd.read_parquet(META_PATH)

# # ===============================
# # RETRIEVAL
# # ===============================
# def retrieve_similar_smiles(k=5):
#     """Random fallback retrieval (robust & fast)"""
#     return pubchem_meta.sample(k)["SMILES"].astype(str).tolist()

# # ===============================
# # SELFIES UTILITIES
# # ===============================
# def smiles_to_selfies(sm):
#     try:
#         return sf.encoder(sm)
#     except Exception:
#         return None

# def selfies_to_smiles(sf_str):
#     try:
#         return sf.decoder(sf_str)
#     except Exception:
#         return None

# # ===============================
# # SELFIES GENERATOR (100% VALID)
# # ===============================
# SELFIES_VOCAB = list(sf.get_semantic_robust_alphabet())
# MAX_LEN = 40

# def generate_selfies(seed_smiles="CCO"):
#     """
#     SELFIES random walk generator
#     ALWAYS decodes to valid SMILES
#     """
#     seed_sf = smiles_to_selfies(seed_smiles)
#     if seed_sf is None:
#         seed_sf = "[C][C][O]"

#     tokens = list(sf.split_selfies(seed_sf))

#     # random extension
#     for _ in range(random.randint(5, MAX_LEN)):
#         tokens.append(random.choice(SELFIES_VOCAB))

#     return "".join(tokens)

# # ===============================
# # VALIDATION + METRICS
# # ===============================
# def validate_smiles(smiles_list):
#     rows = []
#     for sm in smiles_list:
#         mol = Chem.MolFromSmiles(sm)
#         if mol:
#             rows.append({
#                 "SMILES": sm,
#                 "valid": True,
#                 "MolWt": Descriptors.MolWt(mol),
#                 "QED": QED.qed(mol),
#                 "LogP": Descriptors.MolLogP(mol),
#                 "lipinski": (
#                     Descriptors.MolWt(mol) <= 500
#                     and Descriptors.MolLogP(mol) <= 5
#                     and Descriptors.NumHDonors(mol) <= 5
#                     and Descriptors.NumHAcceptors(mol) <= 10
#                 )
#             })
#         else:
#             rows.append({
#                 "SMILES": sm,
#                 "valid": False,
#                 "MolWt": None,
#                 "QED": None,
#                 "LogP": None,
#                 "lipinski": False
#             })
#     return pd.DataFrame(rows)

# # ===============================
# # MAIN PIPELINE (USED BY UI)
# # ===============================
# def generate_for_target(
#     target: str,
#     num_generate: int = 20,
#     retrieve_k: int = 3
# ) -> pd.DataFrame:

#     retrieved = retrieve_similar_smiles(k=retrieve_k)

#     generated_smiles = []
#     for _ in range(num_generate):
#         seed = random.choice(retrieved)
#         sf_str = generate_selfies(seed_smiles=seed)
#         sm = selfies_to_smiles(sf_str)
#         if sm:
#             generated_smiles.append(sm)

#     df = validate_smiles(generated_smiles)

#     df["target_query"] = target
#     df["retrieved_examples"] = "|".join(retrieved)

#     return df







import faiss
import pandas as pd
import torch
import random
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf
import torch.nn as nn

# ===============================
# PATHS
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB = PROJECT_ROOT / "embeddings"
MODELS = PROJECT_ROOT / "models"

FAISS_PATH = EMB / "pubchem_index.faiss"
META_PATH = EMB / "pubchem_metadata.parquet"
MODEL_PATH = MODELS / "selfies_lstm.pt"

# ===============================
# LOAD DATA
# ===============================
pubchem_index = faiss.read_index(str(FAISS_PATH))
pubchem_meta = pd.read_parquet(META_PATH)

# ===============================
# RETRIEVAL
# ===============================
def retrieve_similar_smiles(k=20):
    return pubchem_meta.sample(k)["SMILES"].astype(str).tolist()

# ===============================
# MODEL DEFINITION (MATCH TRAINING)
# ===============================
class SelfiesLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# ===============================
# LOAD CHECKPOINT
# ===============================
ckpt = torch.load(MODEL_PATH, map_location="cpu")

token2idx = ckpt["token2idx"]
idx2token = ckpt["idx2token"]

model = SelfiesLSTM(vocab_size=len(token2idx), layers=3)
model.load_state_dict(ckpt["model"])
model.eval()

# ===============================
# GENERATION
# ===============================
def generate_selfies(seed_smiles, max_len=40, temperature=0.8):
    try:
        seed_sf = sf.encoder(seed_smiles)
    except Exception:
        seed_sf = "[C]"

    tokens = list(sf.split_selfies(seed_sf))
    idxs = [token2idx[t] for t in tokens if t in token2idx]

    x = torch.tensor([idxs], dtype=torch.long)
    hidden = None
    generated = tokens.copy()

    for _ in range(max_len):
        out, hidden = model(x, hidden)
        logits = out[0, -1] / temperature
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        tok = idx2token[idx]

        if tok == "[EOS]":
            break

        generated.append(tok)
        x = torch.tensor([[idx]], dtype=torch.long)

    return "".join(generated)

# ===============================
# VALIDATION
# ===============================
def validate_smiles(smiles_list):
    rows = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            rows.append({
                "SMILES": sm,
                "valid": True,
                "MolWt": Descriptors.MolWt(mol),
                "QED": QED.qed(mol),
                "LogP": Descriptors.MolLogP(mol),
                "lipinski": (
                    Descriptors.MolWt(mol) <= 500 and
                    Descriptors.MolLogP(mol) <= 5 and
                    Descriptors.NumHDonors(mol) <= 5 and
                    Descriptors.NumHAcceptors(mol) <= 10
                )
            })
        else:
            rows.append({
                "SMILES": sm,
                "valid": False,
                "MolWt": None,
                "QED": None,
                "LogP": None,
                "lipinski": False
            })
    return pd.DataFrame(rows)

# ===============================
# MAIN PIPELINE
# ===============================
def generate_for_target(target, num_generate=20, retrieve_k=20):
    retrieved = retrieve_similar_smiles(k=retrieve_k)

    generated = []
    for _ in range(num_generate):
        seed = random.choice(retrieved)
        sf_str = generate_selfies(seed)
        sm = sf.decoder(sf_str)
        if sm:
            generated.append(sm)

    df = validate_smiles(generated)
    df["target_query"] = target
    df["retrieved_examples"] = "|".join(retrieved)
    return df
