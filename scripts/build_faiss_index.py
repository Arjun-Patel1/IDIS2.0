import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ======================================================
# PATHS (DO NOT CHANGE AS REQUESTED)
# ======================================================
DATA_PATH = "C:/Users/arjun/Downloads/IDIS2.0/data/tickets.csv"
MODEL_DIR = "C:/Users/arjun/Downloads/IDIS2.0/models/"

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

# ======================================================
# EMBEDDINGS
# ======================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

# ======================================================
# FAISS INDEX
# ======================================================
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, MODEL_DIR + "faiss.index")

# ======================================================
# SAVE ARTIFACTS (SEPARATE & CLEAN)
# ======================================================
with open(MODEL_DIR + "tickets.pkl", "wb") as f:
    pickle.dump(texts, f)

with open(MODEL_DIR + "labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("✅ FAISS index built correctly")
print("✅ tickets.pkl (text only) saved")
print("✅ labels.pkl saved")
