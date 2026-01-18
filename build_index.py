import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/tickets.csv"
MODEL_DIR = "models/"

df = pd.read_csv(DATA_PATH)

texts = df["text"].tolist()
labels = df["label"].tolist()

model = SentenceTransformer("intfloat/e5-small-v2")

embeddings = model.encode(
    ["passage: " + t for t in texts],
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, MODEL_DIR + "faiss.index")

with open(MODEL_DIR + "texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open(MODEL_DIR + "labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("✅ FAISS index + artifacts built successfully")
