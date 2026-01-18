import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv("data/tickets.csv")

embedder = SentenceTransformer("intfloat/e5-small-v2")

texts = ["query: " + t for t in df["text"].tolist()]
X = embedder.encode(texts, normalize_embeddings=True)

index = faiss.IndexFlatIP(X.shape[1])
index.add(np.array(X))

faiss.write_index(index, "models/faiss.index")

print("✅ FAISS index built")
