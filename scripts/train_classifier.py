import pandas as pd
import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/tickets.csv")

embedder = SentenceTransformer("intfloat/e5-small-v2")

texts = ["query: " + t for t in df["text"].tolist()]
X = embedder.encode(texts, normalize_embeddings=True)
y = df["label"].astype("category").cat.codes

labels = dict(enumerate(df["label"].astype("category").cat.categories))

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

with open("models/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("models/labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Classifier trained")
