import pickle
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# PATHS
# =========================
MODEL_DIR = "models/"
FEEDBACK_FILE = "feedback_store.jsonl"

# =========================
# LOAD MODELS
# =========================
embedder = SentenceTransformer("intfloat/e5-small-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================
# LOAD ARTIFACTS
# =========================
index = faiss.read_index(MODEL_DIR + "faiss.index")

with open(MODEL_DIR + "texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open(MODEL_DIR + "labels.pkl", "rb") as f:
    labels = pickle.load(f)

# =========================
# RETRIEVAL
# =========================
def retrieve_similar(query_embedding, k=3):
    D, I = index.search(query_embedding, k)
    results = []

    for idx in I[0]:
        results.append({
            "text": texts[int(idx)],
            "label": labels[int(idx)]
        })

    return results

# =========================
# OLLAMA EXPLANATION
# =========================
def generate_explanation(query, retrieved, predicted_category):
    prompt = f"""
You are a banking support AI.

Customer issue:
"{query}"

Predicted category: {predicted_category}

Similar historical cases:
{json.dumps(retrieved, indent=2)}

Explain clearly why this issue belongs to this category.
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return res.json().get("response", "Explanation unavailable")
    except Exception as e:
        return f"LLM failed: {str(e)}"

# =========================
# TRUST LEVEL
# =========================
def trust_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.45:
        return "Medium"
    return "Low"

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(text: str):

    # E5 requires prefix
    emb = embedder.encode(
        ["query: " + text],
        normalize_embeddings=True
    ).astype("float32")

    retrieved = retrieve_similar(emb)

    # Cross-encoder reranking
    pairs = [[text, r["text"]] for r in retrieved]
    scores = cross_encoder.predict(pairs)

    best_idx = int(np.argmax(scores))
    predicted_category = retrieved[best_idx]["label"]

    confidence = float(np.max(scores))
    confidence = round(min(confidence, 1.0), 2)

    trust = trust_level(confidence)
    human_review_required = trust == "Low"

    explanation = generate_explanation(
        text, retrieved, predicted_category
    )

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "trust_level": trust,
        "human_review_required": human_review_required,
        "rag_explanation": explanation,
        "similar_cases": retrieved
    }

# =========================
# FEEDBACK STORE
# =========================
def store_feedback(payload: dict):
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
