from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path
from idis_pipeline import run_pipeline  # Your pipeline file
import os
import json
import pickle
import faiss
from fastapi import FastAPI, Request
from datetime import datetime
from idis_pipeline import embedder, index, texts, labels  # these are already loaded in your pipeline

app = FastAPI(title="IDIS 2.0 API")

FEEDBACK_FILE = Path("feedback/feedback_store.jsonl")
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

class PredictRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    query: str
    predicted_category: str
    confidence: float
    trust_level: str
    user_feedback: str
    correct_category: Optional[str] = None

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = run_pipeline(req.text)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        # Append feedback as JSONL
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(req.dict()) + "\n")
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
FEEDBACK_FILE = "feedback/feedback_store.jsonl"
MODEL_DIR = "models/"

@app.post("/feedback")
async def feedback(request: Request):
    data = await request.json()
    data["timestamp"] = str(datetime.now())

    # Save feedback to JSONL
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

    # Self-healing: update embeddings if user corrected category
    if data.get("user_feedback") == "incorrect" and data.get("correct_category"):
        query_text = data["query"]
        correct_label = data["correct_category"]

        # Add to texts and labels
        texts.append(query_text)
        labels.append(correct_label)

        # Encode new embedding and add to FAISS
        new_emb = embedder.encode([query_text], normalize_embeddings=True).astype("float32")
        index.add(new_emb)

        # Save updated artifacts
        with open(os.path.join(MODEL_DIR, "texts.pkl"), "wb") as f:
            pickle.dump(texts, f)
        with open(os.path.join(MODEL_DIR, "labels.pkl"), "wb") as f:
            pickle.dump(labels, f)
        faiss.write_index(index, os.path.join(MODEL_DIR, "faiss.index"))

    return {"status": "success", "message": "Feedback saved"}
