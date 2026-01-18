# 🧠 IDIS 2.0 – Intelligent Document Intelligence System

IDIS 2.0 is an end-to-end AI-powered support ticket classification and explanation system
designed for real-world enterprise workflows.

It combines **semantic search, re-ranking, RAG, and human feedback loops**
to deliver reliable and explainable predictions.

---

## 🚀 Features

- 🔍 **Semantic Retrieval** using E5-small-v2 embeddings
- ⚡ **FAISS Vector Search** for fast similarity lookup
- 🎯 **Cross-Encoder Re-ranking** for high-precision classification
- 🧠 **RAG Explanations** using Ollama (Mistral)
- 🔁 **Self-Healing Loop** via human feedback
- 👤 **Human Review Detection**
- 🌐 **FastAPI Backend**
- 🖥️ **Streamlit Interactive UI**
- 🐳 **Dockerized Deployment**

---

## 🏗️ Architecture

User Query  
→ Sentence Embedding (E5)  
→ FAISS Retrieval  
→ Cross-Encoder Re-ranking  
→ Category Prediction  
→ RAG Explanation (Mistral)  
→ Confidence & Trust Estimation  
→ Human Feedback Storage  

---

## 🧪 Tech Stack

| Layer | Tools |
|-----|------|
| Embeddings | sentence-transformers (E5-small-v2) |
| Re-ranking | Cross-Encoder (MS MARCO MiniLM) |
| Vector DB | FAISS |
| LLM | Ollama (Mistral) |
| Backend | FastAPI |
| UI | Streamlit |
| Deployment | Docker |

---

## ▶️ Run with Docker

```bash
docker build -t idis_project .
docker run -p 8000:8000 -p 8501:8501 idis_project
```
FastAPI → http://localhost:8000/docs

Streamlit UI → http://localhost:8501

📊 Example Output
```
{
  "predicted_category": "atm_issue",
  "confidence": 1.0,
  "trust_level": "High",
  "human_review_required": false,
  "rag_explanation": "...",
  "similar_cases": [...]
}
```
🔁 Feedback Loop

User feedback is stored in:

feedback/feedback_store.jsonl


This data can be used for:

Re-training embeddings

Improving FAISS index

Model evaluation

