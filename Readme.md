# 🧠 IDIS 2.0 – Intelligent Document Intelligence System

IDIS is an **AI-powered customer support issue classification system** that combines **Machine Learning**, **Retrieval-Augmented Generation (RAG)**, and **self-healing feedback loops** to automatically analyze, explain, and prioritize customer support tickets.

This project simulates a **real-world enterprise banking support system** and is designed to be **production-oriented**, not just a demo.

---

## 🚀 Key Features

- 🔍 **Automatic Issue Classification**
  - TF-IDF + Logistic Regression classifier
  - Predicts issue category (e.g., `card_delivery_issue`)
  - Returns confidence score

- 🧠 **RAG-based Explanation (LLM + FAISS)**
  - Retrieves similar historical tickets using FAISS
  - Generates human-readable explanations using LLM (via Ollama)

- ♻️ **Self-Healing RAG Feedback Loop**
  - Stores model explanations and outcomes in JSONL
  - Enables future retraining and prompt refinement
  - Reduces hallucinations over time

- ⚠️ **Operational Decision Engine**
  - Confidence calibration
  - Human-review flag
  - SLA risk detection
  - Priority assignment

- 🌐 **Production-Style Architecture**
  - FastAPI backend
  - Streamlit frontend
  - Modular, scalable design

---

## 🏗️ System Architecture

User Input
↓
Streamlit UI
↓
FastAPI (/predict)
↓
ML Classifier (TF-IDF + Logistic Regression)
↓
FAISS Similarity Search
↓
LLM (RAG Explanation)
↓
Business Rules (Priority, SLA, Review)
↓
Final Structured Output + Feedback Storage

---

## 📂 Project Structure

IDIS2.0/
│
├── app.py # FastAPI backend
├── ui.py # Streamlit UI
├── idis_pipeline.py # Core ML + RAG pipeline
├── requirements.txt
├── README.md
│
├── models/
│ ├── classifier.pkl
│ ├── vectorizer.pkl
│ ├── faiss.index
│ └── tickets.pkl
│
├── feedback/
│ └── rag_feedback.jsonl


---

## 🧪 Example Output

```json
{
  "predicted_category": "card_delivery_issue",
  "confidence": 0.62,
  "calibrated_confidence": 0.59,
  "trust_level": "Medium",
  "priority": true,
  "sla_risk": true,
  "recommended_action": "Escalate to card operations team",
  "human_review_required": true,
  "explanation": "The common issue across the tickets is a delayed debit card delivery...",
  "similar_cases": [
    "I have not received my debit card after two weeks",
    "My debit card delivery is delayed",
    "Card has not arrived yet"
  ]
}
```
▶️ How to Run Locally
1️⃣ Install dependencies
```
pip install -r requirements.txt
```
2️⃣ Start FastAPI backend
```
uvicorn app:app --reload
```
API will run at:
```
http://127.0.0.1:8000
```
3️⃣ Start Streamlit UI
```
streamlit run ui.py
```
♻️ Self-Healing RAG (Feedback Loop)

Each prediction is logged in:
```
feedback/rag_feedback.jsonl
```
Example:
```
{
  "timestamp": "2026-01-09T07:03:22",
  "query": "I have not received my debit card after two weeks",
  "predicted_category": "card_delivery_issue",
  "confidence": 0.62,
  "explanation": "The issue relates to delayed debit card delivery..."
}
```
This data can later be used for:

Prompt refinement

Model retraining

Confidence recalibration

Hallucination reduction

🏦 Real-World Use Cases

Banking & FinTech customer support

Ticket routing and prioritization

SLA breach prediction

Human-agent workload reduction

Explainable AI for enterprise systems

🧠 Tech Stack

Python

scikit-learn

FAISS

SentenceTransformers

FastAPI

Streamlit

Ollama (LLM)

Git / GitHub

🎯 Why This Project Stands Out

✔ Not a toy project
✔ Shows ML + NLP + LLM integration
✔ Demonstrates production thinking
✔ Self-healing architecture
✔ Recruiter-friendly explanations

📌 Author

Arjun Patel
AI / ML Engineer
GitHub: https://github.com/Arjun-Patel1

⭐ If you like this project, give it a star!
This data can later be used for:

Prompt refinement

Model retraining

Confidence recalibration

Hallucination reduction

🏦 Real-World Use Cases

Banking & FinTech customer support

Ticket routing and prioritization

SLA breach prediction

Human-agent workload reduction

Explainable AI for enterprise systems

🧠 Tech Stack

Python

scikit-learn

FAISS

SentenceTransformers

FastAPI

Streamlit

Ollama (LLM)

Git / GitHub

🎯 Why This Project Stands Out

✔ Not a toy project
✔ Shows ML + NLP + LLM integration
✔ Demonstrates production thinking
✔ Self-healing architecture
✔ Recruiter-friendly explanations

📌 Author

Arjun Patel
AI / ML Engineer
linkedin: www.linkedin.com/in/arjunpatel97259
GitHub: https://github.com/Arjun-Patel1

⭐ If you like this project, give it a star!
