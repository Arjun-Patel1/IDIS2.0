'''import faiss
import pickle
import ollama
from sentence_transformers import SentenceTransformer

# ======================================================
# PATHS (DO NOT CHANGE)
# ======================================================
MODEL_DIR = "C:/Users/arjun/Downloads/IDIS2.0/models/"

# ======================================================
# LOAD ARTIFACTS
# ======================================================

# ---- CLASSIFIER ----
with open(MODEL_DIR + "classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open(MODEL_DIR + "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---- FAISS INDEX ----
index = faiss.read_index(MODEL_DIR + "faiss.index")

# ---- TICKETS (TEXT + LABEL TOGETHER) ----
# tickets = List[Tuple[text, label]]
with open(MODEL_DIR + "tickets.pkl", "rb") as f:
    tickets = pickle.load(f)

# ---- EMBEDDING MODEL ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ======================================================
# STEP 1: CATEGORY-AWARE RETRIEVAL
# ======================================================
def retrieve_similar_docs(query, predicted_category, k=10, final_k=3):
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)

    matched_docs = []

    for idx in indices[0]:
        text, label = tickets[idx]
        if label == predicted_category:
            matched_docs.append(text)

    # ---- FALLBACK IF FEW MATCHES ----
    if len(matched_docs) < final_k:
        for idx in indices[0]:
            text, _ = tickets[idx]
            if text not in matched_docs:
                matched_docs.append(text)
            if len(matched_docs) == final_k:
                break

    return matched_docs[:final_k]

# ======================================================
# STEP 2: PROMPT BUILDING (STRICT RAG)
# ======================================================
def build_prompt(query, retrieved_docs):
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)

    return f"""
You are an enterprise banking support analyst.

Customer Issue:
"{query}"

Historical Support Tickets:
{context}

Instructions:
- Identify the common issue across the tickets
- Explain why the customer issue matches them
- Use ONLY the ticket text as evidence
- Do NOT invent new categories or assumptions
- Keep the explanation concise (2–3 sentences)

Return ONLY the explanation text.
"""

# ======================================================
# STEP 3: RAG EXPLANATION (OLLAMA)
# ======================================================
def generate_rag_explanation(query, predicted_category):
    retrieved_docs = retrieve_similar_docs(query, predicted_category)
    prompt = build_prompt(query, retrieved_docs)

    try:
        response = ollama.chat(
            model="mistral:7b-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response["message"]["content"].strip()
    except Exception:
        explanation = (
            "The issue matches historical tickets involving similar customer complaints "
            "based on previous service requests."
        )

    return explanation, retrieved_docs

# ======================================================
# BUSINESS LOGIC
# ======================================================
def assign_priority():
    return {
        "priority": "High",
        "sla_risk": True,
        "recommended_action": "Escalate to card operations team"
    }

def calibrate_confidence(confidence):
    return {
        "calibrated_confidence": round(confidence * 0.95, 2),
        "trust_level": "Medium",
        "human_review_required": bool(confidence < 0.65)  # IMPORTANT: bool()
    }

# ======================================================
# OPTIONAL: MODEL DISAGREEMENT CHECK (ENTERPRISE FEATURE)
# ======================================================
def detect_model_disagreement(predicted_category, retrieved_docs):
    label_counts = {}

    for text, label in tickets:
        if text in retrieved_docs:
            label_counts[label] = label_counts.get(label, 0) + 1

    if not label_counts:
        return {
            "dominant_retrieved_label": predicted_category,
            "model_disagreement": False
        }

    dominant_label = max(label_counts, key=label_counts.get)

    return {
        "dominant_retrieved_label": dominant_label,
        "model_disagreement": dominant_label != predicted_category
    }

# ======================================================
# MAIN PIPELINE (FASTAPI CALLS THIS)
# ======================================================
def run_pipeline(query):
    # ---- CLASSIFICATION (DEFINED FIRST → NO NameError) ----
    X = vectorizer.transform([query])
    predicted_category = clf.predict(X)[0]
    confidence = float(max(clf.predict_proba(X)[0]))  # cast to Python float

    # ---- RAG ----
    explanation, similar_cases = generate_rag_explanation(
        query,
        predicted_category
    )

    # ---- DISAGREEMENT CHECK ----
    disagreement = detect_model_disagreement(
        predicted_category,
        similar_cases
    )

    # ---- FINAL OUTPUT ----
    output = {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "similar_cases": similar_cases
    }

    output.update(assign_priority())
    output.update(calibrate_confidence(confidence))
    output.update(disagreement)

    # Force human review if models disagree
    if disagreement["model_disagreement"]:
        output["human_review_required"] = True
        output["priority"] = "High"

    return output
'''
import faiss
import pickle
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

# ======================================================
# PATH
# ======================================================
MODEL_DIR = "C:/Users/arjun/Downloads/IDIS2.0/models/"

# ======================================================
# LOAD ARTIFACTS
# ======================================================

# ---- CLASSIFIER ----
with open(MODEL_DIR + "classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open(MODEL_DIR + "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---- FAISS INDEX ----
index = faiss.read_index(MODEL_DIR + "faiss.index")

# ---- TICKETS (TEXT + LABEL TOGETHER) ----
with open(MODEL_DIR + "tickets.pkl", "rb") as f:
    tickets = pickle.load(f)  # tickets: list of tuples (text, label)

# ---- EMBEDDER ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ======================================================
# STEP 1: CATEGORY-AWARE RETRIEVAL
# ======================================================
def retrieve_similar_docs(query, predicted_category, k=10, final_k=3):
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)

    matched_docs = []

    for idx in indices[0]:
        text, label = tickets[idx]
        if label == predicted_category:
            matched_docs.append(text)

    # Fallback if insufficient category matches
    if len(matched_docs) < final_k:
        for idx in indices[0]:
            text, _ = tickets[idx]
            if text not in matched_docs:
                matched_docs.append(text)
            if len(matched_docs) == final_k:
                break

    return matched_docs[:final_k]

# ======================================================
# STEP 2: PROMPT BUILDING
# ======================================================
def build_prompt(query, retrieved_docs):
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)

    return f"""
You are an enterprise banking support analyst.

Customer Issue:
"{query}"

Historical Support Tickets:
{context}

Instructions:
- Identify the common issue across the tickets
- Explain why the customer issue matches them
- Use ONLY the ticket text as evidence
- Do NOT invent new categories or assumptions
- Keep the explanation concise (2–3 sentences)

Return ONLY the explanation text.
"""

# ======================================================
# STEP 3: RAG EXPLANATION
# ======================================================
def generate_rag_explanation(query, category):
    retrieved_docs = retrieve_similar_docs(query, category)
    prompt = build_prompt(query, retrieved_docs)

    try:
        response = ollama.chat(
            model="mistral:7b-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = str(response["message"]["content"].strip())
    except Exception:
        explanation = (
            "The issue matches historical tickets involving similar customer complaints "
            "based on previous service requests."
        )

    return explanation, retrieved_docs

# ======================================================
# BUSINESS LOGIC
# ======================================================
def assign_priority():
    return {
        "priority": True,  # will cast later
        "sla_risk": True,
        "recommended_action": "Escalate to card operations team"
    }

def calibrate_confidence(confidence):
    return {
        "calibrated_confidence": float(round(confidence * 0.95, 2)),
        "trust_level": "Medium",
        "human_review_required": bool(confidence < 0.65)
    }

# ======================================================
# MAIN PIPELINE (FASTAPI CALLS THIS)
# ======================================================
def run_pipeline(query):
    # ---- CLASSIFICATION ----
    X = vectorizer.transform([query])
    predicted_category = str(clf.predict(X)[0])
    confidence = float(max(clf.predict_proba(X)[0]))

    # ---- RAG ----
    explanation, similar_cases = generate_rag_explanation(
        query,
        predicted_category
    )

    # ---- OUTPUT DICTIONARY ----
    output = {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "explanation": explanation,
        "similar_cases": list(similar_cases)  # ensure list of strings
    }

    # ---- PRIORITY AND CONFIDENCE ----
    priority_dict = assign_priority()
    confidence_dict = calibrate_confidence(confidence)

    # Cast any numpy.bool_ to native bool
    priority_dict = {k: (bool(v) if isinstance(v, (np.bool_,)) else v) for k, v in priority_dict.items()}
    confidence_dict = {k: (bool(v) if isinstance(v, (np.bool_,)) else v) for k, v in confidence_dict.items()}

    output.update(priority_dict)
    output.update(confidence_dict)

    return output
