import ollama
from services.retriever import retrieve_similar_docs

def build_prompt(query, docs):
    context = "\n".join([f"- {d}" for d in docs])

    return f"""
You are a banking support analyst.

Customer Issue:
"{query}"

Historical Tickets:
{context}

Explain why this issue belongs to the same category.
Use only the provided tickets.
Keep it concise.
"""

def generate_explanation(query):
    docs = retrieve_similar_docs(query)

    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": build_prompt(query, docs)}]
    )

    explanation = response["message"]["content"].strip()
    return explanation, docs
