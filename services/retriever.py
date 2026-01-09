import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index(
    "C:/Users/arjun/Downloads/IDIS2.0/models/faiss.index"
)

# Load stored ticket texts
with open(
    "C:/Users/arjun/Downloads/IDIS2.0/models/tickets.pkl", "rb"
) as f:
    tickets = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_similar_docs(query, k=3):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(tickets[idx])

    return results
