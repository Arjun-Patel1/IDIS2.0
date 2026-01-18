# Use Python 3.11
FROM python:3.11

WORKDIR /app

COPY . .

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential git libglib2.0-0 libsm6 libxrender1 libxext6 libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies without GitHub clone
RUN pip install --no-cache-dir fastapi uvicorn streamlit requests \
    sentence-transformers[dev] faiss-cpu

# Expose ports
EXPOSE 8000 8501

# Run FastAPI + Streamlit together
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]
