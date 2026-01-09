
'''
from fastapi import FastAPI
from idis_pipeline import run_pipeline

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    return run_pipeline(data["text"])
'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from idis_pipeline import run_pipeline

app = FastAPI(
    title="IDIS API",
    description="Intelligent Document Intelligence System – Backend",
    version="1.0"
)

# -----------------------------
# Request schema
# -----------------------------
class TicketRequest(BaseModel):
    text: str

# -----------------------------
# Health check (important for deployment)
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "IDIS API running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(request: TicketRequest):
    try:
        result = run_pipeline(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )