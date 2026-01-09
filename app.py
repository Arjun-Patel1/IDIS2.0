'''from fastapi import FastAPI
from idis_pipeline import run_pipeline

app = FastAPI(title="IDIS API")

@app.post("/predict")
def predict(payload: dict):
    return run_pipeline(payload["text"])
'''
from fastapi import FastAPI
from idis_pipeline import run_pipeline

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    return run_pipeline(data["text"])
