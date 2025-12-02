# src/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

# load artifacts once on startup
model = joblib.load(os.path.join(MODEL_DIR, "plant_health_model.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

app = FastAPI(
    title="Vertical Farming Plant Health API",
    version="0.1",
    description="Predict plant health (healthy/stressed) from sensor readings."
)

class SensorInput(BaseModel):
    temperature: float = Field(..., example=30.0)
    humidity: float = Field(..., example=40.0)
    light_hours: float = Field(..., example=16.0)
    ec: float = Field(..., example=2.4)
    pH: float = Field(..., example=7.5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: SensorInput):
    # convert to DataFrame so model sees correct feature names (no warning)
    df = pd.DataFrame([{
        "temperature": payload.temperature,
        "humidity": payload.humidity,
        "light_hours": payload.light_hours,
        "ec": payload.ec,
        "pH": payload.pH
    }])
    pred_encoded = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    # optionally return probabilities
    probs = model.predict_proba(df)[0].tolist()
    class_names = label_encoder.inverse_transform(range(len(probs)))
    return {
        "prediction": pred_label,
        "probabilities": {str(c): float(p) for c, p in zip(class_names, probs)},
        "input": payload.dict()
    }
