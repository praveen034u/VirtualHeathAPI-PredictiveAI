# main2.py (updated to apply saved scaler before NN)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle

app = FastAPI()

# Load input column list and scaler
with open('daily_input_cols.pkl', 'rb') as f:
    input_cols = pickle.load(f)
with open('daily_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define model architecture
class HealthPredictor(nn.Module):
    def __init__(self, input_dim):
        super(HealthPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Load trained weights
device = torch.device('cpu')
model = HealthPredictor(input_dim=len(input_cols))
model.load_state_dict(torch.load('daily_vitals_nn.pth', map_location=device))
model.eval()

# Pydantic schema
class DailyVitals(BaseModel):
    pulse: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    body_temp: float
    calories_burned: float
    steps: float
    stress_level: float
    hrv: float
    label_diabetes: float
    label_heart: float
    label_sleep_apnea: float
    label_hypertension: float

@app.post("/predict_daily")
async def predict_daily(data: DailyVitals):
    try:
        # Construct feature vector in correct order
        features = [
            data.pulse, data.systolic_bp, data.diastolic_bp, data.spo2,
            data.body_temp, data.calories_burned, data.steps,
            data.stress_level, data.hrv,
            data.label_diabetes, data.label_heart,
            data.label_sleep_apnea, data.label_hypertension
        ]
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        # Apply scaling
        X_scaled = scaler.transform(X)
        # Predict
        with torch.no_grad():
            out = model(torch.from_numpy(X_scaled))
        probs = out.numpy().flatten()
        return {
            "diabetes_probability": float(f"{probs[0]:.4f}"),
            "heart_probability": float(f"{probs[1]:.4f}"),
            "sleep_apnea_probability": float(f"{probs[2]:.4f}"),
            "hypertension_probability": float(f"{probs[3]:.4f}")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# uvicorn main1:app --reload --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/docs

# uvicorn main2:app --reload --host 127.0.0.1 --port 8001
# http://127.0.0.1:8001/docs
