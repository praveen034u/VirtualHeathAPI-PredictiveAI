from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import torch
import torch.nn as nn
import pickle
import traceback

app = FastAPI()

# --- Load models and preprocessors from main1.py ---
with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)
with open("heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)
with open("sleep_apnea_model.pkl", "rb") as f:
    sleep_apnea_model = pickle.load(f)
with open("hypertension_model.pkl", "rb") as f:
    hypertension_model = pickle.load(f)

# --- Load NN model and scaler from main2.py ---
with open('daily_input_cols.pkl', 'rb') as f:
    input_cols = pickle.load(f)
with open('daily_scaler.pkl', 'rb') as f:
    daily_scaler = pickle.load(f)

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

device = torch.device('cpu')
model = HealthPredictor(input_dim=len(input_cols))
model.load_state_dict(torch.load('daily_vitals_nn.pth', map_location=device))
model.eval()

# --- Input schema with defaults ---
class CombinedInput(BaseModel):
    # Health data (main1)
    fasting_glucose: float = Field(110.0)
    cholesterol: float = Field(215.0)
    hba1c: float = Field(6.2)
    hdl: float = Field(45.0)
    ldl: float = Field(130.0)
    triglycerides: float = Field(180.0)
    vitamin_d: float = Field(22.5)
    vitamin_b12: float = Field(350.0)
    iron: float = Field(75.0)
    hemoglobin: float = Field(13.8)
    rbc_count: float = Field(4.7)
    wbc_count: float = Field(6200.0)
    platelet_count: float = Field(240000.0)
    tsh: float = Field(3.1)
    t3: float = Field(1.3)
    t4: float = Field(7.8)
    family_history: int = Field(1)
    smoking: int = Field(0)
    # Daily vitals (main2)
    pulse: float = Field(78.0)
    systolic_bp: float = Field(120.0)
    diastolic_bp: float = Field(80.0)
    spo2: float = Field(98.0)
    body_temp: float = Field(98.6)
    calories_burned: float = Field(2200.0)
    steps: float = Field(8000.0)
    stress_level: float = Field(5.0)
    hrv: float = Field(55.0)

@app.post("/predict_combined")
def predict_combined(data: CombinedInput):
    try:
        # Step 1: Predict labels using main1 features
        health_features = [
            data.fasting_glucose, data.cholesterol, data.hba1c, data.hdl,
            data.ldl, data.triglycerides, data.vitamin_d, data.vitamin_b12,
            data.iron, data.hemoglobin, data.rbc_count, data.wbc_count,
            data.platelet_count, data.tsh, data.t3, data.t4,
            data.family_history, data.smoking
        ]
        health_array = np.array(health_features).reshape(1, -1)
        health_array = scaler.transform(imputer.transform(health_array))

        label_diabetes = diabetes_model.predict(health_array)[0]
        label_heart = heart_model.predict(health_array)[0]
        label_sleep_apnea = sleep_apnea_model.predict(health_array)[0]
        label_hypertension = hypertension_model.predict(health_array)[0]

        # Step 2: Predict final probabilities using daily vitals + labels
        daily_features = [
            data.pulse, data.systolic_bp, data.diastolic_bp, data.spo2,
            data.body_temp, data.calories_burned, data.steps,
            data.stress_level, data.hrv,
            label_diabetes, label_heart,
            label_sleep_apnea, label_hypertension
        ]
        X = np.array(daily_features, dtype=np.float32).reshape(1, -1)
        X_scaled = daily_scaler.transform(X)

        with torch.no_grad():
            probs = model(torch.from_numpy(X_scaled)).numpy().flatten()

        return {
            "diabetes_probability": float(f"{probs[0]:.4f}"),
            "heart_disease_probability": float(f"{probs[1]:.4f}"),
            "sleep_apnea_probability": float(f"{probs[2]:.4f}"),
            "hypertension_probability": float(f"{probs[3]:.4f}")
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# uvicorn main:app --reload --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/docs

# uvicorn main1:app --reload --host 127.0.0.1 --port 8001
# http://127.0.0.1:8001/docs

# uvicorn main2:app --reload --host 127.0.0.1 --port 8002
# http://127.0.0.1:8002/docs