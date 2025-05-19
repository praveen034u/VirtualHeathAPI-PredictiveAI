# DH_main.py (updated with error handling and debug logging)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np
import traceback

app = FastAPI()

# Load saved artifacts
with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
# Load all four models
with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)
with open("heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)
with open("sleep_apnea_model.pkl", "rb") as f:
    sleep_apnea_model = pickle.load(f)
with open("hypertension_model.pkl", "rb") as f:
    hypertension_model = pickle.load(f)

# Define input schema (adjust fields if needed)
class HealthData(BaseModel):
    fasting_glucose: float = 110.0
    cholesterol: float = 215.0
    hba1c: float = 6.2
    hdl: float = 45.0
    ldl: float = 130.0
    triglycerides: float = 180.0
    vitamin_d: float = 22.5
    vitamin_b12: float = 350.0
    iron: float = 75.0
    hemoglobin: float = 13.8
    rbc_count: float = 4.7
    wbc_count: float = 6200.0
    platelet_count: float = 240000.0
    tsh: float = 3.1
    t3: float = 1.3
    t4: float = 7.8
    family_history: int = 1
    smoking: int = 0
    
@app.post("/predict")
def predict(data: HealthData):
    try:
        # Convert input dataclass to 2D array
        features = [
            data.fasting_glucose, data.cholesterol, data.hba1c, data.hdl,
            data.ldl, data.triglycerides, data.vitamin_d, data.vitamin_b12,
            data.iron, data.hemoglobin, data.rbc_count, data.wbc_count,
            data.platelet_count, data.tsh, data.t3, data.t4,
            data.family_history, data.smoking
        ]
        input_array = np.array(features).reshape(1, -1)

        # Impute and scale
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)

        # Predict probabilities or values
        diabetes_prob = diabetes_model.predict(input_scaled)[0]
        heart_prob = heart_model.predict(input_scaled)[0]
        sleep_apnea_prob = sleep_apnea_model.predict(input_scaled)[0]
        hypertension_prob = hypertension_model.predict(input_scaled)[0]

        # Return as JSON
        return {
            "label_diabetes": float(np.round(diabetes_prob, 4)),
            "label_heart_disease": float(np.round(heart_prob, 4)),
            "label_sleep_apnea": float(np.round(sleep_apnea_prob, 4)),
            "label_hypertension": float(np.round(hypertension_prob, 4))
        }
    except Exception as e:
        # Log the full traceback for debugging
        tb = traceback.format_exc()
        # Return a structured error
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "trace": tb}
        )

# uvicorn main1:app --reload --host 127.0.0.1 --port 8000

# http://127.0.0.1:8000/docs

# {
#   "fasting_glucose": 110.0,
#   "cholesterol": 215.0,
#   "hba1c": 6.2,
#   "hdl": 45.0,
#   "ldl": 130.0,
#   "triglycerides": 180.0,
#   "vitamin_d": 22.5,
#   "vitamin_b12": 350.0,
#   "iron": 75.0,
#   "hemoglobin": 13.8,
#   "rbc_count": 4.7,
#   "wbc_count": 6200.0,
#   "platelet_count": 240000.0,
#   "tsh": 3.1,
#   "t3": 1.3,
#   "t4": 7.8,
#   "family_history": 1,
#   "smoking": 0
# }
