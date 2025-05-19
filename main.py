# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load saved model and transformers
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

app = FastAPI()

# Define input schema
class HealthInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict_diabetes(data: HealthInput):
    input_array = np.array([[ 
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

@app.get("/health")
def health_check():
    return {"status": "ok"}

    # Transform input
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "diabetes_prediction": int(prediction),
        "probability": round(probability, 4)
    }
