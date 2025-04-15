# train_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load CSV data
csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
df = pd.read_csv(csv_path)

# Split features and target
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Preprocessing steps
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and transformers
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("Training complete. Files saved: model.pkl, scaler.pkl, imputer.pkl")
