# Updated DH_train_model.py: using regressors for continuous targets
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Load the monthly clinical data
csv_path = os.path.join(os.path.dirname(__file__), "monthly_clinical_data.csv")
data = pd.read_csv(csv_path)

# Select features
features = [
    'fasting_glucose', 'cholesterol', 'hba1c', 'hdl', 'ldl',
    'triglycerides', 'vitamin_d', 'vitamin_b12', 'iron', 'hemoglobin',
    'rbc_count', 'wbc_count', 'platelet_count', 'tsh', 't3', 't4',
    'family_history', 'smoking'
]
X = data[features]

# One-hot encode categorical/text columns
X = pd.get_dummies(X, drop_first=True)

# Impute missing values with column means
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Define continuous targets for regression
targets = {
    "diabetes_model.pkl": data['label_diabetes'],
    "heart_model.pkl": data['label_heart'],
    "sleep_apnea_model.pkl": data['label_sleep_apnea'],
    "hypertension_model.pkl": data['label_hypertension']
}

# Train a regressor for each target and save
for model_file, y in targets.items():
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    with open(model_file, "wb") as f:
        pickle.dump(reg, f)

# Save preprocessing artifacts
with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
