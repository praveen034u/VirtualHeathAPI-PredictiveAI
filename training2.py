# training2.py (final improved version with train/validation split, scaling, no output sigmoid)
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pickle

# Paths
daily_path = os.path.join(os.path.dirname(__file__), "daily_vitals_data.csv")
monthly_path = os.path.join(os.path.dirname(__file__), "monthly_clinical_data.csv")

# Load data
daily = pd.read_csv(daily_path)
monthly = pd.read_csv(monthly_path)

# Merge monthly predictions
monthly_preds = monthly[['user_id', 'label_diabetes', 'label_heart', 'label_sleep_apnea', 'label_hypertension']]
data = pd.merge(daily, monthly_preds, on='user_id', how='left')

# Define feature & target columns
daily_features = ['pulse','systolic_bp','diastolic_bp','spo2','body_temp','calories_burned','steps','stress_level','hrv']
input_cols = daily_features + ['label_diabetes','label_heart','label_sleep_apnea','label_hypertension']
target_cols = ['label_diabetes','label_heart','label_sleep_apnea','label_hypertension']

# Drop missing
data = data.dropna(subset=input_cols + target_cols)

# Scale inputs
scaler = StandardScaler()
X_all = data[input_cols].values.astype(np.float32)
X_scaled_all = scaler.fit_transform(X_all)
data[input_cols] = X_scaled_all

# Save scaler and input_cols for inference
with open('daily_scaler.pkl','wb') as f: pickle.dump(scaler, f)
with open('daily_input_cols.pkl','wb') as f: pickle.dump(input_cols, f)

# PyTorch Dataset
class VitalsDataset(Dataset):
    def __init__(self, df):
        self.X = df[input_cols].values
        self.y = df[target_cols].values
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Create train/validation split
dataset = VitalsDataset(data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)

# Model: no final sigmoid
class HealthPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4)  # raw outputs for regression
        )
    def forward(self, x): return self.net(x)

# Instantiate
model = HealthPredictor(input_dim=len(input_cols))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop with validation monitoring
best_val_loss = float('inf')
best_state = None
for epoch in range(1, 51):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            val_loss += criterion(preds, y_batch).item() * X_batch.size(0)
    val_loss /= val_size

    print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict().copy()

# Save best model weights
torch.save(best_state, 'daily_vitals_nn.pth')
