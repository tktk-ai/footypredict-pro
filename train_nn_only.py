#!/usr/bin/env python3
"""
Neural Network Only Training
Resumes training for just the NN component
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json

# Import feature engineering from train_comprehensive
from train_comprehensive import download_comprehensive_data, engineer_all_features

print("="*70)
print("🧠 FootyPredict Pro - Neural Network Training")
print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

OUTPUT_PATH = Path('./models/trained')
EPOCHS = 300

# Step 1: Load data
print("\n📥 Loading data...")
raw_data = download_comprehensive_data()
df, feature_cols, team_encoder = engineer_all_features(raw_data)

# Prepare features (feature_cols already provided)
X = df[feature_cols].values
y = df['Result'].values  # Result is already encoded in engineer_all_features

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
with open(OUTPUT_PATH / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Features: {X_train.shape[1]}")

# Neural Network
class DeepFootballNet(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# Train
print("\n🧠 Training Deep Neural Network...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = DeepFootballNet(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

best_acc = 0
patience = 0
max_patience = 30

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        acc = (preds == y_test_t).float().mean().item()
    
    scheduler.step(acc)
    
    if acc > best_acc:
        best_acc = acc
        patience = 0
        torch.save(model.state_dict(), str(OUTPUT_PATH / 'nn_football.pt'))
    else:
        patience += 1
    
    if patience >= max_patience:
        print(f'  Early stopping at epoch {epoch+1}')
        break
    
    if (epoch + 1) % 25 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} - Accuracy: {acc:.2%} (best: {best_acc:.2%})')

print(f'\n✅ Neural Network Best Accuracy: {best_acc:.2%}')

# Save results
results = {
    'nn_accuracy': best_acc,
    'features': X_train.shape[1],
    'trained_at': datetime.now().isoformat()
}
with open(OUTPUT_PATH / 'nn_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("✅ Neural Network Training Complete!")
print(f"   Best Accuracy: {best_acc:.2%}")
print(f"   Model saved to: {OUTPUT_PATH / 'nn_football.pt'}")
print("="*70)
