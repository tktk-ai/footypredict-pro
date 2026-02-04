#!/usr/bin/env python3
"""
FootyPredict Pro - Model Training Script (CLI)

Run this script directly to train all models without needing Colab.
Supports CPU and GPU training. Downloads data automatically.

Usage:
    python train_all_models.py                    # Train all models
    python train_all_models.py --gpu              # Force GPU training
    python train_all_models.py --epochs 200       # Custom epochs for NN
    python train_all_models.py --output ./models  # Custom output path

Requirements:
    pip install xgboost lightgbm catboost torch scikit-learn pandas numpy
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def download_football_data():
    """Download historical match data from Football-Data.co.uk"""
    print("\n📥 Downloading historical match data...")
    
    leagues = {
        'E0': 'Premier League',
        'D1': 'Bundesliga',
        'SP1': 'La Liga',
        'I1': 'Serie A',
        'F1': 'Ligue 1'
    }
    
    seasons = ['2324', '2223', '2122', '2021', '1920', '1819', '1718', '1617', '1516', '1415']
    
    all_data = []
    
    for league_code, league_name in leagues.items():
        for season in seasons:
            url = f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'
            try:
                df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
                df['League'] = league_name
                df['Season'] = season
                all_data.append(df)
                print(f'  ✓ {league_name} {season}: {len(df)} matches')
            except Exception as e:
                pass
    
    raw_data = pd.concat(all_data, ignore_index=True)
    print(f'\n📊 Total matches downloaded: {len(raw_data):,}')
    return raw_data


def prepare_features(raw_data):
    """Feature engineering for ML training"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    print("\n🔧 Preparing features...")
    
    columns_needed = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                      'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                      'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'League']
    
    available_cols = [c for c in columns_needed if c in raw_data.columns]
    df = raw_data[available_cols].copy()
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
    
    print(f'  Matches after cleaning: {len(df):,}')
    
    # Encode teams
    team_encoder = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_encoder.fit(all_teams)
    
    df['HomeTeamEnc'] = team_encoder.transform(df['HomeTeam'])
    df['AwayTeamEnc'] = team_encoder.transform(df['AwayTeam'])
    
    # Encode result
    result_map = {'H': 0, 'D': 1, 'A': 2}
    df['Result'] = df['FTR'].map(result_map)
    
    # Encode league
    league_encoder = LabelEncoder()
    df['LeagueEnc'] = league_encoder.fit_transform(df['League'])
    
    # Derived features
    df['GoalDiff'] = df['FTHG'] - df['FTAG']
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    df['Over25'] = (df['TotalGoals'] > 2.5).astype(int)
    
    # Fill numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Feature columns
    feature_cols = ['HomeTeamEnc', 'AwayTeamEnc', 'LeagueEnc']
    
    odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA']
    stat_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
    
    for col in odds_cols + stat_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    X = df[feature_cols].values
    y_result = df['Result'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f'  Features: {len(feature_cols)}')
    print(f'  Teams encoded: {len(all_teams)}')
    
    return X_scaled, y_result, feature_cols, scaler, team_encoder


def train_xgboost(X_train, y_train, X_test, y_test, output_path):
    """Train XGBoost classifier"""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    model.save_model(str(output_path / 'xgb_football.json'))
    print(f'  ✅ XGBoost Accuracy: {acc:.2%}')
    
    return acc


def train_lightgbm(X_train, y_train, X_test, y_test, output_path):
    """Train LightGBM classifier"""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)])
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    model.booster_.save_model(str(output_path / 'lgb_football.txt'))
    print(f'  ✅ LightGBM Accuracy: {acc:.2%}')
    
    return acc


def train_catboost(X_train, y_train, X_test, y_test, output_path):
    """Train CatBoost classifier"""
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function='MultiClass',
        random_seed=42,
        verbose=False
    )
    
    model.fit(X_train, y_train, 
              eval_set=(X_test, y_test))
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    model.save_model(str(output_path / 'cat_football.cbm'))
    print(f'  ✅ CatBoost Accuracy: {acc:.2%}')
    
    return acc


def train_neural_network(X_train, y_train, X_test, y_test, output_path, epochs=100, use_gpu=False):
    """Train PyTorch neural network"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training Neural Network...")
    
    # Device selection
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'  Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print(f'  Using CPU')
    
    # Define model
    class FootballPredictor(nn.Module):
        def __init__(self, input_dim, num_classes=3):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)
    
    # Prepare data
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = FootballPredictor(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_test_t).sum().item() / len(y_test_t)
            scheduler.step(1 - acc)
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), str(output_path / 'nn_football.pt'))
        
        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch+1}/{epochs} - Accuracy: {acc:.2%}')
    
    print(f'  ✅ Neural Network Best Accuracy: {best_acc:.2%}')
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='FootyPredict Pro - Model Training')
    parser.add_argument('--output', type=str, default='./models/trained',
                        help='Output directory for trained models')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for neural network training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for neural network')
    parser.add_argument('--skip-nn', action='store_true',
                        help='Skip neural network training')
    
    args = parser.parse_args()
    
    print('='*60)
    print('🏆 FootyPredict Pro - Model Training')
    print(f'   Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download data
    raw_data = download_football_data()
    
    # Prepare features
    X, y, feature_cols, scaler, team_encoder = prepare_features(raw_data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'\n📊 Training: {len(X_train):,} | Testing: {len(X_test):,}')
    
    # Train models
    results = {}
    
    results['xgboost'] = train_xgboost(X_train, y_train, X_test, y_test, output_path)
    results['lightgbm'] = train_lightgbm(X_train, y_train, X_test, y_test, output_path)
    results['catboost'] = train_catboost(X_train, y_train, X_test, y_test, output_path)
    
    if not args.skip_nn:
        results['neural_net'] = train_neural_network(
            X_train, y_train, X_test, y_test, 
            output_path, epochs=args.epochs, use_gpu=args.gpu
        )
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'total_samples': len(X),
        'features': feature_cols,
        'accuracies': {k: round(v, 4) for k, v in results.items()}
    }
    
    with open(output_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print('\n' + '='*60)
    print('🏆 TRAINING COMPLETE!')
    print('='*60)
    print('\n📊 Model Accuracies:')
    for name, acc in results.items():
        print(f'   {name.capitalize()}: {acc:.2%}')
    
    ensemble_avg = sum(results.values()) / len(results)
    print(f'\n   Ensemble Average: {ensemble_avg:.2%}')
    print(f'\n💾 Models saved to: {output_path}')
    print(f'   - xgb_football.json')
    print(f'   - lgb_football.txt')
    print(f'   - cat_football.cbm')
    if not args.skip_nn:
        print(f'   - nn_football.pt')
    print(f'   - training_metadata.json')
    print('\n✅ Done!')


if __name__ == '__main__':
    main()
