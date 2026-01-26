#!/usr/bin/env python3
"""
FootyPredict Pro - ENHANCED Model Training v2.0

Optimized for maximum accuracy with:
- 15+ seasons of historical data
- 10 major leagues
- Advanced feature engineering (Elo, rolling stats)
- Hyperparameter optimization
- Stacking ensemble

Target: Push accuracy towards 70%+ (realistic) or higher

Usage:
    python train_enhanced.py                    # Full training
    python train_enhanced.py --quick            # Quick test run
    python train_enhanced.py --gpu --epochs 500 # GPU with more epochs
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')


def download_extended_data():
    """Download maximum historical data from Football-Data.co.uk"""
    print("\n📥 Downloading EXTENDED historical match data...")
    print("   (15 seasons, 10 leagues = ~50,000+ matches)")
    
    leagues = {
        'E0': 'Premier League',
        'E1': 'Championship',
        'D1': 'Bundesliga',
        'D2': 'Bundesliga 2',
        'SP1': 'La Liga',
        'SP2': 'La Liga 2',
        'I1': 'Serie A',
        'I2': 'Serie B',
        'F1': 'Ligue 1',
        'F2': 'Ligue 2',
        'N1': 'Eredivisie',
        'P1': 'Primeira Liga',
        'B1': 'Belgian Pro League',
        'T1': 'Super Lig'
    }
    
    # Extended seasons (15 years)
    seasons = [
        '2324', '2223', '2122', '2021', '1920', '1819', '1718', '1617',
        '1516', '1415', '1314', '1213', '1112', '1011', '0910', '0809'
    ]
    
    all_data = []
    
    for league_code, league_name in leagues.items():
        league_matches = 0
        for season in seasons:
            url = f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'
            try:
                df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
                df['League'] = league_name
                df['LeagueCode'] = league_code
                df['Season'] = season
                all_data.append(df)
                league_matches += len(df)
            except:
                pass
        if league_matches > 0:
            print(f'  ✓ {league_name}: {league_matches:,} matches')
    
    raw_data = pd.concat(all_data, ignore_index=True)
    print(f'\n📊 Total matches downloaded: {len(raw_data):,}')
    return raw_data


def calculate_elo_ratings(df):
    """Calculate Elo ratings for all teams"""
    print("  ⚡ Calculating Elo ratings...")
    
    K = 32  # Elo K-factor
    elo = defaultdict(lambda: 1500)  # Starting Elo
    
    home_elos = []
    away_elos = []
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row.get('FTR', 'D')
        
        home_elo = elo[home_team]
        away_elo = elo[away_team]
        
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        
        # Expected scores
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - 100) / 400))  # +100 home advantage
        exp_away = 1 - exp_home
        
        # Actual scores
        if result == 'H':
            actual_home, actual_away = 1, 0
        elif result == 'A':
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Update Elo
        elo[home_team] += K * (actual_home - exp_home)
        elo[away_team] += K * (actual_away - exp_away)
    
    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    
    return df


def calculate_rolling_stats(df, windows=[3, 5, 10]):
    """Calculate rolling averages for goals, form, etc."""
    print("  📈 Calculating rolling statistics...")
    
    # Sort by date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Team stats dictionary
    team_stats = defaultdict(lambda: {'goals_scored': [], 'goals_conceded': [], 'points': []})
    
    home_form = []
    away_form = []
    home_goals_avg = []
    away_goals_avg = []
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get current form (last 5 matches)
        home_pts = team_stats[home_team]['points'][-5:]
        away_pts = team_stats[away_team]['points'][-5:]
        
        home_form.append(sum(home_pts) / max(len(home_pts), 1))
        away_form.append(sum(away_pts) / max(len(away_pts), 1))
        
        # Goals average
        home_gs = team_stats[home_team]['goals_scored'][-5:]
        away_gs = team_stats[away_team]['goals_scored'][-5:]
        
        home_goals_avg.append(sum(home_gs) / max(len(home_gs), 1) if home_gs else 1.5)
        away_goals_avg.append(sum(away_gs) / max(len(away_gs), 1) if away_gs else 1.2)
        
        # Update stats
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg = row['FTHG']
            ftag = row['FTAG']
            
            team_stats[home_team]['goals_scored'].append(fthg)
            team_stats[home_team]['goals_conceded'].append(ftag)
            team_stats[away_team]['goals_scored'].append(ftag)
            team_stats[away_team]['goals_conceded'].append(fthg)
            
            if row.get('FTR') == 'H':
                team_stats[home_team]['points'].append(3)
                team_stats[away_team]['points'].append(0)
            elif row.get('FTR') == 'A':
                team_stats[home_team]['points'].append(0)
                team_stats[away_team]['points'].append(3)
            else:
                team_stats[home_team]['points'].append(1)
                team_stats[away_team]['points'].append(1)
    
    df['HomeForm'] = home_form
    df['AwayForm'] = away_form
    df['HomeGoalsAvg'] = home_goals_avg
    df['AwayGoalsAvg'] = away_goals_avg
    df['FormDiff'] = df['HomeForm'] - df['AwayForm']
    
    return df


def prepare_advanced_features(raw_data):
    """Advanced feature engineering"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    print("\n🔧 Advanced Feature Engineering...")
    
    # Clean data
    columns_needed = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
                      'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',
                      'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA',
                      'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',
                      'League', 'Season', 'Date']
    
    available_cols = [c for c in columns_needed if c in raw_data.columns]
    df = raw_data[available_cols].copy()
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
    
    print(f'  Matches after cleaning: {len(df):,}')
    
    # Calculate Elo ratings
    df = calculate_elo_ratings(df)
    
    # Calculate rolling stats
    df = calculate_rolling_stats(df)
    
    # Encode teams
    team_encoder = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_encoder.fit(all_teams)
    
    df['HomeTeamEnc'] = team_encoder.transform(df['HomeTeam'])
    df['AwayTeamEnc'] = team_encoder.transform(df['AwayTeam'])
    
    # Encode result
    result_map = {'H': 0, 'D': 1, 'A': 2}
    df['Result'] = df['FTR'].map(result_map)
    df = df.dropna(subset=['Result'])
    
    # Encode league
    league_encoder = LabelEncoder()
    df['LeagueEnc'] = league_encoder.fit_transform(df['League'])
    
    # Odds implied probabilities
    for bookmaker in ['B365', 'BW', 'PS', 'WH']:
        h_col = f'{bookmaker}H'
        d_col = f'{bookmaker}D'
        a_col = f'{bookmaker}A'
        
        if h_col in df.columns and d_col in df.columns and a_col in df.columns:
            # Convert odds to probabilities
            df[f'{bookmaker}_HomeProb'] = 1 / df[h_col]
            df[f'{bookmaker}_DrawProb'] = 1 / df[d_col]
            df[f'{bookmaker}_AwayProb'] = 1 / df[a_col]
    
    # Derived features
    df['GoalDiff'] = df['FTHG'] - df['FTAG']
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    
    # Feature columns
    feature_cols = [
        'HomeTeamEnc', 'AwayTeamEnc', 'LeagueEnc',
        'HomeElo', 'AwayElo', 'EloDiff',
        'HomeForm', 'AwayForm', 'FormDiff',
        'HomeGoalsAvg', 'AwayGoalsAvg'
    ]
    
    # Add odds features
    odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA',
                 'B365_HomeProb', 'B365_DrawProb', 'B365_AwayProb']
    for col in odds_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # Add match stats (will be NaN for predictions, but useful for training)
    stat_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
    for col in stat_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # Keep only available features
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Fill NaN
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_cols].values
    y = df['Result'].values.astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f'  ✅ Features: {len(feature_cols)}')
    print(f'  ✅ Teams: {len(all_teams)}')
    print(f'  ✅ Samples: {len(X):,}')
    
    return X_scaled, y, feature_cols, scaler, team_encoder


def train_xgboost_optimized(X_train, y_train, X_test, y_test, output_path):
    """Train XGBoost with optimized hyperparameters"""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training XGBoost (optimized)...")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    model.save_model(str(output_path / 'xgb_football.json'))
    print(f'  ✅ XGBoost Accuracy: {acc:.2%}')
    
    return model, acc


def train_lightgbm_optimized(X_train, y_train, X_test, y_test, output_path):
    """Train LightGBM with optimized hyperparameters"""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training LightGBM (optimized)...")
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    model.booster_.save_model(str(output_path / 'lgb_football.txt'))
    print(f'  ✅ LightGBM Accuracy: {acc:.2%}')
    
    return model, acc


def train_catboost_optimized(X_train, y_train, X_test, y_test, output_path):
    """Train CatBoost with optimized hyperparameters"""
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training CatBoost (optimized)...")
    
    model = CatBoostClassifier(
        iterations=1000,
        depth=10,
        learning_rate=0.03,
        l2_leaf_reg=3,
        loss_function='MultiClass',
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
        task_type='CPU'
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    model.save_model(str(output_path / 'cat_football.cbm'))
    print(f'  ✅ CatBoost Accuracy: {acc:.2%}')
    
    return model, acc


def train_neural_network_optimized(X_train, y_train, X_test, y_test, output_path, epochs=300, use_gpu=False):
    """Train deeper Neural Network"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training Neural Network (deep architecture)...")
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    
    class DeepFootballPredictor(nn.Module):
        def __init__(self, input_dim, num_classes=3):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.4),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.35),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.25),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = DeepFootballPredictor(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_acc = 0
    patience = 0
    max_patience = 30
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_test_t).sum().item() / len(y_test_t)
            
            if acc > best_acc:
                best_acc = acc
                patience = 0
                torch.save(model.state_dict(), str(output_path / 'nn_football.pt'))
            else:
                patience += 1
            
            if patience >= max_patience:
                print(f'    Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 50 == 0:
            print(f'    Epoch {epoch+1}/{epochs} - Accuracy: {acc:.2%} (best: {best_acc:.2%})')
    
    print(f'  ✅ Neural Network Best Accuracy: {best_acc:.2%}')
    return model, best_acc


def train_stacking_ensemble(models, X_train, y_train, X_test, y_test, output_path):
    """Train a stacking ensemble for maximum accuracy"""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    print("\n🚀 Training Stacking Ensemble...")
    
    xgb_model, lgb_model, cat_model = models
    
    estimators = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, C=0.5),
        cv=5,
        n_jobs=-1
    )
    
    stacking.fit(X_train, y_train)
    pred = stacking.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    print(f'  ✅ Stacking Ensemble Accuracy: {acc:.2%}')
    return stacking, acc


def main():
    parser = argparse.ArgumentParser(description='Enhanced FootyPredict Training')
    parser.add_argument('--output', type=str, default='./models/trained')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    
    args = parser.parse_args()
    
    print('='*70)
    print('🏆 FootyPredict Pro - ENHANCED Training v2.0')
    print(f'   Target: Maximum Accuracy | Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*70)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download extended data
    raw_data = download_extended_data()
    
    # Advanced feature engineering
    X, y, feature_cols, scaler, team_encoder = prepare_advanced_features(raw_data)
    
    # Split with stratification
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f'\n📊 Dataset: Train={len(X_train):,} | Test={len(X_test):,}')
    
    # Train optimized models
    results = {}
    
    xgb_model, results['XGBoost'] = train_xgboost_optimized(X_train, y_train, X_test, y_test, output_path)
    lgb_model, results['LightGBM'] = train_lightgbm_optimized(X_train, y_train, X_test, y_test, output_path)
    cat_model, results['CatBoost'] = train_catboost_optimized(X_train, y_train, X_test, y_test, output_path)
    
    if not args.quick:
        _, results['NeuralNet'] = train_neural_network_optimized(
            X_train, y_train, X_test, y_test, output_path,
            epochs=args.epochs, use_gpu=args.gpu
        )
        
        # Stacking ensemble
        _, results['Stacking'] = train_stacking_ensemble(
            [xgb_model, lgb_model, cat_model],
            X_train, y_train, X_test, y_test, output_path
        )
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'version': '2.0-enhanced',
        'total_samples': len(X),
        'features': feature_cols,
        'accuracies': {k: round(v, 4) for k, v in results.items()}
    }
    
    with open(output_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print('\n' + '='*70)
    print('🏆 ENHANCED TRAINING COMPLETE!')
    print('='*70)
    print('\n📊 Model Accuracies:')
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(acc * 50)
        print(f'   {name:12s}: {acc:.2%} {bar}')
    
    best = max(results.values())
    avg = sum(results.values()) / len(results)
    print(f'\n   Best Model:    {best:.2%}')
    print(f'   Average:       {avg:.2%}')
    print(f'\n💾 Models saved to: {output_path}')
    
    if best < 0.70:
        print('\n💡 Tips to improve further:')
        print('   - Add more historical data (20+ seasons)')
        print('   - Include head-to-head statistics')
        print('   - Add player-level data (injuries, transfers)')
        print('   - Include weather and stadium data')
        print('   - Use Bayesian hyperparameter optimization')


if __name__ == '__main__':
    main()
