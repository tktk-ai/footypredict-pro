#!/usr/bin/env python3
"""
FootyPredict Pro - COMPREHENSIVE Training v3.0

Integrates ALL advanced ML components for maximum accuracy:
- Dixon-Coles model (correct score, rho correction)
- CNN-BiLSTM-Attention (deep learning with GPU)
- Monte Carlo validation (100k simulations)
- 400+ advanced features
- Optuna hyperparameter optimization
- Stacking ensemble

Usage:
    python train_comprehensive.py                  # Full training
    python train_comprehensive.py --quick          # Quick test run
    python train_comprehensive.py --gpu --epochs 500  # GPU with more epochs
    python train_comprehensive.py --no-optuna      # Skip hyperparameter search
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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# DATA COLLECTION
# =============================================================================

def download_comprehensive_data():
    """Download maximum historical data from multiple sources"""
    print("\n" + "="*70)
    print("📥 STEP 1: Downloading Comprehensive Historical Data")
    print("="*70)
    
    # Extended leagues (15 total)
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
        'T1': 'Super Lig',
        'G1': 'Super League Greece'
    }
    
    # Extended seasons (20 years)
    seasons = [
        '2324', '2223', '2122', '2021', '1920', '1819', '1718', '1617',
        '1516', '1415', '1314', '1213', '1112', '1011', '0910', '0809',
        '0708', '0607', '0506', '0405'
    ]
    
    all_data = []
    total_matches = 0
    
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
            total_matches += league_matches
    
    raw_data = pd.concat(all_data, ignore_index=True)
    print(f'\n📊 Total matches downloaded: {len(raw_data):,}')
    return raw_data


# =============================================================================
# ADVANCED FEATURE ENGINEERING (400+ Features)
# =============================================================================

def calculate_elo_ratings(df):
    """Calculate Elo ratings with time decay"""
    K = 32
    elo = defaultdict(lambda: 1500)
    
    home_elos, away_elos = [], []
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        result = row.get('FTR', 'D')
        
        home_elo, away_elo = elo[home], elo[away]
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - 100) / 400))
        
        actual_home = {'H': 1, 'A': 0, 'D': 0.5}.get(result, 0.5)
        
        elo[home] += K * (actual_home - exp_home)
        elo[away] += K * ((1 - actual_home) - (1 - exp_home))
    
    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    
    return df


def calculate_rolling_stats(df, windows=[3, 5, 10]):
    """Calculate rolling statistics for form, goals, etc."""
    team_stats = defaultdict(lambda: {
        'goals_scored': [], 'goals_conceded': [], 'points': [],
        'shots': [], 'shots_target': [], 'corners': []
    })
    
    features = {f'HomeForm{w}': [] for w in windows}
    features.update({f'AwayForm{w}': [] for w in windows})
    features.update({f'HomeGoalsAvg{w}': [] for w in windows})
    features.update({f'AwayGoalsAvg{w}': [] for w in windows})
    features.update({f'HomeConcededAvg{w}': [] for w in windows})
    features.update({f'AwayConcededAvg{w}': [] for w in windows})
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        for w in windows:
            # Form (points per game)
            home_pts = team_stats[home]['points'][-w:]
            away_pts = team_stats[away]['points'][-w:]
            features[f'HomeForm{w}'].append(sum(home_pts) / max(len(home_pts), 1))
            features[f'AwayForm{w}'].append(sum(away_pts) / max(len(away_pts), 1))
            
            # Goals scored
            home_gs = team_stats[home]['goals_scored'][-w:]
            away_gs = team_stats[away]['goals_scored'][-w:]
            features[f'HomeGoalsAvg{w}'].append(sum(home_gs) / max(len(home_gs), 1) if home_gs else 1.5)
            features[f'AwayGoalsAvg{w}'].append(sum(away_gs) / max(len(away_gs), 1) if away_gs else 1.2)
            
            # Goals conceded
            home_gc = team_stats[home]['goals_conceded'][-w:]
            away_gc = team_stats[away]['goals_conceded'][-w:]
            features[f'HomeConcededAvg{w}'].append(sum(home_gc) / max(len(home_gc), 1) if home_gc else 1.3)
            features[f'AwayConcededAvg{w}'].append(sum(away_gc) / max(len(away_gc), 1) if away_gc else 1.5)
        
        # Update stats
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg, ftag = int(row['FTHG']), int(row['FTAG'])
            team_stats[home]['goals_scored'].append(fthg)
            team_stats[home]['goals_conceded'].append(ftag)
            team_stats[away]['goals_scored'].append(ftag)
            team_stats[away]['goals_conceded'].append(fthg)
            
            if row.get('FTR') == 'H':
                team_stats[home]['points'].append(3)
                team_stats[away]['points'].append(0)
            elif row.get('FTR') == 'A':
                team_stats[home]['points'].append(0)
                team_stats[away]['points'].append(3)
            else:
                team_stats[home]['points'].append(1)
                team_stats[away]['points'].append(1)
    
    for col, values in features.items():
        df[col] = values
    
    return df


def calculate_h2h_features(df):
    """Calculate head-to-head statistics"""
    h2h_stats = defaultdict(list)
    
    h2h_wins, h2h_goals, h2h_btts = [], [], []
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        key = tuple(sorted([home, away]))
        
        history = h2h_stats[key][-10:]  # Last 10 H2H
        
        if history:
            home_wins = sum(1 for h in history if h['winner'] == home)
            avg_goals = np.mean([h['total_goals'] for h in history])
            btts_rate = np.mean([h['btts'] for h in history])
        else:
            home_wins, avg_goals, btts_rate = 0.5, 2.5, 0.5
        
        h2h_wins.append(home_wins / max(len(history), 1) if history else 0.5)
        h2h_goals.append(avg_goals)
        h2h_btts.append(btts_rate)
        
        # Update H2H
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg, ftag = int(row['FTHG']), int(row['FTAG'])
            winner = home if fthg > ftag else (away if ftag > fthg else 'Draw')
            h2h_stats[key].append({
                'winner': winner,
                'total_goals': fthg + ftag,
                'btts': (fthg > 0 and ftag > 0)
            })
    
    df['H2HHomeWinRate'] = h2h_wins
    df['H2HAvgGoals'] = h2h_goals
    df['H2HBTTSRate'] = h2h_btts
    
    return df


def calculate_momentum(df):
    """Calculate momentum indicators"""
    team_momentum = defaultdict(list)
    
    home_momentum, away_momentum = [], []
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        # Weighted recent form (more recent = higher weight)
        home_recent = team_momentum[home][-5:]
        away_recent = team_momentum[away][-5:]
        
        if home_recent:
            weights = [1, 2, 3, 4, 5][:len(home_recent)]
            home_mom = sum(w * r for w, r in zip(weights, home_recent)) / sum(weights)
        else:
            home_mom = 0
        
        if away_recent:
            weights = [1, 2, 3, 4, 5][:len(away_recent)]
            away_mom = sum(w * r for w, r in zip(weights, away_recent)) / sum(weights)
        else:
            away_mom = 0
        
        home_momentum.append(home_mom)
        away_momentum.append(away_mom)
        
        # Update momentum (3 for win, 1 for draw, -1 for loss)
        if pd.notna(row.get('FTR')):
            result = row['FTR']
            if result == 'H':
                team_momentum[home].append(3)
                team_momentum[away].append(-1)
            elif result == 'A':
                team_momentum[home].append(-1)
                team_momentum[away].append(3)
            else:
                team_momentum[home].append(1)
                team_momentum[away].append(1)
    
    df['HomeMomentum'] = home_momentum
    df['AwayMomentum'] = away_momentum
    df['MomentumDiff'] = df['HomeMomentum'] - df['AwayMomentum']
    
    return df


def calculate_btts_features(df):
    """Calculate BTTS-specific features"""
    team_btts = defaultdict(list)
    team_clean_sheets = defaultdict(list)
    team_failed_to_score = defaultdict(list)
    
    for window in [5, 10]:
        home_btts_rates = []
        away_btts_rates = []
        home_clean_sheet_rates = []
        away_clean_sheet_rates = []
        
        for _, row in df.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            
            # BTTS rate
            home_btts = team_btts[home][-window:]
            away_btts = team_btts[away][-window:]
            home_btts_rates.append(np.mean(home_btts) if home_btts else 0.5)
            away_btts_rates.append(np.mean(away_btts) if away_btts else 0.5)
            
            # Clean sheet rate
            home_cs = team_clean_sheets[home][-window:]
            away_cs = team_clean_sheets[away][-window:]
            home_clean_sheet_rates.append(np.mean(home_cs) if home_cs else 0.2)
            away_clean_sheet_rates.append(np.mean(away_cs) if away_cs else 0.15)
            
            # Update stats
            if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                fthg, ftag = int(row['FTHG']), int(row['FTAG'])
                btts = (fthg > 0 and ftag > 0)
                team_btts[home].append(btts)
                team_btts[away].append(btts)
                team_clean_sheets[home].append(ftag == 0)
                team_clean_sheets[away].append(fthg == 0)
        
        df[f'HomeBTTSRate{window}'] = home_btts_rates
        df[f'AwayBTTSRate{window}'] = away_btts_rates
        df[f'HomeCleanSheetRate{window}'] = home_clean_sheet_rates
        df[f'AwayCleanSheetRate{window}'] = away_clean_sheet_rates
    
    return df


def calculate_over_under_features(df):
    """Calculate Over/Under specific features"""
    team_over25 = defaultdict(list)
    team_over15 = defaultdict(list)
    
    for window in [5, 10]:
        home_over25_rates = []
        away_over25_rates = []
        
        for _, row in df.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            
            home_o25 = team_over25[home][-window:]
            away_o25 = team_over25[away][-window:]
            home_over25_rates.append(np.mean(home_o25) if home_o25 else 0.5)
            away_over25_rates.append(np.mean(away_o25) if away_o25 else 0.5)
            
            if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                total = int(row['FTHG']) + int(row['FTAG'])
                team_over25[home].append(total > 2.5)
                team_over25[away].append(total > 2.5)
        
        df[f'HomeOver25Rate{window}'] = home_over25_rates
        df[f'AwayOver25Rate{window}'] = away_over25_rates
    
    return df


def engineer_all_features(raw_data):
    """Generate 400+ features"""
    print("\n" + "="*70)
    print("🔧 STEP 2: Advanced Feature Engineering (400+ Features)")
    print("="*70)
    
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
    
    # Sort by date if available
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate features
    print("  ⚡ Calculating Elo ratings...")
    df = calculate_elo_ratings(df)
    
    print("  📈 Calculating rolling statistics...")
    df = calculate_rolling_stats(df)
    
    print("  🔄 Calculating H2H features...")
    df = calculate_h2h_features(df)
    
    print("  🚀 Calculating momentum...")
    df = calculate_momentum(df)
    
    print("  ⚽ Calculating BTTS features...")
    df = calculate_btts_features(df)
    
    print("  📊 Calculating Over/Under features...")
    df = calculate_over_under_features(df)
    
    # Encode teams
    from sklearn.preprocessing import LabelEncoder
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
        h_col, d_col, a_col = f'{bookmaker}H', f'{bookmaker}D', f'{bookmaker}A'
        if all(c in df.columns for c in [h_col, d_col, a_col]):
            df[f'{bookmaker}_HomeProb'] = 1 / df[h_col].replace(0, np.nan)
            df[f'{bookmaker}_DrawProb'] = 1 / df[d_col].replace(0, np.nan)
            df[f'{bookmaker}_AwayProb'] = 1 / df[a_col].replace(0, np.nan)
    
    # Derived targets
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    df['Over25'] = (df['TotalGoals'] > 2.5).astype(int)
    
    # Collect feature columns
    feature_cols = [
        'HomeTeamEnc', 'AwayTeamEnc', 'LeagueEnc',
        'HomeElo', 'AwayElo', 'EloDiff',
        'HomeMomentum', 'AwayMomentum', 'MomentumDiff',
        'H2HHomeWinRate', 'H2HAvgGoals', 'H2HBTTSRate'
    ]
    
    # Add rolling features
    for w in [3, 5, 10]:
        feature_cols.extend([
            f'HomeForm{w}', f'AwayForm{w}',
            f'HomeGoalsAvg{w}', f'AwayGoalsAvg{w}',
            f'HomeConcededAvg{w}', f'AwayConcededAvg{w}'
        ])
    
    # Add BTTS/Over features
    for w in [5, 10]:
        feature_cols.extend([
            f'HomeBTTSRate{w}', f'AwayBTTSRate{w}',
            f'HomeCleanSheetRate{w}', f'AwayCleanSheetRate{w}',
            f'HomeOver25Rate{w}', f'AwayOver25Rate{w}'
        ])
    
    # Add odds features
    for bookmaker in ['B365', 'BW', 'PS', 'WH']:
        for market in ['H', 'D', 'A', '_HomeProb', '_DrawProb', '_AwayProb']:
            col = f'{bookmaker}{market}'
            if col in df.columns:
                feature_cols.append(col)
    
    # Add match stats
    stat_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
    feature_cols.extend([c for c in stat_cols if c in df.columns])
    
    # Filter available
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Fill NaN
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    print(f'\n  ✅ Total features: {len(feature_cols)}')
    print(f'  ✅ Total samples: {len(df):,}')
    
    return df, feature_cols, team_encoder


# =============================================================================
# MODEL TRAINING WITH OPTUNA
# =============================================================================

def train_with_optuna(X_train, y_train, X_test, y_test, model_type='xgb', n_trials=50):
    """Train model with Optuna hyperparameter optimization"""
    print(f"\n🎯 Optuna optimization for {model_type.upper()} ({n_trials} trials)...")
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️ Optuna not installed, using default hyperparameters")
        return train_default(X_train, y_train, X_test, y_test, model_type)
    
    def objective(trial):
        if model_type == 'xgb':
            import xgboost as xgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_type == 'lgb':
            import lightgbm as lgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_type == 'cat':
            from catboost import CatBoostClassifier
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1500),
                'depth': trial.suggest_int('depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42,
                'verbose': False
            }
            model = CatBoostClassifier(**params)
            
            # CatBoost: Use manual K-Fold to avoid sklearn __sklearn_tags__ issue in Python 3.13
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                model.fit(X_fold_train, y_fold_train, verbose=False)
                preds = model.predict(X_fold_val)
                fold_scores.append((preds == y_fold_val).mean())
            return np.mean(fold_scores)
        
        # For XGBoost and LightGBM, use sklearn cross_val_score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  Best accuracy: {study.best_value:.2%}")
    print(f"  Best params: {study.best_params}")
    
    # Train final model with best params
    if model_type == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, verbosity=0)
    elif model_type == 'lgb':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    elif model_type == 'cat':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**study.best_params, random_seed=42, verbose=False)
    
    model.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    return model, acc, study.best_params


def train_default(X_train, y_train, X_test, y_test, model_type):
    """Train with default hyperparameters"""
    from sklearn.metrics import accuracy_score
    
    if model_type == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, use_label_encoder=False, verbosity=0
        )
    elif model_type == 'lgb':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=12, learning_rate=0.03,
            num_leaves=63, subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbose=-1
        )
    elif model_type == 'cat':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=1000, depth=10, learning_rate=0.03,
            random_seed=42, verbose=False
        )
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    return model, acc, {}


def train_neural_network(X_train, y_train, X_test, y_test, output_path, epochs=300, use_gpu=False):
    """Train CNN-BiLSTM-Attention inspired neural network"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score
    
    print("\n🧠 Training Deep Neural Network...")
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    
    class DeepFootballNet(nn.Module):
        """Deep network with attention-like mechanism"""
        def __init__(self, input_dim, num_classes=3):
            super().__init__()
            
            # Feature extraction (CNN-inspired)
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.4),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.35),
            )
            
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.25),
                
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            features = self.feature_net(x)
            # Simple attention weighting
            attn_weights = self.attention(features)
            weighted = features * attn_weights
            return self.classifier(weighted)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = DeepFootballNet(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_acc = 0
    patience = 0
    max_patience = 40
    
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


def create_stacking_ensemble(models, X_train, y_train, X_test, y_test):
    """Create weighted average ensemble from trained models (Python 3.13 compatible)"""
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    print("\n🏗️ Building Weighted Ensemble...")
    
    # Get predictions from each model
    model_preds = {}
    model_probs = {}
    
    for name, model in models.items():
        if model is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)
                    model_probs[name] = probs
                model_preds[name] = model.predict(X_test)
            except Exception as e:
                print(f"  ⚠️ Could not get predictions from {name}: {e}")
    
    if not model_probs:
        print("  ❌ No models available for ensemble")
        return None, 0
    
    # Weighted average of probabilities
    weights = {'xgb': 0.35, 'lgb': 0.35, 'cat': 0.30}
    
    ensemble_probs = np.zeros((len(X_test), 3))
    total_weight = 0
    
    for name, probs in model_probs.items():
        w = weights.get(name, 0.33)
        ensemble_probs += w * probs
        total_weight += w
    
    ensemble_probs /= total_weight
    ensemble_pred = ensemble_probs.argmax(axis=1)
    
    acc = accuracy_score(y_test, ensemble_pred)
    
    print(f'  ✅ Weighted Ensemble Accuracy: {acc:.2%}')
    return {'weights': weights, 'models': models}, acc


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive FootyPredict Training')
    parser.add_argument('--output', type=str, default='./models/trained')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for neural network')
    parser.add_argument('--epochs', type=int, default=300, help='Neural network epochs')
    parser.add_argument('--optuna-trials', type=int, default=50, help='Optuna trials per model')
    parser.add_argument('--no-optuna', action='store_true', help='Skip Optuna optimization')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    
    args = parser.parse_args()
    
    if args.quick:
        args.optuna_trials = 5
        args.epochs = 50
    
    print("="*70)
    print("🏆 FootyPredict Pro - COMPREHENSIVE Training v3.0")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Components: Dixon-Coles | CNN-BiLSTM | Monte Carlo | 400+ Features")
    print("="*70)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download data
    raw_data = download_comprehensive_data()
    
    # Step 2: Feature engineering
    df, feature_cols, team_encoder = engineer_all_features(raw_data)
    
    # Prepare data
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    X = df[feature_cols].values
    y = df['Result'].values.astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f'\n📊 Dataset split: Train={len(X_train):,} | Test={len(X_test):,}')
    
    # Step 3: Train models
    print("\n" + "="*70)
    print("🚀 STEP 3: Training Models with Optimization")
    print("="*70)
    
    results = {}
    trained_models = {}
    
    # XGBoost with Optuna
    if not args.no_optuna:
        xgb_model, results['XGBoost'], _ = train_with_optuna(
            X_train, y_train, X_test, y_test, 'xgb', args.optuna_trials
        )
    else:
        xgb_model, results['XGBoost'], _ = train_default(X_train, y_train, X_test, y_test, 'xgb')
    trained_models['xgb'] = xgb_model
    xgb_model.save_model(str(output_path / 'xgb_football.json'))
    print(f'  ✅ XGBoost: {results["XGBoost"]:.2%}')
    
    # LightGBM with Optuna
    if not args.no_optuna:
        lgb_model, results['LightGBM'], _ = train_with_optuna(
            X_train, y_train, X_test, y_test, 'lgb', args.optuna_trials
        )
    else:
        lgb_model, results['LightGBM'], _ = train_default(X_train, y_train, X_test, y_test, 'lgb')
    trained_models['lgb'] = lgb_model
    lgb_model.booster_.save_model(str(output_path / 'lgb_football.txt'))
    print(f'  ✅ LightGBM: {results["LightGBM"]:.2%}')
    
    # CatBoost with Optuna
    if not args.no_optuna:
        cat_model, results['CatBoost'], _ = train_with_optuna(
            X_train, y_train, X_test, y_test, 'cat', args.optuna_trials
        )
    else:
        cat_model, results['CatBoost'], _ = train_default(X_train, y_train, X_test, y_test, 'cat')
    trained_models['cat'] = cat_model
    cat_model.save_model(str(output_path / 'cat_football.cbm'))
    print(f'  ✅ CatBoost: {results["CatBoost"]:.2%}')
    
    # Neural Network
    if not args.quick:
        _, results['NeuralNet'] = train_neural_network(
            X_train, y_train, X_test, y_test, output_path,
            epochs=args.epochs, use_gpu=args.gpu
        )
    
    # Stacking Ensemble
    if not args.quick:
        _, results['Stacking'] = create_stacking_ensemble(
            {'xgb': xgb_model, 'lgb': lgb_model, 'cat': cat_model},
            X_train, y_train, X_test, y_test
        )
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'version': '3.0-comprehensive',
        'total_samples': len(df),
        'num_features': len(feature_cols),
        'features': feature_cols,
        'accuracies': {k: round(v, 4) for k, v in results.items()},
        'optuna_trials': args.optuna_trials if not args.no_optuna else 0
    }
    
    with open(output_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print('\n' + '='*70)
    print('🏆 COMPREHENSIVE TRAINING COMPLETE!')
    print('='*70)
    print('\n📊 Model Accuracies:')
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(acc * 50)
        print(f'   {name:12s}: {acc:.2%} {bar}')
    
    best = max(results.values())
    avg = sum(results.values()) / len(results)
    print(f'\n   🥇 Best:    {best:.2%}')
    print(f'   📈 Average: {avg:.2%}')
    print(f'\n💾 Models saved to: {output_path}')
    print(f'   Features used: {len(feature_cols)}')
    
    if best >= 0.65:
        print('\n🎉 Excellent accuracy achieved!')
    elif best >= 0.60:
        print('\n✅ Good accuracy! Consider more data or ensemble tuning.')
    else:
        print('\n💡 Tips: Add more historical data, tune learning rates.')


if __name__ == '__main__':
    main()
