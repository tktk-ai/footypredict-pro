"""
================================================================================
QUANTUM-ENHANCED FOOTBALL PREDICTION SYSTEM v3.0 - MAXIMUM EDITION
================================================================================

Optimized for Kaggle Dataset: football-match-prediction-features
170 Betting Odds Features → Goal Prediction

Components:
1. Advanced Odds Processing & Feature Engineering
2. Quantum Neural Network (CORE) - Enhanced
3. CatBoost + Pi-Ratings (SOTA)
4. LightGBM + XGBoost Ensemble
5. HIGFormer-Inspired GNN
6. TimesNet Temporal Transformer
7. Bivariate Poisson with Dixon-Coles
8. Deep Ensemble with Monte Carlo Dropout
9. Mixture of Experts (MoE)
10. Neural Architecture Search (NAS) Components
11. Meta-Learning Stacking
12. Confidence-Based Prediction Filtering

Target: 70%+ accuracy on high-confidence predictions
================================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Quantum Computing
import pennylane as qml
from pennylane import numpy as pnp

# Gradient Boosting
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# For reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ================================================================================
# SECTION 1: CONFIGURATION & DATA CLASSES
# ================================================================================

@dataclass
class ModelConfig:
    """Configuration for the prediction system"""
    input_dim: int = 170
    n_classes: int = 3
    sequence_length: int = 10
    n_qubits: int = 12
    n_quantum_layers: int = 6
    quantum_depth: int = 4
    hidden_dims: List[int] = None
    dropout: float = 0.3
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_folds: int = 5
    n_seeds: int = 3
    confidence_threshold: float = 0.55
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]


# ================================================================================
# SECTION 2: ADVANCED ODDS PROCESSING & FEATURE ENGINEERING
# ================================================================================

class AdvancedOddsProcessor:
    """Advanced processing of betting odds features"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.quantile_transformer = QuantileTransformer(output_distribution='normal')
        
    def remove_vig(self, home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
        """Remove bookmaker margin to get true probabilities"""
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        total = home_prob + draw_prob + away_prob
        return home_prob / total, draw_prob / total, away_prob / total
    
    def calculate_market_consensus(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate consensus across multiple bookmakers"""
        features = pd.DataFrame()
        home_cols = [c for c in odds_df.columns if c.endswith('H') and not c.endswith('AHH')]
        draw_cols = [c for c in odds_df.columns if c.endswith('D')]
        away_cols = [c for c in odds_df.columns if c.endswith('A') and not c.endswith('AHA')]
        
        home_probs, draw_probs, away_probs = [], [], []
        
        for h, d, a in zip(home_cols[:10], draw_cols[:10], away_cols[:10]):
            if h in odds_df.columns and d in odds_df.columns and a in odds_df.columns:
                probs = odds_df.apply(
                    lambda row: self.remove_vig(
                        row[h] if row[h] > 1 else 1.01,
                        row[d] if row[d] > 1 else 1.01,
                        row[a] if row[a] > 1 else 1.01
                    ), axis=1
                )
                home_probs.append(probs.apply(lambda x: x[0]))
                draw_probs.append(probs.apply(lambda x: x[1]))
                away_probs.append(probs.apply(lambda x: x[2]))
        
        if home_probs:
            features['consensus_home_prob'] = pd.concat(home_probs, axis=1).mean(axis=1)
            features['consensus_draw_prob'] = pd.concat(draw_probs, axis=1).mean(axis=1)
            features['consensus_away_prob'] = pd.concat(away_probs, axis=1).mean(axis=1)
            features['consensus_std_home'] = pd.concat(home_probs, axis=1).std(axis=1)
            features['consensus_std_draw'] = pd.concat(draw_probs, axis=1).std(axis=1)
            features['consensus_std_away'] = pd.concat(away_probs, axis=1).std(axis=1)
            features['consensus_agreement'] = 1 - (
                features['consensus_std_home'] + 
                features['consensus_std_draw'] + 
                features['consensus_std_away']
            ) / 3
        
        return features
    
    def calculate_odds_value(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Detect value in odds (potential mispricing)"""
        features = pd.DataFrame()
        if 'MaxH' in odds_df.columns and 'AvgH' in odds_df.columns:
            features['value_home'] = odds_df['MaxH'] / odds_df['AvgH'] - 1
            features['value_draw'] = odds_df['MaxD'] / odds_df['AvgD'] - 1
            features['value_away'] = odds_df['MaxA'] / odds_df['AvgA'] - 1
        if 'PSH' in odds_df.columns and 'B365H' in odds_df.columns:
            features['pinnacle_vs_b365_home'] = odds_df['PSH'] / odds_df['B365H'] - 1
            features['pinnacle_vs_b365_away'] = odds_df['PSA'] / odds_df['B365A'] - 1
        return features
    
    def calculate_odds_movement(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate odds movement (opening vs closing)"""
        features = pd.DataFrame()
        opening_cols = ['B365H', 'B365D', 'B365A']
        closing_cols = ['B365CH', 'B365CD', 'B365CA']
        
        for op, cl, name in zip(opening_cols, closing_cols, ['home', 'draw', 'away']):
            if op in odds_df.columns and cl in odds_df.columns:
                features[f'movement_{name}'] = (odds_df[cl] - odds_df[op]) / odds_df[op]
                features[f'movement_{name}_direction'] = (odds_df[cl] < odds_df[op]).astype(int)
        return features
    
    def calculate_asian_handicap_features(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Extract Asian Handicap implied goal difference"""
        features = pd.DataFrame()
        if 'AHh' in odds_df.columns:
            features['ah_line'] = odds_df['AHh']
            features['ah_implied_diff'] = -odds_df['AHh']
            if 'PAHH' in odds_df.columns and 'PAHA' in odds_df.columns:
                features['ah_home_prob'] = 1 / odds_df['PAHH']
                features['ah_away_prob'] = 1 / odds_df['PAHA']
                features['ah_balance'] = features['ah_home_prob'] - features['ah_away_prob']
        return features
    
    def calculate_over_under_features(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Extract Over/Under implied total goals"""
        features = pd.DataFrame()
        if 'P>2.5' in odds_df.columns and 'P<2.5' in odds_df.columns:
            over_prob = 1 / odds_df['P>2.5']
            under_prob = 1 / odds_df['P<2.5']
            total = over_prob + under_prob
            features['ou25_over_prob'] = over_prob / total
            features['ou25_under_prob'] = under_prob / total
            features['ou25_implied_goals'] = 2.5 + (features['ou25_over_prob'] - 0.5) * 2
        return features
    
    def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all odds features"""
        print("Processing odds features...")
        processed = df.copy()
        
        consensus = self.calculate_market_consensus(df)
        processed = pd.concat([processed, consensus], axis=1)
        
        value = self.calculate_odds_value(df)
        processed = pd.concat([processed, value], axis=1)
        
        movement = self.calculate_odds_movement(df)
        processed = pd.concat([processed, movement], axis=1)
        
        ah = self.calculate_asian_handicap_features(df)
        processed = pd.concat([processed, ah], axis=1)
        
        ou = self.calculate_over_under_features(df)
        processed = pd.concat([processed, ou], axis=1)
        
        processed = processed.fillna(processed.median())
        
        if 'consensus_home_prob' in processed.columns:
            processed['home_draw_ratio'] = processed['consensus_home_prob'] / (processed['consensus_draw_prob'] + 0.01)
            processed['home_away_ratio'] = processed['consensus_home_prob'] / (processed['consensus_away_prob'] + 0.01)
            processed['favorite_strength'] = processed[['consensus_home_prob', 'consensus_draw_prob', 'consensus_away_prob']].max(axis=1)
            processed['uncertainty'] = 1 - processed['favorite_strength']
        
        print(f"Total features after processing: {len(processed.columns)}")
        return processed
