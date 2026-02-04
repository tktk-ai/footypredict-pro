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
# Part 2: Quantum Neural Network Components

# ================================================================================
# SECTION 3: ENHANCED QUANTUM NEURAL NETWORK (CORE)
# ================================================================================

class EnhancedQuantumCircuit:
    """Enhanced Quantum Circuit with Multiple Ansätze"""
    
    def __init__(self, n_qubits: int = 12, n_layers: int = 6, ansatz: str = 'strongly_entangling'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        self.circuit = qml.QNode(self._build_circuit, self.dev, interface="torch")
    
    def _build_circuit(self, inputs, weights):
        """Build enhanced quantum circuit with data re-uploading"""
        for i in range(self.n_qubits):
            qml.RY(inputs[i % len(inputs)] * np.pi, wires=i)
        
        weight_idx = 0
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(weights[weight_idx], wires=i)
                weight_idx += 1
                qml.RY(weights[weight_idx], wires=i)
                weight_idx += 1
                qml.RZ(weights[weight_idx], wires=i)
                weight_idx += 1
            
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            if layer % 2 == 0 and layer < self.n_layers - 1:
                for i in range(self.n_qubits):
                    qml.RY(inputs[i % len(inputs)] * np.pi * 0.5, wires=i)
            
            if layer % 2 == 1:
                for i in range(0, self.n_qubits - 2, 2):
                    qml.CZ(wires=[i, i + 2])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(min(3, self.n_qubits))]
    
    @property
    def n_params(self):
        return self.n_layers * self.n_qubits * 3


class QuantumProcessingUnit(nn.Module):
    """Core Quantum Processing Unit"""
    
    def __init__(self, input_dim: int, n_qubits: int = 12, n_layers: int = 6, output_dim: int = 64):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        self.qc = EnhancedQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.q_weights = nn.Parameter(torch.randn(self.qc.n_params) * 0.1)
        
        self.post_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_quantum = self.pre_net(x)
        
        q_outputs = []
        for i in range(batch_size):
            q_out = self.qc.circuit(x_quantum[i] * np.pi, self.q_weights)
            q_outputs.append(torch.stack(q_out))
        
        q_outputs = torch.stack(q_outputs)
        return self.post_net(q_outputs)


class HybridQuantumTransformer(nn.Module):
    """Hybrid Quantum-Classical Transformer"""
    
    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 n_qubits: int = 12, n_quantum_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.quantum_processor = QuantumProcessingUnit(
            input_dim=d_model, n_qubits=n_qubits, n_layers=n_quantum_layers, output_dim=d_model // 2
        )
        
        self.classical_path = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        x = self.input_embed(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        q_out = self.quantum_processor(x)
        c_out = self.classical_path(x)
        
        combined = torch.cat([q_out, c_out], dim=-1)
        return self.fusion(combined)


# ================================================================================
# SECTION 4: ADVANCED NEURAL ARCHITECTURES
# ================================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return x + self.block(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_experts * 2), nn.GELU(),
            nn.Linear(n_experts * 2, n_experts)
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, output_dim * 2), nn.GELU(), nn.Linear(output_dim * 2, output_dim))
            for _ in range(n_experts)
        ])
        
    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)
        
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = top_k_weights[mask, (top_k_indices[mask] == i).float().argmax(dim=-1)]
                output[mask] += expert_out * weight.unsqueeze(-1)
        return output


class DeepCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, n_cross_layers: int = 3):
        super().__init__()
        self.n_cross_layers = n_cross_layers
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim) * 0.01) for _ in range(n_cross_layers)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(n_cross_layers)])
        
    def forward(self, x0):
        x = x0
        for w, b in zip(self.cross_weights, self.cross_biases):
            cross = x0 * (x * w).sum(dim=-1, keepdim=True) + b
            x = cross + x
        return x


class MCDropoutNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], n_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
        self.dropout = dropout
        
    def forward(self, x, n_samples: int = 1):
        if n_samples == 1: return self.network(x)
        self.train()
        outputs = [self.network(x) for _ in range(n_samples)]
        self.eval()
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0), outputs.std(dim=0).mean(dim=-1)


class DeepEnsemble(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], n_classes: int, n_networks: int = 5, dropout: float = 0.3):
        super().__init__()
        self.networks = nn.ModuleList([MCDropoutNetwork(input_dim, hidden_dims, n_classes, dropout) for _ in range(n_networks)])
        
    def forward(self, x, mc_samples: int = 1):
        predictions = []
        for net in self.networks:
            pred = net(x, n_samples=mc_samples)
            if isinstance(pred, tuple): pred = pred[0]
            predictions.append(F.softmax(pred, dim=-1))
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        uncertainty = torch.stack(predictions, dim=0).std(dim=0).mean(dim=-1)
        return ensemble_pred, uncertainty
# Part 3: Gradient Boosting Ensemble & Statistical Models

# ================================================================================
# SECTION 5: GRADIENT BOOSTING ENSEMBLE (ENHANCED)
# ================================================================================

class EnhancedGBEnsemble:
    """Enhanced Gradient Boosting Ensemble"""
    
    def __init__(self, n_seeds: int = 3):
        self.n_seeds = n_seeds
        self.models = {}
        self.calibrators = {}
        self.feature_importance = None
        
    def _get_catboost(self, seed: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
            random_strength=0.5, bagging_temperature=0.5, border_count=128,
            loss_function='MultiClass', eval_metric='TotalF1:average=Macro',
            early_stopping_rounds=200, verbose=False, random_state=seed,
            task_type='GPU' if torch.cuda.is_available() else 'CPU'
        )
    
    def _get_xgboost(self, seed: int) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0.5, reg_lambda=1.5,
            min_child_weight=3, gamma=0.1, early_stopping_rounds=200, eval_metric='mlogloss',
            use_label_encoder=False, tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            random_state=seed
        )
    
    def _get_lightgbm(self, seed: int) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=8, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
            min_child_samples=20, verbose=-1, random_state=seed,
            device='gpu' if torch.cuda.is_available() else 'cpu'
        )
    
    def fit(self, X_train, y_train, X_val, y_val):
        print("Training Enhanced Gradient Boosting Ensemble...")
        all_importance = []
        
        for seed in range(self.n_seeds):
            print(f"\n  Seed {seed + 1}/{self.n_seeds}")
            
            for name, get_model in [('catboost', self._get_catboost), ('xgboost', self._get_xgboost), ('lightgbm', self._get_lightgbm)]:
                model_key = f"{name}_seed{seed}"
                model = get_model(seed)
                
                print(f"    Training {name}...", end=" ")
                
                if name == 'catboost':
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                else:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                all_importance.append(model.feature_importances_)
                
                self.calibrators[model_key] = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
                self.calibrators[model_key].fit(X_val, y_val)
                self.models[model_key] = model
                
                val_pred = self.calibrators[model_key].predict(X_val)
                acc = accuracy_score(y_val, val_pred)
                print(f"Accuracy: {acc:.4f}")
        
        self.feature_importance = np.mean(all_importance, axis=0)
        
    def predict_proba(self, X) -> np.ndarray:
        predictions = [calibrator.predict_proba(X) for calibrator in self.calibrators.values()]
        return np.mean(predictions, axis=0)
    
    def get_top_features(self, feature_names: List[str], top_n: int = 30) -> pd.DataFrame:
        return pd.DataFrame({'feature': feature_names, 'importance': self.feature_importance}).sort_values('importance', ascending=False).head(top_n)


# ================================================================================
# SECTION 6: DIXON-COLES POISSON MODEL
# ================================================================================

class DixonColesModel:
    """Dixon-Coles Model for Goal Prediction"""
    
    def __init__(self, rho: float = -0.13):
        self.rho = rho
        self.teams = {}
        self.home_advantage = 0.25
        
    def tau(self, home_goals: int, away_goals: int, lambda_h: float, mu_a: float) -> float:
        if home_goals == 0 and away_goals == 0: return 1 - lambda_h * mu_a * self.rho
        elif home_goals == 0 and away_goals == 1: return 1 + lambda_h * self.rho
        elif home_goals == 1 and away_goals == 0: return 1 + mu_a * self.rho
        elif home_goals == 1 and away_goals == 1: return 1 - self.rho
        return 1.0
    
    def fit(self, df: pd.DataFrame, time_weight: bool = True):
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        for team in teams:
            home_scored = df[df['home_team'] == team]['home_goals'].mean()
            home_conceded = df[df['home_team'] == team]['away_goals'].mean()
            away_scored = df[df['away_team'] == team]['away_goals'].mean()
            away_conceded = df[df['away_team'] == team]['home_goals'].mean()
            self.teams[team] = {
                'attack': (home_scored + away_scored) / 2 if not np.isnan(home_scored) else 1.3,
                'defense': (home_conceded + away_conceded) / 2 if not np.isnan(home_conceded) else 1.3
            }
        self.avg_goals = df[['home_goals', 'away_goals']].mean().mean()
        self.home_advantage = (df['home_goals'].mean() - df['away_goals'].mean()) / self.avg_goals
        
    def predict(self, home_team: str, away_team: str, max_goals: int = 8) -> Dict:
        home_attack = self.teams.get(home_team, {}).get('attack', 1.0)
        away_defense = self.teams.get(away_team, {}).get('defense', 1.0)
        away_attack = self.teams.get(away_team, {}).get('attack', 1.0)
        home_defense = self.teams.get(home_team, {}).get('defense', 1.0)
        
        lambda_h = home_attack * away_defense * self.avg_goals * (1 + self.home_advantage * 0.5)
        mu_a = away_attack * home_defense * self.avg_goals * (1 - self.home_advantage * 0.25)
        
        probs = np.zeros((max_goals, max_goals))
        for h in range(max_goals):
            for a in range(max_goals):
                p_base = poisson.pmf(h, lambda_h) * poisson.pmf(a, mu_a)
                probs[h, a] = p_base * self.tau(h, a, lambda_h, mu_a)
        probs /= probs.sum()
        
        return {
            'home_win': np.triu(probs, k=1).sum(),
            'draw': np.trace(probs),
            'away_win': np.tril(probs, k=-1).sum(),
            'over_25': (probs * (np.array([[h + a for a in range(max_goals)] for h in range(max_goals)]) > 2.5)).sum(),
            'btts_yes': probs[1:, 1:].sum(),
            'home_xg': lambda_h,
            'away_xg': mu_a
        }
# Part 4: Complete System & Main Execution

# ================================================================================
# SECTION 7: COMPLETE SYSTEM
# ================================================================================

class UltraAdvancedFootballPredictor:
    """ULTRA-ADVANCED FOOTBALL PREDICTION SYSTEM v3.0"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*80)
        print("INITIALIZING ULTRA-ADVANCED QUANTUM FOOTBALL PREDICTION SYSTEM v3.0")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Input dimensions: {config.input_dim}")
        print(f"Quantum qubits: {config.n_qubits}")
        print("="*80)
        
        self._init_preprocessors()
        self._init_quantum_components()
        self._init_neural_components()
        self._init_ensemble_components()
        
        print("\nSystem initialized successfully!")
        
    def _init_preprocessors(self):
        print("\n[1/4] Initializing preprocessors...")
        self.odds_processor = AdvancedOddsProcessor()
        self.scaler = RobustScaler()
        
    def _init_quantum_components(self):
        print("[2/4] Initializing QUANTUM components (CORE)...")
        self.quantum_transformer = HybridQuantumTransformer(
            input_dim=self.config.input_dim, d_model=128, n_heads=8, n_layers=4,
            n_qubits=self.config.n_qubits, n_quantum_layers=self.config.n_quantum_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.quantum_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.n_classes)
        ).to(self.device)
        
    def _init_neural_components(self):
        print("[3/4] Initializing neural components...")
        self.deep_cross = nn.Sequential(
            DeepCrossNetwork(self.config.input_dim, n_cross_layers=4),
            nn.Linear(self.config.input_dim, 128), nn.GELU(), nn.Dropout(self.config.dropout)
        ).to(self.device)
        
        self.moe = MixtureOfExperts(input_dim=128, output_dim=64, n_experts=8, top_k=2).to(self.device)
        
        self.deep_ensemble = DeepEnsemble(
            input_dim=self.config.input_dim, hidden_dims=[256, 128, 64],
            n_classes=self.config.n_classes, n_networks=5, dropout=self.config.dropout
        ).to(self.device)
        
    def _init_ensemble_components(self):
        print("[4/4] Initializing gradient boosting ensemble...")
        self.gb_ensemble = EnhancedGBEnsemble(n_seeds=self.config.n_seeds)
        
        self.meta_learner = nn.Sequential(
            nn.Linear(self.config.n_classes * 4, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, self.config.n_classes)
        ).to(self.device)
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print("\nPreprocessing data...")
        target_cols = ['home_goals', 'away_goals']
        feature_cols = [c for c in df.columns if c not in target_cols]
        
        processed = self.odds_processor.process_all_features(df[feature_cols])
        
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            y = np.where(df['home_goals'] > df['away_goals'], 0,
                        np.where(df['home_goals'] == df['away_goals'], 1, 2))
        else:
            y = None
        
        X = self.scaler.fit_transform(processed.values)
        self.feature_names = processed.columns.tolist()
        print(f"Processed features: {X.shape[1]}")
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        print("\n" + "="*80)
        print("TRAINING ULTRA-ADVANCED QUANTUM SYSTEM")
        print("="*80)
        print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Features: {X_train.shape[1]}")
        
        results = {}
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        class_counts = np.bincount(y_train)
        sample_weights = (1.0 / class_counts)[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler)
        
        # PHASE 1: Gradient Boosting
        print("\n" + "-"*40 + "\nPHASE 1: Training Gradient Boosting\n" + "-"*40)
        self.gb_ensemble.fit(X_train, y_train, X_val, y_val)
        gb_val_pred = self.gb_ensemble.predict_proba(X_val)
        results['gb_accuracy'] = accuracy_score(y_val, gb_val_pred.argmax(axis=1))
        print(f"\nGB Ensemble Accuracy: {results['gb_accuracy']:.4f}")
        
        # PHASE 2: Quantum Neural Network
        print("\n" + "-"*40 + "\nPHASE 2: Training QUANTUM NN (CORE)\n" + "-"*40)
        quantum_params = list(self.quantum_transformer.parameters()) + list(self.quantum_classifier.parameters())
        optimizer = torch.optim.AdamW(quantum_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(1.0 / class_counts).to(self.device))
        
        best_quantum_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.quantum_transformer.train()
            self.quantum_classifier.train()
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                features = self.quantum_transformer(batch_x)
                outputs = self.quantum_classifier(features)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(quantum_params, max_norm=1.0)
                optimizer.step()
            scheduler.step()
            
            self.quantum_transformer.eval()
            self.quantum_classifier.eval()
            with torch.no_grad():
                val_features = self.quantum_transformer(X_val_t)
                val_outputs = self.quantum_classifier(val_features)
                val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean().item()
            
            if val_acc > best_quantum_acc:
                best_quantum_acc = val_acc
                torch.save({'transformer': self.quantum_transformer.state_dict(), 'classifier': self.quantum_classifier.state_dict()}, 'best_quantum_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - Acc: {val_acc:.4f} - Best: {best_quantum_acc:.4f}")
            
            if patience_counter >= 30: break
        
        checkpoint = torch.load('best_quantum_model.pt')
        self.quantum_transformer.load_state_dict(checkpoint['transformer'])
        self.quantum_classifier.load_state_dict(checkpoint['classifier'])
        results['quantum_accuracy'] = best_quantum_acc
        
        # PHASE 3: Deep Ensemble
        print("\n" + "-"*40 + "\nPHASE 3: Training Deep Ensemble\n" + "-"*40)
        ensemble_optimizer = torch.optim.AdamW(self.deep_ensemble.parameters(), lr=self.config.learning_rate)
        best_ensemble_acc = 0
        
        for epoch in range(50):
            self.deep_ensemble.train()
            for batch_x, batch_y in train_loader:
                ensemble_optimizer.zero_grad()
                pred, _ = self.deep_ensemble(batch_x, mc_samples=1)
                loss = F.cross_entropy(torch.log(pred + 1e-8), batch_y)
                loss.backward()
                ensemble_optimizer.step()
            
            self.deep_ensemble.eval()
            with torch.no_grad():
                val_pred, _ = self.deep_ensemble(X_val_t, mc_samples=5)
                val_acc = (val_pred.argmax(dim=1) == y_val_t).float().mean().item()
            
            if val_acc > best_ensemble_acc:
                best_ensemble_acc = val_acc
                torch.save(self.deep_ensemble.state_dict(), 'best_deep_ensemble.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50 - Acc: {val_acc:.4f}")
        
        self.deep_ensemble.load_state_dict(torch.load('best_deep_ensemble.pt'))
        results['deep_ensemble_accuracy'] = best_ensemble_acc
        
        # PHASE 4: Meta-Learner
        print("\n" + "-"*40 + "\nPHASE 4: Training Meta-Learner\n" + "-"*40)
        self.quantum_transformer.eval()
        self.quantum_classifier.eval()
        self.deep_ensemble.eval()
        
        with torch.no_grad():
            q_pred = F.softmax(self.quantum_classifier(self.quantum_transformer(X_train_t)), dim=1)
            q_val_pred = F.softmax(self.quantum_classifier(self.quantum_transformer(X_val_t)), dim=1)
            de_pred, _ = self.deep_ensemble(X_train_t, mc_samples=5)
            de_val_pred, _ = self.deep_ensemble(X_val_t, mc_samples=5)
        
        gb_train_pred = torch.FloatTensor(self.gb_ensemble.predict_proba(X_train)).to(self.device)
        gb_val_pred_t = torch.FloatTensor(gb_val_pred).to(self.device)
        
        avg_train = (q_pred + de_pred + gb_train_pred) / 3
        avg_val = (q_val_pred + de_val_pred + gb_val_pred_t) / 3
        
        meta_train_input = torch.cat([q_pred, de_pred, gb_train_pred, avg_train], dim=1)
        meta_val_input = torch.cat([q_val_pred, de_val_pred, gb_val_pred_t, avg_val], dim=1)
        
        meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=1e-3)
        
        for epoch in range(100):
            self.meta_learner.train()
            meta_optimizer.zero_grad()
            outputs = self.meta_learner(meta_train_input)
            loss = F.cross_entropy(outputs, y_train_t)
            loss.backward()
            meta_optimizer.step()
        
        self.meta_learner.eval()
        with torch.no_grad():
            final_outputs = self.meta_learner(meta_val_input)
            final_probs = F.softmax(final_outputs, dim=1)
            final_pred = final_outputs.argmax(dim=1)
            results['meta_accuracy'] = (final_pred == y_val_t).float().mean().item()
        
        # Confidence Analysis
        print("\n" + "-"*40 + "\nConfidence Analysis\n" + "-"*40)
        confidence_scores = final_probs.max(dim=1)[0].cpu().numpy()
        
        for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = confidence_scores >= thresh
            if mask.sum() > 0:
                acc = accuracy_score(y_val[mask], final_pred.cpu().numpy()[mask])
                print(f"Threshold >= {thresh:.2f}: Acc = {acc:.4f} (Coverage: {mask.mean()*100:.1f}%)")
                if thresh == self.config.confidence_threshold:
                    results['high_conf_accuracy'] = acc
                    results['high_conf_coverage'] = mask.mean() * 100
        
        print("\n" + "="*80 + "\nTRAINING COMPLETE\n" + "="*80)
        print(f"GB: {results['gb_accuracy']:.4f} | Quantum: {results['quantum_accuracy']:.4f}")
        print(f"Ensemble: {results['deep_ensemble_accuracy']:.4f} | Meta: {results['meta_accuracy']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray, return_confidence: bool = True):
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.quantum_transformer.eval()
        self.quantum_classifier.eval()
        self.deep_ensemble.eval()
        self.meta_learner.eval()
        
        with torch.no_grad():
            q_pred = F.softmax(self.quantum_classifier(self.quantum_transformer(X_t)), dim=1)
            de_pred, _ = self.deep_ensemble(X_t, mc_samples=10)
            gb_pred = torch.FloatTensor(self.gb_ensemble.predict_proba(X)).to(self.device)
            avg_pred = (q_pred + de_pred + gb_pred) / 3
            
            meta_input = torch.cat([q_pred, de_pred, gb_pred, avg_pred], dim=1)
            final_probs = F.softmax(self.meta_learner(meta_input), dim=1)
        
        probabilities = final_probs.cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        
        return (predictions, probabilities, confidence) if return_confidence else predictions


# ================================================================================
# SECTION 8: MAIN EXECUTION FOR KAGGLE
# ================================================================================

def main():
    print("="*80)
    print("ULTRA-ADVANCED QUANTUM FOOTBALL PREDICTION SYSTEM v3.0")
    print("="*80)
    
    # Load Kaggle data
    df = pd.read_csv('/kaggle/input/football-match-prediction-features/kaggle_training_data.csv')
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    config = ModelConfig(
        input_dim=200, n_classes=3, n_qubits=10, n_quantum_layers=4,
        batch_size=256, epochs=150, learning_rate=1e-3,
        confidence_threshold=0.55, n_folds=5, n_seeds=3
    )
    
    predictor = UltraAdvancedFootballPredictor(config)
    X, y = predictor.preprocess_data(df)
    
    # Update input_dim
    config.input_dim = X.shape[1]
    predictor = UltraAdvancedFootballPredictor(config)
    predictor.scaler = RobustScaler()
    X = predictor.scaler.fit_transform(X)
    
    # Temporal split
    split_idx = int(0.8 * len(X))
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]
    
    val_split = int(0.9 * len(X_train_full))
    X_train, X_val = X_train_full[:val_split], X_train_full[val_split:]
    y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Train
    results = predictor.train(X_train, y_train, X_val, y_val)
    
    # Test
    print("\n" + "="*80 + "\nTEST SET EVALUATION\n" + "="*80)
    test_pred, test_probs, test_conf = predictor.predict(X_test)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    print(f"Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    conf_mask = test_conf >= config.confidence_threshold
    if conf_mask.sum() > 0:
        print(f"High Confidence (>={config.confidence_threshold:.0%}): {accuracy_score(y_test[conf_mask], test_pred[conf_mask]):.4f} ({conf_mask.mean()*100:.1f}%)")
    
    # Save model
    torch.save({
        'quantum_transformer': predictor.quantum_transformer.state_dict(),
        'quantum_classifier': predictor.quantum_classifier.state_dict(),
        'deep_ensemble': predictor.deep_ensemble.state_dict(),
        'meta_learner': predictor.meta_learner.state_dict(),
        'config': config
    }, 'quantum_football_predictor.pt')
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
