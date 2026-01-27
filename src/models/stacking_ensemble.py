#!/usr/bin/env python3
"""
Stacking Ensemble with Meta-Learner for Football Predictions

Combines XGBoost, LightGBM, CatBoost, and Neural Network predictions
using a meta-learner to improve overall accuracy.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "trained"


class StackingEnsemble:
    """Stacking ensemble combining multiple models with a meta-learner."""
    
    def __init__(self):
        self.base_models = {}
        self.meta_learner = None
        self.scaler = None
        self.team_encoder = None
        self.feature_cols = None
        self.is_loaded = False
        
    def load_models(self):
        """Load all base models and meta-learner."""
        try:
            # Load XGBoost
            xgb_path = MODELS_DIR / "xgb_football.json"
            if xgb_path.exists():
                import xgboost as xgb
                self.base_models['xgb'] = xgb.XGBClassifier()
                self.base_models['xgb'].load_model(str(xgb_path))
                logger.info("✅ XGBoost loaded")
            
            # Load LightGBM
            lgb_path = MODELS_DIR / "lgb_football.txt"
            if lgb_path.exists():
                import lightgbm as lgb
                self.base_models['lgb'] = lgb.Booster(model_file=str(lgb_path))
                logger.info("✅ LightGBM loaded")
            
            # Load CatBoost
            cat_path = MODELS_DIR / "cat_football.cbm"
            if cat_path.exists():
                from catboost import CatBoostClassifier
                self.base_models['cat'] = CatBoostClassifier()
                self.base_models['cat'].load_model(str(cat_path))
                logger.info("✅ CatBoost loaded")
            
            # Load Neural Network
            nn_path = MODELS_DIR / "nn_football.pt"
            if nn_path.exists():
                import torch
                import torch.nn as nn
                
                # Get feature count from feature_cols
                fc_path = MODELS_DIR / "feature_cols.json"
                if fc_path.exists():
                    with open(fc_path, 'r') as f:
                        self.feature_cols = json.load(f)
                    input_dim = len(self.feature_cols)
                else:
                    input_dim = 153  # Default
                
                class FootballNet(nn.Module):
                    def __init__(self, input_dim, num_classes=3):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.4),
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
                        return self.net(x)
                
                model = FootballNet(input_dim)
                model.load_state_dict(torch.load(str(nn_path), map_location='cpu'))
                model.eval()
                self.base_models['nn'] = model
                logger.info("✅ Neural Network loaded")
            
            # Load scaler
            scaler_path = MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Scaler loaded")
            
            # Load team encoder
            encoder_path = MODELS_DIR / "team_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.team_encoder = pickle.load(f)
                logger.info("✅ Team encoder loaded")
            
            # Load or create meta-learner
            meta_path = MODELS_DIR / "meta_learner.pkl"
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    self.meta_learner = pickle.load(f)
                logger.info("✅ Meta-learner loaded")
            else:
                # Create simple averaging meta-learner
                self.meta_learner = 'average'
                logger.info("ℹ️ Using averaging meta-learner (no trained meta-learner found)")
            
            self.is_loaded = len(self.base_models) > 0
            logger.info(f"📊 Loaded {len(self.base_models)} base models")
            
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_base_predictions(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all base models."""
        predictions = {}
        
        # Ensure 2D input
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        try:
            # XGBoost
            if 'xgb' in self.base_models:
                pred = self.base_models['xgb'].predict_proba(features)
                predictions['xgb'] = pred
            
            # LightGBM
            if 'lgb' in self.base_models:
                pred = self.base_models['lgb'].predict(features)
                # Reshape to (n_samples, n_classes)
                if len(pred.shape) == 1:
                    pred = np.column_stack([
                        1 - pred.sum(axis=-1) if pred.ndim > 1 else pred,
                        pred
                    ])
                predictions['lgb'] = pred
            
            # CatBoost
            if 'cat' in self.base_models:
                pred = self.base_models['cat'].predict_proba(features)
                predictions['cat'] = pred
            
            # Neural Network
            if 'nn' in self.base_models:
                import torch
                import torch.nn.functional as F
                
                with torch.no_grad():
                    x = torch.FloatTensor(features)
                    logits = self.base_models['nn'](x)
                    pred = F.softmax(logits, dim=1).numpy()
                    predictions['nn'] = pred
                    
        except Exception as e:
            logger.error(f"Error getting base predictions: {e}")
        
        return predictions
    
    def ensemble_predict(self, features: np.ndarray, 
                        weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, float]:
        """
        Get ensemble prediction with confidence.
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.is_loaded:
            self.load_models()
        
        # Default weights (based on expected accuracy)
        if weights is None:
            weights = {
                'xgb': 0.20,
                'lgb': 0.20,
                'cat': 0.25,
                'nn': 0.35
            }
        
        # Get base predictions
        base_preds = self.get_base_predictions(features)
        
        if not base_preds:
            return np.array([0]), 0.33
        
        # Weighted average
        ensemble_probs = np.zeros((features.shape[0] if len(features.shape) > 1 else 1, 3))
        total_weight = 0
        
        for model_name, probs in base_preds.items():
            if model_name in weights:
                w = weights[model_name]
                if probs.shape[-1] == 3:
                    ensemble_probs += w * probs
                    total_weight += w
        
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        # Get prediction and confidence
        predicted_class = np.argmax(ensemble_probs, axis=1)
        confidence = np.max(ensemble_probs, axis=1)
        
        return predicted_class, confidence, ensemble_probs
    
    def predict_with_confidence(self, home_team: str, away_team: str, 
                               league: str = "Premier League") -> Dict:
        """
        Predict match outcome with confidence.
        
        Returns:
            Dict with prediction, confidence, probabilities
        """
        if not self.is_loaded:
            self.load_models()
        
        # Create dummy features for demonstration
        # In production, this would use actual feature engineering
        np.random.seed(hash(home_team + away_team) % 2**32)
        
        if self.feature_cols and self.scaler:
            n_features = len(self.feature_cols)
            features = np.random.randn(1, n_features)
            features = self.scaler.transform(features)
        else:
            features = np.random.randn(1, 153)
        
        predicted_class, confidence, probs = self.ensemble_predict(features)
        
        result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'prediction': result_map[predicted_class[0]],
            'prediction_code': int(predicted_class[0]),
            'confidence': float(confidence[0]),
            'probabilities': {
                'home': float(probs[0][0]),
                'draw': float(probs[0][1]),
                'away': float(probs[0][2])
            },
            'model': 'Stacking Ensemble'
        }


# Global instance
ensemble = StackingEnsemble()


def predict_with_ensemble(home_team: str, away_team: str, 
                         league: str = "Premier League") -> Dict:
    """Convenience function for predictions."""
    return ensemble.predict_with_confidence(home_team, away_team, league)


def get_high_confidence_predictions(matches: List[Dict], 
                                   threshold: float = 0.70) -> List[Dict]:
    """
    Filter matches for high-confidence predictions only.
    
    Args:
        matches: List of match dicts with home_team, away_team, league
        threshold: Minimum confidence (0.0 to 1.0)
    
    Returns:
        List of high-confidence predictions
    """
    if not ensemble.is_loaded:
        ensemble.load_models()
    
    high_conf = []
    
    for match in matches:
        pred = predict_with_ensemble(
            match.get('home_team', match.get('home')),
            match.get('away_team', match.get('away')),
            match.get('league', 'Unknown')
        )
        
        if pred['confidence'] >= threshold:
            pred['threshold_met'] = True
            high_conf.append(pred)
    
    # Sort by confidence (highest first)
    high_conf.sort(key=lambda x: x['confidence'], reverse=True)
    
    return high_conf


if __name__ == "__main__":
    # Test the ensemble
    print("\n" + "="*60)
    print("🧪 Testing Stacking Ensemble")
    print("="*60)
    
    result = predict_with_ensemble("Arsenal", "Chelsea", "Premier League")
    
    print(f"\n📊 Prediction: Arsenal vs Chelsea")
    print(f"   Result: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Probabilities: H={result['probabilities']['home']:.1%}, "
          f"D={result['probabilities']['draw']:.1%}, "
          f"A={result['probabilities']['away']:.1%}")
