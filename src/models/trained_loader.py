"""
Trained Model Loader

Loads models trained on Kaggle and exported to models/trained/
Supports: XGBoost, LightGBM, CatBoost, PyTorch, ONNX
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
TRAINED_DIR = MODELS_DIR / "trained"
CONFIG_DIR = MODELS_DIR / "config"


class TrainedModelLoader:
    """Load models trained on Kaggle"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.elo_ratings: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self.scaler = None
        self._loaded = False
    
    def load_all(self) -> bool:
        """Load all available trained models"""
        try:
            self._load_config()
            self._load_xgboost()
            self._load_lightgbm()
            self._load_catboost()
            self._load_neural_net()
            self._load_onnx()
            self._loaded = len(self.models) > 0
            logger.info(f"Loaded {len(self.models)} trained models")
            return self._loaded
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _load_config(self):
        """Load encoders, elo ratings, and metadata"""
        # Encoders
        enc_path = CONFIG_DIR / "encoders.pkl"
        if enc_path.exists():
            with open(enc_path, 'rb') as f:
                data = pickle.load(f)
                self.encoders = data
                self.scaler = data.get('scaler')
            logger.info("Loaded encoders")
        
        # Elo ratings
        elo_path = CONFIG_DIR / "elo_ratings.json"
        if elo_path.exists():
            with open(elo_path, 'r') as f:
                self.elo_ratings = json.load(f)
            logger.info(f"Loaded {len(self.elo_ratings)} team Elo ratings")
        
        # Metadata
        meta_path = CONFIG_DIR / "model_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Loaded model metadata")
    
    def _load_xgboost(self):
        """Load XGBoost model"""
        path = TRAINED_DIR / "xgb_football.json"
        if path.exists():
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier()
                model.load_model(str(path))
                self.models['xgb'] = model
                logger.info("Loaded XGBoost model")
            except ImportError:
                logger.warning("XGBoost not installed")
    
    def _load_lightgbm(self):
        """Load LightGBM model"""
        path = TRAINED_DIR / "lgb_football.txt"
        if path.exists():
            try:
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(path))
                self.models['lgb'] = model
                logger.info("Loaded LightGBM model")
            except ImportError:
                logger.warning("LightGBM not installed")
    
    def _load_catboost(self):
        """Load CatBoost model"""
        path = TRAINED_DIR / "cat_football.cbm"
        if path.exists():
            try:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(str(path))
                self.models['cat'] = model
                logger.info("Loaded CatBoost model")
            except ImportError:
                logger.warning("CatBoost not installed")
    
    def _load_neural_net(self):
        """Load PyTorch neural network"""
        path = TRAINED_DIR / "nn_football.pt"
        if path.exists():
            try:
                import torch
                import torch.nn as nn
                
                class FootballNet(nn.Module):
                    """FootballNet architecture matching the trained v7 checkpoint.
                    
                    Architecture: 153 features → 256 → 128 → 64 → 3 classes
                    With BatchNorm and Dropout for regularization.
                    """
                    def __init__(self, input_dim=153, num_classes=3):
                        super().__init__()
                        self.net = nn.Sequential(
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
                        return self.net(x)
                
                model = FootballNet()
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                self.models['nn'] = model
                logger.info("Loaded PyTorch neural network")
            except ImportError:
                logger.warning("PyTorch not installed")
    
    def _load_onnx(self):
        """Load ONNX model for fast inference"""
        path = TRAINED_DIR / "football_transformer.onnx"
        if path.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(path))
                self.models['onnx'] = session
                logger.info("Loaded ONNX transformer")
            except ImportError:
                logger.warning("ONNX Runtime not installed")
    
    def get_elo(self, team: str) -> float:
        """Get Elo rating for a team"""
        if team in self.elo_ratings:
            return self.elo_ratings[team]
        # Fuzzy match
        team_lower = team.lower()
        for t, elo in self.elo_ratings.items():
            if t.lower() in team_lower or team_lower in t.lower():
                return elo
        return 1500.0  # Default
    
    def build_features(self, home_team: str, away_team: str, league: str = 'premier_league') -> np.ndarray:
        """Build comprehensive 153-feature vector for prediction."""
        try:
            # Use comprehensive feature builder
            from .comprehensive_features import build_match_features
            features = build_match_features(home_team, away_team, league)
            logger.debug(f"Built {features.shape[1]} features for {home_team} vs {away_team}")
            return features
        except Exception as e:
            logger.warning(f"Comprehensive features failed, using fallback: {e}")
            # Fallback to basic features
            home_elo = self.get_elo(home_team)
            away_elo = self.get_elo(away_team)
            
            # Encode teams
            team_enc = self.encoders.get('team_enc')
            if team_enc:
                try:
                    home_enc = team_enc.transform([home_team])[0]
                    away_enc = team_enc.transform([away_team])[0]
                except:
                    home_enc, away_enc = 0, 0
            else:
                home_enc, away_enc = 0, 0
            
            # Build basic feature vector
            import datetime
            now = datetime.datetime.now()
            features = np.array([
                home_enc, away_enc,
                home_elo, away_elo,
                home_elo - away_elo,
                now.year, now.month, now.weekday()
            ], dtype=np.float32)
            
            return features.reshape(1, -1)
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Get ensemble prediction"""
        if not self._loaded:
            self.load_all()
        
        if not self.models:
            return {'error': 'No trained models available'}
        
        features = self.build_features(home_team, away_team)
        
        # Ensemble weights
        weights = self.metadata.get('ensemble_weights', {
            'xgb': 0.3, 'lgb': 0.3, 'cat': 0.25, 'nn': 0.15
        })
        
        probs = np.zeros(3)
        total_weight = 0
        
        # XGBoost
        if 'xgb' in self.models:
            probs += weights.get('xgb', 0.3) * self.models['xgb'].predict_proba(features)[0]
            total_weight += weights.get('xgb', 0.3)
        
        # LightGBM (skip if feature count mismatch)
        if 'lgb' in self.models:
            try:
                lgb_raw = self.models['lgb'].predict(features)
                # Handle different output shapes
                if lgb_raw.ndim == 1:
                    lgb_probs = lgb_raw
                elif lgb_raw.ndim == 2:
                    lgb_probs = lgb_raw[0]
                else:
                    lgb_probs = np.array([lgb_raw, 0.3, 0.3])
                
                # Normalize if needed
                if len(lgb_probs) >= 3:
                    lgb_probs = lgb_probs[:3]
                    lgb_probs = lgb_probs / lgb_probs.sum()
                    probs += weights.get('lgb', 0.3) * lgb_probs
                    total_weight += weights.get('lgb', 0.3)
            except Exception as e:
                # Feature mismatch - skip this model
                logger.debug(f"LightGBM skipped: {e}")
        
        # CatBoost (skip if feature count mismatch)
        if 'cat' in self.models:
            try:
                cat_probs = self.models['cat'].predict_proba(features)[0]
                probs += weights.get('cat', 0.25) * cat_probs
                total_weight += weights.get('cat', 0.25)
            except Exception as e:
                logger.debug(f"CatBoost skipped: {e}")
        
        # Neural Net (skip if scaler or feature issues)
        if 'nn' in self.models:
            try:
                import torch
                if self.scaler:
                    scaled = self.scaler.transform(features)
                else:
                    scaled = features
                with torch.no_grad():
                    nn_out = torch.softmax(self.models['nn'](torch.FloatTensor(scaled)), dim=1).numpy()[0]
                probs += weights.get('nn', 0.15) * nn_out
                total_weight += weights.get('nn', 0.15)
            except Exception as e:
                logger.debug(f"Neural Net skipped: {e}")
        
        if total_weight > 0:
            probs = probs / total_weight
        
        # Normalize
        probs = probs / probs.sum()
        
        # Get classes
        classes = self.metadata.get('classes', ['A', 'D', 'H'])
        pred_idx = probs.argmax()
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': float(probs[classes.index('H')] if 'H' in classes else probs[0]),
            'draw_prob': float(probs[classes.index('D')] if 'D' in classes else probs[1]),
            'away_win_prob': float(probs[classes.index('A')] if 'A' in classes else probs[2]),
            'predicted_outcome': classes[pred_idx].replace('H', 'Home Win').replace('A', 'Away Win').replace('D', 'Draw'),
            'confidence': float(probs[pred_idx]),
            'models_used': list(self.models.keys())
        }


# Global instance
_loader: Optional[TrainedModelLoader] = None

def get_trained_loader() -> TrainedModelLoader:
    global _loader
    if _loader is None:
        _loader = TrainedModelLoader()
        _loader.load_all()
    return _loader

def predict_with_trained(home: str, away: str) -> Dict:
    return get_trained_loader().predict(home, away)
