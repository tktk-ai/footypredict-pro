"""
Advanced Models Integration

Integrates the trained XGBoost/LightGBM models into the prediction system.
Provides unified API for all advanced market predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "advanced"


@dataclass
class AdvancedPrediction:
    """Prediction result from advanced models"""
    market: str
    prediction: str
    confidence: float
    probability: float
    model: str


class AdvancedModelsPredictor:
    """Unified predictor using trained advanced models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
        self.odds_cols = ['B365H', 'B365D', 'B365A']
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all trained models"""
        if not MODELS_DIR.exists():
            logger.warning(f"Models directory not found: {MODELS_DIR}")
            return
        
        # Load scaler
        scaler_file = MODELS_DIR / "feature_scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info("Loaded feature scaler")
        
        # Load models
        model_files = {
            'result_xgb': 'xgb_result_1x2.joblib',
            'result_rf': 'result_1x2_model.joblib',
            'goals_xgb': 'xgb_goals_over25.joblib',
            'goals_gb': 'goals_over25_model.joblib',
            'btts_lgb': 'lgb_btts.joblib',
            'btts_rf': 'btts_model.joblib',
        }
        
        for name, filename in model_files.items():
            filepath = MODELS_DIR / filename
            if filepath.exists():
                self.models[name] = joblib.load(filepath)
                logger.info(f"Loaded model: {name}")
        
        logger.info(f"Loaded {len(self.models)} advanced models")
    
    def _create_features(self, home_stats: Dict, away_stats: Dict, 
                        odds: Optional[Dict] = None) -> np.ndarray:
        """Create feature vector from match stats"""
        features = [
            home_stats.get('shots', 12),
            away_stats.get('shots', 10),
            home_stats.get('shots_on_target', 5),
            away_stats.get('shots_on_target', 4),
            home_stats.get('fouls', 11),
            away_stats.get('fouls', 12),
            home_stats.get('corners', 5),
            away_stats.get('corners', 4),
            home_stats.get('yellow_cards', 1.5),
            away_stats.get('yellow_cards', 1.5),
        ]
        
        if odds:
            features.extend([
                odds.get('home', 2.0),
                odds.get('draw', 3.3),
                odds.get('away', 3.5)
            ])
        else:
            features.extend([2.0, 3.3, 3.5])
        
        return np.array(features).reshape(1, -1)
    
    def _create_features_from_odds(self, home_odds: float, draw_odds: float, 
                                   away_odds: float) -> np.ndarray:
        """Create features primarily from odds - must match 20 features from training"""
        # 12 stats + 6 odds + 2 averages = 20 features
        features = [
            12, 10,      # HS, AS (shots)
            5, 4,        # HST, AST (shots on target)
            11, 12,      # HF, AF (fouls)
            5, 4,        # HC, AC (corners)
            1.5, 1.5,    # HY, AY (yellow cards)
            0.1, 0.1,    # HR, AR (red cards)
            home_odds, draw_odds, away_odds,  # B365H, B365D, B365A
            home_odds * 0.95, draw_odds * 0.95, away_odds * 0.95,  # BWH, BWD, BWA (alternate odds)
            1.5, 1.2     # home_goals_avg, away_goals_avg
        ]
        return np.array(features).reshape(1, -1)
    
    def predict_result(self, home_odds: float = 2.0, draw_odds: float = 3.3, 
                      away_odds: float = 3.5, 
                      home_stats: Optional[Dict] = None,
                      away_stats: Optional[Dict] = None) -> AdvancedPrediction:
        """Predict match result (1X2)"""
        if home_stats and away_stats:
            X = self._create_features(home_stats, away_stats, 
                                     {'home': home_odds, 'draw': draw_odds, 'away': away_odds})
        else:
            X = self._create_features_from_odds(home_odds, draw_odds, away_odds)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Use XGBoost model preferentially, fallback to RF
        model = self.models.get('result_xgb') or self.models.get('result_rf')
        model_name = 'xgb' if 'result_xgb' in self.models else 'rf'
        
        if model is None:
            return AdvancedPrediction(
                market='1X2', prediction='H', confidence=0.45, 
                probability=0.45, model='fallback'
            )
        
        proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_map = {0: 'H', 1: 'D', 2: 'A'}
        
        return AdvancedPrediction(
            market='1X2',
            prediction=pred_map[pred_idx],
            confidence=float(proba[pred_idx]),
            probability=float(proba[pred_idx]),
            model=f'result_{model_name}'
        )
    
    def predict_over_25(self, home_odds: float = 2.0, draw_odds: float = 3.3,
                       away_odds: float = 3.5,
                       home_stats: Optional[Dict] = None,
                       away_stats: Optional[Dict] = None) -> AdvancedPrediction:
        """Predict Over/Under 2.5 goals"""
        if home_stats and away_stats:
            X = self._create_features(home_stats, away_stats, 
                                     {'home': home_odds, 'draw': draw_odds, 'away': away_odds})
        else:
            X = self._create_features_from_odds(home_odds, draw_odds, away_odds)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        model = self.models.get('goals_xgb') or self.models.get('goals_gb')
        model_name = 'xgb' if 'goals_xgb' in self.models else 'gb'
        
        if model is None:
            return AdvancedPrediction(
                market='Over 2.5', prediction='Over', confidence=0.50,
                probability=0.50, model='fallback'
            )
        
        proba = model.predict_proba(X)[0]
        over_prob = proba[1] if len(proba) > 1 else 0.5
        
        return AdvancedPrediction(
            market='Over 2.5',
            prediction='Over' if over_prob > 0.5 else 'Under',
            confidence=float(max(over_prob, 1 - over_prob)),
            probability=float(over_prob),
            model=f'goals_{model_name}'
        )
    
    def predict_btts(self, home_odds: float = 2.0, draw_odds: float = 3.3,
                    away_odds: float = 3.5,
                    home_stats: Optional[Dict] = None,
                    away_stats: Optional[Dict] = None) -> AdvancedPrediction:
        """Predict Both Teams To Score"""
        if home_stats and away_stats:
            X = self._create_features(home_stats, away_stats,
                                     {'home': home_odds, 'draw': draw_odds, 'away': away_odds})
        else:
            X = self._create_features_from_odds(home_odds, draw_odds, away_odds)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        model = self.models.get('btts_lgb') or self.models.get('btts_rf')
        model_name = 'lgb' if 'btts_lgb' in self.models else 'rf'
        
        if model is None:
            return AdvancedPrediction(
                market='BTTS', prediction='Yes', confidence=0.50,
                probability=0.50, model='fallback'
            )
        
        proba = model.predict_proba(X)[0]
        yes_prob = proba[1] if len(proba) > 1 else 0.5
        
        return AdvancedPrediction(
            market='BTTS',
            prediction='Yes' if yes_prob > 0.5 else 'No',
            confidence=float(max(yes_prob, 1 - yes_prob)),
            probability=float(yes_prob),
            model=f'btts_{model_name}'
        )
    
    def predict_all_markets(self, home_odds: float = 2.0, draw_odds: float = 3.3,
                           away_odds: float = 3.5,
                           home_stats: Optional[Dict] = None,
                           away_stats: Optional[Dict] = None) -> Dict[str, AdvancedPrediction]:
        """Predict all available markets"""
        return {
            'result': self.predict_result(home_odds, draw_odds, away_odds, home_stats, away_stats),
            'over_25': self.predict_over_25(home_odds, draw_odds, away_odds, home_stats, away_stats),
            'btts': self.predict_btts(home_odds, draw_odds, away_odds, home_stats, away_stats)
        }
    
    def get_model_info(self) -> Dict:
        """Get loaded models information"""
        return {
            'loaded_models': list(self.models.keys()),
            'has_scaler': self.scaler is not None,
            'models_dir': str(MODELS_DIR)
        }


# Global predictor instance
_predictor: Optional[AdvancedModelsPredictor] = None

def get_advanced_predictor() -> AdvancedModelsPredictor:
    """Get global advanced predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = AdvancedModelsPredictor()
    return _predictor

def advanced_predict(home_odds: float = 2.0, draw_odds: float = 3.3, 
                    away_odds: float = 3.5) -> Dict:
    """Quick prediction using advanced models"""
    predictor = get_advanced_predictor()
    predictions = predictor.predict_all_markets(home_odds, draw_odds, away_odds)
    
    return {
        'result': {
            'prediction': predictions['result'].prediction,
            'confidence': predictions['result'].confidence,
            'model': predictions['result'].model
        },
        'over_25': {
            'prediction': predictions['over_25'].prediction,
            'probability': predictions['over_25'].probability,
            'model': predictions['over_25'].model
        },
        'btts': {
            'prediction': predictions['btts'].prediction,
            'probability': predictions['btts'].probability,
            'model': predictions['btts'].model
        }
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = AdvancedModelsPredictor()
    print(f"Model info: {predictor.get_model_info()}")
    
    # Test prediction
    predictions = predictor.predict_all_markets(1.8, 3.5, 4.5)
    for market, pred in predictions.items():
        print(f"{market}: {pred.prediction} ({pred.confidence:.2%}) [{pred.model}]")
