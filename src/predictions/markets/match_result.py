"""
Match Result Predictor
Specialized predictor for 1X2 market.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MatchResultPredictor:
    """
    Specialized 1X2 match result predictor.
    
    Combines multiple signals:
    - Statistical models
    - ML predictions
    - Form analysis
    """
    
    def __init__(self):
        self.model_weights = {
            'poisson': 0.25,
            'xgboost': 0.30,
            'form': 0.15,
            'elo': 0.20,
            'h2h': 0.10
        }
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        features: Dict = None,
        model_predictions: Dict[str, Dict] = None
    ) -> Dict:
        """
        Predict match result.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            features: Match features dict
            model_predictions: Predictions from various models
        """
        features = features or {}
        model_predictions = model_predictions or {}
        
        # Collect predictions from different sources
        predictions = []
        weights = []
        
        for model_name, pred in model_predictions.items():
            if '1x2' in pred:
                predictions.append([
                    pred['1x2'].get('home', 0.33),
                    pred['1x2'].get('draw', 0.33),
                    pred['1x2'].get('away', 0.34)
                ])
                weights.append(self.model_weights.get(model_name, 1.0))
        
        if not predictions:
            # Use basic analysis
            return self._basic_prediction(features)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        combined = np.average(predictions, axis=0, weights=weights)
        
        # Normalize
        combined = combined / combined.sum()
        
        # Determine confidence
        max_prob = max(combined)
        confidence = 'high' if max_prob > 0.55 else ('medium' if max_prob > 0.40 else 'low')
        
        result = ['H', 'D', 'A'][np.argmax(combined)]
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            '1x2': {
                'home': round(float(combined[0]), 4),
                'draw': round(float(combined[1]), 4),
                'away': round(float(combined[2]), 4)
            },
            'prediction': result,
            'confidence': confidence,
            'odds_implied': {
                'home': round(1 / combined[0], 2) if combined[0] > 0 else None,
                'draw': round(1 / combined[1], 2) if combined[1] > 0 else None,
                'away': round(1 / combined[2], 2) if combined[2] > 0 else None
            }
        }
    
    def _basic_prediction(self, features: Dict) -> Dict:
        """Basic prediction from features only."""
        home_strength = features.get('home_attack', 1.0)
        away_strength = features.get('away_attack', 1.0)
        
        home_adv = 0.15  # Home advantage
        
        # Simple strength-based calculation
        home_prob = 0.4 + home_adv + 0.1 * (home_strength - away_strength)
        away_prob = 0.3 - home_adv + 0.1 * (away_strength - home_strength)
        draw_prob = 1 - home_prob - away_prob
        
        # Clip to valid range
        home_prob = max(0.1, min(0.8, home_prob))
        away_prob = max(0.1, min(0.8, away_prob))
        draw_prob = max(0.1, 1 - home_prob - away_prob)
        
        # Renormalize
        total = home_prob + draw_prob + away_prob
        
        return {
            '1x2': {
                'home': round(home_prob / total, 4),
                'draw': round(draw_prob / total, 4),
                'away': round(away_prob / total, 4)
            },
            'method': 'basic'
        }
    
    def set_model_weight(self, model: str, weight: float):
        """Set weight for a model."""
        self.model_weights[model] = weight


_predictor: Optional[MatchResultPredictor] = None

def get_predictor() -> MatchResultPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MatchResultPredictor()
    return _predictor
