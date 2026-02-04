"""
Prediction Confidence Calculator
Estimates confidence in predictions.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Calculates confidence scores for predictions.
    
    Factors:
    - Model agreement
    - Probability margin
    - Data quality
    - Historical accuracy
    """
    
    def __init__(self):
        self.historical_accuracy = {}
    
    def calculate(
        self,
        predictions: Dict,
        model_predictions: Dict[str, Dict] = None,
        data_quality: float = 1.0
    ) -> Dict:
        """
        Calculate confidence metrics.
        
        Args:
            predictions: Combined prediction
            model_predictions: Individual model predictions
            data_quality: Data quality score (0-1)
        """
        confidence = {
            'overall': 0.5,
            'factors': {}
        }
        
        if '1x2' in predictions:
            probs = predictions['1x2']
            
            # 1. Probability margin
            sorted_probs = sorted([probs['home'], probs['draw'], probs['away']], reverse=True)
            margin = sorted_probs[0] - sorted_probs[1]
            confidence['factors']['probability_margin'] = round(margin, 3)
            
            # 2. Model agreement (if multiple models)
            if model_predictions:
                agreement = self._calculate_agreement(model_predictions)
                confidence['factors']['model_agreement'] = round(agreement, 3)
            else:
                agreement = 0.5
            
            # 3. Max probability
            max_prob = max(probs.values())
            confidence['factors']['max_probability'] = round(max_prob, 3)
            
            # 4. Data quality
            confidence['factors']['data_quality'] = round(data_quality, 3)
            
            # Combine factors
            overall = (
                0.3 * (margin * 2) +  # Margin scaled
                0.3 * agreement +
                0.25 * max_prob +
                0.15 * data_quality
            )
            
            confidence['overall'] = round(min(max(overall, 0), 1), 3)
            confidence['level'] = self._get_level(confidence['overall'])
        
        return confidence
    
    def _calculate_agreement(
        self,
        model_predictions: Dict[str, Dict]
    ) -> float:
        """Calculate agreement between models."""
        predictions_1x2 = []
        
        for name, pred in model_predictions.items():
            if '1x2' in pred:
                probs = pred['1x2']
                winner = max(probs, key=probs.get)
                predictions_1x2.append((winner, probs[winner]))
        
        if len(predictions_1x2) < 2:
            return 0.5
        
        # Check if all agree on winner
        winners = [p[0] for p in predictions_1x2]
        mode_winner = max(set(winners), key=winners.count)
        agreement_rate = winners.count(mode_winner) / len(winners)
        
        # Also consider probability spread
        if all(w == mode_winner for w in winners):
            # Full agreement - check probability variance
            probs = [p[1] for p in predictions_1x2]
            variance = np.var(probs)
            agreement = 1.0 - min(variance * 10, 0.2)
        else:
            agreement = agreement_rate * 0.8
        
        return agreement
    
    def _get_level(self, score: float) -> str:
        """Convert confidence score to level."""
        if score >= 0.75:
            return 'very_high'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.45:
            return 'medium'
        elif score >= 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def update_accuracy(
        self,
        prediction_type: str,
        was_correct: bool
    ):
        """Update historical accuracy tracking."""
        if prediction_type not in self.historical_accuracy:
            self.historical_accuracy[prediction_type] = {'correct': 0, 'total': 0}
        
        self.historical_accuracy[prediction_type]['total'] += 1
        if was_correct:
            self.historical_accuracy[prediction_type]['correct'] += 1
    
    def get_historical_accuracy(self, prediction_type: str) -> float:
        """Get historical accuracy for a prediction type."""
        if prediction_type not in self.historical_accuracy:
            return 0.5
        
        stats = self.historical_accuracy[prediction_type]
        return stats['correct'] / stats['total'] if stats['total'] > 0 else 0.5


_calculator: Optional[ConfidenceCalculator] = None

def get_calculator() -> ConfidenceCalculator:
    global _calculator
    if _calculator is None:
        _calculator = ConfidenceCalculator()
    return _calculator


def calculate_confidence(predictions: Dict, model_predictions: Dict = None) -> Dict:
    """Quick function to calculate confidence."""
    return get_calculator().calculate(predictions, model_predictions)
