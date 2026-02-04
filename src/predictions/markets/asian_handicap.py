"""
Asian Handicap Predictor
Predicts Asian handicap markets.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class AsianHandicapPredictor:
    """
    Predicts Asian Handicap markets.
    
    Supports:
    - Whole line (0, -1, +1, etc.)
    - Half line (-0.5, +1.5, etc.)
    - Quarter line (-0.25, -0.75, etc.)
    """
    
    COMMON_LINES = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    
    def __init__(self):
        pass
    
    def predict(
        self,
        home_xg: float,
        away_xg: float,
        lines: list = None
    ) -> Dict:
        """
        Predict Asian Handicap probabilities.
        
        Args:
            home_xg: Expected home goals
            away_xg: Expected away goals
            lines: Lines to calculate (negative = home gives handicap)
        """
        lines = lines or self.COMMON_LINES
        predictions = {}
        
        for line in lines:
            result = self._calculate_ah_prob(home_xg, away_xg, line)
            predictions[str(line)] = result
        
        # Find fair line (close to 50/50)
        fair_line = min(
            lines,
            key=lambda l: abs(predictions[str(l)]['home'] - 0.5)
        )
        
        return {
            'lines': predictions,
            'fair_line': fair_line,
            'fair_line_probs': predictions[str(fair_line)]
        }
    
    def _calculate_ah_prob(
        self,
        home_xg: float,
        away_xg: float,
        line: float
    ) -> Dict:
        """Calculate AH probabilities for a specific line."""
        # For half lines (no push possible)
        if line % 1 == 0.5 or line % 1 == -0.5:
            return self._half_line(home_xg, away_xg, line)
        
        # For quarter lines
        elif line % 0.5 == 0.25 or line % 0.5 == 0.75:
            return self._quarter_line(home_xg, away_xg, line)
        
        # For whole lines (push possible)
        else:
            return self._whole_line(home_xg, away_xg, line)
    
    def _half_line(
        self,
        home_xg: float,
        away_xg: float,
        line: float
    ) -> Dict:
        """Half line - no push."""
        home_win = 0
        
        for h in range(10):
            for a in range(10):
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                margin = h - a
                
                if margin > -line:
                    home_win += prob
        
        return {
            'home': round(home_win, 4),
            'away': round(1 - home_win, 4),
            'push': 0
        }
    
    def _whole_line(
        self,
        home_xg: float,
        away_xg: float,
        line: float
    ) -> Dict:
        """Whole line - push possible."""
        home_win = push = 0
        
        for h in range(10):
            for a in range(10):
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                margin = h - a
                
                if margin > -line:
                    home_win += prob
                elif margin == -line:
                    push += prob
        
        away_win = 1 - home_win - push
        
        return {
            'home': round(home_win, 4),
            'away': round(away_win, 4),
            'push': round(push, 4)
        }
    
    def _quarter_line(
        self,
        home_xg: float,
        away_xg: float,
        line: float
    ) -> Dict:
        """Quarter line - split stake."""
        lower = np.floor(line * 2) / 2
        upper = np.ceil(line * 2) / 2
        
        lower_result = self._half_line(home_xg, away_xg, lower)
        upper_result = self._half_line(home_xg, away_xg, upper)
        
        return {
            'home': round((lower_result['home'] + upper_result['home']) / 2, 4),
            'away': round((lower_result['away'] + upper_result['away']) / 2, 4),
            'push': 0
        }
    
    def get_recommended_line(
        self,
        home_xg: float,
        away_xg: float,
        target_prob: float = 0.6
    ) -> Dict:
        """Find recommended line for target probability."""
        for line in sorted(self.COMMON_LINES, key=abs):
            prob = self._calculate_ah_prob(home_xg, away_xg, line)
            
            if prob['home'] >= target_prob:
                return {
                    'line': line,
                    'home_prob': prob['home'],
                    'target_met': True
                }
        
        return {'line': None, 'target_met': False}


_predictor: Optional[AsianHandicapPredictor] = None

def get_predictor() -> AsianHandicapPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AsianHandicapPredictor()
    return _predictor
