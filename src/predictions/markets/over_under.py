"""
Over/Under Goals Predictor
Predicts over/under goal totals.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class OverUnderPredictor:
    """
    Predicts over/under goal markets.
    
    Supports:
    - O/U 0.5 to 5.5
    - Team totals
    - Half totals
    """
    
    LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    def __init__(self):
        pass
    
    def predict(
        self,
        home_xg: float,
        away_xg: float,
        lines: list = None
    ) -> Dict:
        """
        Predict over/under probabilities.
        
        Args:
            home_xg: Expected home goals
            away_xg: Expected away goals
            lines: Lines to calculate (default: 0.5 to 5.5)
        """
        lines = lines or self.LINES
        total_xg = home_xg + away_xg
        
        predictions = {}
        
        # Calculate cumulative Poisson
        for line in lines:
            under_prob = 0
            for goals in range(int(line) + 1):
                under_prob += self._poisson_total_prob(goals, home_xg, away_xg)
            
            over_prob = 1 - under_prob
            
            predictions[f'over_{line}'] = round(over_prob, 4)
            predictions[f'under_{line}'] = round(under_prob, 4)
        
        # Most likely total
        total_probs = {}
        for t in range(10):
            total_probs[t] = self._poisson_total_prob(t, home_xg, away_xg)
        
        most_likely = max(total_probs, key=total_probs.get)
        
        return {
            'lines': predictions,
            'expected_total': round(total_xg, 2),
            'most_likely_total': most_likely,
            'most_likely_prob': round(total_probs[most_likely], 4),
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2)
        }
    
    def _poisson_total_prob(
        self,
        total: int,
        home_xg: float,
        away_xg: float
    ) -> float:
        """Calculate probability of exactly 'total' goals."""
        prob = 0
        for h in range(total + 1):
            a = total - h
            prob += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
        return prob
    
    def predict_team_total(
        self,
        team: str,
        xg: float,
        lines: list = None
    ) -> Dict:
        """Predict team over/under."""
        lines = lines or [0.5, 1.5, 2.5]
        
        predictions = {}
        for line in lines:
            under = sum(poisson.pmf(g, xg) for g in range(int(line) + 1))
            predictions[f'{team}_over_{line}'] = round(1 - under, 4)
            predictions[f'{team}_under_{line}'] = round(under, 4)
        
        return predictions
    
    def predict_first_half(
        self,
        home_xg: float,
        away_xg: float,
        ht_factor: float = 0.42
    ) -> Dict:
        """Predict first half over/under."""
        # Typical first half is ~42% of total xG
        ht_home = home_xg * ht_factor
        ht_away = away_xg * ht_factor
        
        return self.predict(ht_home, ht_away, lines=[0.5, 1.5, 2.5])


_predictor: Optional[OverUnderPredictor] = None

def get_predictor() -> OverUnderPredictor:
    global _predictor
    if _predictor is None:
        _predictor = OverUnderPredictor()
    return _predictor
