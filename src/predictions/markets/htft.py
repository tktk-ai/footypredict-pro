"""
HT/FT (Half-Time/Full-Time) Predictor
Predicts half-time and full-time result combinations.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class HTFTPredictor:
    """
    Predicts Half-Time/Full-Time markets.
    
    All 9 combinations:
    H/H, H/D, H/A, D/H, D/D, D/A, A/H, A/D, A/A
    """
    
    HTFT_COMBOS = [
        ('H', 'H'), ('H', 'D'), ('H', 'A'),
        ('D', 'H'), ('D', 'D'), ('D', 'A'),
        ('A', 'H'), ('A', 'D'), ('A', 'A')
    ]
    
    def __init__(self):
        self.ht_factor = 0.42  # First half scoring ratio
    
    def predict(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict:
        """Predict HT/FT probabilities."""
        # Half-time expected goals
        home_ht_xg = home_xg * self.ht_factor
        away_ht_xg = away_xg * self.ht_factor
        
        # Second half expected goals
        home_sh_xg = home_xg * (1 - self.ht_factor)
        away_sh_xg = away_xg * (1 - self.ht_factor)
        
        predictions = {}
        
        for ht, ft in self.HTFT_COMBOS:
            prob = self._calculate_htft_prob(
                ht, ft,
                home_ht_xg, away_ht_xg,
                home_sh_xg, away_sh_xg
            )
            predictions[f'{ht}/{ft}'] = round(prob, 4)
        
        # Find most likely
        best = max(predictions, key=predictions.get)
        
        return {
            'probabilities': predictions,
            'most_likely': best,
            'most_likely_prob': predictions[best]
        }
    
    def _calculate_htft_prob(
        self,
        ht_result: str,
        ft_result: str,
        home_ht_xg: float,
        away_ht_xg: float,
        home_sh_xg: float,
        away_sh_xg: float
    ) -> float:
        """Calculate probability for a specific HT/FT combo."""
        prob = 0
        
        # Iterate over possible HT scores
        for ht_h in range(6):
            for ht_a in range(6):
                # Check if HT score matches HT result
                if not self._matches_result(ht_h, ht_a, ht_result):
                    continue
                
                ht_prob = poisson.pmf(ht_h, home_ht_xg) * poisson.pmf(ht_a, away_ht_xg)
                
                # Iterate over possible SH scores
                for sh_h in range(6):
                    for sh_a in range(6):
                        ft_h = ht_h + sh_h
                        ft_a = ht_a + sh_a
                        
                        if not self._matches_result(ft_h, ft_a, ft_result):
                            continue
                        
                        sh_prob = poisson.pmf(sh_h, home_sh_xg) * poisson.pmf(sh_a, away_sh_xg)
                        prob += ht_prob * sh_prob
        
        return prob
    
    def _matches_result(
        self,
        home_goals: int,
        away_goals: int,
        result: str
    ) -> bool:
        """Check if score matches result."""
        if result == 'H':
            return home_goals > away_goals
        elif result == 'A':
            return home_goals < away_goals
        else:  # 'D'
            return home_goals == away_goals
    
    def predict_turnaround(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict:
        """Predict probability of result turnaround."""
        preds = self.predict(home_xg, away_xg)['probabilities']
        
        turnarounds = {
            'home_turnaround': (preds['D/H'] + preds['A/H']),  # Losing/drawing HT, win FT
            'away_turnaround': (preds['D/A'] + preds['H/A']),
            'any_turnaround': sum(preds[k] for k in ['H/A', 'H/D', 'D/H', 'D/A', 'A/H', 'A/D'])
        }
        
        return {k: round(v, 4) for k, v in turnarounds.items()}


_predictor: Optional[HTFTPredictor] = None

def get_predictor() -> HTFTPredictor:
    global _predictor
    if _predictor is None:
        _predictor = HTFTPredictor()
    return _predictor
