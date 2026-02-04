"""
BTTS (Both Teams To Score) Predictor
Predicts whether both teams will score.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class BTTSPredictor:
    """
    Predicts Both Teams To Score markets.
    
    Uses:
    - Poisson model
    - Historical BTTS rates
    - Form-based adjustments
    """
    
    def __init__(self):
        self.default_btts_rate = 0.52  # League average
    
    def predict_from_xg(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict:
        """
        Predict BTTS from expected goals.
        """
        # P(BTTS) = P(home >= 1) * P(away >= 1)
        home_scores = 1 - poisson.pmf(0, home_xg)
        away_scores = 1 - poisson.pmf(0, away_xg)
        
        btts_yes = home_scores * away_scores
        btts_no = 1 - btts_yes
        
        return {
            'btts_yes': round(btts_yes, 4),
            'btts_no': round(btts_no, 4),
            'home_scores': round(home_scores, 4),
            'away_scores': round(away_scores, 4),
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2)
        }
    
    def predict_from_form(
        self,
        home_btts_rate: float,
        away_btts_rate: float,
        home_score_rate: float,
        away_score_rate: float,
        home_concede_rate: float,
        away_concede_rate: float
    ) -> Dict:
        """
        Predict BTTS from team form.
        """
        # Combined scoring probability
        home_scores_prob = (home_score_rate + away_concede_rate) / 2
        away_scores_prob = (away_score_rate + home_concede_rate) / 2
        
        # BTTS using form
        btts_form = home_scores_prob * away_scores_prob
        
        # Adjust with historical BTTS rates
        btts_historical = (home_btts_rate + away_btts_rate) / 2
        
        # Weighted combination
        btts_combined = 0.6 * btts_form + 0.4 * btts_historical
        
        return {
            'btts_yes': round(btts_combined, 4),
            'btts_no': round(1 - btts_combined, 4),
            'method': 'form_based'
        }
    
    def predict_combined(
        self,
        home_xg: float,
        away_xg: float,
        home_btts_rate: float = None,
        away_btts_rate: float = None
    ) -> Dict:
        """Combine xG and form-based predictions."""
        xg_pred = self.predict_from_xg(home_xg, away_xg)
        
        if home_btts_rate is not None and away_btts_rate is not None:
            form_btts = (home_btts_rate + away_btts_rate) / 2
            combined = 0.7 * xg_pred['btts_yes'] + 0.3 * form_btts
        else:
            combined = xg_pred['btts_yes']
        
        return {
            'btts_yes': round(combined, 4),
            'btts_no': round(1 - combined, 4),
            'xg_btts': xg_pred['btts_yes'],
            'form_btts': (home_btts_rate + away_btts_rate) / 2 if home_btts_rate else None
        }


_predictor: Optional[BTTSPredictor] = None

def get_predictor() -> BTTSPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BTTSPredictor()
    return _predictor
