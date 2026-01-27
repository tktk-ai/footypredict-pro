"""
Correct Score Predictor
Predicts exact final scores.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class CorrectScorePredictor:
    """
    Predicts exact match scores.
    
    Uses Poisson distribution with modifications for:
    - Score correlations
    - Low-scoring bias
    """
    
    MAX_GOALS = 8
    
    def __init__(self, rho: float = -0.1):
        self.rho = rho  # Score correlation parameter
    
    def predict(
        self,
        home_xg: float,
        away_xg: float,
        use_correlation: bool = True
    ) -> Dict:
        """
        Predict correct score probabilities.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            use_correlation: Apply score correlation adjustment
        """
        # Base Poisson probabilities
        score_probs = {}
        
        for h in range(self.MAX_GOALS + 1):
            for a in range(self.MAX_GOALS + 1):
                base_prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                
                if use_correlation:
                    # Dixon-Coles correlation adjustment for low scores
                    if h == 0 and a == 0:
                        base_prob *= (1 - self.rho * home_xg * away_xg)
                    elif h == 0 and a == 1:
                        base_prob *= (1 + self.rho * home_xg)
                    elif h == 1 and a == 0:
                        base_prob *= (1 + self.rho * away_xg)
                    elif h == 1 and a == 1:
                        base_prob *= (1 - self.rho)
                
                score_probs[(h, a)] = max(0, base_prob)
        
        # Normalize
        total = sum(score_probs.values())
        if total > 0:
            score_probs = {k: v / total for k, v in score_probs.items()}
        
        # Sort by probability
        sorted_scores = sorted(
            score_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top predictions
        top_10 = sorted_scores[:10]
        
        return {
            'top_scores': {
                f"{s[0]}-{s[1]}": round(p, 4)
                for s, p in top_10
            },
            'most_likely': f"{top_10[0][0][0]}-{top_10[0][0][1]}",
            'most_likely_prob': round(top_10[0][1], 4),
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2)
        }
    
    def get_score_probability(
        self,
        home_xg: float,
        away_xg: float,
        home_goals: int,
        away_goals: int
    ) -> float:
        """Get probability for a specific score."""
        return poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
    
    def get_score_groups(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict:
        """Get probabilities for score groups."""
        predictions = self.predict(home_xg, away_xg, use_correlation=True)
        
        # Score groups
        low_scoring = 0  # 0-0, 1-0, 0-1, 1-1
        high_scoring = 0  # 4+ total goals
        draws = 0
        home_wins = 0
        away_wins = 0
        
        for h in range(self.MAX_GOALS + 1):
            for a in range(self.MAX_GOALS + 1):
                prob = self.get_score_probability(home_xg, away_xg, h, a)
                
                if h + a <= 2 and h <= 1 and a <= 1:
                    low_scoring += prob
                if h + a >= 4:
                    high_scoring += prob
                if h == a:
                    draws += prob
                elif h > a:
                    home_wins += prob
                else:
                    away_wins += prob
        
        return {
            'low_scoring': round(low_scoring, 4),
            'high_scoring': round(high_scoring, 4),
            'any_draw': round(draws, 4),
            'home_win': round(home_wins, 4),
            'away_win': round(away_wins, 4)
        }


_predictor: Optional[CorrectScorePredictor] = None

def get_predictor() -> CorrectScorePredictor:
    global _predictor
    if _predictor is None:
        _predictor = CorrectScorePredictor()
    return _predictor
