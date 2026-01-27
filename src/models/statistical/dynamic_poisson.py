"""
Dynamic Poisson Model
Time-varying Poisson model for goal predictions.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
from scipy.stats import poisson
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class DynamicPoissonModel:
    """
    Dynamic Poisson model with time-varying attack/defense ratings.
    
    Features:
    - Exponential decay for older matches
    - Time-varying parameters
    - Match importance weighting
    """
    
    def __init__(
        self,
        decay_rate: float = 0.002,
        home_advantage: float = 0.25
    ):
        self.decay_rate = decay_rate
        self.home_advantage = home_advantage
        self.attack_ratings = {}
        self.defense_ratings = {}
        self.league_avg_goals = 2.7
        self.is_fitted = False
        
    def fit(self, matches: list) -> 'DynamicPoissonModel':
        """
        Fit the model to historical matches.
        
        Args:
            matches: List of match dicts with home_team, away_team, 
                    home_goals, away_goals, days_ago
        """
        teams = set()
        for match in matches:
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        # Initialize ratings
        for team in teams:
            self.attack_ratings[team] = 1.0
            self.defense_ratings[team] = 1.0
        
        # Calculate league average
        total_goals = sum(m['home_goals'] + m['away_goals'] for m in matches)
        self.league_avg_goals = total_goals / (2 * len(matches)) if matches else 2.7
        
        # Iteratively update ratings
        for _ in range(50):  # Iterations
            for match in matches:
                weight = np.exp(-self.decay_rate * match.get('days_ago', 0))
                
                home = match['home_team']
                away = match['away_team']
                home_goals = match['home_goals']
                away_goals = match['away_goals']
                
                # Expected goals
                exp_home = (self.attack_ratings[home] * 
                           self.defense_ratings[away] * 
                           self.league_avg_goals * 
                           np.exp(self.home_advantage))
                
                exp_away = (self.attack_ratings[away] * 
                           self.defense_ratings[home] * 
                           self.league_avg_goals)
                
                # Update ratings
                lr = 0.01 * weight
                
                self.attack_ratings[home] *= (1 + lr * (home_goals / max(exp_home, 0.1) - 1))
                self.attack_ratings[away] *= (1 + lr * (away_goals / max(exp_away, 0.1) - 1))
                
                self.defense_ratings[home] *= (1 + lr * (away_goals / max(exp_away, 0.1) - 1))
                self.defense_ratings[away] *= (1 + lr * (home_goals / max(exp_home, 0.1) - 1))
        
        # Normalize ratings
        mean_attack = np.mean(list(self.attack_ratings.values()))
        mean_defense = np.mean(list(self.defense_ratings.values()))
        
        for team in teams:
            self.attack_ratings[team] /= mean_attack
            self.defense_ratings[team] /= mean_defense
        
        self.is_fitted = True
        logger.info(f"Fitted DynamicPoisson on {len(matches)} matches, {len(teams)} teams")
        
        return self
    
    def predict_xg(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float]:
        """Predict expected goals for a match."""
        home_attack = self.attack_ratings.get(home_team, 1.0)
        home_defense = self.defense_ratings.get(home_team, 1.0)
        away_attack = self.attack_ratings.get(away_team, 1.0)
        away_defense = self.defense_ratings.get(away_team, 1.0)
        
        home_xg = (home_attack * away_defense * 
                  self.league_avg_goals * 
                  np.exp(self.home_advantage))
        
        away_xg = away_attack * home_defense * self.league_avg_goals
        
        return home_xg, away_xg
    
    def predict_score_probability(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int
    ) -> float:
        """Predict probability of a specific score."""
        home_xg, away_xg = self.predict_xg(home_team, away_team)
        
        return (poisson.pmf(home_goals, home_xg) * 
                poisson.pmf(away_goals, away_xg))
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 8
    ) -> Dict:
        """Full match prediction with all markets."""
        home_xg, away_xg = self.predict_xg(home_team, away_team)
        
        # Score matrix
        score_probs = {}
        home_win = draw = away_win = 0.0
        btts = 0.0
        over_1_5 = over_2_5 = over_3_5 = 0.0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = (poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg))
                score_probs[(h, a)] = prob
                
                if h > a:
                    home_win += prob
                elif h < a:
                    away_win += prob
                else:
                    draw += prob
                
                if h > 0 and a > 0:
                    btts += prob
                
                if h + a > 1.5:
                    over_1_5 += prob
                if h + a > 2.5:
                    over_2_5 += prob
                if h + a > 3.5:
                    over_3_5 += prob
        
        # Top correct scores
        sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            '1x2': {
                'home': round(home_win, 4),
                'draw': round(draw, 4),
                'away': round(away_win, 4)
            },
            'btts': {
                'yes': round(btts, 4),
                'no': round(1 - btts, 4)
            },
            'over_under': {
                'over_1.5': round(over_1_5, 4),
                'over_2.5': round(over_2_5, 4),
                'over_3.5': round(over_3_5, 4)
            },
            'correct_scores': {
                f"{s[0]}-{s[1]}": round(p, 4)
                for s, p in sorted_scores
            }
        }
    
    def get_team_ratings(self, team: str) -> Dict:
        """Get ratings for a specific team."""
        return {
            'team': team,
            'attack': round(self.attack_ratings.get(team, 1.0), 3),
            'defense': round(self.defense_ratings.get(team, 1.0), 3)
        }


# Global instance
_model: Optional[DynamicPoissonModel] = None


def get_model() -> DynamicPoissonModel:
    """Get or create Dynamic Poisson model."""
    global _model
    if _model is None:
        _model = DynamicPoissonModel()
    return _model


def predict_match(home: str, away: str) -> Dict:
    """Quick function to predict match."""
    return get_model().predict_match(home, away)
