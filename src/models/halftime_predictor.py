"""
Halftime Predictor
===================

Predicts halftime-specific markets:
- First half result (1X2)
- First half over/under
- First half BTTS
"""

import numpy as np
from scipy.stats import poisson
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HalftimePredictor:
    """
    Predicts halftime-specific outcomes.
    """
    
    # Halftime goals are typically ~42% of full-time goals
    HT_GOALS_RATIO = 0.42
    
    # Halftime home advantage is stronger
    HT_HOME_ADVANTAGE = 1.15
    
    def __init__(self):
        self.team_stats = {}
    
    def predict_ht_result(
        self, 
        home_team: str, 
        away_team: str, 
        home_ft_xg: float = 1.45, 
        away_ft_xg: float = 1.15
    ) -> Dict[str, float]:
        """
        Predict halftime result probabilities.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_ft_xg: Full-time expected goals for home team
            away_ft_xg: Full-time expected goals for away team
        
        Returns:
            Dictionary with HT result probabilities
        """
        # Convert to halftime expected goals
        home_ht_xg = home_ft_xg * self.HT_GOALS_RATIO * self.HT_HOME_ADVANTAGE
        away_ht_xg = away_ft_xg * self.HT_GOALS_RATIO
        
        # Calculate outcome probabilities using Poisson
        max_goals = 4
        
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p_h = poisson.pmf(h, home_ht_xg)
                p_a = poisson.pmf(a, away_ht_xg)
                prob = p_h * p_a
                
                if h > a:
                    home_win_prob += prob
                elif h < a:
                    away_win_prob += prob
                else:
                    draw_prob += prob
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        
        return {
            'home_win': home_win_prob / total * 100,
            'draw': draw_prob / total * 100,
            'away_win': away_win_prob / total * 100,
        }
    
    def predict_ht_over_under(
        self, 
        home_team: str, 
        away_team: str, 
        home_ft_xg: float = 1.45, 
        away_ft_xg: float = 1.15
    ) -> Dict[str, float]:
        """
        Predict halftime over/under probabilities.
        """
        home_ht_xg = home_ft_xg * self.HT_GOALS_RATIO
        away_ht_xg = away_ft_xg * self.HT_GOALS_RATIO
        total_ht_xg = home_ht_xg + away_ht_xg
        
        # Calculate probabilities for different totals
        over_05 = 1 - poisson.pmf(0, total_ht_xg)
        over_15 = 1 - poisson.cdf(1, total_ht_xg)
        over_25 = 1 - poisson.cdf(2, total_ht_xg)
        
        return {
            'over_05': over_05 * 100,
            'under_05': (1 - over_05) * 100,
            'over_15': over_15 * 100,
            'under_15': (1 - over_15) * 100,
            'over_25': over_25 * 100,
            'under_25': (1 - over_25) * 100,
        }
    
    def predict_ht_btts(
        self, 
        home_team: str, 
        away_team: str, 
        home_ft_xg: float = 1.45, 
        away_ft_xg: float = 1.15
    ) -> Dict[str, float]:
        """
        Predict halftime BTTS probability.
        """
        home_ht_xg = home_ft_xg * self.HT_GOALS_RATIO
        away_ht_xg = away_ft_xg * self.HT_GOALS_RATIO
        
        # P(home scores) = 1 - P(0 goals)
        home_scores = 1 - poisson.pmf(0, home_ht_xg)
        away_scores = 1 - poisson.pmf(0, away_ht_xg)
        
        # P(both score) = P(home scores) * P(away scores) assuming independence
        btts_yes = home_scores * away_scores
        
        return {
            'yes': btts_yes * 100,
            'no': (1 - btts_yes) * 100,
        }
    
    def predict(
        self, 
        home_team: str, 
        away_team: str, 
        home_ft_xg: float = 1.45, 
        away_ft_xg: float = 1.15
    ) -> Dict:
        """
        Get complete halftime prediction.
        """
        result = self.predict_ht_result(home_team, away_team, home_ft_xg, away_ft_xg)
        over_under = self.predict_ht_over_under(home_team, away_team, home_ft_xg, away_ft_xg)
        btts = self.predict_ht_btts(home_team, away_team, home_ft_xg, away_ft_xg)
        
        # Determine best picks
        best_picks = []
        
        # HT result
        best_result = max(result.items(), key=lambda x: x[1])
        if best_result[1] > 40:
            best_picks.append({
                'market': 'HT Result',
                'prediction': best_result[0].replace('_', ' ').title(),
                'probability': best_result[1],
            })
        
        # HT Over 0.5
        if over_under['over_05'] > 60:
            best_picks.append({
                'market': 'HT Over 0.5',
                'prediction': 'Yes',
                'probability': over_under['over_05'],
            })
        
        # HT BTTS
        if btts['yes'] > 30:
            best_picks.append({
                'market': 'HT BTTS',
                'prediction': 'Yes',
                'probability': btts['yes'],
            })
        elif btts['no'] > 70:
            best_picks.append({
                'market': 'HT BTTS',
                'prediction': 'No',
                'probability': btts['no'],
            })
        
        # Sort by probability
        best_picks.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'ht_result': result,
            'ht_over_under': over_under,
            'ht_btts': btts,
            'best_picks': best_picks[:3],
        }


# Singleton
_predictor = None


def get_halftime_predictor() -> HalftimePredictor:
    """Get singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = HalftimePredictor()
    return _predictor


if __name__ == "__main__":
    predictor = HalftimePredictor()
    
    result = predictor.predict('Liverpool', 'Manchester City', 1.6, 1.4)
    
    print(f"\n⚽ {result['home_team']} vs {result['away_team']} - HALFTIME PREDICTIONS")
    
    print("\n📊 HT Result:")
    for k, v in result['ht_result'].items():
        print(f"  {k}: {v:.1f}%")
    
    print("\n📊 HT Over/Under:")
    for k, v in result['ht_over_under'].items():
        print(f"  {k}: {v:.1f}%")
    
    print("\n📊 HT BTTS:")
    for k, v in result['ht_btts'].items():
        print(f"  {k}: {v:.1f}%")
    
    print("\n🎯 Best Picks:")
    for pick in result['best_picks']:
        print(f"  {pick['market']}: {pick['prediction']} ({pick['probability']:.1f}%)")
