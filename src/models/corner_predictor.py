"""
Corner Predictor
=================

Predicts corner-related markets:
- Total corners over/under
- Corner handicap
- Team corner totals
"""

import numpy as np
from scipy.stats import poisson, norm
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerPredictor:
    """
    Predicts corner markets using statistical models.
    """
    
    # Average corners per game (based on historical data)
    LEAGUE_AVERAGES = {
        'Premier League': {'home': 5.2, 'away': 4.3, 'total': 9.5},
        'Bundesliga': {'home': 5.0, 'away': 4.1, 'total': 9.1},
        'La Liga': {'home': 5.4, 'away': 4.0, 'total': 9.4},
        'Serie A': {'home': 5.1, 'away': 4.2, 'total': 9.3},
        'Ligue 1': {'home': 4.8, 'away': 4.0, 'total': 8.8},
        'Eredivisie': {'home': 5.5, 'away': 4.5, 'total': 10.0},
        'default': {'home': 5.1, 'away': 4.2, 'total': 9.3},
    }
    
    # Standard deviation
    CORNER_STD = 3.0
    
    def __init__(self):
        self.team_stats = {}
    
    def _get_league_avg(self, league: str) -> Dict[str, float]:
        """Get league-specific corner averages."""
        league_lower = league.lower() if league else ''
        
        for name, stats in self.LEAGUE_AVERAGES.items():
            if name.lower() in league_lower or league_lower in name.lower():
                return stats
        
        return self.LEAGUE_AVERAGES['default']
    
    def predict_total_corners(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = ''
    ) -> Dict[str, float]:
        """
        Predict total corners over/under.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
        
        Returns:
            Dictionary with over/under probabilities
        """
        avg = self._get_league_avg(league)
        expected_total = avg['total']
        
        # Use normal distribution for total corners
        over_75 = 1 - norm.cdf(7.5, expected_total, self.CORNER_STD)
        over_85 = 1 - norm.cdf(8.5, expected_total, self.CORNER_STD)
        over_95 = 1 - norm.cdf(9.5, expected_total, self.CORNER_STD)
        over_105 = 1 - norm.cdf(10.5, expected_total, self.CORNER_STD)
        over_115 = 1 - norm.cdf(11.5, expected_total, self.CORNER_STD)
        over_125 = 1 - norm.cdf(12.5, expected_total, self.CORNER_STD)
        
        return {
            'expected_total': expected_total,
            'over_75': over_75 * 100,
            'under_75': (1 - over_75) * 100,
            'over_85': over_85 * 100,
            'under_85': (1 - over_85) * 100,
            'over_95': over_95 * 100,
            'under_95': (1 - over_95) * 100,
            'over_105': over_105 * 100,
            'under_105': (1 - over_105) * 100,
            'over_115': over_115 * 100,
            'under_115': (1 - over_115) * 100,
            'over_125': over_125 * 100,
            'under_125': (1 - over_125) * 100,
        }
    
    def predict_corner_handicap(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = ''
    ) -> Dict[str, float]:
        """
        Predict corner handicap outcomes.
        
        Home team typically wins corners at home.
        """
        avg = self._get_league_avg(league)
        
        expected_diff = avg['home'] - avg['away']  # Home advantage in corners
        diff_std = 2.5  # Standard deviation of corner difference
        
        # Handicap probabilities
        results = {}
        
        for handicap in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            # Home covers handicap if (home_corners - away_corners) > handicap
            home_covers = 1 - norm.cdf(handicap, expected_diff, diff_std)
            
            results[f'home_{handicap:.1f}'] = home_covers * 100
            results[f'away_{-handicap:.1f}'] = (1 - home_covers) * 100
        
        return {
            'expected_difference': expected_diff,
            **results
        }
    
    def predict_team_corners(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = ''
    ) -> Dict:
        """
        Predict individual team corner totals.
        """
        avg = self._get_league_avg(league)
        
        home_expected = avg['home']
        away_expected = avg['away']
        
        std = 2.0  # Individual team corner std
        
        home_over_35 = 1 - norm.cdf(3.5, home_expected, std)
        home_over_45 = 1 - norm.cdf(4.5, home_expected, std)
        home_over_55 = 1 - norm.cdf(5.5, home_expected, std)
        
        away_over_25 = 1 - norm.cdf(2.5, away_expected, std)
        away_over_35 = 1 - norm.cdf(3.5, away_expected, std)
        away_over_45 = 1 - norm.cdf(4.5, away_expected, std)
        
        return {
            'home_team': {
                'expected': home_expected,
                'over_35': home_over_35 * 100,
                'over_45': home_over_45 * 100,
                'over_55': home_over_55 * 100,
            },
            'away_team': {
                'expected': away_expected,
                'over_25': away_over_25 * 100,
                'over_35': away_over_35 * 100,
                'over_45': away_over_45 * 100,
            }
        }
    
    def predict(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = ''
    ) -> Dict:
        """
        Get complete corner prediction.
        """
        total = self.predict_total_corners(home_team, away_team, league)
        handicap = self.predict_corner_handicap(home_team, away_team, league)
        team = self.predict_team_corners(home_team, away_team, league)
        
        # Best picks
        best_picks = []
        
        # Total corners
        if total['over_85'] > 60:
            best_picks.append({
                'market': 'Total Corners Over 8.5',
                'prediction': 'Over',
                'probability': total['over_85'],
            })
        elif total['under_105'] > 60:
            best_picks.append({
                'market': 'Total Corners Under 10.5',
                'prediction': 'Under',
                'probability': total['under_105'],
            })
        
        # Home team corners
        if team['home_team']['over_45'] > 55:
            best_picks.append({
                'market': f'{home_team} Over 4.5 Corners',
                'prediction': 'Over',
                'probability': team['home_team']['over_45'],
            })
        
        best_picks.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'total_corners': total,
            'corner_handicap': handicap,
            'team_corners': team,
            'best_picks': best_picks[:3],
        }


# Singleton
_predictor = None


def get_corner_predictor() -> CornerPredictor:
    """Get singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CornerPredictor()
    return _predictor


if __name__ == "__main__":
    predictor = CornerPredictor()
    
    result = predictor.predict('Arsenal', 'Chelsea', 'Premier League')
    
    print(f"\n⚽ {result['home_team']} vs {result['away_team']} - CORNER PREDICTIONS")
    
    print(f"\n📊 Total Corners (Expected: {result['total_corners']['expected_total']:.1f}):")
    print(f"  Over 8.5: {result['total_corners']['over_85']:.1f}%")
    print(f"  Over 9.5: {result['total_corners']['over_95']:.1f}%")
    print(f"  Over 10.5: {result['total_corners']['over_105']:.1f}%")
    
    print(f"\n📊 Team Corners:")
    print(f"  {result['home_team']}: {result['team_corners']['home_team']['expected']:.1f} expected")
    print(f"  {result['away_team']}: {result['team_corners']['away_team']['expected']:.1f} expected")
    
    print("\n🎯 Best Picks:")
    for pick in result['best_picks']:
        print(f"  {pick['market']}: {pick['prediction']} ({pick['probability']:.1f}%)")
