"""
Unit tests for the Football Prediction System

Tests for:
- ELO rating calculations
- Prediction probability validation
- H2H analyzer
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predictor import PredictionEngine, ELORatingSystem
from src.advanced_predictions import HeadToHeadAnalyzer
from src.goals_predictor import PoissonGoalPredictor


class TestELORatingSystem:
    """Tests for the ELO rating system"""
    
    def setup_method(self):
        self.elo = ELORatingSystem()
    
    def test_default_rating(self):
        """New teams should get default rating of 1500"""
        rating = self.elo.get_rating("UnknownTeam")
        assert rating == 1500
    
    def test_set_and_get_rating(self):
        """Should be able to set and retrieve rating"""
        self.elo.set_rating("TestTeam", 1800)
        assert self.elo.get_rating("TestTeam") == 1800
    
    def test_predict_returns_valid_probabilities(self):
        """Predictions should sum to approximately 1.0"""
        home_win, draw, away_win = self.elo.predict(1600, 1400)
        
        total = home_win + draw + away_win
        assert 0.99 <= total <= 1.01, f"Probabilities sum to {total}, expected ~1.0"
    
    def test_higher_elo_favored(self):
        """Team with higher ELO should have higher win probability"""
        home_win, draw, away_win = self.elo.predict(1700, 1300)
        assert home_win > away_win, "Higher ELO team should be favored"
    
    def test_equal_elo_balanced(self):
        """Equal ELO should result in balanced probabilities with home advantage"""
        home_win, draw, away_win = self.elo.predict(1500, 1500)
        
        # Home team should still have slight advantage
        assert home_win >= away_win, "Home team should have at least equal probability"


class TestPredictionEngine:
    """Tests for the prediction engine"""
    
    def setup_method(self):
        self.engine = PredictionEngine()
    
    def test_predict_returns_valid_result(self):
        """Prediction should return valid PredictionResult"""
        result = self.engine.predict_match("Bayern", "Dortmund", "bundesliga")
        
        assert hasattr(result, 'home_win_prob')
        assert hasattr(result, 'draw_prob')
        assert hasattr(result, 'away_win_prob')
        assert hasattr(result, 'predicted_outcome')
        assert hasattr(result, 'confidence')
    
    def test_probabilities_sum_to_one(self):
        """Prediction probabilities should sum to 1.0"""
        result = self.engine.predict_match("Liverpool", "Arsenal", "premier_league")
        
        total = result.home_win_prob + result.draw_prob + result.away_win_prob
        assert 0.99 <= total <= 1.01, f"Probabilities sum to {total}"
    
    def test_probabilities_in_range(self):
        """Each probability should be between 0 and 1"""
        result = self.engine.predict_match("Real Madrid", "Barcelona", "la_liga")
        
        assert 0 <= result.home_win_prob <= 1
        assert 0 <= result.draw_prob <= 1
        assert 0 <= result.away_win_prob <= 1
    
    def test_predicted_outcome_valid(self):
        """Predicted outcome should be Home Win, Draw, or Away Win"""
        result = self.engine.predict_match("Inter", "Juventus", "serie_a")
        
        assert result.predicted_outcome in ['Home Win', 'Draw', 'Away Win']


class TestHeadToHeadAnalyzer:
    """Tests for the H2H analyzer"""
    
    def setup_method(self):
        self.h2h = HeadToHeadAnalyzer()
    
    def test_known_matchup_found(self):
        """Known matchup should return data"""
        result = self.h2h.get_full_h2h("Bayern", "Dortmund")
        
        assert result['found'] == True
        assert result['total_matches'] > 0
    
    def test_unknown_matchup_not_found(self):
        """Unknown matchup should return found=False"""
        result = self.h2h.get_full_h2h("UnknownTeam1", "UnknownTeam2")
        
        assert result['found'] == False
    
    def test_h2h_factor_in_range(self):
        """H2H factors should be capped at ±0.1"""
        factors = self.h2h.get_h2h_factor("Bayern", "Dortmund")
        
        for key, value in factors.items():
            assert -0.1 <= value <= 0.1, f"Factor {key}={value} out of range"
    
    def test_fuzzy_matching(self):
        """Fuzzy matching should work for partial team names"""
        result = self.h2h.get_full_h2h("Bayern München", "Borussia Dortmund")
        
        assert result['found'] == True
    
    def test_last_5_matches_structure(self):
        """Last 5 matches should have correct structure"""
        result = self.h2h.get_full_h2h("Bayern", "Dortmund")
        
        if result['found']:
            for match in result['last_5_matches']:
                assert 'home_score' in match
                assert 'away_score' in match
                assert 'result' in match
                assert match['result'] in ['H', 'D', 'A']


class TestGoalPredictor:
    """Tests for goal predictions"""
    
    def setup_method(self):
        self.predictor = PoissonGoalPredictor()
    
    def test_expected_goals_positive(self):
        """Expected goals should be positive"""
        result = self.predictor.predict_goals("Bayern", "Dortmund", "bundesliga")
        
        assert result.home_xg > 0
        assert result.away_xg > 0
        assert result.total_xg > 0
    
    def test_over_under_probabilities_valid(self):
        """Over/Under probabilities should be between 0 and 1"""
        result = self.predictor.predict_goals("Liverpool", "Arsenal", "premier_league")
        
        assert 0 <= result.over_2_5 <= 1, f"Over 2.5 = {result.over_2_5} out of range"
        assert 0 <= result.over_1_5 <= 1, f"Over 1.5 = {result.over_1_5} out of range"
        assert 0 <= result.over_3_5 <= 1, f"Over 3.5 = {result.over_3_5} out of range"
    
    def test_btts_probabilities_valid(self):
        """BTTS probabilities should sum to 1"""
        result = self.predictor.predict_goals("Real Madrid", "Barcelona", "la_liga")
        
        total = result.btts_yes + result.btts_no
        assert 0.99 <= total <= 1.01, f"BTTS probabilities sum to {total}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
