"""
Mock Models - Fallback predictors when HuggingFace is unavailable

These provide reasonable predictions using statistical methods
until real ML models are loaded.
"""

import random
from typing import Dict, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class MockPrediction:
    """Prediction output from mock models"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float
    model_name: str
    
    def to_dict(self) -> Dict:
        return {
            'home_win_prob': self.home_win_prob,
            'draw_prob': self.draw_prob,
            'away_win_prob': self.away_win_prob,
            'confidence': self.confidence,
            'model_name': self.model_name
        }


class MockPodosPredictor:
    """
    Mock version of Podos transformer.
    Uses Elo-like calculations for predictions.
    """
    
    def __init__(self):
        self.name = "mock_podos"
        self.base_elo = 1500
        self.home_advantage = 100
        
        # Simple team strength database (can be expanded)
        self.team_strengths = {
            # Top teams
            'Bayern': 1900, 'Bayern Munich': 1900, 'FC Bayern Munich': 1900,
            'Real Madrid': 1880, 'Barcelona': 1860, 'FC Barcelona': 1860,
            'Man City': 1870, 'Manchester City': 1870,
            'Liverpool': 1840,
            'PSG': 1820, 'Paris Saint-Germain': 1820,
            'Juventus': 1780,
            'Inter Milan': 1770, 'Inter': 1770,
            'Dortmund': 1760, 'Borussia Dortmund': 1760,
            'Arsenal': 1800,
            'Chelsea': 1750,
            
            # Mid-table
            'Tottenham': 1700,
            'Atletico Madrid': 1720,
            'Sevilla': 1680,
            'Roma': 1660,
            'Napoli': 1750,
            'AC Milan': 1720,
            'Lyon': 1650,
            'Marseille': 1640,
            
            # Lower teams have default
        }
    
    def get_elo(self, team: str) -> float:
        """Get Elo rating for a team"""
        # Try exact match first
        if team in self.team_strengths:
            return self.team_strengths[team]
        
        # Try partial match
        team_lower = team.lower()
        for known_team, elo in self.team_strengths.items():
            if known_team.lower() in team_lower or team_lower in known_team.lower():
                return elo
        
        # Default for unknown teams
        return self.base_elo
    
    def predict(self, home_team: str, away_team: str, **kwargs) -> MockPrediction:
        """
        Predict match outcome using Elo-based calculation.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            **kwargs: Additional features (odds, form, etc.)
            
        Returns:
            MockPrediction with probabilities
        """
        home_elo = self.get_elo(home_team) + self.home_advantage
        away_elo = self.get_elo(away_team)
        
        # Elo expected score formula
        elo_diff = home_elo - away_elo
        
        # Convert to probabilities using logistic function
        # Higher diff = higher home win probability
        home_advantage_factor = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Adjust for draws (football has high draw rate ~25%)
        draw_base = 0.25
        
        # Scale home/away probs
        home_win_prob = home_advantage_factor * (1 - draw_base)
        away_win_prob = (1 - home_advantage_factor) * (1 - draw_base)
        
        # Increase draw probability for close matches
        closeness = 1 - abs(home_win_prob - away_win_prob)
        draw_prob = draw_base + (closeness * 0.1)
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Confidence based on Elo difference
        confidence = min(0.95, 0.5 + abs(elo_diff) / 800)
        
        # Add small random variance to simulate model uncertainty
        variance = random.uniform(-0.03, 0.03)
        home_win_prob = max(0.05, min(0.90, home_win_prob + variance))
        
        # Re-normalize after variance
        total = home_win_prob + draw_prob + away_win_prob
        
        return MockPrediction(
            home_win_prob=home_win_prob / total,
            draw_prob=draw_prob / total,
            away_win_prob=away_win_prob / total,
            confidence=confidence,
            model_name=self.name
        )


class MockFootballerPredictor:
    """
    Mock version of FootballerModel.
    Uses form-based predictions.
    """
    
    def __init__(self):
        self.name = "mock_footballer"
    
    def predict(self, home_team: str, away_team: str, 
                home_form: Optional[float] = None,
                away_form: Optional[float] = None,
                **kwargs) -> MockPrediction:
        """
        Predict based on team form.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_form: Home team recent form (0-1)
            away_form: Away team recent form (0-1)
            
        Returns:
            MockPrediction
        """
        # Default form if not provided
        home_form = home_form or 0.5
        away_form = away_form or 0.5
        
        # Base probabilities from form
        form_diff = home_form - away_form
        
        # Home advantage
        home_advantage = 0.1
        
        # Calculate probabilities
        home_win_prob = 0.35 + (form_diff * 0.3) + home_advantage
        away_win_prob = 0.30 - (form_diff * 0.3)
        draw_prob = 1 - home_win_prob - away_win_prob
        
        # Clamp values
        home_win_prob = max(0.1, min(0.8, home_win_prob))
        away_win_prob = max(0.1, min(0.7, away_win_prob))
        draw_prob = max(0.1, min(0.4, draw_prob))
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        
        # Confidence based on form difference
        confidence = 0.5 + abs(form_diff) * 0.4
        
        return MockPrediction(
            home_win_prob=home_win_prob / total,
            draw_prob=draw_prob / total,
            away_win_prob=away_win_prob / total,
            confidence=min(0.9, confidence),
            model_name=self.name
        )


class MockXGBoostPredictor:
    """
    Mock XGBoost-style predictor.
    Uses feature-based prediction with odds integration.
    """
    
    def __init__(self):
        self.name = "mock_xgboost"
    
    def predict(self, home_team: str, away_team: str,
                home_odds: Optional[float] = None,
                draw_odds: Optional[float] = None,
                away_odds: Optional[float] = None,
                **kwargs) -> MockPrediction:
        """
        Predict using odds if available, otherwise use defaults.
        """
        # If odds provided, convert to probabilities
        if home_odds and draw_odds and away_odds:
            # Implied probabilities from odds
            total_implied = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            home_win_prob = (1/home_odds) / total_implied
            draw_prob = (1/draw_odds) / total_implied
            away_win_prob = (1/away_odds) / total_implied
            confidence = 0.75  # Higher confidence when using market odds
        else:
            # Default balanced prediction with home advantage
            home_win_prob = 0.42
            draw_prob = 0.28
            away_win_prob = 0.30
            confidence = 0.55
        
        return MockPrediction(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            confidence=confidence,
            model_name=self.name
        )


def create_mock_predictor(model_name: str) -> Any:
    """
    Factory function to create mock predictors.
    
    Args:
        model_name: Name of model to mock
        
    Returns:
        Mock predictor instance
    """
    predictors = {
        'podos': MockPodosPredictor,
        'footballer': MockFootballerPredictor,
        'xgboost': MockXGBoostPredictor
    }
    
    if model_name in predictors:
        return predictors[model_name]()
    
    # Default to Podos-style
    return MockPodosPredictor()
