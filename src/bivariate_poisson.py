"""
Bivariate Poisson Model for Enhanced Draw Prediction

The bivariate Poisson model improves on independent Poisson by considering
correlation between team scores. The diagonal-inflated variant specifically
enhances draw probability estimation.

Research shows:
- With λ3=0.05: +3.3% more draws predicted than independent Poisson
- With λ3=0.20: +14% more draws predicted
- Best model for Bundesliga and EPL in half-season forecasting
"""

import numpy as np
from scipy.stats import poisson
from scipy.special import factorial
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BivariatePrediction:
    """Bivariate Poisson prediction output"""
    home_team: str
    away_team: str
    
    # Probabilities
    home_win: float
    draw: float
    away_win: float
    
    # Parameters
    lambda1: float  # Home independent component
    lambda2: float  # Away independent component
    lambda3: float  # Correlation parameter
    
    # Score matrix
    score_matrix: np.ndarray
    correct_scores: Dict[str, float]
    
    # Goals
    over_2_5: float
    btts_yes: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['score_matrix'] = self.score_matrix.tolist()
        return result


class BivariatePoissonModel:
    """
    Bivariate Poisson model for football score prediction.
    
    Unlike independent Poisson, this model considers correlation between
    home and away team scores through a shared λ3 parameter.
    
    Joint probability: P(X=x, Y=y) where X,Y follow bivariate Poisson
    with parameters (λ1, λ2, λ3)
    
    The model predicts MORE DRAWS than independent Poisson, which is
    empirically more accurate.
    """
    
    MAX_GOALS = 8
    
    def __init__(self, correlation: float = 0.1):
        """
        Initialize bivariate Poisson model.
        
        Args:
            correlation: λ3 parameter (0.05 to 0.20 typical)
        """
        self.correlation = correlation
        self.params = {}
        self._init_team_params()
    
    def _init_team_params(self):
        """Initialize parameters for known teams."""
        teams = {
            'manchester city': (1.9, 0.8),
            'liverpool': (1.8, 0.9),
            'arsenal': (1.7, 0.9),
            'chelsea': (1.5, 1.0),
            'manchester united': (1.4, 1.1),
            'tottenham': (1.4, 1.2),
            'real madrid': (2.0, 0.8),
            'barcelona': (1.9, 0.9),
            'bayern munich': (2.2, 0.7),
            'psg': (1.9, 0.8),
            'paris saint germain': (1.9, 0.8),
            'inter milan': (1.6, 0.9),
            'juventus': (1.5, 0.9),
            'napoli': (1.7, 1.0),
            'borussia dortmund': (1.7, 1.2),
            'atletico madrid': (1.3, 0.7),
        }
        
        for team, (attack, defense) in teams.items():
            self.params[f'{team}_attack'] = attack
            self.params[f'{team}_defense'] = defense
    
    @staticmethod
    def bivariate_poisson_pmf(x: int, y: int, 
                              lambda1: float, lambda2: float, 
                              lambda3: float) -> float:
        """
        Bivariate Poisson probability mass function.
        
        P(X=x, Y=y) = exp(-(λ1+λ2+λ3)) × Σ[k=0 to min(x,y)] 
                      (λ1^(x-k) × λ2^(y-k) × λ3^k) / ((x-k)! × (y-k)! × k!)
        
        Args:
            x: Home goals
            y: Away goals
            lambda1: Home team intensity (independent component)
            lambda2: Away team intensity (independent component)
            lambda3: Covariance parameter (correlation)
        """
        min_xy = min(x, y)
        prob = 0.0
        
        for k in range(min_xy + 1):
            try:
                term = (np.exp(-(lambda1 + lambda2 + lambda3)) * 
                        (lambda1 ** (x - k)) * (lambda2 ** (y - k)) * (lambda3 ** k) /
                        (factorial(x - k) * factorial(y - k) * factorial(k)))
                prob += term
            except (OverflowError, ValueError):
                continue
        
        return max(0, prob)
    
    def get_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Get expected goals for each team."""
        home_key = home_team.lower().strip()
        away_key = away_team.lower().strip()
        
        home_attack = self.params.get(f'{home_key}_attack', 1.35)
        away_defense = self.params.get(f'{away_key}_defense', 1.0)
        away_attack = self.params.get(f'{away_key}_attack', 1.2)
        home_defense = self.params.get(f'{home_key}_defense', 1.0)
        
        # Home advantage factor
        home_factor = 1.1
        
        home_xg = home_attack * (away_defense / 1.0) * home_factor
        away_xg = away_attack * (home_defense / 1.0)
        
        return max(0.3, min(4.0, home_xg)), max(0.2, min(3.5, away_xg))
    
    def calculate_score_matrix(self, lambda1: float, lambda2: float,
                                lambda3: float = None) -> np.ndarray:
        """
        Calculate the full score probability matrix.
        
        Returns NxN matrix with P(home=i, away=j)
        """
        if lambda3 is None:
            lambda3 = self.correlation
        
        # Adjust lambdas for correlation (joint model)
        adj_lambda1 = max(0.1, lambda1 - lambda3)
        adj_lambda2 = max(0.1, lambda2 - lambda3)
        
        matrix = np.zeros((self.MAX_GOALS, self.MAX_GOALS))
        
        for i in range(self.MAX_GOALS):
            for j in range(self.MAX_GOALS):
                matrix[i, j] = self.bivariate_poisson_pmf(
                    i, j, adj_lambda1, adj_lambda2, lambda3
                )
        
        # Normalize
        total = matrix.sum()
        if total > 0:
            matrix /= total
        
        return matrix
    
    def predict(self, home_team: str, away_team: str) -> BivariatePrediction:
        """Generate match prediction using bivariate Poisson."""
        home_xg, away_xg = self.get_expected_goals(home_team, away_team)
        
        # Calculate score matrix
        matrix = self.calculate_score_matrix(home_xg, away_xg, self.correlation)
        
        # Extract probabilities
        home_win = 0
        away_win = 0
        for i in range(self.MAX_GOALS):
            for j in range(self.MAX_GOALS):
                if i > j:
                    home_win += matrix[i, j]
                elif i < j:
                    away_win += matrix[i, j]
        
        draw = np.trace(matrix)
        
        # Over 2.5
        over_2_5 = sum(matrix[i, j] for i in range(self.MAX_GOALS) 
                       for j in range(self.MAX_GOALS) if i + j > 2)
        
        # BTTS
        btts_yes = sum(matrix[i, j] for i in range(1, self.MAX_GOALS) 
                       for j in range(1, self.MAX_GOALS))
        
        # Correct scores (top 15)
        scores = {}
        for i in range(6):
            for j in range(6):
                scores[f'{i}-{j}'] = matrix[i, j]
        correct_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15])
        
        return BivariatePrediction(
            home_team=home_team,
            away_team=away_team,
            home_win=round(home_win, 4),
            draw=round(draw, 4),
            away_win=round(away_win, 4),
            lambda1=round(home_xg, 3),
            lambda2=round(away_xg, 3),
            lambda3=round(self.correlation, 3),
            score_matrix=matrix,
            correct_scores={k: round(v, 4) for k, v in correct_scores.items()},
            over_2_5=round(over_2_5, 4),
            btts_yes=round(btts_yes, 4)
        )


class DiagonalInflatedBivariatePoissonModel:
    """
    Diagonal-Inflated Bivariate Poisson for enhanced draw prediction.
    
    This model adds an inflation factor to diagonal entries (draws)
    in the score matrix, improving draw probability estimation.
    
    Research: "This inflation improves in precision the estimation of draws."
    Best for: La Liga specifically, generally good for draw-heavy leagues.
    """
    
    MAX_GOALS = 8
    
    def __init__(self, 
                 correlation: float = 0.08,
                 inflation_factor: float = 0.12):
        """
        Initialize diagonal-inflated model.
        
        Args:
            correlation: λ3 parameter
            inflation_factor: How much to inflate draw probabilities (0.1-0.2 typical)
        """
        self.correlation = correlation
        self.inflation_factor = inflation_factor
        self.base_model = BivariatePoissonModel(correlation)
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """
        Generate prediction with diagonal inflation.
        """
        # Get base bivariate prediction
        home_xg, away_xg = self.base_model.get_expected_goals(home_team, away_team)
        
        # Calculate base matrix
        matrix = self.base_model.calculate_score_matrix(home_xg, away_xg, self.correlation)
        
        # Apply diagonal inflation
        for i in range(min(6, self.MAX_GOALS)):
            matrix[i, i] *= (1 + self.inflation_factor)
        
        # Renormalize
        matrix /= matrix.sum()
        
        # Extract probabilities
        home_win = sum(matrix[i, j] for i in range(self.MAX_GOALS) 
                       for j in range(self.MAX_GOALS) if i > j)
        away_win = sum(matrix[i, j] for i in range(self.MAX_GOALS) 
                       for j in range(self.MAX_GOALS) if i < j)
        draw = np.trace(matrix)
        
        # Over/Under
        over_2_5 = sum(matrix[i, j] for i in range(self.MAX_GOALS) 
                       for j in range(self.MAX_GOALS) if i + j > 2)
        
        # BTTS
        btts_yes = sum(matrix[i, j] for i in range(1, self.MAX_GOALS) 
                       for j in range(1, self.MAX_GOALS))
        
        # Correct scores
        scores = {}
        for i in range(6):
            for j in range(6):
                scores[f'{i}-{j}'] = round(matrix[i, j], 4)
        correct_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15])
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win': round(home_win, 4),
            'draw': round(draw, 4),
            'away_win': round(away_win, 4),
            'home_xg': round(home_xg, 3),
            'away_xg': round(away_xg, 3),
            'correlation': self.correlation,
            'inflation': self.inflation_factor,
            'correct_scores': correct_scores,
            'over_2_5': round(over_2_5, 4),
            'btts_yes': round(btts_yes, 4),
            'model': 'diagonal_inflated_bivariate_poisson'
        }
    
    def compare_with_independent(self, home_team: str, away_team: str) -> Dict:
        """
        Compare predictions with independent Poisson.
        
        Shows the improvement in draw prediction.
        """
        home_xg, away_xg = self.base_model.get_expected_goals(home_team, away_team)
        
        # Independent Poisson
        ind_home_win = 0
        ind_draw = 0
        ind_away_win = 0
        
        for i in range(self.MAX_GOALS):
            for j in range(self.MAX_GOALS):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    ind_home_win += prob
                elif i < j:
                    ind_away_win += prob
                else:
                    ind_draw += prob
        
        # Diagonal-inflated bivariate
        di_pred = self.predict(home_team, away_team)
        
        return {
            'match': f'{home_team} vs {away_team}',
            'independent_poisson': {
                'home_win': round(ind_home_win, 4),
                'draw': round(ind_draw, 4),
                'away_win': round(ind_away_win, 4)
            },
            'diagonal_inflated_bp': {
                'home_win': di_pred['home_win'],
                'draw': di_pred['draw'],
                'away_win': di_pred['away_win']
            },
            'draw_improvement': f"+{round((di_pred['draw'] - ind_draw) * 100, 1)}%"
        }


# Ensemble that combines Dixon-Coles with Bivariate Poisson
class StatisticalEnsemble:
    """
    Ensemble combining Dixon-Coles and Bivariate Poisson models.
    
    Weights:
    - Dixon-Coles: 50% (best for correct score)
    - Bivariate Poisson: 30% (better draw estimation)
    - Diagonal-Inflated BP: 20% (for draw-heavy scenarios)
    """
    
    def __init__(self):
        from .dixon_coles import DixonColesModel
        
        self.dixon_coles = DixonColesModel()
        self.bivariate = BivariatePoissonModel(correlation=0.08)
        self.diagonal_inflated = DiagonalInflatedBivariatePoissonModel(
            correlation=0.08, 
            inflation_factor=0.12
        )
        
        self.weights = {
            'dixon_coles': 0.50,
            'bivariate': 0.30,
            'diagonal_inflated': 0.20
        }
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Get ensemble prediction."""
        # Get individual predictions
        dc_pred = self.dixon_coles.predict(home_team, away_team)
        bp_pred = self.bivariate.predict(home_team, away_team)
        di_pred = self.diagonal_inflated.predict(home_team, away_team)
        
        # Weighted ensemble for 1X2
        home_win = (
            self.weights['dixon_coles'] * dc_pred.home_win +
            self.weights['bivariate'] * bp_pred.home_win +
            self.weights['diagonal_inflated'] * di_pred['home_win']
        )
        
        draw = (
            self.weights['dixon_coles'] * dc_pred.draw +
            self.weights['bivariate'] * bp_pred.draw +
            self.weights['diagonal_inflated'] * di_pred['draw']
        )
        
        away_win = (
            self.weights['dixon_coles'] * dc_pred.away_win +
            self.weights['bivariate'] * bp_pred.away_win +
            self.weights['diagonal_inflated'] * di_pred['away_win']
        )
        
        # Use Dixon-Coles for detailed markets (best for correct score)
        return {
            'home_team': home_team,
            'away_team': away_team,
            '1x2': {
                'home_win': round(home_win, 4),
                'draw': round(draw, 4),
                'away_win': round(away_win, 4)
            },
            'recommendation': max({'home': home_win, 'draw': draw, 'away': away_win}.items(), 
                                  key=lambda x: x[1])[0],
            'expected_goals': {
                'home': dc_pred.home_xg,
                'away': dc_pred.away_xg,
                'total': round(dc_pred.home_xg + dc_pred.away_xg, 2)
            },
            'correct_scores': dc_pred.correct_scores,
            'over_under': {
                'over_1_5': dc_pred.over_1_5,
                'over_2_5': dc_pred.over_2_5,
                'over_3_5': dc_pred.over_3_5
            },
            'btts': {
                'yes': dc_pred.btts_yes,
                'no': dc_pred.btts_no
            },
            'htft': self.dixon_coles.predict_htft(home_team, away_team),
            'model_breakdown': {
                'dixon_coles': {
                    'home_win': dc_pred.home_win,
                    'draw': dc_pred.draw,
                    'away_win': dc_pred.away_win
                },
                'bivariate_poisson': {
                    'home_win': bp_pred.home_win,
                    'draw': bp_pred.draw,
                    'away_win': bp_pred.away_win
                },
                'diagonal_inflated': {
                    'home_win': di_pred['home_win'],
                    'draw': di_pred['draw'],
                    'away_win': di_pred['away_win']
                }
            },
            'weights': self.weights
        }


# Global instances
bivariate_model = BivariatePoissonModel()
diagonal_inflated_model = DiagonalInflatedBivariatePoissonModel()


def predict_bivariate(home_team: str, away_team: str) -> Dict:
    """Get bivariate Poisson prediction."""
    return bivariate_model.predict(home_team, away_team).to_dict()


def predict_with_draw_enhancement(home_team: str, away_team: str) -> Dict:
    """Get diagonal-inflated prediction for better draw estimation."""
    return diagonal_inflated_model.predict(home_team, away_team)


def compare_draw_models(home_team: str, away_team: str) -> Dict:
    """Compare independent vs bivariate for draw prediction."""
    return diagonal_inflated_model.compare_with_independent(home_team, away_team)
