"""
Bayesian Hierarchical Model
Hierarchical Bayesian model for football predictions.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class BayesianHierarchicalModel:
    """
    Bayesian hierarchical model for football predictions.
    
    Implements a hierarchical structure:
    - League-level parameters
    - Team-level parameters
    - Match-level predictions
    """
    
    def __init__(
        self,
        prior_attack_mean: float = 0.0,
        prior_attack_std: float = 0.5,
        prior_defense_mean: float = 0.0,
        prior_defense_std: float = 0.5
    ):
        self.prior_attack = (prior_attack_mean, prior_attack_std)
        self.prior_defense = (prior_defense_mean, prior_defense_std)
        
        # Posteriors
        self.attack_params = {}  # team -> (mean, std)
        self.defense_params = {}  # team -> (mean, std)
        
        # League-level hyperparameters
        self.league_attack_mean = 0.0
        self.league_defense_mean = 0.0
        self.home_advantage = 0.3
        self.baseline_goals = 1.35
        
        self.is_fitted = False
    
    def fit(
        self,
        matches: List[Dict],
        n_iterations: int = 1000
    ) -> 'BayesianHierarchicalModel':
        """
        Fit the model using variational inference approximation.
        """
        # Collect teams
        teams = set()
        for match in matches:
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        # Initialize posteriors from priors
        for team in teams:
            self.attack_params[team] = list(self.prior_attack)
            self.defense_params[team] = list(self.prior_defense)
        
        # Simplified variational inference
        for iteration in range(n_iterations):
            np.random.shuffle(matches)
            
            for match in matches:
                home = match['home_team']
                away = match['away_team']
                home_goals = match['home_goals']
                away_goals = match['away_goals']
                
                # Sample from current posteriors
                home_attack = np.random.normal(*self.attack_params[home])
                home_defense = np.random.normal(*self.defense_params[home])
                away_attack = np.random.normal(*self.attack_params[away])
                away_defense = np.random.normal(*self.defense_params[away])
                
                # Expected goals (log link)
                home_lambda = np.exp(
                    self.home_advantage + 
                    home_attack - 
                    away_defense + 
                    np.log(self.baseline_goals)
                )
                away_lambda = np.exp(
                    away_attack - 
                    home_defense + 
                    np.log(self.baseline_goals)
                )
                
                # Update posteriors (simplified gradient)
                lr = 0.001 * (1 - iteration / n_iterations)
                
                # Attack updates
                home_attack_grad = home_goals - home_lambda
                away_attack_grad = away_goals - away_lambda
                
                self.attack_params[home][0] += lr * home_attack_grad
                self.attack_params[away][0] += lr * away_attack_grad
                
                # Defense updates
                home_defense_grad = away_goals - away_lambda
                away_defense_grad = home_goals - home_lambda
                
                self.defense_params[home][0] += lr * home_defense_grad
                self.defense_params[away][0] += lr * away_defense_grad
                
                # Shrink variances gradually
                shrink_factor = 0.9999
                self.attack_params[home][1] *= shrink_factor
                self.attack_params[away][1] *= shrink_factor
                self.defense_params[home][1] *= shrink_factor
                self.defense_params[away][1] *= shrink_factor
        
        self.is_fitted = True
        logger.info(f"Fitted BayesianHierarchical on {len(matches)} matches")
        
        return self
    
    def predict_xg(
        self,
        home_team: str,
        away_team: str,
        n_samples: int = 1000
    ) -> Tuple[float, float, float, float]:
        """
        Predict expected goals with uncertainty.
        
        Returns: (home_xg_mean, home_xg_std, away_xg_mean, away_xg_std)
        """
        home_attack = self.attack_params.get(home_team, self.prior_attack)
        home_defense = self.defense_params.get(home_team, self.prior_defense)
        away_attack = self.attack_params.get(away_team, self.prior_attack)
        away_defense = self.defense_params.get(away_team, self.prior_defense)
        
        home_xgs = []
        away_xgs = []
        
        for _ in range(n_samples):
            h_att = np.random.normal(*home_attack)
            h_def = np.random.normal(*home_defense)
            a_att = np.random.normal(*away_attack)
            a_def = np.random.normal(*away_defense)
            
            home_lambda = np.exp(self.home_advantage + h_att - a_def + np.log(self.baseline_goals))
            away_lambda = np.exp(a_att - h_def + np.log(self.baseline_goals))
            
            home_xgs.append(home_lambda)
            away_xgs.append(away_lambda)
        
        return (
            np.mean(home_xgs),
            np.std(home_xgs),
            np.mean(away_xgs),
            np.std(away_xgs)
        )
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        n_samples: int = 10000
    ) -> Dict:
        """Predict match with full posterior sampling."""
        home_attack = self.attack_params.get(home_team, self.prior_attack)
        home_defense = self.defense_params.get(home_team, self.prior_defense)
        away_attack = self.attack_params.get(away_team, self.prior_attack)
        away_defense = self.defense_params.get(away_team, self.prior_defense)
        
        home_wins = draws = away_wins = 0
        btts = over_25 = 0
        total_home = total_away = 0
        
        for _ in range(n_samples):
            h_att = np.random.normal(*home_attack)
            h_def = np.random.normal(*home_defense)
            a_att = np.random.normal(*away_attack)
            a_def = np.random.normal(*away_defense)
            
            home_lambda = np.exp(self.home_advantage + h_att - a_def + np.log(self.baseline_goals))
            away_lambda = np.exp(a_att - h_def + np.log(self.baseline_goals))
            
            home_goals = np.random.poisson(home_lambda)
            away_goals = np.random.poisson(away_lambda)
            
            total_home += home_goals
            total_away += away_goals
            
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals < away_goals:
                away_wins += 1
            else:
                draws += 1
            
            if home_goals > 0 and away_goals > 0:
                btts += 1
            
            if home_goals + away_goals > 2.5:
                over_25 += 1
        
        return {
            'home_xg': round(total_home / n_samples, 2),
            'away_xg': round(total_away / n_samples, 2),
            '1x2': {
                'home': round(home_wins / n_samples, 4),
                'draw': round(draws / n_samples, 4),
                'away': round(away_wins / n_samples, 4)
            },
            'btts': round(btts / n_samples, 4),
            'over_2.5': round(over_25 / n_samples, 4),
            'uncertainty': {
                'home_attack_std': round(home_attack[1], 3),
                'away_attack_std': round(away_attack[1], 3)
            }
        }
    
    def get_team_posteriors(self, team: str) -> Dict:
        """Get posterior parameters for a team."""
        return {
            'team': team,
            'attack': {
                'mean': round(self.attack_params.get(team, self.prior_attack)[0], 3),
                'std': round(self.attack_params.get(team, self.prior_attack)[1], 3)
            },
            'defense': {
                'mean': round(self.defense_params.get(team, self.prior_defense)[0], 3),
                'std': round(self.defense_params.get(team, self.prior_defense)[1], 3)
            }
        }
    
    def get_credible_interval(
        self,
        home_team: str,
        away_team: str,
        confidence: float = 0.9
    ) -> Dict:
        """Get credible intervals for predictions."""
        h_xg_mean, h_xg_std, a_xg_mean, a_xg_std = self.predict_xg(home_team, away_team)
        
        z = stats.norm.ppf((1 + confidence) / 2) if SCIPY_AVAILABLE else 1.645
        
        return {
            'home_xg': {
                'mean': round(h_xg_mean, 2),
                'lower': round(h_xg_mean - z * h_xg_std, 2),
                'upper': round(h_xg_mean + z * h_xg_std, 2)
            },
            'away_xg': {
                'mean': round(a_xg_mean, 2),
                'lower': round(a_xg_mean - z * a_xg_std, 2),
                'upper': round(a_xg_mean + z * a_xg_std, 2)
            },
            'confidence': confidence
        }


# Global instance
_model: Optional[BayesianHierarchicalModel] = None


def get_model() -> BayesianHierarchicalModel:
    """Get or create Bayesian model."""
    global _model
    if _model is None:
        _model = BayesianHierarchicalModel()
    return _model
