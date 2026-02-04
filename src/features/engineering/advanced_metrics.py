"""
Advanced Metrics Module
Calculates advanced football analytics metrics.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AdvancedMetricsCalculator:
    """
    Calculates advanced football analytics.
    
    Metrics include:
    - Expected Goals (xG) based features
    - Pressing intensity
    - Shot quality
    - Progressive actions
    - Elo ratings
    """
    
    INITIAL_ELO = 1500
    K_FACTOR = 32
    
    def __init__(self, matches_df: pd.DataFrame = None):
        self.matches = matches_df
        self.elo_ratings = {}
        
    def set_matches(self, matches_df: pd.DataFrame):
        """Set match data."""
        self.matches = matches_df.copy()
        if 'match_date' in self.matches.columns:
            self.matches = self.matches.sort_values('match_date')
        self._compute_elo_ratings()
    
    def _compute_elo_ratings(self):
        """Compute Elo ratings for all teams."""
        if self.matches is None:
            return
        
        for _, match in self.matches.iterrows():
            home = match.get('home_team')
            away = match.get('away_team')
            home_goals = match.get('home_goals', 0)
            away_goals = match.get('away_goals', 0)
            
            if not home or not away:
                continue
            
            # Initialize if needed
            if home not in self.elo_ratings:
                self.elo_ratings[home] = self.INITIAL_ELO
            if away not in self.elo_ratings:
                self.elo_ratings[away] = self.INITIAL_ELO
            
            # Calculate expected scores
            home_elo = self.elo_ratings[home]
            away_elo = self.elo_ratings[away]
            
            exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            exp_away = 1 - exp_home
            
            # Actual result
            if home_goals > away_goals:
                actual_home, actual_away = 1, 0
            elif home_goals < away_goals:
                actual_home, actual_away = 0, 1
            else:
                actual_home, actual_away = 0.5, 0.5
            
            # Update ratings
            self.elo_ratings[home] += self.K_FACTOR * (actual_home - exp_home)
            self.elo_ratings[away] += self.K_FACTOR * (actual_away - exp_away)
    
    def get_elo(self, team: str) -> float:
        """Get current Elo rating for a team."""
        return self.elo_ratings.get(team, self.INITIAL_ELO)
    
    def calculate_xg_overperformance(
        self,
        team: str,
        n_matches: int = 10
    ) -> Dict:
        """Calculate how much a team over/underperforms their xG."""
        if self.matches is None:
            return {}
        
        # Get team's matches with xG data
        home = self.matches[self.matches['home_team'] == team].tail(n_matches)
        away = self.matches[self.matches['away_team'] == team].tail(n_matches)
        
        xg_cols_home = ['home_xg', 'xg_home']
        xg_cols_away = ['away_xg', 'xg_away']
        
        total_xg = 0
        total_goals = 0
        matches = 0
        
        for col in xg_cols_home:
            if col in home.columns:
                total_xg += home[col].sum()
                total_goals += home['home_goals'].sum()
                matches += len(home)
                break
        
        for col in xg_cols_away:
            if col in away.columns:
                total_xg += away[col].sum()
                total_goals += away['away_goals'].sum()
                matches += len(away)
                break
        
        if matches == 0 or total_xg == 0:
            return {'overperformance': 0, 'xg_ratio': 1.0}
        
        return {
            'total_xg': total_xg,
            'total_goals': total_goals,
            'overperformance': total_goals - total_xg,
            'xg_ratio': total_goals / total_xg,
            'matches': matches
        }
    
    def calculate_shot_quality(self, team: str, n_matches: int = 10) -> Dict:
        """Calculate shot quality metrics."""
        if self.matches is None:
            return {}
        
        home = self.matches[self.matches['home_team'] == team].tail(n_matches)
        away = self.matches[self.matches['away_team'] == team].tail(n_matches)
        
        metrics = {
            'shots_per_game': 0,
            'shots_on_target_per_game': 0,
            'shot_accuracy': 0,
            'goals_per_shot': 0,
        }
        
        total_shots = 0
        total_sot = 0
        total_goals = 0
        matches = 0
        
        # Home matches
        if 'home_shots' in home.columns:
            total_shots += home['home_shots'].sum()
            matches += len(home)
        if 'home_shots_on_target' in home.columns:
            total_sot += home['home_shots_on_target'].sum()
        total_goals += home['home_goals'].sum() if 'home_goals' in home.columns else 0
        
        # Away matches
        if 'away_shots' in away.columns:
            total_shots += away['away_shots'].sum()
            matches += len(away)
        if 'away_shots_on_target' in away.columns:
            total_sot += away['away_shots_on_target'].sum()
        total_goals += away['away_goals'].sum() if 'away_goals' in away.columns else 0
        
        if matches > 0:
            metrics['shots_per_game'] = total_shots / matches
            metrics['shots_on_target_per_game'] = total_sot / matches
        
        if total_shots > 0:
            metrics['shot_accuracy'] = total_sot / total_shots
            metrics['goals_per_shot'] = total_goals / total_shots
        
        return metrics
    
    def calculate_defensive_metrics(self, team: str, n_matches: int = 10) -> Dict:
        """Calculate defensive quality metrics."""
        if self.matches is None:
            return {}
        
        home = self.matches[self.matches['home_team'] == team].tail(n_matches)
        away = self.matches[self.matches['away_team'] == team].tail(n_matches)
        
        metrics = {
            'goals_conceded_per_game': 0,
            'shots_conceded_per_game': 0,
            'clean_sheet_rate': 0,
        }
        
        total_conceded = 0
        total_shots_conceded = 0
        clean_sheets = 0
        matches = 0
        
        # Home
        if len(home) > 0:
            total_conceded += home['away_goals'].sum()
            if 'away_shots' in home.columns:
                total_shots_conceded += home['away_shots'].sum()
            clean_sheets += (home['away_goals'] == 0).sum()
            matches += len(home)
        
        # Away
        if len(away) > 0:
            total_conceded += away['home_goals'].sum()
            if 'home_shots' in away.columns:
                total_shots_conceded += away['home_shots'].sum()
            clean_sheets += (away['home_goals'] == 0).sum()
            matches += len(away)
        
        if matches > 0:
            metrics['goals_conceded_per_game'] = total_conceded / matches
            metrics['shots_conceded_per_game'] = total_shots_conceded / matches
            metrics['clean_sheet_rate'] = clean_sheets / matches
        
        return metrics
    
    def get_match_features(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Get advanced metric features for a match."""
        features = {}
        
        # Elo ratings
        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)
        
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        features['elo_home_prob'] = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        
        # xG overperformance
        home_xg = self.calculate_xg_overperformance(home_team)
        away_xg = self.calculate_xg_overperformance(away_team)
        
        features['home_xg_ratio'] = home_xg.get('xg_ratio', 1.0)
        features['away_xg_ratio'] = away_xg.get('xg_ratio', 1.0)
        
        # Shot quality
        home_shots = self.calculate_shot_quality(home_team)
        away_shots = self.calculate_shot_quality(away_team)
        
        features['home_shot_accuracy'] = home_shots.get('shot_accuracy', 0)
        features['away_shot_accuracy'] = away_shots.get('shot_accuracy', 0)
        features['home_goals_per_shot'] = home_shots.get('goals_per_shot', 0)
        features['away_goals_per_shot'] = away_shots.get('goals_per_shot', 0)
        
        # Defensive metrics
        home_defense = self.calculate_defensive_metrics(home_team)
        away_defense = self.calculate_defensive_metrics(away_team)
        
        features['home_clean_sheet_rate'] = home_defense.get('clean_sheet_rate', 0)
        features['away_clean_sheet_rate'] = away_defense.get('clean_sheet_rate', 0)
        
        return features


# Global instance
_calculator: Optional[AdvancedMetricsCalculator] = None


def get_calculator(matches_df: pd.DataFrame = None) -> AdvancedMetricsCalculator:
    """Get or create advanced metrics calculator."""
    global _calculator
    if _calculator is None:
        _calculator = AdvancedMetricsCalculator()
    if matches_df is not None:
        _calculator.set_matches(matches_df)
    return _calculator


def get_team_elo(team: str, matches_df: pd.DataFrame) -> float:
    """Quick function to get team Elo."""
    return get_calculator(matches_df).get_elo(team)
