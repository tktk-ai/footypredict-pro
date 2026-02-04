"""
Team Features Module
Generates team-level features for predictions.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TeamFeatureGenerator:
    """
    Generates team-level features from historical match data.
    
    Features include:
    - Rolling averages (goals, shots, possession)
    - Attack/defense ratings
    - Home/away splits
    - Form indicators
    """
    
    ROLLING_WINDOWS = [3, 5, 10, 20]
    
    def __init__(self, matches_df: pd.DataFrame = None):
        self.matches = matches_df
        self.team_stats = {}
        
    def set_matches(self, matches_df: pd.DataFrame):
        """Set match data for feature generation."""
        self.matches = matches_df.copy()
        if 'match_date' in self.matches.columns:
            self.matches = self.matches.sort_values('match_date')
        self._compute_team_stats()
    
    def _compute_team_stats(self):
        """Compute rolling statistics for all teams."""
        if self.matches is None:
            return
        
        teams = set(self.matches['home_team'].unique()) | set(self.matches['away_team'].unique())
        
        for team in teams:
            self.team_stats[team] = self._compute_single_team_stats(team)
    
    def _compute_single_team_stats(self, team: str) -> Dict:
        """Compute statistics for a single team."""
        # Get all matches for team
        home_matches = self.matches[self.matches['home_team'] == team].copy()
        away_matches = self.matches[self.matches['away_team'] == team].copy()
        
        # Standardize columns for combining
        home_matches['is_home'] = True
        home_matches['team_goals'] = home_matches['home_goals']
        home_matches['opp_goals'] = home_matches['away_goals']
        
        away_matches['is_home'] = False
        away_matches['team_goals'] = away_matches['away_goals']
        away_matches['opp_goals'] = away_matches['home_goals']
        
        all_matches = pd.concat([home_matches, away_matches]).sort_values('match_date')
        
        if len(all_matches) == 0:
            return {}
        
        stats = {
            'matches_played': len(all_matches),
            'home_matches': len(home_matches),
            'away_matches': len(away_matches),
        }
        
        # Calculate rolling averages
        for window in self.ROLLING_WINDOWS:
            if len(all_matches) >= window:
                recent = all_matches.tail(window)
                
                stats[f'goals_scored_avg_{window}'] = recent['team_goals'].mean()
                stats[f'goals_conceded_avg_{window}'] = recent['opp_goals'].mean()
                stats[f'goals_diff_avg_{window}'] = (recent['team_goals'] - recent['opp_goals']).mean()
                
                # Points
                points = recent.apply(
                    lambda r: 3 if r['team_goals'] > r['opp_goals']
                             else (1 if r['team_goals'] == r['opp_goals'] else 0),
                    axis=1
                )
                stats[f'ppg_{window}'] = points.mean()
                
                # Win/Draw/Loss rates
                stats[f'win_rate_{window}'] = (recent['team_goals'] > recent['opp_goals']).mean()
                stats[f'draw_rate_{window}'] = (recent['team_goals'] == recent['opp_goals']).mean()
                stats[f'loss_rate_{window}'] = (recent['team_goals'] < recent['opp_goals']).mean()
                
                # Clean sheets and BTTS
                stats[f'clean_sheet_rate_{window}'] = (recent['opp_goals'] == 0).mean()
                stats[f'failed_to_score_rate_{window}'] = (recent['team_goals'] == 0).mean()
                stats[f'btts_rate_{window}'] = ((recent['team_goals'] > 0) & (recent['opp_goals'] > 0)).mean()
                
                # Over/Under
                total_goals = recent['team_goals'] + recent['opp_goals']
                stats[f'over_2.5_rate_{window}'] = (total_goals > 2.5).mean()
                stats[f'over_1.5_rate_{window}'] = (total_goals > 1.5).mean()
        
        # Home/Away splits
        if len(home_matches) > 0:
            stats['home_goals_avg'] = home_matches['home_goals'].mean()
            stats['home_conceded_avg'] = home_matches['away_goals'].mean()
            stats['home_win_rate'] = (home_matches['home_goals'] > home_matches['away_goals']).mean()
        
        if len(away_matches) > 0:
            stats['away_goals_avg'] = away_matches['away_goals'].mean()
            stats['away_conceded_avg'] = away_matches['home_goals'].mean()
            stats['away_win_rate'] = (away_matches['away_goals'] > away_matches['home_goals']).mean()
        
        return stats
    
    def get_team_features(self, team: str) -> Dict:
        """Get features for a specific team."""
        return self.team_stats.get(team, {})
    
    def get_match_features(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Get combined features for a match."""
        home_stats = self.get_team_features(home_team)
        away_stats = self.get_team_features(away_team)
        
        features = {}
        
        # Add home team features with prefix
        for key, value in home_stats.items():
            features[f'home_{key}'] = value
        
        # Add away team features with prefix
        for key, value in away_stats.items():
            features[f'away_{key}'] = value
        
        # Add difference features
        for window in self.ROLLING_WINDOWS:
            if f'goals_scored_avg_{window}' in home_stats and f'goals_scored_avg_{window}' in away_stats:
                features[f'attack_diff_{window}'] = (
                    home_stats[f'goals_scored_avg_{window}'] - 
                    away_stats[f'goals_conceded_avg_{window}']
                )
                features[f'defense_diff_{window}'] = (
                    away_stats[f'goals_scored_avg_{window}'] - 
                    home_stats[f'goals_conceded_avg_{window}']
                )
                features[f'ppg_diff_{window}'] = (
                    home_stats.get(f'ppg_{window}', 0) - 
                    away_stats.get(f'ppg_{window}', 0)
                )
        
        return features
    
    def generate_all_features(self) -> pd.DataFrame:
        """Generate features for all matches."""
        if self.matches is None:
            return pd.DataFrame()
        
        features_list = []
        
        for _, row in self.matches.iterrows():
            match_features = self.get_match_features(row['home_team'], row['away_team'])
            match_features['match_id'] = row.get('match_id', f"{row['home_team']}_{row['away_team']}")
            features_list.append(match_features)
        
        return pd.DataFrame(features_list)


# Global instance
_generator: Optional[TeamFeatureGenerator] = None


def get_generator(matches_df: pd.DataFrame = None) -> TeamFeatureGenerator:
    """Get or create team feature generator."""
    global _generator
    if _generator is None:
        _generator = TeamFeatureGenerator()
    if matches_df is not None:
        _generator.set_matches(matches_df)
    return _generator


def generate_team_features(
    home_team: str,
    away_team: str,
    matches_df: pd.DataFrame
) -> Dict:
    """Quick function to generate match features."""
    generator = get_generator(matches_df)
    return generator.get_match_features(home_team, away_team)
