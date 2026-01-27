"""
Player Features Module
Aggregates player-level data into team features.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PlayerFeatureGenerator:
    """
    Generates player-aggregated features for team predictions.
    
    Features include:
    - Squad quality ratings
    - Key player availability
    - Goal/assist contributions
    - Experience metrics
    """
    
    def __init__(self, player_data: pd.DataFrame = None):
        self.player_data = player_data
        self.team_squads = {}
        
    def set_player_data(self, player_data: pd.DataFrame):
        """Set player data for feature generation."""
        self.player_data = player_data.copy()
        self._build_team_squads()
    
    def _build_team_squads(self):
        """Build team squad mappings."""
        if self.player_data is None or 'team' not in self.player_data.columns:
            return
        
        for team in self.player_data['team'].unique():
            team_players = self.player_data[self.player_data['team'] == team]
            self.team_squads[team] = team_players
    
    def get_squad_strength(self, team: str) -> Dict:
        """Calculate squad strength metrics."""
        if team not in self.team_squads:
            return self._empty_squad_features()
        
        squad = self.team_squads[team]
        
        features = {
            'squad_size': len(squad),
        }
        
        # Average ratings if available
        if 'rating' in squad.columns:
            features['avg_rating'] = squad['rating'].mean()
            features['max_rating'] = squad['rating'].max()
            features['min_rating'] = squad['rating'].min()
        
        # Goal contributions
        if 'goals' in squad.columns:
            features['total_goals'] = squad['goals'].sum()
            features['avg_goals'] = squad['goals'].mean()
            features['top_scorer_goals'] = squad['goals'].max()
        
        if 'assists' in squad.columns:
            features['total_assists'] = squad['assists'].sum()
            features['avg_assists'] = squad['assists'].mean()
        
        # Experience
        if 'appearances' in squad.columns:
            features['total_appearances'] = squad['appearances'].sum()
            features['avg_appearances'] = squad['appearances'].mean()
        
        if 'age' in squad.columns:
            features['avg_age'] = squad['age'].mean()
            features['youngest'] = squad['age'].min()
            features['oldest'] = squad['age'].max()
        
        # Market value if available
        if 'market_value' in squad.columns:
            features['total_value'] = squad['market_value'].sum()
            features['avg_value'] = squad['market_value'].mean()
        
        # xG/xA if available
        if 'xg' in squad.columns:
            features['total_xg'] = squad['xg'].sum()
            features['avg_xg'] = squad['xg'].mean()
        
        if 'xa' in squad.columns:
            features['total_xa'] = squad['xa'].sum()
        
        return features
    
    def _empty_squad_features(self) -> Dict:
        """Return empty squad features."""
        return {
            'squad_size': 0,
            'avg_rating': 0,
            'total_goals': 0,
            'total_assists': 0,
            'avg_age': 0,
        }
    
    def get_key_players(self, team: str, n: int = 5) -> List[Dict]:
        """Get top N key players for a team."""
        if team not in self.team_squads:
            return []
        
        squad = self.team_squads[team].copy()
        
        # Score players by importance
        if 'rating' in squad.columns:
            squad['importance'] = squad['rating']
        elif 'goals' in squad.columns and 'assists' in squad.columns:
            squad['importance'] = squad['goals'] * 1.5 + squad['assists']
        else:
            squad['importance'] = 0
        
        top_players = squad.nlargest(n, 'importance')
        
        return top_players.to_dict('records')
    
    def get_missing_player_impact(
        self,
        team: str,
        missing_players: List[str]
    ) -> float:
        """Estimate impact of missing players."""
        if team not in self.team_squads or not missing_players:
            return 0.0
        
        squad = self.team_squads[team]
        
        if 'player_name' not in squad.columns:
            return 0.0
        
        # Find missing players in squad
        missing = squad[squad['player_name'].isin(missing_players)]
        
        if len(missing) == 0:
            return 0.0
        
        # Calculate impact based on contributions
        total_goals = squad['goals'].sum() if 'goals' in squad.columns else 1
        missing_goals = missing['goals'].sum() if 'goals' in missing.columns else 0
        
        goal_impact = missing_goals / max(total_goals, 1)
        
        return min(goal_impact * 0.5, 0.3)  # Cap at 30% impact
    
    def get_match_features(
        self,
        home_team: str,
        away_team: str,
        home_missing: List[str] = None,
        away_missing: List[str] = None
    ) -> Dict:
        """Get player-based features for a match."""
        home_strength = self.get_squad_strength(home_team)
        away_strength = self.get_squad_strength(away_team)
        
        features = {}
        
        # Add team features
        for key, value in home_strength.items():
            features[f'home_{key}'] = value
        for key, value in away_strength.items():
            features[f'away_{key}'] = value
        
        # Add differences
        if 'avg_rating' in home_strength and 'avg_rating' in away_strength:
            features['rating_diff'] = home_strength['avg_rating'] - away_strength['avg_rating']
        
        if 'total_goals' in home_strength and 'total_goals' in away_strength:
            features['goals_contribution_diff'] = home_strength['total_goals'] - away_strength['total_goals']
        
        # Missing player impact
        if home_missing:
            features['home_injury_impact'] = self.get_missing_player_impact(home_team, home_missing)
        if away_missing:
            features['away_injury_impact'] = self.get_missing_player_impact(away_team, away_missing)
        
        return features


# Global instance
_generator: Optional[PlayerFeatureGenerator] = None


def get_generator(player_data: pd.DataFrame = None) -> PlayerFeatureGenerator:
    """Get or create player feature generator."""
    global _generator
    if _generator is None:
        _generator = PlayerFeatureGenerator()
    if player_data is not None:
        _generator.set_player_data(player_data)
    return _generator
