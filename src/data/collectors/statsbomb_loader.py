"""
StatsBomb Open Data Loader
Loads free event data from StatsBomb's open dataset.

Part of the complete blueprint implementation.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

STATSBOMB_GITHUB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


class StatsBombLoader:
    """
    Loader for StatsBomb open data.
    
    Provides:
    - Match events (shots, passes, etc.)
    - Player performance data
    - Detailed event locations
    - 360 freeze frames (where available)
    """
    
    def __init__(self, cache_dir: str = "data/cache/statsbomb"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        import requests
        self.session = requests.Session()
    
    def get_competitions(self) -> pd.DataFrame:
        """Get list of available competitions."""
        url = f"{STATSBOMB_GITHUB}/competitions.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get competitions: {e}")
            return pd.DataFrame()
    
    def get_matches(
        self,
        competition_id: int,
        season_id: int
    ) -> pd.DataFrame:
        """Get matches for a competition/season."""
        url = f"{STATSBOMB_GITHUB}/matches/{competition_id}/{season_id}.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get matches: {e}")
            return pd.DataFrame()
    
    def get_lineups(self, match_id: int) -> Dict:
        """Get lineups for a match."""
        url = f"{STATSBOMB_GITHUB}/lineups/{match_id}.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get lineups: {e}")
            return {}
    
    def get_events(self, match_id: int) -> pd.DataFrame:
        """
        Get all events for a match.
        
        Returns DataFrame with columns:
        - id, type, minute, second, team, player, location, etc.
        """
        cache_file = self.cache_dir / f"events_{match_id}.parquet"
        
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        
        url = f"{STATSBOMB_GITHUB}/events/{match_id}.json"
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            df = pd.json_normalize(data)
            
            df.to_parquet(cache_file)
            return df
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return pd.DataFrame()
    
    def get_shots(self, match_id: int) -> pd.DataFrame:
        """Get shot events for a match."""
        events = self.get_events(match_id)
        
        if events.empty:
            return pd.DataFrame()
        
        shots = events[events['type.name'] == 'Shot'].copy()
        
        # Extract key columns
        shot_cols = [
            'id', 'minute', 'second', 'team.name', 'player.name',
            'location', 'shot.statsbomb_xg', 'shot.outcome.name',
            'shot.body_part.name', 'shot.technique.name'
        ]
        
        available_cols = [c for c in shot_cols if c in shots.columns]
        return shots[available_cols]
    
    def get_passes(self, match_id: int) -> pd.DataFrame:
        """Get pass events for a match."""
        events = self.get_events(match_id)
        
        if events.empty:
            return pd.DataFrame()
        
        return events[events['type.name'] == 'Pass'].copy()
    
    def calculate_match_xg(self, match_id: int) -> Dict:
        """Calculate total xG for each team in a match."""
        shots = self.get_shots(match_id)
        
        if shots.empty:
            return {}
        
        xg_col = 'shot.statsbomb_xg'
        if xg_col not in shots.columns:
            return {}
        
        xg_by_team = shots.groupby('team.name')[xg_col].sum().to_dict()
        
        return {
            'match_id': match_id,
            'xg_by_team': xg_by_team,
            'total_shots': len(shots)
        }
    
    def get_player_match_summary(
        self,
        match_id: int,
        player_name: str
    ) -> Dict:
        """Get player summary for a match."""
        events = self.get_events(match_id)
        
        if events.empty or 'player.name' not in events.columns:
            return {}
        
        player_events = events[events['player.name'] == player_name]
        
        if player_events.empty:
            return {}
        
        summary = {
            'player': player_name,
            'match_id': match_id,
            'total_events': len(player_events),
            'event_types': player_events['type.name'].value_counts().to_dict()
        }
        
        # Add xG if player had shots
        shots = player_events[player_events['type.name'] == 'Shot']
        if not shots.empty and 'shot.statsbomb_xg' in shots.columns:
            summary['total_xg'] = shots['shot.statsbomb_xg'].sum()
            summary['shots'] = len(shots)
            summary['goals'] = len(shots[shots['shot.outcome.name'] == 'Goal'])
        
        return summary
    
    def get_available_data_summary(self) -> Dict:
        """Get summary of available StatsBomb open data."""
        competitions = self.get_competitions()
        
        if competitions.empty:
            return {}
        
        return {
            'total_competitions': len(competitions),
            'competitions': competitions.groupby('competition_name').agg({
                'season_name': 'count',
                'match_available': 'sum'
            }).to_dict()
        }


# Global instance
_loader: Optional[StatsBombLoader] = None


def get_loader() -> StatsBombLoader:
    """Get or create StatsBomb loader."""
    global _loader
    if _loader is None:
        _loader = StatsBombLoader()
    return _loader


def get_match_xg(match_id: int) -> Dict:
    """Quick function to get match xG."""
    return get_loader().calculate_match_xg(match_id)
