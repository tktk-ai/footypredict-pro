"""
Understat API Client
Fetches expected goals (xG) data from understat.com.

Part of the complete blueprint implementation.
"""

import pandas as pd
import requests
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"


class UnderstatAPI:
    """
    API client for Understat xG data.
    
    Provides:
    - Match xG/xGA
    - Player xG/xA
    - Shot maps and locations
    - Team xG trends
    """
    
    LEAGUES = {
        'epl': 'EPL',
        'la-liga': 'La_liga',
        'bundesliga': 'Bundesliga',
        'serie-a': 'Serie_A',
        'ligue-1': 'Ligue_1',
        'rfpl': 'RFPL',
    }
    
    def __init__(self, cache_dir: str = "data/cache/understat"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
    
    def _extract_json_data(self, html: str, var_name: str) -> Optional[Dict]:
        """Extract JSON data from JavaScript variable in HTML."""
        pattern = rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        
        if match:
            json_str = match.group(1)
            # Decode escaped characters
            json_str = json_str.encode().decode('unicode_escape')
            return json.loads(json_str)
        return None
    
    def fetch_league_data(
        self,
        league: str,
        season: str = "2024"
    ) -> Dict:
        """
        Fetch league-level xG data.
        
        Returns team standings with xG/xGA.
        """
        if league not in self.LEAGUES:
            logger.warning(f"Unknown league: {league}")
            return {}
        
        league_code = self.LEAGUES[league]
        cache_file = self.cache_dir / f"league_{league}_{season}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        url = f"{UNDERSTAT_BASE}/league/{league_code}/{season}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract teamsData
            teams_data = self._extract_json_data(response.text, 'teamsData')
            
            if teams_data:
                with open(cache_file, 'w') as f:
                    json.dump(teams_data, f)
                return teams_data
            
        except Exception as e:
            logger.error(f"Failed to fetch league data: {e}")
        
        return {}
    
    def fetch_team_data(self, team_id: int) -> Dict:
        """Fetch detailed team xG data."""
        cache_file = self.cache_dir / f"team_{team_id}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        url = f"{UNDERSTAT_BASE}/team/{team_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = {
                'matches': self._extract_json_data(response.text, 'datesData'),
                'players': self._extract_json_data(response.text, 'playersData'),
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch team data: {e}")
        
        return {}
    
    def fetch_match_shots(self, match_id: int) -> List[Dict]:
        """Fetch shot data for a specific match."""
        url = f"{UNDERSTAT_BASE}/match/{match_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            shots_data = self._extract_json_data(response.text, 'shotsData')
            
            if shots_data:
                return shots_data.get('h', []) + shots_data.get('a', [])
            
        except Exception as e:
            logger.error(f"Failed to fetch match shots: {e}")
        
        return []
    
    def get_team_xg_summary(self, league: str, team_name: str) -> Dict:
        """
        Get xG summary for a specific team.
        
        Returns:
            Dict with xG, xGA, xPts, etc.
        """
        league_data = self.fetch_league_data(league)
        
        for team_id, team_data in league_data.items():
            if team_data.get('title', '').lower() == team_name.lower():
                history = team_data.get('history', [])
                
                if not history:
                    return {}
                
                # Calculate averages
                total_xg = sum(float(m.get('xG', 0)) for m in history)
                total_xga = sum(float(m.get('xGA', 0)) for m in history)
                matches = len(history)
                
                return {
                    'team': team_name,
                    'team_id': team_id,
                    'matches': matches,
                    'total_xg': round(total_xg, 2),
                    'total_xga': round(total_xga, 2),
                    'avg_xg': round(total_xg / matches, 2) if matches > 0 else 0,
                    'avg_xga': round(total_xga / matches, 2) if matches > 0 else 0,
                    'xg_diff': round(total_xg - total_xga, 2),
                }
        
        return {}
    
    def get_upcoming_match_xg(
        self,
        home_team: str,
        away_team: str,
        league: str
    ) -> Dict:
        """
        Predict xG for an upcoming match based on team averages.
        """
        home_stats = self.get_team_xg_summary(league, home_team)
        away_stats = self.get_team_xg_summary(league, away_team)
        
        if not home_stats or not away_stats:
            return {'error': 'Team data not found'}
        
        # Simple prediction based on attack vs defense
        predicted_home_xg = (home_stats['avg_xg'] + away_stats['avg_xga']) / 2
        predicted_away_xg = (away_stats['avg_xg'] + home_stats['avg_xga']) / 2
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_xg': round(predicted_home_xg, 2),
            'predicted_away_xg': round(predicted_away_xg, 2),
            'predicted_total_xg': round(predicted_home_xg + predicted_away_xg, 2),
            'home_stats': home_stats,
            'away_stats': away_stats,
        }


# Global instance
_api: Optional[UnderstatAPI] = None


def get_api() -> UnderstatAPI:
    """Get or create Understat API client."""
    global _api
    if _api is None:
        _api = UnderstatAPI()
    return _api


def get_match_xg_prediction(home: str, away: str, league: str = 'epl') -> Dict:
    """Quick function to get xG prediction."""
    return get_api().get_upcoming_match_xg(home, away, league)
