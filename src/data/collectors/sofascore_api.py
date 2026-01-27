"""
Sofascore API Client
Fetches live and historical data from Sofascore.

Part of the complete blueprint implementation.
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

SOFASCORE_API = "https://api.sofascore.com/api/v1"


class SofascoreAPI:
    """
    API client for Sofascore data.
    
    Provides:
    - Live match data
    - Team form and statistics
    - Player ratings
    - Head-to-head data
    """
    
    TOURNAMENT_IDS = {
        'premier-league': 17,
        'la-liga': 8,
        'bundesliga': 35,
        'serie-a': 23,
        'ligue-1': 34,
        'eredivisie': 37,
        'primeira-liga': 238,
        'championship': 18,
        'champions-league': 7,
        'europa-league': 679,
    }
    
    def __init__(self, cache_dir: str = "data/cache/sofascore"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def get_scheduled_events(
        self,
        date: str = None,
        tournament_id: int = None
    ) -> List[Dict]:
        """
        Get scheduled matches for a date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            tournament_id: Filter by tournament
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{SOFASCORE_API}/sport/football/scheduled-events/{date}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            events = data.get('events', [])
            
            if tournament_id:
                events = [e for e in events 
                         if e.get('tournament', {}).get('id') == tournament_id]
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get scheduled events: {e}")
            return []
    
    def get_team_info(self, team_id: int) -> Dict:
        """Get team information and statistics."""
        url = f"{SOFASCORE_API}/team/{team_id}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.json().get('team', {})
        except Exception as e:
            logger.error(f"Failed to get team info: {e}")
            return {}
    
    def get_team_form(self, team_id: int, tournament_id: int = None) -> List[Dict]:
        """Get team's recent form (last 10 matches)."""
        url = f"{SOFASCORE_API}/team/{team_id}/events/last/0"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            events = response.json().get('events', [])[:10]
            
            if tournament_id:
                events = [e for e in events 
                         if e.get('tournament', {}).get('id') == tournament_id]
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get team form: {e}")
            return []
    
    def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """Get head-to-head history between two teams."""
        url = f"{SOFASCORE_API}/team/{team1_id}/events/total"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            events = response.json().get('events', [])
            
            # Filter for h2h
            h2h = []
            for event in events:
                home_id = event.get('homeTeam', {}).get('id')
                away_id = event.get('awayTeam', {}).get('id')
                
                if (home_id == team2_id or away_id == team2_id):
                    h2h.append(event)
                    if len(h2h) >= limit:
                        break
            
            return h2h
            
        except Exception as e:
            logger.error(f"Failed to get H2H: {e}")
            return []
    
    def get_live_event(self, event_id: int) -> Dict:
        """Get live match statistics."""
        url = f"{SOFASCORE_API}/event/{event_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get('event', {})
        except Exception as e:
            logger.error(f"Failed to get live event: {e}")
            return {}
    
    def get_event_statistics(self, event_id: int) -> Dict:
        """Get detailed match statistics."""
        url = f"{SOFASCORE_API}/event/{event_id}/statistics"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get('statistics', [])
        except Exception as e:
            logger.error(f"Failed to get event statistics: {e}")
            return {}
    
    def get_player_ratings(self, event_id: int) -> Dict:
        """Get player ratings for a match."""
        url = f"{SOFASCORE_API}/event/{event_id}/lineups"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return {
                'home': data.get('home', {}),
                'away': data.get('away', {})
            }
        except Exception as e:
            logger.error(f"Failed to get player ratings: {e}")
            return {}
    
    def search_team(self, query: str) -> List[Dict]:
        """Search for a team by name."""
        url = f"{SOFASCORE_API}/search/teams/{query}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get('teams', [])
        except Exception as e:
            logger.error(f"Failed to search team: {e}")
            return []
    
    def get_tournament_standings(self, tournament_id: int, season_id: int) -> List[Dict]:
        """Get league standings."""
        url = f"{SOFASCORE_API}/tournament/{tournament_id}/season/{season_id}/standings/total"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            standings = response.json().get('standings', [])
            if standings:
                return standings[0].get('rows', [])
            return []
            
        except Exception as e:
            logger.error(f"Failed to get standings: {e}")
            return []


# Global instance
_api: Optional[SofascoreAPI] = None


def get_api() -> SofascoreAPI:
    """Get or create Sofascore API client."""
    global _api
    if _api is None:
        _api = SofascoreAPI()
    return _api


def get_today_matches(tournament: str = None) -> List[Dict]:
    """Get today's matches."""
    api = get_api()
    tournament_id = SofascoreAPI.TOURNAMENT_IDS.get(tournament) if tournament else None
    return api.get_scheduled_events(tournament_id=tournament_id)


def get_team_recent_form(team_name: str) -> List[Dict]:
    """Get team's recent form by name."""
    api = get_api()
    teams = api.search_team(team_name)
    if teams:
        return api.get_team_form(teams[0]['id'])
    return []
