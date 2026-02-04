"""
Multi-Source Data Aggregator

Fetches football data from multiple free APIs to expand league coverage.
Sources: Football-Data.org, OpenLigaDB, TheSportsDB, Sports Open Data
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FootballDataAggregator:
    """Aggregate football data from multiple free sources."""
    
    def __init__(self):
        self.sources = {
            'football_data': FootballDataOrg(),
            'openligadb': OpenLigaDB(),
            'thesportsdb': TheSportsDB(),
            'sports_open_data': SportsOpenData()
        }
        
    def get_fixtures(self, league: str, days_ahead: int = 7) -> List[Dict]:
        """Get fixtures from all available sources."""
        all_fixtures = []
        
        for name, source in self.sources.items():
            try:
                fixtures = source.get_fixtures(league, days_ahead)
                for f in fixtures:
                    f['source'] = name
                all_fixtures.extend(fixtures)
            except Exception as e:
                logger.debug(f"{name} fixtures error: {e}")
        
        # Deduplicate by team names
        seen = set()
        unique = []
        for f in all_fixtures:
            key = (f.get('home_team', ''), f.get('away_team', ''), f.get('date', ''))
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        return unique
    
    def get_team_stats(self, team: str, league: str = None) -> Dict:
        """Get team statistics from available sources."""
        for name, source in self.sources.items():
            try:
                stats = source.get_team_stats(team, league)
                if stats:
                    return stats
            except Exception as e:
                logger.debug(f"{name} stats error for {team}: {e}")
        
        return {}
    
    def get_league_standings(self, league: str) -> List[Dict]:
        """Get league standings."""
        for name, source in self.sources.items():
            try:
                standings = source.get_standings(league)
                if standings:
                    return standings
            except Exception as e:
                logger.debug(f"{name} standings error: {e}")
        
        return []


class FootballDataOrg:
    """Football-Data.org API wrapper."""
    
    BASE_URL = "https://api.football-data.org/v4"
    
    LEAGUES = {
        'premier_league': 'PL',
        'bundesliga': 'BL1',
        'la_liga': 'PD',
        'serie_a': 'SA',
        'ligue_1': 'FL1',
        'eredivisie': 'DED',
        'primeira_liga': 'PPL',
        'championship': 'ELC'
    }
    
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_DATA_API_KEY')
        
    def _request(self, endpoint: str) -> Dict:
        if not self.api_key:
            return {}
        
        headers = {'X-Auth-Token': self.api_key}
        try:
            resp = requests.get(f"{self.BASE_URL}{endpoint}", headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Football-Data.org error: {e}")
        
        return {}
    
    def get_fixtures(self, league: str, days_ahead: int = 7) -> List[Dict]:
        code = self.LEAGUES.get(league)
        if not code:
            return []
        
        today = datetime.now().strftime('%Y-%m-%d')
        end = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        data = self._request(f"/competitions/{code}/matches?dateFrom={today}&dateTo={end}")
        
        fixtures = []
        for match in data.get('matches', []):
            fixtures.append({
                'home_team': match.get('homeTeam', {}).get('name', ''),
                'away_team': match.get('awayTeam', {}).get('name', ''),
                'date': match.get('utcDate', '')[:10],
                'time': match.get('utcDate', '')[11:16],
                'league': league,
                'status': match.get('status', 'SCHEDULED')
            })
        
        return fixtures
    
    def get_team_stats(self, team: str, league: str = None) -> Dict:
        # Would need team ID lookup
        return {}
    
    def get_standings(self, league: str) -> List[Dict]:
        code = self.LEAGUES.get(league)
        if not code:
            return []
        
        data = self._request(f"/competitions/{code}/standings")
        
        standings = []
        for standing in data.get('standings', []):
            if standing.get('type') == 'TOTAL':
                for entry in standing.get('table', []):
                    standings.append({
                        'position': entry.get('position'),
                        'team': entry.get('team', {}).get('name'),
                        'points': entry.get('points'),
                        'played': entry.get('playedGames'),
                        'won': entry.get('won'),
                        'draw': entry.get('draw'),
                        'lost': entry.get('lost'),
                        'gf': entry.get('goalsFor'),
                        'ga': entry.get('goalsAgainst'),
                        'gd': entry.get('goalDifference')
                    })
        
        return standings


class OpenLigaDB:
    """OpenLigaDB API - Free, no auth required. Mainly German leagues."""
    
    BASE_URL = "https://api.openligadb.de"
    
    LEAGUES = {
        'bundesliga': 'bl1',
        'bundesliga_2': 'bl2',
        'dfb_pokal': 'dfb'
    }
    
    def get_fixtures(self, league: str, days_ahead: int = 7) -> List[Dict]:
        code = self.LEAGUES.get(league)
        if not code:
            return []
        
        try:
            # Get current matchday
            resp = requests.get(f"{self.BASE_URL}/getmatchdata/{code}/2025", timeout=10)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            fixtures = []
            
            today = datetime.now().date()
            end_date = today + timedelta(days=days_ahead)
            
            for match in data:
                match_date = datetime.fromisoformat(match.get('matchDateTime', '').replace('Z', '')).date()
                if today <= match_date <= end_date:
                    fixtures.append({
                        'home_team': match.get('team1', {}).get('teamName', ''),
                        'away_team': match.get('team2', {}).get('teamName', ''),
                        'date': str(match_date),
                        'time': match.get('matchDateTime', '')[11:16],
                        'league': league
                    })
            
            return fixtures
        except Exception as e:
            logger.debug(f"OpenLigaDB error: {e}")
            return []
    
    def get_team_stats(self, team: str, league: str = None) -> Dict:
        return {}
    
    def get_standings(self, league: str) -> List[Dict]:
        code = self.LEAGUES.get(league)
        if not code:
            return []
        
        try:
            resp = requests.get(f"{self.BASE_URL}/getbltable/{code}/2025", timeout=10)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            standings = []
            
            for i, entry in enumerate(data, 1):
                standings.append({
                    'position': i,
                    'team': entry.get('teamName'),
                    'points': entry.get('points'),
                    'played': entry.get('matches'),
                    'won': entry.get('won'),
                    'draw': entry.get('draw'),
                    'lost': entry.get('lost'),
                    'gf': entry.get('goals'),
                    'ga': entry.get('opponentGoals'),
                    'gd': entry.get('goalDiff')
                })
            
            return standings
        except Exception as e:
            logger.debug(f"OpenLigaDB standings error: {e}")
            return []


class TheSportsDB:
    """TheSportsDB API - Free tier with 100 req/min limit."""
    
    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
    
    LEAGUES = {
        'premier_league': '4328',
        'la_liga': '4335',
        'serie_a': '4332',
        'bundesliga': '4331',
        'ligue_1': '4334',
        'eredivisie': '4337',
        'primeira_liga': '4344',
        'mls': '4346',
        'championship': '4329'
    }
    
    def get_fixtures(self, league: str, days_ahead: int = 7) -> List[Dict]:
        league_id = self.LEAGUES.get(league)
        if not league_id:
            return []
        
        try:
            resp = requests.get(
                f"{self.BASE_URL}/eventsnextleague.php?id={league_id}",
                timeout=10
            )
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            fixtures = []
            
            today = datetime.now().date()
            end_date = today + timedelta(days=days_ahead)
            
            for event in (data.get('events') or []):
                try:
                    match_date = datetime.strptime(event.get('dateEvent', ''), '%Y-%m-%d').date()
                    if today <= match_date <= end_date:
                        fixtures.append({
                            'home_team': event.get('strHomeTeam', ''),
                            'away_team': event.get('strAwayTeam', ''),
                            'date': event.get('dateEvent', ''),
                            'time': event.get('strTime', '00:00')[:5],
                            'league': league,
                            'venue': event.get('strVenue', '')
                        })
                except:
                    pass
            
            return fixtures
        except Exception as e:
            logger.debug(f"TheSportsDB error: {e}")
            return []
    
    def get_team_stats(self, team: str, league: str = None) -> Dict:
        try:
            resp = requests.get(
                f"{self.BASE_URL}/searchteams.php?t={team}",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                teams = data.get('teams', [])
                if teams:
                    t = teams[0]
                    return {
                        'name': t.get('strTeam'),
                        'stadium': t.get('strStadium'),
                        'capacity': t.get('intStadiumCapacity'),
                        'formed': t.get('intFormedYear'),
                        'league': t.get('strLeague')
                    }
        except Exception as e:
            logger.debug(f"TheSportsDB team error: {e}")
        
        return {}
    
    def get_standings(self, league: str) -> List[Dict]:
        league_id = self.LEAGUES.get(league)
        if not league_id:
            return []
        
        try:
            resp = requests.get(
                f"{self.BASE_URL}/lookuptable.php?l={league_id}&s=2025-2026",
                timeout=10
            )
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            standings = []
            
            for entry in (data.get('table') or []):
                standings.append({
                    'position': int(entry.get('intRank', 0)),
                    'team': entry.get('strTeam'),
                    'points': int(entry.get('intPoints', 0)),
                    'played': int(entry.get('intPlayed', 0)),
                    'won': int(entry.get('intWin', 0)),
                    'draw': int(entry.get('intDraw', 0)),
                    'lost': int(entry.get('intLoss', 0)),
                    'gf': int(entry.get('intGoalsFor', 0)),
                    'ga': int(entry.get('intGoalsAgainst', 0)),
                    'gd': int(entry.get('intGoalDifference', 0))
                })
            
            return standings
        except Exception as e:
            logger.debug(f"TheSportsDB standings error: {e}")
            return []


class SportsOpenData:
    """Sports Open Data - Community-driven, no auth required."""
    
    BASE_URL = "https://sports-open-data.api.sportradar.com/soccer/trial/v4/en"
    
    # Note: This API may have limited free access
    LEAGUES = {
        'serie_a': 'sr:competition:23',
        'la_liga': 'sr:competition:8'
    }
    
    def get_fixtures(self, league: str, days_ahead: int = 7) -> List[Dict]:
        # Sports Open Data requires specific implementation
        return []
    
    def get_team_stats(self, team: str, league: str = None) -> Dict:
        return {}
    
    def get_standings(self, league: str) -> List[Dict]:
        return []


# Global instance
_aggregator: Optional[FootballDataAggregator] = None


def get_data_aggregator() -> FootballDataAggregator:
    """Get or create data aggregator singleton."""
    global _aggregator
    if _aggregator is None:
        _aggregator = FootballDataAggregator()
    return _aggregator


def get_all_fixtures(league: str, days: int = 7) -> List[Dict]:
    """Get fixtures from all sources."""
    return get_data_aggregator().get_fixtures(league, days)


def get_standings(league: str) -> List[Dict]:
    """Get league standings."""
    return get_data_aggregator().get_league_standings(league)
