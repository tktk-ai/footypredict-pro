"""
Football Data API Integrations

This module provides integrations with multiple free football APIs:
- API-Football (api-football.com)
- Football-Data.org
- OpenLigaDB (no API key required)
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib


@dataclass
class Team:
    """Standardized team representation"""
    id: str
    name: str
    short_name: Optional[str] = None
    logo_url: Optional[str] = None


@dataclass
class Match:
    """Standardized match representation"""
    id: str
    home_team: Team
    away_team: Team
    kickoff: datetime
    league: str
    league_id: str
    season: str
    status: str  # scheduled, live, finished
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    venue: Optional[str] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['kickoff'] = self.kickoff.isoformat()
        return d


class CacheManager:
    """Simple file-based cache to avoid hitting API rate limits"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.json"
    
    def get(self, key: str, max_age_minutes: int = 60) -> Optional[Any]:
        """Get cached data if not expired"""
        path = self._get_cache_path(key)
        if not path.exists():
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            cached_at = datetime.fromisoformat(data['cached_at'])
            if datetime.now() - cached_at > timedelta(minutes=max_age_minutes):
                return None
            
            return data['value']
        except:
            return None
    
    def set(self, key: str, value: Any):
        """Cache data with timestamp"""
        path = self._get_cache_path(key)
        with open(path, 'w') as f:
            json.dump({
                'cached_at': datetime.now().isoformat(),
                'value': value
            }, f)


class OpenLigaDBClient:
    """
    OpenLigaDB - Completely free, no API key required!
    https://www.openligadb.de/
    
    Best for: German Bundesliga and other leagues
    """
    
    BASE_URL = "https://api.openligadb.de"
    
    LEAGUES = {
        # German Leagues (Free - No API Key!)
        'bundesliga': 'bl1',
        'bundesliga2': 'bl2',
        '3liga': 'bl3',
        'dfb_pokal': 'dfb2025',  # DFB-Pokal Cup
        
        # Other Leagues available on OpenLigaDB
        'euro_2024': 'em2024',
        'world_cup_2022': 'wm2022',
        'champions_league': 'ucl2025',
        'europa_league': 'uel2025',
    }
    
    def __init__(self):
        self.cache = CacheManager()
        self.session = requests.Session()
    
    def get_current_matchday(self, league: str = 'bundesliga') -> List[Match]:
        """Get current matchday fixtures"""
        league_code = self.LEAGUES.get(league, league)
        
        cache_key = f"openliga_current_{league_code}"
        cached = self.cache.get(cache_key, max_age_minutes=30)
        if cached:
            return self._parse_matches(cached, league)
        
        url = f"{self.BASE_URL}/getmatchdata/{league_code}"
        response = self.session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            self.cache.set(cache_key, data)
            return self._parse_matches(data, league)
        
        return []
    
    def get_upcoming_matches(self, league: str = 'bundesliga', days: int = 7) -> List[Match]:
        """Get upcoming matches for the next N days"""
        # OpenLigaDB returns current matchday, filter for upcoming
        matches = self.get_current_matchday(league)
        now = datetime.now()
        cutoff = now + timedelta(days=days)
        
        return [m for m in matches if now <= m.kickoff <= cutoff]
    
    def get_standings(self, league: str = 'bundesliga', season: int = None) -> List[Dict]:
        """Get current league standings"""
        league_code = self.LEAGUES.get(league, league)
        season = season or datetime.now().year
        
        cache_key = f"openliga_standings_{league_code}_{season}"
        cached = self.cache.get(cache_key, max_age_minutes=60)
        if cached:
            return cached
        
        url = f"{self.BASE_URL}/getbltable/{league_code}/{season}"
        response = self.session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            self.cache.set(cache_key, data)
            return data
        
        return []
    
    def _parse_matches(self, data: List[Dict], league: str) -> List[Match]:
        """Parse OpenLigaDB response into Match objects"""
        matches = []
        for m in data:
            try:
                # Parse datetime
                kickoff_str = m.get('matchDateTime') or m.get('matchDateTimeUTC')
                if kickoff_str:
                    kickoff = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                else:
                    continue
                
                # Determine status
                if m.get('matchIsFinished'):
                    status = 'finished'
                elif kickoff <= datetime.now():
                    status = 'live'
                else:
                    status = 'scheduled'
                
                # Get scores
                results = m.get('matchResults', [])
                home_score = away_score = None
                for r in results:
                    if r.get('resultTypeID') == 2:  # Final result
                        home_score = r.get('pointsTeam1')
                        away_score = r.get('pointsTeam2')
                
                match = Match(
                    id=f"openliga_{m.get('matchID')}",
                    home_team=Team(
                        id=str(m['team1']['teamId']),
                        name=m['team1']['teamName'],
                        short_name=m['team1'].get('shortName'),
                        logo_url=m['team1'].get('teamIconUrl')
                    ),
                    away_team=Team(
                        id=str(m['team2']['teamId']),
                        name=m['team2']['teamName'],
                        short_name=m['team2'].get('shortName'),
                        logo_url=m['team2'].get('teamIconUrl')
                    ),
                    kickoff=kickoff,
                    league=league,
                    league_id=m.get('leagueId', ''),
                    season=str(m.get('leagueSeason', '')),
                    status=status,
                    home_score=home_score,
                    away_score=away_score,
                    venue=m.get('location', {}).get('locationCity') if m.get('location') else None
                )
                matches.append(match)
            except Exception as e:
                print(f"Error parsing match: {e}")
                continue
        
        return matches


class FootballDataOrgClient:
    """
    Football-Data.org - Free tier with major leagues
    https://www.football-data.org/
    
    Free tier: 10 requests/minute
    Leagues: EPL, La Liga, Bundesliga, Serie A, Ligue 1, Champions League
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    
    LEAGUES = {
        'premier_league': 'PL',
        'la_liga': 'PD',
        'bundesliga': 'BL1',
        'serie_a': 'SA',
        'ligue_1': 'FL1',
        'champions_league': 'CL',
        'eredivisie': 'DED',
        'primeira_liga': 'PPL',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FOOTBALL_DATA_API_KEY')
        self.cache = CacheManager()
        self.session = requests.Session()
        if self.api_key:
            self.session.headers['X-Auth-Token'] = self.api_key
    
    def get_upcoming_matches(self, league: str = 'premier_league', days: int = 7) -> List[Match]:
        """Get upcoming fixtures for a league"""
        if not self.api_key:
            print("Warning: Football-Data.org requires API key")
            return []
        
        league_code = self.LEAGUES.get(league, league)
        
        cache_key = f"fdo_matches_{league_code}_{days}"
        cached = self.cache.get(cache_key, max_age_minutes=30)
        if cached:
            return self._parse_matches(cached, league)
        
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = f"{self.BASE_URL}/competitions/{league_code}/matches"
        params = {
            'dateFrom': date_from,
            'dateTo': date_to,
            'status': 'SCHEDULED'
        }
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data)
                return self._parse_matches(data, league)
        except Exception as e:
            print(f"Football-Data.org error: {e}")
        
        return []
    
    def get_standings(self, league: str = 'premier_league') -> List[Dict]:
        """Get current standings"""
        if not self.api_key:
            return []
        
        league_code = self.LEAGUES.get(league, league)
        
        cache_key = f"fdo_standings_{league_code}"
        cached = self.cache.get(cache_key, max_age_minutes=60)
        if cached:
            return cached
        
        url = f"{self.BASE_URL}/competitions/{league_code}/standings"
        
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data)
                return data.get('standings', [])
        except Exception as e:
            print(f"Football-Data.org standings error: {e}")
        
        return []
    
    def get_team_matches(self, team_id: int, limit: int = 10) -> List[Dict]:
        """Get recent matches for a team"""
        if not self.api_key:
            return []
        
        cache_key = f"fdo_team_matches_{team_id}_{limit}"
        cached = self.cache.get(cache_key, max_age_minutes=60)
        if cached:
            return cached
        
        url = f"{self.BASE_URL}/teams/{team_id}/matches"
        params = {'limit': limit, 'status': 'FINISHED'}
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data.get('matches', []))
                return data.get('matches', [])
        except:
            pass
        
        return []
    
    def get_head_to_head(self, match_id: int) -> Dict:
        """
        Get head-to-head data for a specific match
        Returns: Dict with H2H statistics
        """
        if not self.api_key:
            return {}
        
        cache_key = f"fdo_h2h_{match_id}"
        cached = self.cache.get(cache_key, max_age_minutes=1440)  # 24hr cache
        if cached:
            return cached
        
        url = f"{self.BASE_URL}/matches/{match_id}/head2head"
        params = {'limit': 10}
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data)
                return data
        except Exception as e:
            print(f"H2H fetch error: {e}")
        
        return {}
    
    def get_team_by_name(self, team_name: str) -> Optional[Dict]:
        """
        Search for team by name to get team ID
        """
        if not self.api_key:
            return None
        
        cache_key = f"fdo_team_search_{team_name.lower()}"
        cached = self.cache.get(cache_key, max_age_minutes=1440)
        if cached:
            return cached
        
        # Try to find team in any league we support
        for league_code in self.LEAGUES.values():
            url = f"{self.BASE_URL}/competitions/{league_code}/teams"
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    for team in data.get('teams', []):
                        if (team_name.lower() in team['name'].lower() or 
                            team_name.lower() in team.get('shortName', '').lower() or
                            team_name.lower() == team.get('tla', '').lower()):
                            self.cache.set(cache_key, team)
                            return team
            except:
                continue
        
        return None
    
    def get_finished_matches(self, league: str = 'premier_league', limit: int = 50) -> List[Dict]:
        """Get finished matches for training data"""
        if not self.api_key:
            return []
        
        league_code = self.LEAGUES.get(league, league)
        
        cache_key = f"fdo_finished_{league_code}_{limit}"
        cached = self.cache.get(cache_key, max_age_minutes=60)
        if cached:
            return cached
        
        url = f"{self.BASE_URL}/competitions/{league_code}/matches"
        params = {'status': 'FINISHED', 'limit': limit}
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                self.cache.set(cache_key, matches)
                return matches
        except Exception as e:
            print(f"Error fetching finished matches: {e}")
        
        return []
    
    def get_live_standings_parsed(self, league: str = 'premier_league') -> Dict[str, int]:
        """
        Get live standings as team name -> position dict
        For use in predictions
        """
        standings_raw = self.get_standings(league)
        
        positions = {}
        if standings_raw:
            for table in standings_raw:
                if table.get('type') == 'TOTAL':
                    for entry in table.get('table', []):
                        team_name = entry.get('team', {}).get('name', '')
                        short_name = entry.get('team', {}).get('shortName', '')
                        position = entry.get('position', 10)
                        
                        positions[team_name] = position
                        if short_name:
                            positions[short_name] = position
        
        return positions
    
    def _parse_matches(self, data: Dict, league: str) -> List[Match]:
        """Parse Football-Data.org response"""
        matches = []
        for m in data.get('matches', []):
            try:
                kickoff = datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00'))
                
                status_map = {
                    'SCHEDULED': 'scheduled',
                    'TIMED': 'scheduled',
                    'IN_PLAY': 'live',
                    'PAUSED': 'live',
                    'FINISHED': 'finished',
                }
                
                match = Match(
                    id=f"fdo_{m['id']}",
                    home_team=Team(
                        id=str(m['homeTeam']['id']),
                        name=m['homeTeam']['name'],
                        short_name=m['homeTeam'].get('tla'),
                        logo_url=m['homeTeam'].get('crest')
                    ),
                    away_team=Team(
                        id=str(m['awayTeam']['id']),
                        name=m['awayTeam']['name'],
                        short_name=m['awayTeam'].get('tla'),
                        logo_url=m['awayTeam'].get('crest')
                    ),
                    kickoff=kickoff,
                    league=league,
                    league_id=str(m.get('competition', {}).get('id', '')),
                    season=str(m.get('season', {}).get('id', '')),
                    status=status_map.get(m.get('status'), 'scheduled'),
                    home_score=m.get('score', {}).get('fullTime', {}).get('home'),
                    away_score=m.get('score', {}).get('fullTime', {}).get('away'),
                    venue=m.get('venue')
                )
                matches.append(match)
            except Exception as e:
                print(f"Error parsing FDO match: {e}")
                continue
        
        return matches


class APIFootballClient:
    """
    API-Football (RapidAPI) - Comprehensive football data
    https://www.api-football.com/
    
    Free tier: 100 requests/day
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    LEAGUES = {
        'premier_league': 39,
        'la_liga': 140,
        'bundesliga': 78,
        'serie_a': 135,
        'ligue_1': 61,
        'champions_league': 2,
        'europa_league': 3,
        'mls': 253,
        'eredivisie': 88,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY')
        self.cache = CacheManager()
        self.session = requests.Session()
        if self.api_key:
            self.session.headers['x-apisports-key'] = self.api_key
    
    def get_upcoming_matches(self, league: str = 'premier_league', days: int = 7) -> List[Match]:
        """Get upcoming fixtures"""
        if not self.api_key:
            print("Warning: API-Football requires API key")
            return []
        
        league_id = self.LEAGUES.get(league, league)
        
        cache_key = f"apifb_matches_{league_id}_{days}"
        cached = self.cache.get(cache_key, max_age_minutes=30)
        if cached:
            return self._parse_matches(cached, league)
        
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = f"{self.BASE_URL}/fixtures"
        params = {
            'league': league_id,
            'season': datetime.now().year,
            'from': date_from,
            'to': date_to
        }
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data)
                return self._parse_matches(data, league)
        except Exception as e:
            print(f"API-Football error: {e}")
        
        return []
    
    def _parse_matches(self, data: Dict, league: str) -> List[Match]:
        """Parse API-Football response"""
        matches = []
        for m in data.get('response', []):
            try:
                fixture = m['fixture']
                teams = m['teams']
                goals = m['goals']
                
                kickoff = datetime.fromtimestamp(fixture['timestamp'])
                
                status_map = {
                    'NS': 'scheduled',
                    'TBD': 'scheduled',
                    '1H': 'live',
                    '2H': 'live',
                    'HT': 'live',
                    'FT': 'finished',
                    'AET': 'finished',
                    'PEN': 'finished',
                }
                
                match = Match(
                    id=f"apifb_{fixture['id']}",
                    home_team=Team(
                        id=str(teams['home']['id']),
                        name=teams['home']['name'],
                        logo_url=teams['home'].get('logo')
                    ),
                    away_team=Team(
                        id=str(teams['away']['id']),
                        name=teams['away']['name'],
                        logo_url=teams['away'].get('logo')
                    ),
                    kickoff=kickoff,
                    league=league,
                    league_id=str(m.get('league', {}).get('id', '')),
                    season=str(m.get('league', {}).get('season', '')),
                    status=status_map.get(fixture.get('status', {}).get('short'), 'scheduled'),
                    home_score=goals.get('home'),
                    away_score=goals.get('away'),
                    venue=fixture.get('venue', {}).get('name')
                )
                matches.append(match)
            except Exception as e:
                print(f"Error parsing API-Football match: {e}")
                continue
        
        return matches


class DataAggregator:
    """
    Aggregates data from multiple sources with fallback
    """
    
    def __init__(self):
        self.openliga = OpenLigaDBClient()
        self.fdo = FootballDataOrgClient()
        self.apifb = APIFootballClient()
    
    def get_upcoming_matches(self, leagues: List[str] = None, days: int = 7) -> List[Match]:
        """Get upcoming matches from all available sources"""
        if leagues is None:
            leagues = ['bundesliga', 'premier_league', 'la_liga', 'serie_a', 'ligue_1']
        
        all_matches = []
        seen_ids = set()
        
        for league in leagues:
            matches = []
            
            # Try OpenLigaDB first (free, no key needed)
            if league in ['bundesliga', 'bundesliga2', '3liga', 'dfb_pokal', 
                          'champions_league', 'europa_league', 'euro_2024', 'world_cup_2022']:
                matches = self.openliga.get_upcoming_matches(league, days)
            
            # Try Football-Data.org
            if not matches:
                matches = self.fdo.get_upcoming_matches(league, days)
            
            # Try API-Football as fallback
            if not matches:
                matches = self.apifb.get_upcoming_matches(league, days)
            
            # Deduplicate
            for m in matches:
                key = f"{m.home_team.name}_{m.away_team.name}_{m.kickoff.date()}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_matches.append(m)
        
        # Sort by kickoff time
        all_matches.sort(key=lambda x: x.kickoff)
        
        return all_matches
    
    def get_all_standings(self) -> Dict[str, List[Dict]]:
        """Get standings for all available leagues"""
        standings = {}
        
        # OpenLigaDB
        for league in ['bundesliga']:
            data = self.openliga.get_standings(league)
            if data:
                standings[league] = data
        
        # Football-Data.org
        for league in ['premier_league', 'la_liga', 'serie_a', 'ligue_1']:
            data = self.fdo.get_standings(league)
            if data:
                standings[league] = data
        
        return standings
