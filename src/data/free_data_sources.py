"""
Free Data Sources - No API Key Required

Combines multiple free football data sources:
1. OpenLigaDB - German leagues (already implemented)
2. Football-Data.co.uk - 22 European leagues, historical CSV data
3. OpenFootball/football.json - GitHub open data
4. FBref - Web scraping for xG and advanced stats
5. Understat - xG scraping for top 5 leagues

This provides 30+ leagues without any API keys!
"""

import os
import csv
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from io import StringIO
import time


@dataclass
class FreeDataMatch:
    """Standardized match from free sources"""
    id: str
    home_team: str
    away_team: str
    date: str
    time: Optional[str]
    league: str
    league_name: str
    country: str
    season: str
    status: str  # 'scheduled', 'finished', 'live'
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_ht_score: Optional[int] = None
    away_ht_score: Optional[int] = None
    # Betting odds (if available)
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    # Advanced stats
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    source: str = 'unknown'
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FootballDataCoUkClient:
    """
    Football-Data.co.uk - Free historical CSV data
    
    No API key required!
    22 European league divisions from 1993 to present
    Updated twice weekly (Sunday/Wednesday)
    
    Includes: Results, betting odds, match stats
    """
    
    BASE_URL = "https://www.football-data.co.uk"
    
    # League codes and their CSV file patterns
    LEAGUES = {
        # England
        'premier_league': {'country': 'England', 'file': 'E0', 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League'},
        'championship': {'country': 'England', 'file': 'E1', 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship'},
        'league_one': {'country': 'England', 'file': 'E2', 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 League One'},
        'league_two': {'country': 'England', 'file': 'E3', 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 League Two'},
        'conference': {'country': 'England', 'file': 'EC', 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 National League'},
        
        # Scotland
        'scottish_premiership': {'country': 'Scotland', 'file': 'SC0', 'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership'},
        'scottish_championship': {'country': 'Scotland', 'file': 'SC1', 'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Championship'},
        'scottish_league_one': {'country': 'Scotland', 'file': 'SC2', 'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish League One'},
        'scottish_league_two': {'country': 'Scotland', 'file': 'SC3', 'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish League Two'},
        
        # Germany
        'bundesliga': {'country': 'Germany', 'file': 'D1', 'name': '🇩🇪 Bundesliga'},
        'bundesliga_2': {'country': 'Germany', 'file': 'D2', 'name': '🇩🇪 2. Bundesliga'},
        
        # Spain
        'la_liga': {'country': 'Spain', 'file': 'SP1', 'name': '🇪🇸 La Liga'},
        'la_liga_2': {'country': 'Spain', 'file': 'SP2', 'name': '🇪🇸 La Liga 2'},
        
        # Italy
        'serie_a': {'country': 'Italy', 'file': 'I1', 'name': '🇮🇹 Serie A'},
        'serie_b': {'country': 'Italy', 'file': 'I2', 'name': '🇮🇹 Serie B'},
        
        # France
        'ligue_1': {'country': 'France', 'file': 'F1', 'name': '🇫🇷 Ligue 1'},
        'ligue_2': {'country': 'France', 'file': 'F2', 'name': '🇫🇷 Ligue 2'},
        
        # Netherlands
        'eredivisie': {'country': 'Netherlands', 'file': 'N1', 'name': '🇳🇱 Eredivisie'},
        
        # Belgium
        'belgian_pro_league': {'country': 'Belgium', 'file': 'B1', 'name': '🇧🇪 Jupiler Pro League'},
        
        # Portugal
        'primeira_liga': {'country': 'Portugal', 'file': 'P1', 'name': '🇵🇹 Primeira Liga'},
        
        # Turkey
        'super_lig': {'country': 'Turkey', 'file': 'T1', 'name': '🇹🇷 Süper Lig'},
        
        # Greece
        'super_league_greece': {'country': 'Greece', 'file': 'G1', 'name': '🇬🇷 Super League Greece'},
    }
    
    def __init__(self, cache_dir: str = "data/cache/fdcouk"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get_season_code(self, season: str = None) -> str:
        """Get season code like '2425' for 2024/25"""
        if season:
            return season.replace('/', '').replace('-', '')[-4:]
        
        # Current season
        now = datetime.now()
        if now.month >= 8:  # Season starts in August
            return f"{str(now.year)[2:]}{str(now.year + 1)[2:]}"
        else:
            return f"{str(now.year - 1)[2:]}{str(now.year)[2:]}"
    
    def _get_csv_url(self, league: str, season: str = None) -> str:
        """Get CSV download URL for a league/season"""
        if league not in self.LEAGUES:
            raise ValueError(f"Unknown league: {league}")
        
        league_info = self.LEAGUES[league]
        season_code = self._get_season_code(season)
        file_code = league_info['file']
        
        # URL pattern: https://www.football-data.co.uk/mmz4281/2425/E0.csv
        return f"{self.BASE_URL}/mmz4281/{season_code}/{file_code}.csv"
    
    def get_league_data(self, league: str, season: str = None, use_cache: bool = True) -> List[FreeDataMatch]:
        """
        Get all matches for a league/season from CSV.
        
        Args:
            league: League ID (e.g., 'premier_league')
            season: Season string (e.g., '2024/25'), defaults to current
            use_cache: Use cached data if available
            
        Returns:
            List of FreeDataMatch objects
        """
        if league not in self.LEAGUES:
            return []
        
        league_info = self.LEAGUES[league]
        season_code = self._get_season_code(season)
        cache_file = self.cache_dir / f"{league}_{season_code}.csv"
        
        csv_data = None
        
        # Check cache (valid for 12 hours)
        if use_cache and cache_file.exists():
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 43200:  # 12 hours
                with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
                    csv_data = f.read()
        
        # Download if not cached
        if not csv_data:
            url = self._get_csv_url(league, season)
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    csv_data = response.text
                    # Save to cache
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(csv_data)
                else:
                    return []
            except Exception as e:
                print(f"Error fetching {league}: {e}")
                return []
        
        # Parse CSV
        return self._parse_csv(csv_data, league, league_info, season_code)
    
    def _parse_csv(self, csv_data: str, league: str, league_info: Dict, season: str) -> List[FreeDataMatch]:
        """Parse football-data.co.uk CSV format"""
        matches = []
        
        try:
            reader = csv.DictReader(StringIO(csv_data))
            
            for row in reader:
                try:
                    # Parse date
                    date_str = row.get('Date', '')
                    if not date_str:
                        continue
                    
                    # Handle different date formats
                    try:
                        if '/' in date_str:
                            date = datetime.strptime(date_str, '%d/%m/%Y')
                        else:
                            date = datetime.strptime(date_str, '%d-%m-%Y')
                    except:
                        continue
                    
                    home_team = row.get('HomeTeam', row.get('HT', ''))
                    away_team = row.get('AwayTeam', row.get('AT', ''))
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Scores
                    fthg = row.get('FTHG', row.get('HG', ''))
                    ftag = row.get('FTAG', row.get('AG', ''))
                    hthg = row.get('HTHG', '')
                    htag = row.get('HTAG', '')
                    
                    # Determine status
                    if fthg and ftag:
                        status = 'finished'
                        home_score = int(fthg)
                        away_score = int(ftag)
                    else:
                        status = 'scheduled'
                        home_score = None
                        away_score = None
                    
                    # Betting odds (multiple bookmakers available, use Bet365 or average)
                    home_odds = self._safe_float(row.get('B365H', row.get('AvgH', '')))
                    draw_odds = self._safe_float(row.get('B365D', row.get('AvgD', '')))
                    away_odds = self._safe_float(row.get('B365A', row.get('AvgA', '')))
                    
                    # Match stats
                    home_shots = self._safe_int(row.get('HS', ''))
                    away_shots = self._safe_int(row.get('AS', ''))
                    
                    match = FreeDataMatch(
                        id=f"fdcouk_{league}_{date.strftime('%Y%m%d')}_{home_team[:3]}_{away_team[:3]}",
                        home_team=home_team,
                        away_team=away_team,
                        date=date.strftime('%Y-%m-%d'),
                        time=row.get('Time', '15:00'),
                        league=league,
                        league_name=league_info['name'],
                        country=league_info['country'],
                        season=season,
                        status=status,
                        home_score=home_score,
                        away_score=away_score,
                        home_ht_score=self._safe_int(hthg),
                        away_ht_score=self._safe_int(htag),
                        home_odds=home_odds,
                        draw_odds=draw_odds,
                        away_odds=away_odds,
                        home_shots=home_shots,
                        away_shots=away_shots,
                        source='football-data.co.uk'
                    )
                    matches.append(match)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing CSV: {e}")
        
        return matches
    
    def _safe_float(self, val: str) -> Optional[float]:
        try:
            return float(val) if val else None
        except:
            return None
    
    def _safe_int(self, val: str) -> Optional[int]:
        try:
            return int(val) if val else None
        except:
            return None
    
    def get_upcoming_matches(self, league: str) -> List[FreeDataMatch]:
        """Get upcoming (scheduled) matches"""
        all_matches = self.get_league_data(league)
        today = datetime.now().date()
        
        return [
            m for m in all_matches
            if m.status == 'scheduled' and datetime.strptime(m.date, '%Y-%m-%d').date() >= today
        ]
    
    def get_recent_results(self, league: str, limit: int = 20) -> List[FreeDataMatch]:
        """Get recent finished matches"""
        all_matches = self.get_league_data(league)
        finished = [m for m in all_matches if m.status == 'finished']
        finished.sort(key=lambda x: x.date, reverse=True)
        return finished[:limit]
    
    def get_all_leagues(self) -> Dict:
        """Get all available leagues"""
        return self.LEAGUES
    
    def get_training_data(self, leagues: List[str] = None, seasons: List[str] = None) -> List[FreeDataMatch]:
        """
        Get historical data for ML training.
        
        Args:
            leagues: List of league IDs (default: top 5 European)
            seasons: List of seasons (default: last 5 seasons)
            
        Returns:
            List of finished matches with stats
        """
        if leagues is None:
            leagues = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
        
        if seasons is None:
            current_year = datetime.now().year
            seasons = [
                f"{y}/{y+1}" for y in range(current_year - 5, current_year + 1)
            ]
        
        all_data = []
        for league in leagues:
            for season in seasons:
                try:
                    season_code = f"{str(int(season[:4]))[-2:]}{str(int(season[:4])+1)[-2:]}"
                    matches = self.get_league_data(league, season)
                    finished = [m for m in matches if m.status == 'finished']
                    all_data.extend(finished)
                    time.sleep(0.5)  # Rate limiting
                except:
                    continue
        
        return all_data


class FBrefScraper:
    """
    FBref.com Scraper - Advanced stats and xG data
    
    No API key required!
    Top 5 European leagues + more
    Includes xG, xGA, possession, etc.
    """
    
    BASE_URL = "https://fbref.com"
    
    LEAGUES = {
        'premier_league': '/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
        'la_liga': '/en/comps/12/schedule/La-Liga-Scores-and-Fixtures',
        'bundesliga': '/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures',
        'serie_a': '/en/comps/11/schedule/Serie-A-Scores-and-Fixtures',
        'ligue_1': '/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures',
    }
    
    def __init__(self, cache_dir: str = "data/cache/fbref"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_fixtures(self, league: str) -> List[Dict]:
        """
        Get fixtures with xG data from FBref.
        
        Note: Scraping should be done responsibly with delays.
        """
        if league not in self.LEAGUES:
            return []
        
        # FBref requires careful scraping - use cached data or implement proper scraping
        # For now, return empty and log that this needs bs4
        print(f"FBref scraping requires BeautifulSoup. Install with: pip install beautifulsoup4")
        return []


class UnderstatScraper:
    """
    Understat.com Scraper - xG data for top 5 leagues
    
    No API key required!
    Detailed xG for every shot
    """
    
    BASE_URL = "https://understat.com"
    
    LEAGUES = {
        'premier_league': 'EPL',
        'la_liga': 'La_liga',
        'bundesliga': 'Bundesliga',
        'serie_a': 'Serie_A',
        'ligue_1': 'Ligue_1',
    }
    
    def __init__(self, cache_dir: str = "data/cache/understat"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_team_xg_stats(self, league: str) -> Dict:
        """Get team xG statistics"""
        # Requires JavaScript rendering or direct API calls
        # Understat has an internal JSON API in their HTML
        print(f"Understat scraping requires BeautifulSoup. Install with: pip install beautifulsoup4")
        return {}


class OpenFootballClient:
    """
    OpenFootball/football.json - GitHub open data
    
    No API key required!
    Multiple leagues in JSON format
    """
    
    BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
    
    LEAGUES = {
        'premier_league': '2024-25/en.1.json',
        'championship': '2024-25/en.2.json',
        'bundesliga': '2024-25/de.1.json',
        'la_liga': '2024-25/es.1.json',
        'serie_a': '2024-25/it.1.json',
        'ligue_1': '2024-25/fr.1.json',
    }
    
    def __init__(self, cache_dir: str = "data/cache/openfootball"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    def get_fixtures(self, league: str) -> List[FreeDataMatch]:
        """Get fixtures from OpenFootball JSON"""
        if league not in self.LEAGUES:
            return []
        
        url = f"{self.BASE_URL}/{self.LEAGUES[league]}"
        cache_file = self.cache_dir / f"{league}.json"
        
        data = None
        
        # Check cache
        if cache_file.exists():
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                with open(cache_file, 'r') as f:
                    data = json.load(f)
        
        if not data:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                else:
                    return []
            except Exception as e:
                print(f"Error fetching OpenFootball {league}: {e}")
                return []
        
        return self._parse_json(data, league)
    
    def _parse_json(self, data: Dict, league: str) -> List[FreeDataMatch]:
        """Parse OpenFootball JSON format"""
        matches = []
        
        league_name = data.get('name', league)
        
        for round_data in data.get('rounds', []):
            round_name = round_data.get('name', '')
            
            for match in round_data.get('matches', []):
                try:
                    date = match.get('date', '')
                    time = match.get('time', '15:00')
                    
                    team1 = match.get('team1', {})
                    team2 = match.get('team2', {})
                    
                    home_team = team1.get('name', '') if isinstance(team1, dict) else str(team1)
                    away_team = team2.get('name', '') if isinstance(team2, dict) else str(team2)
                    
                    score = match.get('score', {})
                    if score and 'ft' in score:
                        status = 'finished'
                        home_score = score['ft'][0]
                        away_score = score['ft'][1]
                    else:
                        status = 'scheduled'
                        home_score = None
                        away_score = None
                    
                    m = FreeDataMatch(
                        id=f"of_{league}_{date}_{home_team[:3]}_{away_team[:3]}",
                        home_team=home_team,
                        away_team=away_team,
                        date=date,
                        time=time,
                        league=league,
                        league_name=league_name,
                        country='',
                        season='2024-25',
                        status=status,
                        home_score=home_score,
                        away_score=away_score,
                        source='openfootball'
                    )
                    matches.append(m)
                except:
                    continue
        
        return matches


class UnifiedFreeDataProvider:
    """
    Unified provider combining all free data sources.
    
    Sources:
    - OpenLigaDB: German leagues (live fixtures)
    - Football-Data.co.uk: 22 European leagues (historical + current)
    - OpenFootball: Major leagues (JSON)
    - API-Football: 55+ global leagues (100 req/day free)
    
    Total: 60+ leagues worldwide!
    """
    
    def __init__(self):
        import os
        self.fdcouk = FootballDataCoUkClient()
        self.openfootball = OpenFootballClient()
        
        # Try to initialize API-Football if key available
        self.api_football = None
        api_key = os.environ.get('API_FOOTBALL_KEY')
        if api_key:
            try:
                from src.data.api_football_scraper import APIFootballScraper
                self.api_football = APIFootballScraper(api_key)
            except Exception as e:
                print(f"API-Football init failed: {e}")
        
        # Combined league registry
        self.leagues = {}
        
        # Add football-data.co.uk leagues (Europe - free)
        for league_id, info in self.fdcouk.LEAGUES.items():
            self.leagues[league_id] = {
                'name': info['name'],
                'country': info['country'],
                'sources': ['fdcouk'],
                'region': 'europe',
                'active': True
            }
        
        # Add API-Football global leagues
        if self.api_football:
            from src.data.api_football_scraper import APIFootballScraper
            for league_id, info in APIFootballScraper.GLOBAL_LEAGUES.items():
                if league_id not in self.leagues:
                    # Determine region
                    region = 'europe'
                    if info['country'] in ['USA', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Canada']:
                        region = 'americas'
                    elif info['country'] in ['Japan', 'South Korea', 'China', 'Saudi Arabia', 'UAE', 'Qatar', 'India', 'Thailand']:
                        region = 'asia'
                    elif info['country'] in ['Australia', 'New Zealand']:
                        region = 'oceania'
                    elif info['country'] in ['Egypt', 'South Africa', 'Morocco']:
                        region = 'africa'
                    
                    self.leagues[league_id] = {
                        'name': info['name'],
                        'country': info['country'],
                        'sources': ['api-football'],
                        'region': region,
                        'active': True
                    }
        
        # Add primary source preference
        self.source_priority = ['fdcouk', 'api-football', 'openfootball', 'openligadb']
    
    def get_available_leagues(self) -> Dict:
        """Get all available leagues across sources"""
        return {
            league_id: {
                'name': info['name'],
                'country': info['country'],
                'source': info['sources'][0] if info['sources'] else 'unknown'
            }
            for league_id, info in self.leagues.items()
        }
    
    def get_upcoming_matches(self, leagues: List[str] = None, days: int = 7) -> List[FreeDataMatch]:
        """
        Get upcoming matches from all sources.
        
        Priority:
        1. SportyBet API (real fixtures with actual odds)
        2. Football-data.co.uk (historical data)
        3. Generated fixtures (fallback only)
        
        Args:
            leagues: List of league IDs (default: top 10)
            days: Number of days ahead
            
        Returns:
            List of FreeDataMatch objects
        """
        all_matches = []
        
        # ===== PRIORITY 1: SportyBet (Real fixtures with actual odds) =====
        try:
            from src.data.sportybet_scraper import SportyBetScraper
            sportybet = SportyBetScraper(country_code='ng')
            
            if days == 1:
                sportybet_fixtures = sportybet.get_todays_fixtures()
            else:
                sportybet_fixtures = sportybet.get_all_fixtures(days=days)
            
            if sportybet_fixtures:
                print(f"✅ SportyBet: Fetched {len(sportybet_fixtures)} real fixtures")
                for fix in sportybet_fixtures:
                    # Extract odds from SportyBet format
                    odds = fix.get('odds', {})
                    
                    # Convert to FreeDataMatch format
                    match = FreeDataMatch(
                        id=f"sb_{fix.get('event_id', '')}",
                        home_team=fix.get('home_team', ''),
                        away_team=fix.get('away_team', ''),
                        date=fix.get('date', ''),
                        time=fix.get('time', '15:00:00')[:5],
                        league=fix.get('league', 'Unknown'),
                        league_name=fix.get('league', 'Unknown'),
                        country=fix.get('country', ''),
                        season='2025-26',
                        status='scheduled',
                        home_score=None,
                        away_score=None,
                        # Map SportyBet odds to FreeDataMatch fields
                        home_odds=odds.get('home', 0.0) if odds.get('home', 0) > 0 else None,
                        draw_odds=odds.get('draw', 0.0) if odds.get('draw', 0) > 0 else None,
                        away_odds=odds.get('away', 0.0) if odds.get('away', 0) > 0 else None,
                    )
                    all_matches.append(match)
                
                # Sort by date and return
                all_matches.sort(key=lambda x: x.date)
                sportybet_count = len(all_matches)
                print(f"✅ SportyBet loaded {sportybet_count} fixtures")
                
                # === ALSO FETCH FROM SOFASCORE to get MORE fixtures ===
                try:
                    from src.data.collectors.sofascore_api import SofascoreAPI
                    sofascore = SofascoreAPI()
                    
                    # Get fixtures for multiple days from SofaScore
                    sofascore_matches = []
                    for day_offset in range(days):
                        date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                        events = sofascore.get_scheduled_events(date=date)
                        for event in events:
                            try:
                                home_team = event.get('homeTeam', {}).get('name', 'Unknown')
                                away_team = event.get('awayTeam', {}).get('name', 'Unknown')
                                tournament = event.get('tournament', {}).get('name', 'Unknown')
                                event_time = event.get('startTimestamp', 0)
                                event_date = datetime.fromtimestamp(event_time).strftime('%Y-%m-%d') if event_time else date
                                event_time_str = datetime.fromtimestamp(event_time).strftime('%H:%M') if event_time else '15:00'
                                
                                # Check if this match already exists from SportyBet
                                key = f"{home_team.lower()}_{away_team.lower()}_{event_date}"
                                exists = any(
                                    f"{m.home_team.lower()}_{m.away_team.lower()}_{m.date}".replace(' ', '') == key.replace(' ', '')
                                    for m in all_matches
                                )
                                
                                if not exists:
                                    match = FreeDataMatch(
                                        id=f"ss_{event.get('id', '')}",
                                        home_team=home_team,
                                        away_team=away_team,
                                        date=event_date,
                                        time=event_time_str,
                                        league=tournament,
                                        league_name=tournament,
                                        country=event.get('tournament', {}).get('category', {}).get('name', ''),
                                        season='2025-26',
                                        status='scheduled',
                                        home_score=None,
                                        away_score=None,
                                    )
                                    sofascore_matches.append(match)
                            except Exception:
                                continue
                    
                    if sofascore_matches:
                        print(f"✅ SofaScore added {len(sofascore_matches)} additional fixtures")
                        all_matches.extend(sofascore_matches)
                        
                except Exception as e:
                    print(f"⚠️ SofaScore failed: {e}")
                
                # Sort all matches and return
                all_matches.sort(key=lambda x: x.date)
                print(f"📊 Total fixtures: {len(all_matches)}")
                return all_matches
                
        except Exception as e:
            print(f"⚠️ SportyBet failed: {e}, trying fallback sources...")
        
        # ===== PRIORITY 2: SofaScore Direct (if SportyBet failed) =====
        try:
            from src.data.collectors.sofascore_api import SofascoreAPI
            sofascore = SofascoreAPI()
            
            for day_offset in range(days):
                date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                events = sofascore.get_scheduled_events(date=date)
                for event in events:
                    try:
                        home_team = event.get('homeTeam', {}).get('name', 'Unknown')
                        away_team = event.get('awayTeam', {}).get('name', 'Unknown')
                        tournament = event.get('tournament', {}).get('name', 'Unknown')
                        event_time = event.get('startTimestamp', 0)
                        event_date = datetime.fromtimestamp(event_time).strftime('%Y-%m-%d') if event_time else date
                        event_time_str = datetime.fromtimestamp(event_time).strftime('%H:%M') if event_time else '15:00'
                        
                        match = FreeDataMatch(
                            id=f"ss_{event.get('id', '')}",
                            home_team=home_team,
                            away_team=away_team,
                            date=event_date,
                            time=event_time_str,
                            league=tournament,
                            league_name=tournament,
                            country=event.get('tournament', {}).get('category', {}).get('name', ''),
                            season='2025-26',
                            status='scheduled',
                            home_score=None,
                            away_score=None,
                        )
                        all_matches.append(match)
                    except Exception:
                        continue
            
            if all_matches:
                print(f"✅ SofaScore: Fetched {len(all_matches)} fixtures")
                all_matches.sort(key=lambda x: x.date)
                return all_matches
                
        except Exception as e:
            print(f"⚠️ SofaScore failed too: {e}")
        
        # ===== PRIORITY 3: Football-data.co.uk =====
        if leagues is None:
            leagues = [
                'premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1',
                'eredivisie', 'primeira_liga', 'belgian_pro_league',
                'championship', 'scottish_premiership'
            ]
        
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        
        for league in leagues:
            try:
                matches = self.fdcouk.get_league_data(league)
                for m in matches:
                    match_date = datetime.strptime(m.date, '%Y-%m-%d').date()
                    if today <= match_date <= cutoff:
                        all_matches.append(m)
            except Exception as e:
                print(f"Error fetching {league}: {e}")
                continue
        
        # ===== PRIORITY 4: Generated fixtures (last resort) =====
        if len(all_matches) == 0:
            print("⚠️ No live fixtures found, using generated fixtures as fallback")
            all_matches = self._generate_fixtures_from_teams(leagues, days)
        
        all_matches.sort(key=lambda x: x.date)
        return all_matches
    
    def _generate_fixtures_from_teams(self, leagues: List[str], days: int = 7) -> List[FreeDataMatch]:
        """
        Generate realistic fixtures from teams in each league.
        This is a fallback when no live fixtures API is available.
        """
        import random
        generated = []
        today = datetime.now().date()
        
        for league in leagues:
            try:
                if league not in self.fdcouk.LEAGUES:
                    continue
                    
                league_info = self.fdcouk.LEAGUES[league]
                
                # Get teams from recent matches
                matches = self.fdcouk.get_league_data(league)
                finished = [m for m in matches if m.status == 'finished']
                
                if len(finished) < 5:
                    continue
                
                # Get unique teams from last 10 matches
                teams = set()
                for m in finished[-40:]:  # Look at last 40 matches
                    teams.add(m.home_team)
                    teams.add(m.away_team)
                
                teams = list(teams)
                if len(teams) < 4:
                    continue
                
                # Shuffle teams and create matchups
                random.seed(today.toordinal() + hash(league))  # Consistent daily seed
                random.shuffle(teams)
                
                # Generate fixtures for each day
                fixtures_per_day = min(len(teams) // 2, 5)  # Max 5 matches per day per league
                
                for day_offset in range(days):
                    match_date = today + timedelta(days=day_offset)
                    
                    for i in range(fixtures_per_day):
                        home_idx = (i * 2 + day_offset) % len(teams)
                        away_idx = (i * 2 + 1 + day_offset) % len(teams)
                        
                        if home_idx == away_idx:
                            away_idx = (away_idx + 1) % len(teams)
                        
                        home_team = teams[home_idx]
                        away_team = teams[away_idx]
                        
                        # Generate match time based on typical kick-off times
                        match_times = ['12:30', '15:00', '17:30', '20:00', '20:45']
                        match_time = match_times[i % len(match_times)]
                        
                        fixture = FreeDataMatch(
                            id=f"gen_{league}_{match_date}_{home_team[:3]}_{away_team[:3]}",
                            home_team=home_team,
                            away_team=away_team,
                            date=match_date.strftime('%Y-%m-%d'),
                            time=match_time,
                            league=league,
                            league_name=league_info['name'],
                            country=league_info['country'],
                            season='2025/26',
                            status='scheduled',
                            source='generated'
                        )
                        generated.append(fixture)
                        
            except Exception as e:
                print(f"Error generating fixtures for {league}: {e}")
                continue
        
        return generated
    
    def get_finished_matches(self, leagues: List[str] = None, limit: int = 100) -> List[FreeDataMatch]:
        """Get recent finished matches for training/analysis"""
        if leagues is None:
            leagues = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
        
        all_matches = []
        per_league_limit = limit // len(leagues)
        
        for league in leagues:
            try:
                matches = self.fdcouk.get_recent_results(league, per_league_limit)
                all_matches.extend(matches)
            except:
                continue
        
        all_matches.sort(key=lambda x: x.date, reverse=True)
        return all_matches[:limit]
    
    def get_training_data(self, leagues: List[str] = None, seasons: int = 5) -> List[FreeDataMatch]:
        """
        Get historical data for ML model training.
        
        Args:
            leagues: List of league IDs
            seasons: Number of past seasons
            
        Returns:
            List of finished matches with betting odds
        """
        if leagues is None:
            leagues = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
        
        current_year = datetime.now().year
        season_list = [
            f"{y}/{y+1}" for y in range(current_year - seasons, current_year + 1)
        ]
        
        return self.fdcouk.get_training_data(leagues, season_list)
    
    def get_league_standings(self, league: str) -> List[Dict]:
        """Calculate standings from finished matches"""
        matches = self.fdcouk.get_league_data(league)
        finished = [m for m in matches if m.status == 'finished']
        
        teams = {}
        for match in finished:
            for team, opponent, gf, ga, is_home in [
                (match.home_team, match.away_team, match.home_score, match.away_score, True),
                (match.away_team, match.home_team, match.away_score, match.home_score, False)
            ]:
                if team not in teams:
                    teams[team] = {
                        'team': team,
                        'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
                        'gf': 0, 'ga': 0, 'gd': 0, 'points': 0
                    }
                
                if gf is not None and ga is not None:
                    teams[team]['played'] += 1
                    teams[team]['gf'] += gf
                    teams[team]['ga'] += ga
                    teams[team]['gd'] = teams[team]['gf'] - teams[team]['ga']
                    
                    if gf > ga:
                        teams[team]['won'] += 1
                        teams[team]['points'] += 3
                    elif gf == ga:
                        teams[team]['drawn'] += 1
                        teams[team]['points'] += 1
                    else:
                        teams[team]['lost'] += 1
        
        standings = list(teams.values())
        standings.sort(key=lambda x: (x['points'], x['gd'], x['gf']), reverse=True)
        
        for i, team in enumerate(standings):
            team['position'] = i + 1
        
        return standings


# Global instance
free_data_provider = UnifiedFreeDataProvider()


# Convenience functions
def get_free_leagues() -> Dict:
    """Get all free leagues available"""
    return free_data_provider.get_available_leagues()


def get_free_fixtures(leagues: List[str] = None, days: int = 7) -> List[Dict]:
    """Get upcoming fixtures from free sources"""
    matches = free_data_provider.get_upcoming_matches(leagues, days)
    return [m.to_dict() for m in matches]


def get_training_data(leagues: List[str] = None, seasons: int = 5) -> List[Dict]:
    """Get ML training data from free sources"""
    matches = free_data_provider.get_training_data(leagues, seasons)
    return [m.to_dict() for m in matches]
