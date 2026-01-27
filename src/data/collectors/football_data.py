"""
Football-Data.co.uk Data Collector
Collects historical match data from football-data.co.uk.

Part of the complete blueprint implementation.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Base URLs
FDCOUK_BASE = "https://www.football-data.co.uk"
FDCOUK_DATA_URL = "https://www.football-data.co.uk/mmz4281"


class FootballDataCollector:
    """
    Collector for football-data.co.uk historical data.
    
    Provides:
    - Historical match results
    - Betting odds from multiple bookmakers
    - Basic match statistics
    """
    
    LEAGUES = {
        # England
        'E0': ('england', 'premier-league'),
        'E1': ('england', 'championship'),
        'E2': ('england', 'league-one'),
        'E3': ('england', 'league-two'),
        
        # Germany
        'D1': ('germany', 'bundesliga'),
        'D2': ('germany', '2-bundesliga'),
        
        # Spain
        'SP1': ('spain', 'la-liga'),
        'SP2': ('spain', 'segunda'),
        
        # Italy
        'I1': ('italy', 'serie-a'),
        'I2': ('italy', 'serie-b'),
        
        # France
        'F1': ('france', 'ligue-1'),
        'F2': ('france', 'ligue-2'),
        
        # Netherlands
        'N1': ('netherlands', 'eredivisie'),
        
        # Belgium
        'B1': ('belgium', 'jupiler-league'),
        
        # Portugal
        'P1': ('portugal', 'primeira-liga'),
        
        # Turkey
        'T1': ('turkey', 'super-lig'),
        
        # Greece
        'G1': ('greece', 'super-league'),
    }
    
    SEASONS = [
        '2425', '2324', '2223', '2122', '2021', '1920',
        '1819', '1718', '1617', '1516', '1415', '1314'
    ]
    
    def __init__(self, cache_dir: str = "data/cache/fdcouk"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_season_data(
        self,
        league_code: str,
        season: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch season data for a specific league.
        
        Args:
            league_code: League code (e.g., 'E0', 'D1')
            season: Season code (e.g., '2425')
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with match data
        """
        cache_file = self.cache_dir / f"{league_code}_{season}.csv"
        
        # Check cache
        if use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Build URL
        url = f"{FDCOUK_DATA_URL}/{season}/{league_code}.csv"
        
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(response.text)
            
            df = pd.read_csv(cache_file)
            logger.info(f"Fetched {len(df)} matches for {league_code} {season}")
            return df
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {league_code} {season}: {e}")
            return None
    
    def fetch_all_leagues(
        self,
        seasons: List[str] = None,
        leagues: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data for all specified leagues and seasons.
        """
        seasons = seasons or self.SEASONS[:3]  # Last 3 seasons by default
        leagues = leagues or list(self.LEAGUES.keys())
        
        all_data = []
        
        for league in leagues:
            for season in seasons:
                df = self.fetch_season_data(league, season)
                if df is not None and len(df) > 0:
                    df['league_code'] = league
                    df['season'] = season
                    
                    if league in self.LEAGUES:
                        country, league_name = self.LEAGUES[league]
                        df['country'] = country
                        df['league_name'] = league_name
                    
                    all_data.append(df)
                
                time.sleep(0.5)  # Rate limiting
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to common format."""
        column_mapping = {
            'Date': 'match_date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',
            'HTHG': 'home_goals_ht',
            'HTAG': 'away_goals_ht',
            'HTR': 'result_ht',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_on_target',
            'AST': 'away_shots_on_target',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HF': 'home_fouls',
            'AF': 'away_fouls',
            'HY': 'home_yellow',
            'AY': 'away_yellow',
            'HR': 'home_red',
            'AR': 'away_red',
            # Odds
            'B365H': 'odds_home_b365',
            'B365D': 'odds_draw_b365',
            'B365A': 'odds_away_b365',
            'BWH': 'odds_home_bwin',
            'BWD': 'odds_draw_bwin',
            'BWA': 'odds_away_bwin',
            'PSH': 'odds_home_pinnacle',
            'PSD': 'odds_draw_pinnacle',
            'PSA': 'odds_away_pinnacle',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Parse date
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'], format='%d/%m/%Y', errors='coerce')
        
        return df
    
    def get_upcoming_fixtures(self) -> pd.DataFrame:
        """Get upcoming fixtures from the current season."""
        # Current season
        current_season = '2526'  # Adjust as needed
        
        fixtures = []
        for league in self.LEAGUES:
            url = f"{FDCOUK_BASE}/fixtures/{league}.csv"
            try:
                df = pd.read_csv(url)
                df['league_code'] = league
                fixtures.append(df)
            except Exception:
                continue
        
        if fixtures:
            return pd.concat(fixtures, ignore_index=True)
        return pd.DataFrame()


# Global instance
_collector: Optional[FootballDataCollector] = None


def get_collector() -> FootballDataCollector:
    """Get or create football data collector."""
    global _collector
    if _collector is None:
        _collector = FootballDataCollector()
    return _collector


def fetch_historical_data(
    leagues: List[str] = None,
    seasons: List[str] = None
) -> pd.DataFrame:
    """Quick function to fetch historical data."""
    collector = get_collector()
    return collector.fetch_all_leagues(seasons, leagues)
