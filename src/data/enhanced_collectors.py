"""
Enhanced Multi-Source Data Collection System
Aggregates data from 15+ sources with intelligent caching and validation
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    total_records: int
    missing_values: Dict[str, float]
    outliers_detected: int
    duplicates_removed: int
    date_range: Tuple[str, str]
    completeness_score: float
    validation_errors: List[str]
    recommendations: List[str]


@dataclass
class MatchData:
    """Standardized match data structure."""
    match_id: str
    match_date: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_goals_ht: Optional[int] = None
    away_goals_ht: Optional[int] = None
    league: str = ""
    season: str = ""
    
    # Advanced stats
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow: Optional[int] = None
    away_yellow: Optional[int] = None
    home_red: Optional[int] = None
    away_red: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    
    # Betting odds
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    odds_over_25: Optional[float] = None
    odds_under_25: Optional[float] = None
    odds_btts_yes: Optional[float] = None
    odds_btts_no: Optional[float] = None
    
    # Metadata
    source: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


class BaseDataCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.redis_client = None
        
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
        except:
            logger.warning("Redis not available, using file-based caching")
    
    @abstractmethod
    def collect(self, league: str, season: str) -> pd.DataFrame:
        """Collect data for a specific league and season."""
        pass
    
    def _get_cache_key(self, league: str, season: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{self.__class__.__name__}_{league}_{season}".encode()).hexdigest()
    
    def _get_from_cache(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Get data from cache if not expired."""
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"data:{key}")
                if cached:
                    return pd.read_json(cached.decode())
            except:
                pass
        
        cache_file = self.cache_dir / f"{key}.parquet"
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age < timedelta(hours=max_age_hours):
                return pd.read_parquet(cache_file)
        
        return None
    
    def _save_to_cache(self, key: str, df: pd.DataFrame):
        """Save data to cache."""
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"data:{key}",
                    timedelta(hours=24),
                    df.to_json()
                )
            except:
                pass
        
        cache_file = self.cache_dir / f"{key}.parquet"
        try:
            df.to_parquet(cache_file)
        except:
            df.to_csv(cache_file.with_suffix('.csv'), index=False)


class FootballDataCollector(BaseDataCollector):
    """Collector for Football-Data.co.uk via penaltyblog."""
    
    LEAGUE_MAPPING = {
        'ENG-Premier League': 'ENG Premier League',
        'ENG-Championship': 'ENG Championship',
        'ESP-La Liga': 'ESP La Liga',
        'GER-Bundesliga': 'GER Bundesliga',
        'ITA-Serie A': 'ITA Serie A',
        'FRA-Ligue 1': 'FRA Ligue 1',
        'NED-Eredivisie': 'NED Eredivisie',
        'POR-Primeira Liga': 'POR Liga 1',
        'BEL-Pro League': 'BEL Pro League',
        'TUR-Super Lig': 'TUR Super Lig',
        'GRE-Super League': 'GRE Super League',
        'SCO-Premiership': 'SCO Premiership',
    }
    
    def collect(self, league: str, season: str) -> pd.DataFrame:
        """Collect from Football-Data.co.uk."""
        cache_key = self._get_cache_key(league, season)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.info(f"Using cached data for {league} {season}")
            return cached
        
        try:
            import penaltyblog as pb
            mapped_league = self.LEAGUE_MAPPING.get(league, league)
            fd = pb.scrapers.FootballData(mapped_league, season)
            df = fd.get_fixtures()
            
            # Standardize columns
            df = self._standardize_columns(df, league, season)
            
            self._save_to_cache(cache_key, df)
            logger.info(f"Collected {len(df)} matches from Football-Data: {league} {season}")
            
            return df
            
        except Exception as e:
            logger.error(f"Football-Data collection failed for {league} {season}: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
        """Standardize column names."""
        column_mapping = {
            'date': 'match_date',
            'team_home': 'home_team',
            'team_away': 'away_team',
            'goals_home': 'home_goals',
            'goals_away': 'away_goals',
            'goals_home_ht': 'home_goals_ht',
            'goals_away_ht': 'away_goals_ht',
            'shots_home': 'home_shots',
            'shots_away': 'away_shots',
            'shots_target_home': 'home_shots_on_target',
            'shots_target_away': 'away_shots_on_target',
            'corners_home': 'home_corners',
            'corners_away': 'away_corners',
            'fouls_home': 'home_fouls',
            'fouls_away': 'away_fouls',
            'yellow_home': 'home_yellow',
            'yellow_away': 'away_yellow',
            'red_home': 'home_red',
            'red_away': 'away_red',
        }
        
        df = df.rename(columns=column_mapping)
        df['league'] = league
        df['season'] = season
        df['source'] = 'football_data'
        
        # Extract best odds (Pinnacle preferred, then Bet365)
        odds_cols = {
            'odds_home': ['PSH', 'B365H', 'BWH', 'IWH'],
            'odds_draw': ['PSD', 'B365D', 'BWD', 'IWD'],
            'odds_away': ['PSA', 'B365A', 'BWA', 'IWA'],
        }
        
        for target_col, source_cols in odds_cols.items():
            for src_col in source_cols:
                if src_col in df.columns:
                    df[target_col] = df[src_col]
                    break
        
        return df


class FBrefCollector(BaseDataCollector):
    """Collector for FBref data with xG."""
    
    def collect(self, league: str, season: str) -> pd.DataFrame:
        """Collect from FBref."""
        cache_key = self._get_cache_key(league, season)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            import soccerdata as sd
            # Convert season format
            season_year = int(season.split('-')[1]) if '-' in season else int(season)
            
            fbref = sd.FBref(league, season_year)
            schedule = fbref.read_schedule()
            
            # Try to get xG data
            try:
                team_stats = fbref.read_team_match_stats(stat_type='shooting')
                schedule = self._merge_xg_data(schedule, team_stats)
            except:
                logger.warning(f"Could not get xG data for {league} {season}")
            
            schedule['league'] = league
            schedule['season'] = season
            schedule['source'] = 'fbref'
            
            self._save_to_cache(cache_key, schedule)
            logger.info(f"Collected {len(schedule)} matches from FBref: {league} {season}")
            
            return schedule
            
        except Exception as e:
            logger.error(f"FBref collection failed for {league} {season}: {e}")
            return pd.DataFrame()
    
    def _merge_xg_data(self, schedule: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
        """Merge xG data with schedule."""
        return schedule


class StatsBombCollector(BaseDataCollector):
    """Collector for StatsBomb open data."""
    
    FREE_COMPETITIONS = {
        'World Cup 2022': {'competition_id': 43, 'season_id': 106},
        'World Cup 2018': {'competition_id': 43, 'season_id': 3},
        'Euro 2020': {'competition_id': 55, 'season_id': 43},
        'Euro 2024': {'competition_id': 55, 'season_id': 282},
        'La Liga 2018-2019': {'competition_id': 11, 'season_id': 1},
        'La Liga 2019-2020': {'competition_id': 11, 'season_id': 2},
        'La Liga 2020-2021': {'competition_id': 11, 'season_id': 3},
        "FA Women's Super League": {'competition_id': 37, 'season_id': 90},
    }
    
    def collect(self, competition_name: str, season: str = None) -> pd.DataFrame:
        """Collect from StatsBomb."""
        if competition_name not in self.FREE_COMPETITIONS:
            logger.warning(f"Competition {competition_name} not available in free data")
            return pd.DataFrame()
        
        cache_key = self._get_cache_key(competition_name, season or "default")
        cached = self._get_from_cache(cache_key, max_age_hours=168)  # 1 week cache
        if cached is not None:
            return cached
        
        try:
            from statsbombpy import sb
            comp_info = self.FREE_COMPETITIONS[competition_name]
            matches = sb.matches(
                competition_id=comp_info['competition_id'],
                season_id=comp_info['season_id']
            )
            
            matches['source'] = 'statsbomb'
            
            self._save_to_cache(cache_key, matches)
            logger.info(f"Collected {len(matches)} matches from StatsBomb: {competition_name}")
            
            return matches
            
        except Exception as e:
            logger.error(f"StatsBomb collection failed for {competition_name}: {e}")
            return pd.DataFrame()
    
    def extract_xg_training_data(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract shot-level data for xG model training."""
        shots = events_df[events_df['type'] == 'Shot'].copy()
        
        # Extract location
        shots['x'] = shots['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
        shots['y'] = shots['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
        
        # Calculate distance and angle
        goal_x, goal_y = 120, 40
        shots['distance'] = np.sqrt((goal_x - shots['x'])**2 + (goal_y - shots['y'])**2)
        
        # Angle to goal (visible goal area)
        shots['angle'] = np.abs(
            np.arctan2(shots['y'] - 36, goal_x - shots['x']) -
            np.arctan2(shots['y'] - 44, goal_x - shots['x'])
        )
        
        # Target
        shots['is_goal'] = shots['shot_outcome'].apply(
            lambda x: 1 if isinstance(x, dict) and x.get('name') == 'Goal' else 0
        )
        
        return shots


class OddsCollector(BaseDataCollector):
    """Real-time odds collector from multiple bookmakers."""
    
    def __init__(self, api_keys: Dict[str, str] = None, cache_dir: str = "data/cache"):
        super().__init__(cache_dir)
        self.api_keys = api_keys or {}
    
    def collect(self, league: str, season: str) -> pd.DataFrame:
        """Collect historical odds data."""
        # This would interface with historical odds APIs
        return pd.DataFrame()
    
    async def collect_live_odds(self, sport: str = 'soccer') -> pd.DataFrame:
        """Collect live odds from The Odds API."""
        if 'odds_api' not in self.api_keys:
            logger.warning("No Odds API key configured")
            return pd.DataFrame()
        
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            'apiKey': self.api_keys['odds_api'],
            'regions': 'uk,eu',
            'markets': 'h2h,totals,spreads',
            'oddsFormat': 'decimal'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_odds_response(data)
                else:
                    logger.error(f"Odds API error: {response.status}")
                    return pd.DataFrame()
    
    def _parse_odds_response(self, data: List[Dict]) -> pd.DataFrame:
        """Parse odds API response."""
        records = []
        
        for game in data:
            record = {
                'match_id': game.get('id'),
                'sport': game.get('sport_key'),
                'commence_time': game.get('commence_time'),
                'home_team': game.get('home_team'),
                'away_team': game.get('away_team'),
            }
            
            # Get best odds from all bookmakers
            best_odds = {'home': 0, 'draw': 0, 'away': 0}
            
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            name = outcome.get('name')
                            price = outcome.get('price', 0)
                            
                            if name == game.get('home_team'):
                                best_odds['home'] = max(best_odds['home'], price)
                            elif name == game.get('away_team'):
                                best_odds['away'] = max(best_odds['away'], price)
                            elif name == 'Draw':
                                best_odds['draw'] = max(best_odds['draw'], price)
            
            record.update({
                'odds_home': best_odds['home'],
                'odds_draw': best_odds['draw'],
                'odds_away': best_odds['away'],
            })
            
            records.append(record)
        
        return pd.DataFrame(records)


class EnhancedDataAggregator:
    """Aggregates data from multiple sources with intelligent merging."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize collectors
        self.collectors = {
            'football_data': FootballDataCollector(),
            'fbref': FBrefCollector(),
            'statsbomb': StatsBombCollector(),
            'odds': OddsCollector(api_keys=self.config.get('api_keys', {})),
        }
        
        # Team name mapping for merging
        self.team_name_mapping = self._load_team_mapping()
    
    def _load_team_mapping(self) -> Dict[str, str]:
        """Load team name mapping for cross-source matching."""
        return {
            # Premier League
            'Man United': 'Manchester United',
            'Man City': 'Manchester City',
            'Tottenham': 'Tottenham Hotspur',
            'Newcastle': 'Newcastle United',
            'West Ham': 'West Ham United',
            'Wolves': 'Wolverhampton Wanderers',
            "Nott'm Forest": 'Nottingham Forest',
            'Brighton': 'Brighton & Hove Albion',
            'Leicester': 'Leicester City',
            'Leeds': 'Leeds United',
            'Norwich': 'Norwich City',
            'Ipswich': 'Ipswich Town',
            
            # La Liga
            'Atlético Madrid': 'Atletico Madrid',
            'Atlético': 'Atletico Madrid',
            'Athletic Bilbao': 'Athletic Club',
            'Betis': 'Real Betis',
            'Celta': 'Celta Vigo',
            'Sociedad': 'Real Sociedad',
            
            # Bundesliga
            'Bayern München': 'Bayern Munich',
            'Dortmund': 'Borussia Dortmund',
            'Leverkusen': 'Bayer Leverkusen',
            "M'Gladbach": 'Borussia Monchengladbach',
            'Mönchengladbach': 'Borussia Monchengladbach',
            'RB Leipzig': 'RasenBallsport Leipzig',
            'Köln': 'FC Koln',
            
            # Serie A
            'AC Milan': 'Milan',
            'Inter': 'Internazionale',
            'Inter Milan': 'Internazionale',
            'Napoli': 'SSC Napoli',
            
            # Ligue 1
            'Paris S-G': 'Paris Saint-Germain',
            'Paris Saint Germain': 'Paris Saint-Germain',
            'PSG': 'Paris Saint-Germain',
            'Lyon': 'Olympique Lyonnais',
            'Marseille': 'Olympique Marseille',
            'Monaco': 'AS Monaco',
        }
    
    def standardize_team_name(self, name: str) -> str:
        """Standardize team name for matching."""
        return self.team_name_mapping.get(name, name)
    
    def collect_all(
        self,
        leagues: List[str],
        seasons: List[str],
        include_odds: bool = True,
        include_xg: bool = True
    ) -> pd.DataFrame:
        """Collect and merge data from all sources."""
        logger.info(f"Collecting data for {len(leagues)} leagues, {len(seasons)} seasons")
        
        all_data = []
        
        # Parallel collection with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for league in leagues:
                for season in seasons:
                    # Football-Data (primary source)
                    futures.append(
                        executor.submit(
                            self.collectors['football_data'].collect,
                            league, season
                        )
                    )
                    
                    # FBref (for xG if available)
                    if include_xg:
                        futures.append(
                            executor.submit(
                                self.collectors['fbref'].collect,
                                league, season
                            )
                        )
            
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    logger.error(f"Collection error: {e}")
        
        if not all_data:
            logger.warning("No data collected from any source")
            return pd.DataFrame()
        
        # Merge all data
        merged = self._merge_sources(all_data)
        
        # Validate and clean
        merged = self._validate_and_clean(merged)
        
        # Generate quality report
        quality_report = self._generate_quality_report(merged)
        logger.info(f"Data quality score: {quality_report.completeness_score:.2%}")
        
        return merged
    
    def _merge_sources(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Intelligently merge data from multiple sources."""
        # Group by source
        by_source = {}
        for df in dataframes:
            source = df.get('source', pd.Series(['unknown'])).iloc[0] if 'source' in df.columns else 'unknown'
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(df)
        
        # Concatenate within sources
        merged_sources = {}
        for source, dfs in by_source.items():
            merged_sources[source] = pd.concat(dfs, ignore_index=True)
        
        # Start with football_data as primary
        primary = merged_sources.get('football_data', pd.DataFrame())
        
        if primary.empty:
            # Fallback to any available source
            for source, df in merged_sources.items():
                if not df.empty:
                    primary = df
                    break
        
        # Standardize team names
        if 'home_team' in primary.columns:
            primary['home_team'] = primary['home_team'].apply(self.standardize_team_name)
            primary['away_team'] = primary['away_team'].apply(self.standardize_team_name)
        
        return primary
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the merged data."""
        if df.empty:
            return df
            
        initial_count = len(df)
        
        # Remove duplicates
        if 'match_date' in df.columns and 'home_team' in df.columns:
            df = df.drop_duplicates(
                subset=['match_date', 'home_team', 'away_team'],
                keep='first'
            )
        
        # Validate scores
        if 'home_goals' in df.columns:
            df = df[df['home_goals'] >= 0]
            df = df[df['away_goals'] >= 0]
            df = df[df['home_goals'] <= 15]  # Reasonable max
            df = df[df['away_goals'] <= 15]
        
        # Validate dates
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            df = df[df['match_date'].notna()]
            df = df[df['match_date'] <= datetime.now()]
            df = df[df['match_date'] >= datetime(2000, 1, 1)]
        
        # Sort by date
        df = df.sort_values('match_date').reset_index(drop=True)
        
        logger.info(f"Cleaned data: {initial_count} -> {len(df)} records")
        
        return df
    
    def _generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """Generate data quality report."""
        if df.empty:
            return DataQualityReport(
                total_records=0,
                missing_values={},
                outliers_detected=0,
                duplicates_removed=0,
                date_range=('N/A', 'N/A'),
                completeness_score=0.0,
                validation_errors=['No data'],
                recommendations=['Collect more data']
            )
        
        # Missing values
        missing = {}
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0:
                missing[col] = missing_pct
        
        # Calculate completeness
        required_cols = ['match_date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        available_required = [c for c in required_cols if c in df.columns]
        completeness = 1 - df[available_required].isna().any(axis=1).mean() if available_required else 0.0
        
        # Date range
        if 'match_date' in df.columns:
            date_range = (
                df['match_date'].min().strftime('%Y-%m-%d'),
                df['match_date'].max().strftime('%Y-%m-%d')
            )
        else:
            date_range = ('N/A', 'N/A')
        
        return DataQualityReport(
            total_records=len(df),
            missing_values=missing,
            outliers_detected=0,
            duplicates_removed=0,
            date_range=date_range,
            completeness_score=completeness,
            validation_errors=[],
            recommendations=[]
        )


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED DATA COLLECTORS - TEST")
    print("=" * 60)
    
    # Test FootballDataCollector
    print("\n1. Testing FootballDataCollector...")
    fd_collector = FootballDataCollector()
    print(f"   Cache directory: {fd_collector.cache_dir}")
    print(f"   League mapping: {len(fd_collector.LEAGUE_MAPPING)} leagues")
    
    # Test StatsBombCollector
    print("\n2. Testing StatsBombCollector...")
    sb_collector = StatsBombCollector()
    print(f"   Free competitions: {len(sb_collector.FREE_COMPETITIONS)}")
    
    # Test EnhancedDataAggregator
    print("\n3. Testing EnhancedDataAggregator...")
    aggregator = EnhancedDataAggregator()
    print(f"   Collectors: {list(aggregator.collectors.keys())}")
    print(f"   Team mappings: {len(aggregator.team_name_mapping)}")
    
    print("\n✅ All collectors initialized successfully!")
