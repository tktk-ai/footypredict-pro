"""
Comprehensive Data Collector for Football Match Data
=====================================================

Collects historical match data from multiple free sources:
- OpenLigaDB (Bundesliga)
- TheSportsDB (multiple leagues)
- Football-Data.co.uk (historical with odds)

Target: 200,000+ matches for improved model training.
"""

import requests
import pandas as pd
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class OpenLigaDBCollector:
    """Collect data from OpenLigaDB (German leagues)"""
    
    BASE_URL = "https://api.openligadb.de"
    
    LEAGUES = {
        'bl1': 'Bundesliga',
        'bl2': 'Bundesliga 2',
        'bl3': '3. Liga',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.matches = []
    
    def get_season_matches(self, league: str, season: int) -> List[Dict]:
        """Get all matches for a league and season."""
        url = f"{self.BASE_URL}/getmatchdata/{league}/{season}"
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                matches = response.json()
                return [self._parse_match(m, league) for m in matches if m.get('matchIsFinished')]
        except Exception as e:
            logger.error(f"Error fetching {league} {season}: {e}")
        return []
    
    def _parse_match(self, match: Dict, league: str) -> Dict:
        """Parse OpenLigaDB match to standard format."""
        results = match.get('matchResults', [])
        final = results[-1] if results else {}
        
        return {
            'source': 'openligadb',
            'match_id': f"oldb_{match.get('matchID')}",
            'date': match.get('matchDateTime', '')[:10],
            'time': match.get('matchDateTime', '')[11:16] if len(match.get('matchDateTime', '')) > 10 else '',
            'league': self.LEAGUES.get(league, league),
            'season': match.get('leagueSeason', ''),
            'matchday': match.get('group', {}).get('groupOrderID', 0),
            'home_team': match.get('team1', {}).get('teamName', ''),
            'away_team': match.get('team2', {}).get('teamName', ''),
            'home_score': final.get('pointsTeam1', 0),
            'away_score': final.get('pointsTeam2', 0),
            'finished': match.get('matchIsFinished', False),
        }
    
    def collect_all(self, start_season: int = 2015, end_season: int = 2026) -> pd.DataFrame:
        """Collect all matches from all leagues and seasons."""
        all_matches = []
        
        for league in self.LEAGUES.keys():
            for season in range(start_season, end_season):
                logger.info(f"Collecting {league} {season}...")
                matches = self.get_season_matches(league, season)
                all_matches.extend(matches)
                time.sleep(0.5)  # Rate limiting
        
        df = pd.DataFrame(all_matches)
        logger.info(f"OpenLigaDB: Collected {len(df)} matches")
        return df


class FootballDataUKCollector:
    """Collect historical data from Football-Data.co.uk"""
    
    BASE_URL = "https://www.football-data.co.uk"
    
    LEAGUES = {
        'E0': ('England', 'Premier League'),
        'E1': ('England', 'Championship'),
        'E2': ('England', 'League One'),
        'E3': ('England', 'League Two'),
        'SP1': ('Spain', 'La Liga'),
        'SP2': ('Spain', 'Segunda'),
        'I1': ('Italy', 'Serie A'),
        'I2': ('Italy', 'Serie B'),
        'D1': ('Germany', 'Bundesliga'),
        'D2': ('Germany', 'Bundesliga 2'),
        'F1': ('France', 'Ligue 1'),
        'F2': ('France', 'Ligue 2'),
        'N1': ('Netherlands', 'Eredivisie'),
        'B1': ('Belgium', 'Jupiler League'),
        'P1': ('Portugal', 'Primeira Liga'),
        'T1': ('Turkey', 'Super Lig'),
        'G1': ('Greece', 'Super League'),
        'SC0': ('Scotland', 'Premiership'),
    }
    
    # Column mappings
    RESULT_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
    ODDS_COLS = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA']
    STATS_COLS = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_season_data(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Download season data CSV."""
        # Season format: 2324 for 2023-24
        url = f"{self.BASE_URL}/mmz4281/{season}/{league}.csv"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), encoding='utf-8', on_bad_lines='skip')
                return df
        except Exception as e:
            # Try alternate URL format
            pass
        
        return None
    
    def _parse_dataframe(self, df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
        """Parse Football-Data.co.uk DataFrame to standard format."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        country, league_name = self.LEAGUES.get(league, ('Unknown', league))
        
        # Select available columns
        available_cols = [c for c in self.RESULT_COLS + self.ODDS_COLS + self.STATS_COLS if c in df.columns]
        df = df[available_cols].copy()
        
        # Rename and standardize
        df = df.rename(columns={
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_score',
            'FTAG': 'away_score',
            'FTR': 'result',
            'HTHG': 'ht_home_score',
            'HTAG': 'ht_away_score',
            'HTR': 'ht_result',
            'Date': 'date',
        })
        
        # Add metadata
        df['source'] = 'football-data.co.uk'
        df['league'] = league_name
        df['country'] = country
        df['season'] = season
        df['finished'] = True
        
        # Parse date
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            except:
                pass
        
        return df
    
    def collect_all(self, start_season: int = 2015, end_season: int = 2025) -> pd.DataFrame:
        """Collect all available data."""
        all_dfs = []
        
        for league in self.LEAGUES.keys():
            for year in range(start_season, end_season):
                # Season format: 1516 for 2015-16
                season = f"{str(year)[-2:]}{str(year + 1)[-2:]}"
                
                logger.info(f"Collecting {league} {season}...")
                df = self.get_season_data(league, season)
                
                if df is not None and not df.empty:
                    parsed = self._parse_dataframe(df, league, season)
                    if not parsed.empty:
                        all_dfs.append(parsed)
                
                time.sleep(0.3)  # Rate limiting
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Football-Data.co.uk: Collected {len(combined)} matches")
            return combined
        
        return pd.DataFrame()


class TheSportsDBCollector:
    """Collect data from TheSportsDB API"""
    
    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
    
    LEAGUE_IDS = {
        '4328': 'Premier League',
        '4331': 'Bundesliga',
        '4332': 'Serie A',
        '4334': 'Ligue 1',
        '4335': 'La Liga',
        '4344': 'Primeira Liga',
        '4337': 'Eredivisie',
        '4338': 'Belgian Pro League',
        '4339': 'Turkish Super Lig',
        '4346': 'MLS',
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_past_events(self, league_id: str) -> List[Dict]:
        """Get past events for a league."""
        url = f"{self.BASE_URL}/eventspastleague.php?id={league_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', []) or []
                return [self._parse_event(e) for e in events if e.get('intHomeScore') is not None]
        except Exception as e:
            logger.error(f"Error fetching league {league_id}: {e}")
        
        return []
    
    def _parse_event(self, event: Dict) -> Dict:
        """Parse TheSportsDB event to standard format."""
        return {
            'source': 'thesportsdb',
            'match_id': f"tsdb_{event.get('idEvent')}",
            'date': event.get('dateEvent', ''),
            'time': event.get('strTime', '')[:5] if event.get('strTime') else '',
            'league': event.get('strLeague', ''),
            'season': event.get('strSeason', ''),
            'home_team': event.get('strHomeTeam', ''),
            'away_team': event.get('strAwayTeam', ''),
            'home_score': int(event.get('intHomeScore', 0) or 0),
            'away_score': int(event.get('intAwayScore', 0) or 0),
            'finished': True,
        }
    
    def collect_all(self) -> pd.DataFrame:
        """Collect all available data from leagues."""
        all_matches = []
        
        for league_id in self.LEAGUE_IDS.keys():
            logger.info(f"Collecting TheSportsDB league {league_id}...")
            matches = self.get_past_events(league_id)
            all_matches.extend(matches)
            time.sleep(1)  # Rate limiting
        
        df = pd.DataFrame(all_matches)
        logger.info(f"TheSportsDB: Collected {len(df)} matches")
        return df


class UnifiedDataCollector:
    """
    Unified data collector that combines all sources.
    """
    
    def __init__(self):
        self.openliga = OpenLigaDBCollector()
        self.footballdata = FootballDataUKCollector()
        self.sportsdb = TheSportsDBCollector()
    
    def collect_all(self, save: bool = True) -> pd.DataFrame:
        """Collect data from all sources and combine."""
        logger.info("=" * 60)
        logger.info("Starting comprehensive data collection...")
        logger.info("=" * 60)
        
        dfs = []
        
        # Collect from each source
        try:
            logger.info("\n📡 Collecting from OpenLigaDB...")
            df_openliga = self.openliga.collect_all()
            if not df_openliga.empty:
                dfs.append(df_openliga)
        except Exception as e:
            logger.error(f"OpenLigaDB error: {e}")
        
        try:
            logger.info("\n📡 Collecting from Football-Data.co.uk...")
            df_footballdata = self.footballdata.collect_all()
            if not df_footballdata.empty:
                dfs.append(df_footballdata)
        except Exception as e:
            logger.error(f"Football-Data.co.uk error: {e}")
        
        try:
            logger.info("\n📡 Collecting from TheSportsDB...")
            df_sportsdb = self.sportsdb.collect_all()
            if not df_sportsdb.empty:
                dfs.append(df_sportsdb)
        except Exception as e:
            logger.error(f"TheSportsDB error: {e}")
        
        # Combine all dataframes
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            
            # Standardize columns
            combined = self._standardize(combined)
            
            # Remove duplicates
            combined = combined.drop_duplicates(
                subset=['date', 'home_team', 'away_team'],
                keep='first'
            )
            
            logger.info(f"\n✅ Total matches collected: {len(combined)}")
            
            if save:
                # Save to file
                output_path = PROCESSED_DATA_DIR / "training_data_unified.csv"
                combined.to_csv(output_path, index=False)
                logger.info(f"💾 Saved to {output_path}")
                
                # Also save as parquet for faster loading
                parquet_path = PROCESSED_DATA_DIR / "training_data_unified.parquet"
                combined.to_parquet(parquet_path, index=False)
                logger.info(f"💾 Saved to {parquet_path}")
            
            return combined
        
        return pd.DataFrame()
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        # Ensure required columns exist
        required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Convert scores to int
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)
        
        # Calculate derived fields
        df['total_goals'] = df['home_score'] + df['away_score']
        df['result'] = df.apply(lambda r: 'H' if r['home_score'] > r['away_score'] 
                                else ('A' if r['away_score'] > r['home_score'] else 'D'), axis=1)
        df['btts'] = (df['home_score'] > 0) & (df['away_score'] > 0)
        df['over_25'] = df['total_goals'] > 2.5
        df['over_15'] = df['total_goals'] > 1.5
        
        return df
    
    def get_stats(self) -> Dict:
        """Get statistics about collected data."""
        path = PROCESSED_DATA_DIR / "training_data_unified.csv"
        
        if not path.exists():
            return {'status': 'no_data', 'total_matches': 0}
        
        df = pd.read_csv(path)
        
        return {
            'status': 'ready',
            'total_matches': len(df),
            'leagues': df['league'].nunique() if 'league' in df.columns else 0,
            'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'unknown',
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
        }


# Quick collection functions
def collect_data() -> pd.DataFrame:
    """Run full data collection."""
    collector = UnifiedDataCollector()
    return collector.collect_all(save=True)


def get_training_data() -> pd.DataFrame:
    """Load existing training data."""
    path = PROCESSED_DATA_DIR / "training_data_unified.parquet"
    if path.exists():
        return pd.read_parquet(path)
    
    csv_path = PROCESSED_DATA_DIR / "training_data_unified.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    logger.warning("No training data found. Run collect_data() first.")
    return pd.DataFrame()


if __name__ == "__main__":
    # Run data collection
    df = collect_data()
    
    print("\n" + "=" * 60)
    print("📊 DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total matches: {len(df)}")
    print(f"Leagues: {df['league'].nunique() if 'league' in df.columns else 0}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A")
    
    if 'source' in df.columns:
        print("\nBy source:")
        for source, count in df['source'].value_counts().items():
            print(f"  {source}: {count:,}")
