"""
Data Source Integration
Connects all blueprint data collectors to the main application.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import collectors
try:
    from src.data.collectors.football_data import get_collector as get_fdcouk_collector
except ImportError:
    get_fdcouk_collector = None
    logger.warning("Football-data collector not available")

try:
    from src.data.collectors.fbref_scraper import get_scraper as get_fbref_scraper
except ImportError:
    get_fbref_scraper = None
    logger.warning("FBRef scraper not available")

try:
    from src.data.collectors.understat_api import get_api as get_understat_api
except ImportError:
    get_understat_api = None
    logger.warning("Understat API not available")

try:
    from src.data.collectors.sofascore_api import get_api as get_sofascore_api
except ImportError:
    get_sofascore_api = None
    logger.warning("Sofascore API not available")

try:
    from src.data.collectors.statsbomb_loader import get_loader as get_statsbomb_loader
except ImportError:
    get_statsbomb_loader = None
    logger.warning("StatsBomb loader not available")


class DataSourceManager:
    """
    Manages all data sources and provides unified data access.
    
    Connects:
    - Football-Data.co.uk (historical results, odds)
    - FBRef (advanced stats)
    - Understat (xG data)
    - Sofascore (live data)
    - StatsBomb (open data)
    """
    
    def __init__(self):
        self.collectors = {}
        self._initialize_collectors()
        
    def _initialize_collectors(self):
        """Initialize all available collectors."""
        if get_fdcouk_collector:
            try:
                self.collectors['football_data'] = get_fdcouk_collector()
                logger.info("✅ Football-Data.co.uk collector initialized")
            except Exception as e:
                logger.error(f"Failed to init football-data: {e}")
        
        if get_fbref_scraper:
            try:
                self.collectors['fbref'] = get_fbref_scraper()
                logger.info("✅ FBRef scraper initialized")
            except Exception as e:
                logger.error(f"Failed to init fbref: {e}")
        
        if get_understat_api:
            try:
                self.collectors['understat'] = get_understat_api()
                logger.info("✅ Understat API initialized")
            except Exception as e:
                logger.error(f"Failed to init understat: {e}")
        
        if get_sofascore_api:
            try:
                self.collectors['sofascore'] = get_sofascore_api()
                logger.info("✅ Sofascore API initialized")
            except Exception as e:
                logger.error(f"Failed to init sofascore: {e}")
        
        if get_statsbomb_loader:
            try:
                self.collectors['statsbomb'] = get_statsbomb_loader()
                logger.info("✅ StatsBomb loader initialized")
            except Exception as e:
                logger.error(f"Failed to init statsbomb: {e}")
    
    def get_status(self) -> Dict:
        """Get status of all data sources."""
        return {
            'sources': list(self.collectors.keys()),
            'count': len(self.collectors),
            'available': {
                'football_data': 'football_data' in self.collectors,
                'fbref': 'fbref' in self.collectors,
                'understat': 'understat' in self.collectors,
                'sofascore': 'sofascore' in self.collectors,
                'statsbomb': 'statsbomb' in self.collectors,
            }
        }
    
    def fetch_upcoming_fixtures(
        self,
        days_ahead: int = 7,
        leagues: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch upcoming fixtures from all sources.
        
        Args:
            days_ahead: Number of days to look ahead
            leagues: Specific leagues to filter
            
        Returns:
            Combined DataFrame of upcoming fixtures
        """
        all_fixtures = []
        
        # Try Sofascore first (best for live data)
        if 'sofascore' in self.collectors:
            try:
                fixtures = self.collectors['sofascore'].get_fixtures(days=days_ahead)
                if fixtures is not None and len(fixtures) > 0:
                    fixtures['source'] = 'sofascore'
                    all_fixtures.append(fixtures)
                    logger.info(f"Got {len(fixtures)} fixtures from Sofascore")
            except Exception as e:
                logger.error(f"Sofascore fixtures error: {e}")
        
        # Try Football-Data
        if 'football_data' in self.collectors:
            try:
                fixtures = self.collectors['football_data'].get_upcoming_fixtures()
                if fixtures is not None and len(fixtures) > 0:
                    fixtures['source'] = 'football_data'
                    all_fixtures.append(fixtures)
                    logger.info(f"Got {len(fixtures)} fixtures from Football-Data")
            except Exception as e:
                logger.error(f"Football-Data fixtures error: {e}")
        
        if all_fixtures:
            combined = pd.concat(all_fixtures, ignore_index=True)
            # Remove duplicates based on teams and date
            if 'home_team' in combined.columns and 'away_team' in combined.columns:
                combined = combined.drop_duplicates(
                    subset=['home_team', 'away_team'], 
                    keep='first'
                )
            return combined
        
        return pd.DataFrame()
    
    def fetch_historical_data(
        self,
        seasons: List[str] = None,
        leagues: List[str] = None
    ) -> pd.DataFrame:
        """Fetch historical match data from all sources."""
        all_data = []
        
        # Football-Data.co.uk (primary source)
        if 'football_data' in self.collectors:
            try:
                data = self.collectors['football_data'].fetch_all_leagues(
                    seasons=seasons,
                    leagues=leagues
                )
                if data is not None and len(data) > 0:
                    all_data.append(data)
                    logger.info(f"Got {len(data)} matches from Football-Data")
            except Exception as e:
                logger.error(f"Football-Data historical error: {e}")
        
        # StatsBomb (free open data)
        if 'statsbomb' in self.collectors:
            try:
                data = self.collectors['statsbomb'].load_competitions()
                if data is not None and len(data) > 0:
                    all_data.append(data)
                    logger.info(f"Got {len(data)} matches from StatsBomb")
            except Exception as e:
                logger.error(f"StatsBomb error: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def fetch_xg_data(
        self,
        league: str = None,
        team: str = None
    ) -> pd.DataFrame:
        """Fetch expected goals data from Understat."""
        if 'understat' not in self.collectors:
            logger.warning("Understat API not available")
            return pd.DataFrame()
        
        try:
            return self.collectors['understat'].get_team_xg(
                league=league,
                team=team
            )
        except Exception as e:
            logger.error(f"Understat xG error: {e}")
            return pd.DataFrame()
    
    def fetch_advanced_stats(
        self,
        league: str = None,
        season: str = None
    ) -> pd.DataFrame:
        """Fetch advanced statistics from FBRef."""
        if 'fbref' not in self.collectors:
            logger.warning("FBRef scraper not available")
            return pd.DataFrame()
        
        try:
            return self.collectors['fbref'].get_league_stats(
                league=league,
                season=season
            )
        except Exception as e:
            logger.error(f"FBRef stats error: {e}")
            return pd.DataFrame()
    
    def refresh_all_data(self) -> Dict:
        """Refresh data from all sources."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'total_fixtures': 0,
            'total_historical': 0
        }
        
        # Fetch fixtures
        fixtures = self.fetch_upcoming_fixtures(days_ahead=14)
        results['total_fixtures'] = len(fixtures)
        
        # Fetch historical
        historical = self.fetch_historical_data(seasons=['2425', '2324'])
        results['total_historical'] = len(historical)
        
        # Source status
        results['sources'] = self.get_status()
        
        logger.info(f"Data refresh complete: {results['total_fixtures']} fixtures, {results['total_historical']} historical")
        
        return results


# Global instance
_manager: Optional[DataSourceManager] = None


def get_data_manager() -> DataSourceManager:
    """Get or create the data source manager."""
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager


def fetch_all_fixtures(days: int = 7) -> pd.DataFrame:
    """Convenience function to fetch upcoming fixtures."""
    return get_data_manager().fetch_upcoming_fixtures(days_ahead=days)


def get_data_status() -> Dict:
    """Get current data source status."""
    return get_data_manager().get_status()
