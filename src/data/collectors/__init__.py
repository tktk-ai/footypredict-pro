"""Data Collectors Package - Multiple data sources for football statistics."""

from .football_data import FootballDataCollector, get_collector as get_fdcouk
from .fbref_scraper import FBRefScraper, get_scraper as get_fbref
from .understat_api import UnderstatAPI, get_api as get_understat
from .sofascore_api import SofascoreAPI, get_api as get_sofascore
from .statsbomb_loader import StatsBombLoader, get_loader as get_statsbomb

__all__ = [
    'FootballDataCollector', 'get_fdcouk',
    'FBRefScraper', 'get_fbref',
    'UnderstatAPI', 'get_understat',
    'SofascoreAPI', 'get_sofascore',
    'StatsBombLoader', 'get_statsbomb'
]
