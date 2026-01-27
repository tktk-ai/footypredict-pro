"""
FBRef Data Scraper
Scrapes advanced statistics from fbref.com.

Part of the complete blueprint implementation.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time
import re

logger = logging.getLogger(__name__)

FBREF_BASE = "https://fbref.com/en"


class FBRefScraper:
    """
    Scraper for FBRef advanced football statistics.
    
    Provides:
    - xG (Expected Goals)
    - xA (Expected Assists)
    - Passing statistics
    - Defensive actions
    - Possession metrics
    """
    
    COMPETITION_IDS = {
        'premier-league': '9',
        'la-liga': '12',
        'bundesliga': '20',
        'serie-a': '11',
        'ligue-1': '13',
        'eredivisie': '23',
        'primeira-liga': '32',
        'championship': '10',
    }
    
    def __init__(self, cache_dir: str = "data/cache/fbref"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        
    def fetch_match_report(self, match_url: str) -> Optional[Dict]:
        """Fetch detailed match report."""
        try:
            response = self.session.get(match_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract xG values
            xg_data = {}
            xg_elements = soup.find_all('div', {'class': 'score_xg'})
            if len(xg_elements) >= 2:
                xg_data['home_xg'] = float(xg_elements[0].text.strip())
                xg_data['away_xg'] = float(xg_elements[1].text.strip())
            
            return xg_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch match report: {e}")
            return None
    
    def fetch_team_stats(
        self,
        competition: str,
        season: str = "2024-2025"
    ) -> pd.DataFrame:
        """
        Fetch team-level statistics for a competition.
        """
        if competition not in self.COMPETITION_IDS:
            logger.warning(f"Unknown competition: {competition}")
            return pd.DataFrame()
        
        comp_id = self.COMPETITION_IDS[competition]
        cache_file = self.cache_dir / f"team_stats_{competition}_{season.replace('-', '')}.csv"
        
        if cache_file.exists():
            return pd.read_csv(cache_file)
        
        url = f"{FBREF_BASE}/comps/{comp_id}/{season}/stats/{season}-{competition}-Stats"
        
        try:
            logger.info(f"Fetching team stats: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse tables
            tables = pd.read_html(response.text)
            
            if tables:
                df = tables[0]
                df.to_csv(cache_file, index=False)
                return df
            
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
        
        return pd.DataFrame()
    
    def fetch_player_stats(
        self,
        competition: str,
        season: str = "2024-2025",
        stat_type: str = "standard"
    ) -> pd.DataFrame:
        """
        Fetch player statistics.
        
        Args:
            stat_type: 'standard', 'shooting', 'passing', 'gca', 'defense', 'possession'
        """
        if competition not in self.COMPETITION_IDS:
            return pd.DataFrame()
        
        comp_id = self.COMPETITION_IDS[competition]
        cache_file = self.cache_dir / f"player_{stat_type}_{competition}_{season.replace('-', '')}.csv"
        
        if cache_file.exists():
            return pd.read_csv(cache_file)
        
        stat_urls = {
            'standard': 'stats',
            'shooting': 'shooting',
            'passing': 'passing',
            'gca': 'gca',  # Goal-creating actions
            'defense': 'defense',
            'possession': 'possession',
        }
        
        url = f"{FBREF_BASE}/comps/{comp_id}/{season}/{stat_urls.get(stat_type, 'stats')}/{season}-{competition}-Stats"
        
        try:
            logger.info(f"Fetching player {stat_type} stats")
            time.sleep(3)  # Rate limiting
            
            response = self.session.get(url, timeout=30)
            tables = pd.read_html(response.text)
            
            if tables:
                df = tables[0]
                # Flatten multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns]
                
                df.to_csv(cache_file, index=False)
                return df
                
        except Exception as e:
            logger.error(f"Failed to fetch player stats: {e}")
        
        return pd.DataFrame()
    
    def fetch_xg_data(
        self,
        competition: str,
        season: str = "2024-2025"
    ) -> pd.DataFrame:
        """Fetch match-level xG data."""
        if competition not in self.COMPETITION_IDS:
            return pd.DataFrame()
        
        comp_id = self.COMPETITION_IDS[competition]
        cache_file = self.cache_dir / f"xg_{competition}_{season.replace('-', '')}.csv"
        
        if cache_file.exists():
            return pd.read_csv(cache_file)
        
        url = f"{FBREF_BASE}/comps/{comp_id}/{season}/schedule/{season}-{competition}-Scores-and-Fixtures"
        
        try:
            logger.info(f"Fetching xG data for {competition}")
            time.sleep(3)
            
            response = self.session.get(url, timeout=30)
            tables = pd.read_html(response.text)
            
            for table in tables:
                if 'xG' in str(table.columns):
                    table.to_csv(cache_file, index=False)
                    return table
            
        except Exception as e:
            logger.error(f"Failed to fetch xG data: {e}")
        
        return pd.DataFrame()
    
    def get_team_xg_stats(self, team: str, competition: str) -> Dict:
        """Get xG statistics for a specific team."""
        df = self.fetch_xg_data(competition)
        
        if df.empty:
            return {}
        
        # Filter for team
        home_matches = df[df['Home'] == team] if 'Home' in df.columns else pd.DataFrame()
        away_matches = df[df['Away'] == team] if 'Away' in df.columns else pd.DataFrame()
        
        xg_stats = {
            'team': team,
            'matches': len(home_matches) + len(away_matches),
            'avg_xg_for': 0,
            'avg_xg_against': 0,
        }
        
        # Calculate averages (simplified)
        if 'xG' in df.columns:
            if not home_matches.empty:
                xg_stats['home_xg_avg'] = home_matches['xG'].mean()
            if not away_matches.empty:
                xg_stats['away_xg_avg'] = away_matches['xG'].mean()
        
        return xg_stats


# Global instance
_scraper: Optional[FBRefScraper] = None


def get_scraper() -> FBRefScraper:
    """Get or create FBRef scraper."""
    global _scraper
    if _scraper is None:
        _scraper = FBRefScraper()
    return _scraper


def fetch_team_xg(team: str, competition: str = 'premier-league') -> Dict:
    """Quick function to get team xG stats."""
    return get_scraper().get_team_xg_stats(team, competition)
