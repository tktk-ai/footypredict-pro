"""
GitHub Data Collector

Fetches football datasets from GitHub repositories and web sources:
- football.csv - Open public domain football data
- jokecamp/FootballData - JSON/CSV odds data
- understat.com - xG data
- fbref.com - Advanced statistics
"""

import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging
import json
import time

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "github"


class GitHubCollector:
    """Clones and processes GitHub football datasets"""
    
    # GitHub raw file URLs for direct download
    GITHUB_SOURCES = {
        "football_csv": {
            "base_url": "https://raw.githubusercontent.com/openfootball/football.json/master",
            "files": ["2023-24/en.1.json", "2023-24/de.1.json", "2023-24/es.1.json"],
            "format": "json"
        },
        "jokecamp_football": {
            "base_url": "https://raw.githubusercontent.com/jokecamp/FootballData/master",
            "files": ["openFootballData/stadiums.json", "openFootballData/countries.json"],
            "format": "json"
        }
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_github_file(self, url: str, output_name: str) -> bool:
        """Download a single file from GitHub"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            output_path = self.output_dir / output_name
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✓ Downloaded: {output_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def fetch_football_csv(self) -> pd.DataFrame:
        """Fetch data from football.csv / openfootball project"""
        all_data = []
        
        for source_name, config in self.GITHUB_SOURCES.items():
            base_url = config["base_url"]
            
            for file_path in config["files"]:
                url = f"{base_url}/{file_path}"
                output_name = f"{source_name}_{file_path.replace('/', '_')}"
                
                if self.download_github_file(url, output_name):
                    # Parse based on format
                    file_path = self.output_dir / output_name
                    if config["format"] == "json":
                        try:
                            with open(file_path) as f:
                                data = json.load(f)
                            # Convert to DataFrame if it's match data
                            if isinstance(data, dict) and "matches" in data:
                                df = pd.DataFrame(data["matches"])
                                all_data.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to parse {file_path}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def fetch_understat_xg(self, leagues: List[str] = None, seasons: List[str] = None) -> pd.DataFrame:
        """Fetch xG data from understat.com API"""
        try:
            from understatapi import UnderstatClient
            
            if leagues is None:
                leagues = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
            
            if seasons is None:
                seasons = ["2024", "2023", "2022", "2021", "2020"]
            
            all_data = []
            
            with UnderstatClient() as client:
                for league in leagues:
                    for season in seasons:
                        try:
                            logger.info(f"Fetching xG: {league} {season}")
                            
                            # Get league fixtures
                            fixtures = client.league(league).get_match_data(season)
                            
                            for match in fixtures:
                                all_data.append({
                                    'league': league,
                                    'season': season,
                                    'home_team': match.get('h', {}).get('title', ''),
                                    'away_team': match.get('a', {}).get('title', ''),
                                    'home_goals': match.get('goals', {}).get('h', 0),
                                    'away_goals': match.get('goals', {}).get('a', 0),
                                    'home_xg': float(match.get('xG', {}).get('h', 0)),
                                    'away_xg': float(match.get('xG', {}).get('a', 0)),
                                    'date': match.get('datetime', '')
                                })
                            
                            time.sleep(0.5)  # Rate limiting
                            
                        except Exception as e:
                            logger.warning(f"Failed to get {league} {season}: {e}")
            
            if all_data:
                df = pd.DataFrame(all_data)
                output_file = self.output_dir / "understat_xg_data.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"✓ Saved {len(df)} xG records to {output_file}")
                return df
                
        except ImportError:
            logger.warning("understatapi not installed, skipping xG data")
        except Exception as e:
            logger.error(f"Error fetching xG data: {e}")
        
        return pd.DataFrame()
    
    def fetch_fbref_stats(self, league_url: str = None) -> pd.DataFrame:
        """Fetch advanced stats from fbref.com"""
        try:
            # Use pandas read_html to scrape tables
            if league_url is None:
                league_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
            
            logger.info(f"Fetching stats from {league_url}")
            
            tables = pd.read_html(league_url)
            
            # Usually the main stats table is one of the first
            if tables:
                df = tables[0]
                output_file = self.output_dir / "fbref_stats.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"✓ Saved {len(df)} rows to {output_file}")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching fbref stats: {e}")
        
        return pd.DataFrame()
    
    def download_all(self) -> Dict[str, pd.DataFrame]:
        """Download all GitHub and web data sources"""
        results = {}
        
        # GitHub sources
        logger.info("Fetching GitHub data...")
        results["github"] = self.fetch_football_csv()
        
        # Understat xG
        logger.info("Fetching Understat xG data...")
        results["understat_xg"] = self.fetch_understat_xg()
        
        # FBRef stats (optional, may require more handling)
        # results["fbref"] = self.fetch_fbref_stats()
        
        return results
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get all GitHub data combined"""
        all_dfs = []
        
        for csv_file in self.output_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()


# Convenience function
def collect_github_data() -> pd.DataFrame:
    """Download and return all GitHub football data"""
    collector = GitHubCollector()
    collector.download_all()
    return collector.get_combined_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = GitHubCollector()
    
    print("Downloading data from GitHub and web sources...")
    results = collector.download_all()
    
    for name, df in results.items():
        if not df.empty:
            print(f"  {name}: {len(df)} rows")
        else:
            print(f"  {name}: No data")
