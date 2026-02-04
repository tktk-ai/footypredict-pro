"""
Kaggle Data Collector

Downloads and processes football datasets from Kaggle:
- Club Football 2000-2025 (500K+ matches)
- 30K+ Matches with Odds
- International Football Results 1872-2025
- EPL with xG data
"""

import os
import subprocess
import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "kaggle"


class KaggleDataCollector:
    """Downloads and merges Kaggle football datasets"""
    
    # Priority datasets for football prediction
    DATASETS = {
        "azathoth42/football-data": {
            "name": "Club Football Match Data 2000-2025",
            "files": ["matches.csv"],
            "priority": 1
        },
        "secareanualin/football-events": {
            "name": "30K+ Football Matches with Events",
            "files": ["events.csv", "ginf.csv"],
            "priority": 2
        },
        "martj42/international-football-results-from-1872-to-2017": {
            "name": "International Football Results",
            "files": ["results.csv", "shootouts.csv", "goalscorers.csv"],
            "priority": 3
        },
        "davidcariboo/player-scores": {
            "name": "Player Performance Data",
            "files": ["appearances.csv", "players.csv", "games.csv"],
            "priority": 4
        },
        "technika148/football-statistics": {
            "name": "Football Statistics Dataset",
            "files": ["football_data.csv"],
            "priority": 5
        }
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_kaggle_auth()
    
    def _check_kaggle_auth(self) -> bool:
        """Check if Kaggle API credentials are configured"""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            logger.warning("Kaggle credentials not found at ~/.kaggle/kaggle.json")
            logger.info("Will use direct download URLs instead")
            return False
        return True
    
    def download_dataset(self, dataset_id: str) -> bool:
        """Download a single dataset using Kaggle API"""
        try:
            output_path = self.output_dir / dataset_id.replace("/", "_")
            output_path.mkdir(parents=True, exist_ok=True)
            
            cmd = f"kaggle datasets download -d {dataset_id} -p {output_path} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {dataset_id}")
                return True
            else:
                logger.error(f"✗ Failed to download {dataset_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_id}: {e}")
            return False
    
    def download_via_url(self, url: str, filename: str) -> bool:
        """Download dataset via direct URL (fallback method)"""
        try:
            import requests
            
            output_path = self.output_dir / filename
            logger.info(f"Downloading {filename}...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Unzip if needed
            if filename.endswith('.zip'):
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir / filename.replace('.zip', ''))
                os.remove(output_path)
            
            logger.info(f"✓ Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False
    
    def download_all(self, use_api: bool = True) -> dict:
        """Download all configured datasets"""
        results = {}
        
        if use_api and self._check_kaggle_auth():
            for dataset_id in self.DATASETS:
                results[dataset_id] = self.download_dataset(dataset_id)
        else:
            # Fallback to direct Football-Data.co.uk downloads
            logger.info("Using direct downloads from Football-Data.co.uk")
            results.update(self._download_football_data_co_uk())
        
        return results
    
    def _download_football_data_co_uk(self) -> dict:
        """Download historical data from Football-Data.co.uk"""
        import requests
        
        results = {}
        base_url = "https://www.football-data.co.uk/mmz4281"
        
        # League codes and seasons
        leagues = {
            "E0": "Premier League",
            "E1": "Championship", 
            "SP1": "La Liga",
            "D1": "Bundesliga",
            "I1": "Serie A",
            "F1": "Ligue 1",
            "N1": "Eredivisie",
            "P1": "Primeira Liga",
            "B1": "Jupiler Pro League"
        }
        
        seasons = [
            "2425", "2324", "2223", "2122", "2021", 
            "1920", "1819", "1718", "1617", "1516",
            "1415", "1314", "1213", "1112", "1011",
            "0910", "0809", "0708", "0607", "0506"
        ]
        
        all_data = []
        
        for season in seasons:
            for league_code, league_name in leagues.items():
                url = f"{base_url}/{season}/{league_code}.csv"
                try:
                    df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
                    df['Season'] = f"20{season[:2]}-20{season[2:]}"
                    df['LeagueCode'] = league_code
                    df['LeagueName'] = league_name
                    all_data.append(df)
                    logger.info(f"✓ Downloaded: {league_name} {season}")
                    results[f"{league_code}_{season}"] = True
                except Exception as e:
                    logger.debug(f"Skipped {league_code} {season}: {e}")
                    results[f"{league_code}_{season}"] = False
        
        # Combine and save
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            output_file = self.output_dir / "football_data_all_leagues.csv"
            combined.to_csv(output_file, index=False)
            logger.info(f"✓ Saved combined data: {len(combined)} matches to {output_file}")
        
        return results
    
    def load_all_data(self) -> pd.DataFrame:
        """Load and combine all downloaded datasets into one DataFrame"""
        all_dfs = []
        
        # Load combined football-data.co.uk
        combined_file = self.output_dir / "football_data_all_leagues.csv"
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} matches from football-data.co.uk")
        
        # Load any Kaggle datasets
        for dataset_id, info in self.DATASETS.items():
            dataset_dir = self.output_dir / dataset_id.replace("/", "_")
            if dataset_dir.exists():
                for csv_file in dataset_dir.glob("*.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        all_dfs.append(df)
                        logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_file}: {e}")
        
        if all_dfs:
            # Merge all datasets
            combined = pd.concat(all_dfs, ignore_index=True)
            return combined
        
        return pd.DataFrame()
    
    def get_stats(self) -> dict:
        """Get statistics about collected data"""
        combined_file = self.output_dir / "football_data_all_leagues.csv"
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            return {
                "total_matches": len(df),
                "leagues": df['LeagueName'].nunique() if 'LeagueName' in df.columns else 0,
                "seasons": df['Season'].nunique() if 'Season' in df.columns else 0,
                "teams": len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())) if 'HomeTeam' in df.columns else 0,
                "date_range": f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else "Unknown"
            }
        return {"error": "No data collected yet"}


# Convenience functions
def collect_kaggle_data() -> pd.DataFrame:
    """Download and return all Kaggle football data"""
    collector = KaggleDataCollector()
    collector.download_all()
    return collector.load_all_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = KaggleDataCollector()
    print("Downloading football data...")
    results = collector.download_all()
    
    print("\nDownload results:")
    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {key}")
    
    print("\nData statistics:")
    stats = collector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
