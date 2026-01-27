"""
Data Merger

Combines all collected datasets into a unified training dataset:
- Standardizes team names across sources
- Aligns column schemas
- Merges historical data with xG and odds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class DataMerger:
    """Merges all collected datasets into unified training data"""
    
    # Standard column mapping
    COLUMN_MAPPING = {
        # Date columns
        'Date': 'date', 'date': 'date', 'datetime': 'date', 'match_date': 'date',
        # Team columns  
        'HomeTeam': 'home_team', 'home_team': 'home_team', 'home': 'home_team',
        'AwayTeam': 'away_team', 'away_team': 'away_team', 'away': 'away_team',
        'h': 'home_team', 'a': 'away_team',
        # Goals
        'FTHG': 'home_goals', 'FTAG': 'away_goals',
        'home_goals': 'home_goals', 'away_goals': 'away_goals',
        'HG': 'home_goals', 'AG': 'away_goals',
        # Half time
        'HTHG': 'ht_home_goals', 'HTAG': 'ht_away_goals',
        # Result
        'FTR': 'result', 'result': 'result',
        # xG
        'home_xg': 'home_xg', 'away_xg': 'away_xg',
        'xG_home': 'home_xg', 'xG_away': 'away_xg',
        # Shots
        'HS': 'home_shots', 'AS': 'away_shots',
        'HST': 'home_shots_target', 'AST': 'away_shots_target',
        # Other stats
        'HF': 'home_fouls', 'AF': 'away_fouls',
        'HC': 'home_corners', 'AC': 'away_corners',
        'HY': 'home_yellows', 'AY': 'away_yellows',
        'HR': 'home_reds', 'AR': 'away_reds',
        # Odds
        'B365H': 'odds_home', 'B365D': 'odds_draw', 'B365A': 'odds_away',
        'PSH': 'odds_home_ps', 'PSD': 'odds_draw_ps', 'PSA': 'odds_away_ps',
        # League
        'Div': 'league_code', 'LeagueName': 'league', 'league': 'league',
        'Season': 'season', 'season': 'season'
    }
    
    # Known team name variations
    TEAM_ALIASES = {
        'man united': 'manchester united',
        'man utd': 'manchester united',
        'manchester utd': 'manchester united',
        'man city': 'manchester city',
        'manchester c': 'manchester city',
        'spurs': 'tottenham',
        'tottenham hotspur': 'tottenham',
        'wolves': 'wolverhampton',
        'wolverhampton wanderers': 'wolverhampton',
        'west ham': 'west ham united',
        'brighton': 'brighton and hove albion',
        'brighton hove': 'brighton and hove albion',
        'nottm forest': 'nottingham forest',
        "nottingham": "nottingham forest",
        'newcastle utd': 'newcastle united',
        'sheffield utd': 'sheffield united',
        'leicester': 'leicester city',
        'crystal palace': 'crystal palace',
        'bournemouth': 'afc bournemouth',
        'bayern': 'bayern munich',
        'bayern münchen': 'bayern munich',
        'dortmund': 'borussia dortmund',
        'borussia m.gladbach': 'borussia monchengladbach',
        'gladbach': 'borussia monchengladbach',
        'atletico': 'atletico madrid',
        'atlético madrid': 'atletico madrid',
        'real': 'real madrid',
        'barca': 'barcelona',
        'milan': 'ac milan',
        'inter': 'inter milan',
        'internazionale': 'inter milan',
        'napoli': 'ssc napoli',
        'juventus': 'juventus fc',
        'roma': 'as roma',
        'psg': 'paris saint-germain',
        'paris sg': 'paris saint-germain',
        'lyon': 'olympique lyon',
        'marseille': 'olympique marseille'
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.team_index = {}  # For fuzzy matching cache
    
    def standardize_team_name(self, name: str) -> str:
        """Standardize team name to canonical form"""
        if not isinstance(name, str):
            return str(name)
        
        name_lower = name.lower().strip()
        
        # Check aliases first
        if name_lower in self.TEAM_ALIASES:
            return self.TEAM_ALIASES[name_lower]
        
        # Check fuzzy match cache
        if name_lower in self.team_index:
            return self.team_index[name_lower]
        
        # Try fuzzy match against known aliases
        match, score = process.extractOne(name_lower, list(self.TEAM_ALIASES.keys()))
        if score > 85:
            canonical = self.TEAM_ALIASES[match]
            self.team_index[name_lower] = canonical
            return canonical
        
        # Return title case version
        return name.strip().title()
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard names"""
        rename_map = {}
        for old_name in df.columns:
            if old_name in self.COLUMN_MAPPING:
                rename_map[old_name] = self.COLUMN_MAPPING[old_name]
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def load_kaggle_data(self) -> pd.DataFrame:
        """Load data from Kaggle collector"""
        kaggle_dir = RAW_DATA_DIR / "kaggle"
        combined_file = kaggle_dir / "football_data_all_leagues.csv"
        
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            df['source'] = 'kaggle'
            logger.info(f"Loaded {len(df)} matches from Kaggle data")
            return df
        
        return pd.DataFrame()
    
    def load_huggingface_data(self) -> pd.DataFrame:
        """Load data from HuggingFace collector"""
        hf_dir = RAW_DATA_DIR / "huggingface"
        
        all_dfs = []
        for csv_file in hf_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source'] = 'huggingface'
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} rows from HuggingFace data")
            return combined
        
        return pd.DataFrame()
    
    def load_github_data(self) -> pd.DataFrame:
        """Load data from GitHub collector"""
        github_dir = RAW_DATA_DIR / "github"
        
        all_dfs = []
        for csv_file in github_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source'] = 'github'
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} rows from GitHub data")
            return combined
        
        return pd.DataFrame()
    
    def load_existing_data(self) -> pd.DataFrame:
        """Load existing training data"""
        existing_file = DATA_DIR / "comprehensive_training_data.csv"
        
        if existing_file.exists():
            df = pd.read_csv(existing_file)
            df['source'] = 'existing'
            logger.info(f"Loaded {len(df)} matches from existing training data")
            return df
        
        return pd.DataFrame()
    
    def merge_all_sources(self) -> pd.DataFrame:
        """Merge all data sources into unified dataset"""
        sources = []
        
        # Load from each source
        kaggle = self.load_kaggle_data()
        if not kaggle.empty:
            sources.append(('kaggle', kaggle))
        
        hf = self.load_huggingface_data()
        if not hf.empty:
            sources.append(('huggingface', hf))
        
        github = self.load_github_data()
        if not github.empty:
            sources.append(('github', github))
        
        existing = self.load_existing_data()
        if not existing.empty:
            sources.append(('existing', existing))
        
        if not sources:
            logger.warning("No data sources found")
            return pd.DataFrame()
        
        # Process each source
        processed = []
        for name, df in sources:
            logger.info(f"Processing {name}: {len(df)} rows")
            
            # Standardize columns
            df = self.standardize_columns(df)
            
            # Standardize team names
            if 'home_team' in df.columns:
                df['home_team'] = df['home_team'].apply(self.standardize_team_name)
            if 'away_team' in df.columns:
                df['away_team'] = df['away_team'].apply(self.standardize_team_name)
            
            processed.append(df)
        
        # Combine all
        combined = pd.concat(processed, ignore_index=True)
        
        # Remove duplicates (same date + teams)
        if all(col in combined.columns for col in ['date', 'home_team', 'away_team']):
            before = len(combined)
            combined = combined.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
            logger.info(f"Removed {before - len(combined)} duplicates")
        
        # Sort by date
        if 'date' in combined.columns:
            combined = combined.sort_values('date', ascending=False)
        
        return combined
    
    def create_master_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """Create the master training dataset"""
        logger.info("Merging all data sources...")
        
        combined = self.merge_all_sources()
        
        if combined.empty:
            return pd.DataFrame(), {"error": "No data to merge"}
        
        # Save master dataset
        output_file = self.output_dir / "master_training_data.csv"
        combined.to_csv(output_file, index=False)
        
        # Calculate stats
        stats = {
            "total_matches": len(combined),
            "sources": combined['source'].value_counts().to_dict() if 'source' in combined.columns else {},
            "teams": len(set(combined.get('home_team', [])) | set(combined.get('away_team', []))),
            "columns": len(combined.columns),
            "output_file": str(output_file)
        }
        
        logger.info(f"✓ Created master dataset: {len(combined)} matches")
        logger.info(f"  Saved to: {output_file}")
        
        return combined, stats


def merge_all_data() -> pd.DataFrame:
    """Convenience function to merge all collected data"""
    merger = DataMerger()
    df, stats = merger.create_master_dataset()
    print(f"Merged {stats.get('total_matches', 0)} matches from {len(stats.get('sources', {}))} sources")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    merger = DataMerger()
    df, stats = merger.create_master_dataset()
    
    print("\nMaster Dataset Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
