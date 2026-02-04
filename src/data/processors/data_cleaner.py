"""
Data Cleaner Module
Standardizes and cleans football match data from various sources.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Standardizes data from multiple sources into a common format.
    
    Handles:
    - Team name normalization
    - Date parsing
    - Missing value imputation
    - Outlier detection
    - Column standardization
    """
    
    # Team name aliases (common variations)
    TEAM_ALIASES = {
        # England
        'man united': 'manchester united',
        'man utd': 'manchester united',
        'man city': 'manchester city',
        'wolves': 'wolverhampton',
        'spurs': 'tottenham',
        'brighton': 'brighton and hove albion',
        'west ham': 'west ham united',
        'newcastle': 'newcastle united',
        'nott\'m forest': 'nottingham forest',
        'nottingham': 'nottingham forest',
        'leicester': 'leicester city',
        
        # Germany
        'bayern': 'bayern munich',
        'bayern munchen': 'bayern munich',
        'dortmund': 'borussia dortmund',
        'bvb': 'borussia dortmund',
        'm\'gladbach': 'borussia monchengladbach',
        'gladbach': 'borussia monchengladbach',
        'leverkusen': 'bayer leverkusen',
        'rb leipzig': 'leipzig',
        'wolfsburg': 'vfl wolfsburg',
        
        # Spain
        'real': 'real madrid',
        'barca': 'barcelona',
        'atleti': 'atletico madrid',
        'atletico': 'atletico madrid',
        
        # Italy
        'inter': 'inter milan',
        'internazionale': 'inter milan',
        'juve': 'juventus',
        'ac milan': 'milan',
        'napoli': 'ssc napoli',
        
        # France
        'psg': 'paris saint-germain',
        'paris': 'paris saint-germain',
        'monaco': 'as monaco',
        'marseille': 'olympique marseille',
        'lyon': 'olympique lyon',
    }
    
    STANDARD_COLUMNS = {
        'match_date': 'datetime64[ns]',
        'home_team': 'string',
        'away_team': 'string',
        'home_goals': 'int64',
        'away_goals': 'int64',
        'result': 'string',
        'league': 'string',
        'season': 'string',
    }
    
    def __init__(self):
        self.team_mapping = {}
        self._build_team_mapping()
    
    def _build_team_mapping(self):
        """Build team name mapping from aliases."""
        for alias, canonical in self.TEAM_ALIASES.items():
            self.team_mapping[alias.lower()] = canonical.lower()
    
    def normalize_team_name(self, team: str) -> str:
        """Normalize team name to standard form."""
        if not team:
            return ''
        
        team_lower = team.lower().strip()
        
        # Check aliases
        if team_lower in self.team_mapping:
            return self.team_mapping[team_lower].title()
        
        # Basic normalization
        team_clean = re.sub(r'\s+', ' ', team_lower)
        team_clean = re.sub(r'fc$|^fc\s', '', team_clean).strip()
        
        return team_clean.title()
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date from various formats."""
        if pd.isna(date_str):
            return None
        
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%d %b %Y',
            '%b %d, %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        
        # Try pandas
        try:
            return pd.to_datetime(date_str)
        except Exception:
            return None
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize a DataFrame.
        
        Steps:
        1. Normalize column names
        2. Parse dates
        3. Normalize team names
        4. Handle missing values
        5. Calculate derived columns
        """
        df = df.copy()
        
        # 1. Lowercase column names
        df.columns = df.columns.str.lower().str.strip()
        
        # 2. Parse dates
        date_cols = ['date', 'match_date', 'matchdate', 'date_time']
        for col in date_cols:
            if col in df.columns:
                df['match_date'] = pd.to_datetime(df[col], errors='coerce')
                break
        
        # 3. Normalize team names
        team_cols = ['home_team', 'away_team', 'hometeam', 'awayteam', 'home', 'away']
        if 'hometeam' in df.columns and 'home_team' not in df.columns:
            df['home_team'] = df['hometeam']
        if 'awayteam' in df.columns and 'away_team' not in df.columns:
            df['away_team'] = df['awayteam']
        
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].apply(self.normalize_team_name)
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].apply(self.normalize_team_name)
        
        # 4. Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # 5. Calculate result if missing
        if 'result' not in df.columns and all(c in df.columns for c in ['home_goals', 'away_goals']):
            df['result'] = df.apply(
                lambda r: 'H' if r['home_goals'] > r['away_goals'] 
                         else ('A' if r['home_goals'] < r['away_goals'] else 'D'),
                axis=1
            )
        
        return df
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: List[str] = None
    ) -> pd.DataFrame:
        """Remove duplicate matches."""
        subset = subset or ['match_date', 'home_team', 'away_team']
        available_subset = [c for c in subset if c in df.columns]
        
        if not available_subset:
            return df
        
        original_len = len(df)
        df = df.drop_duplicates(subset=available_subset, keep='last')
        
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        return df
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers in a column.
        
        Returns boolean Series indicating outliers.
        """
        if column not in df.columns:
            return pd.Series([False] * len(df))
        
        values = df[column].dropna()
        
        if method == 'iqr':
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (df[column] < lower) | (df[column] > upper)
        
        elif method == 'zscore':
            mean = values.mean()
            std = values.std()
            return abs((df[column] - mean) / std) > threshold
        
        return pd.Series([False] * len(df))
    
    def impute_missing(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'mean'
    ) -> pd.DataFrame:
        """Impute missing values."""
        df = df.copy()
        
        if column not in df.columns:
            return df
        
        if method == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif method == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif method == 'mode':
            df[column] = df[column].fillna(df[column].mode().iloc[0] if not df[column].mode().empty else 0)
        elif method == 'zero':
            df[column] = df[column].fillna(0)
        
        return df


# Global instance
_cleaner: Optional[DataCleaner] = None


def get_cleaner() -> DataCleaner:
    """Get or create data cleaner."""
    global _cleaner
    if _cleaner is None:
        _cleaner = DataCleaner()
    return _cleaner


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quick function to clean match data."""
    return get_cleaner().clean_dataframe(df)
