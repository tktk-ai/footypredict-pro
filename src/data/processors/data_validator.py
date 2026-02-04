"""
Data Validator Module
Validates football match data for quality and consistency.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats
        }


class DataValidator:
    """
    Validates football match data quality.
    
    Checks:
    - Required columns presence
    - Data type correctness
    - Value ranges
    - Logical consistency
    - Date validity
    """
    
    REQUIRED_COLUMNS = [
        'match_date', 'home_team', 'away_team', 'home_goals', 'away_goals'
    ]
    
    OPTIONAL_COLUMNS = [
        'result', 'league', 'season', 'home_goals_ht', 'away_goals_ht',
        'home_shots', 'away_shots', 'home_corners', 'away_corners'
    ]
    
    # Valid ranges for numeric columns
    VALUE_RANGES = {
        'home_goals': (0, 15),
        'away_goals': (0, 15),
        'home_goals_ht': (0, 10),
        'away_goals_ht': (0, 10),
        'home_shots': (0, 50),
        'away_shots': (0, 50),
        'home_corners': (0, 25),
        'away_corners': (0, 25),
        'home_possession': (0, 100),
        'away_possession': (0, 100),
    }
    
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate entire DataFrame.
        
        Returns ValidationResult with errors, warnings, and stats.
        """
        errors = []
        warnings = []
        stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'null_counts': {},
            'type_issues': [],
        }
        
        # Check required columns
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, stats)
        
        # Check data types
        type_issues = self._check_data_types(df)
        if type_issues:
            warnings.extend(type_issues)
            stats['type_issues'] = type_issues
        
        # Check null values
        null_counts = df.isnull().sum()
        significant_nulls = null_counts[null_counts > 0]
        stats['null_counts'] = significant_nulls.to_dict()
        
        for col, count in significant_nulls.items():
            pct = count / len(df) * 100
            if col in self.REQUIRED_COLUMNS:
                if pct > 10:
                    errors.append(f"High null rate ({pct:.1f}%) in required column: {col}")
                else:
                    warnings.append(f"Null values ({count}) in column: {col}")
        
        # Check value ranges
        range_issues = self._check_value_ranges(df)
        if range_issues:
            warnings.extend(range_issues)
        
        # Check logical consistency
        logic_issues = self._check_logical_consistency(df)
        if logic_issues:
            if self.strict:
                errors.extend(logic_issues)
            else:
                warnings.extend(logic_issues)
        
        # Check date validity
        date_issues = self._check_dates(df)
        if date_issues:
            warnings.extend(date_issues)
        
        # Determine overall validity
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, stats)
    
    def _check_data_types(self, df: pd.DataFrame) -> List[str]:
        """Check data types are as expected."""
        issues = []
        
        expected_types = {
            'home_goals': ['int64', 'int32', 'float64'],
            'away_goals': ['int64', 'int32', 'float64'],
            'home_team': ['object', 'string'],
            'away_team': ['object', 'string'],
        }
        
        for col, valid_types in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type not in valid_types:
                    issues.append(f"Column {col} has type {actual_type}, expected {valid_types}")
        
        return issues
    
    def _check_value_ranges(self, df: pd.DataFrame) -> List[str]:
        """Check values are within expected ranges."""
        issues = []
        
        for col, (min_val, max_val) in self.VALUE_RANGES.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    issues.append(
                        f"Column {col} has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                    )
        
        return issues
    
    def _check_logical_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check logical consistency of data."""
        issues = []
        
        # Check result matches goals
        if all(c in df.columns for c in ['home_goals', 'away_goals', 'result']):
            expected_result = df.apply(
                lambda r: 'H' if r['home_goals'] > r['away_goals']
                         else ('A' if r['home_goals'] < r['away_goals'] else 'D'),
                axis=1
            )
            mismatches = (df['result'] != expected_result).sum()
            if mismatches > 0:
                issues.append(f"Result doesn't match goals for {mismatches} rows")
        
        # Check HT goals <= FT goals
        if all(c in df.columns for c in ['home_goals', 'home_goals_ht']):
            invalid_ht = (df['home_goals_ht'] > df['home_goals']).sum()
            if invalid_ht > 0:
                issues.append(f"HT goals > FT goals for {invalid_ht} home team rows")
        
        if all(c in df.columns for c in ['away_goals', 'away_goals_ht']):
            invalid_ht = (df['away_goals_ht'] > df['away_goals']).sum()
            if invalid_ht > 0:
                issues.append(f"HT goals > FT goals for {invalid_ht} away team rows")
        
        # Check possession sums to 100 (roughly)
        if all(c in df.columns for c in ['home_possession', 'away_possession']):
            total_poss = df['home_possession'] + df['away_possession']
            invalid_poss = ((total_poss < 95) | (total_poss > 105)).sum()
            if invalid_poss > 0:
                issues.append(f"Possession doesn't sum to ~100% for {invalid_poss} rows")
        
        # Check same team playing home and away
        if all(c in df.columns for c in ['home_team', 'away_team']):
            same_team = (df['home_team'] == df['away_team']).sum()
            if same_team > 0:
                issues.append(f"Same team as home and away for {same_team} rows")
        
        return issues
    
    def _check_dates(self, df: pd.DataFrame) -> List[str]:
        """Check date validity."""
        issues = []
        
        if 'match_date' in df.columns:
            # Check for future dates (beyond reasonable)
            future_cutoff = pd.Timestamp.now() + pd.Timedelta(days=30)
            future_dates = (df['match_date'] > future_cutoff).sum()
            if future_dates > 0:
                issues.append(f"{future_dates} matches have dates > 30 days in future")
            
            # Check for very old dates
            old_cutoff = pd.Timestamp('2000-01-01')
            old_dates = (df['match_date'] < old_cutoff).sum()
            if old_dates > 0:
                issues.append(f"{old_dates} matches have dates before 2000")
        
        return issues
    
    def validate_prediction_data(
        self,
        home_team: str,
        away_team: str,
        league: str
    ) -> Tuple[bool, List[str]]:
        """Validate data for making a prediction."""
        errors = []
        
        if not home_team or len(home_team) < 2:
            errors.append("Invalid home team name")
        
        if not away_team or len(away_team) < 2:
            errors.append("Invalid away team name")
        
        if home_team == away_team:
            errors.append("Home and away team cannot be the same")
        
        if not league:
            errors.append("League is required")
        
        return len(errors) == 0, errors


# Global instance
_validator: Optional[DataValidator] = None


def get_validator(strict: bool = False) -> DataValidator:
    """Get or create data validator."""
    global _validator
    if _validator is None:
        _validator = DataValidator(strict)
    return _validator


def validate_match_data(df: pd.DataFrame) -> ValidationResult:
    """Quick function to validate match data."""
    return get_validator().validate(df)
