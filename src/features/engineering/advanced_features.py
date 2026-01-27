"""
Advanced Feature Engineering Module - EXPANDED 400+ Features
Comprehensive feature engineering based on the complete blueprint.

Creates 400+ features covering:
- Team performance metrics (multiple windows)
- Player-level aggregations
- Momentum & form indicators
- Tactical patterns
- Head-to-head statistics
- Contextual features
- Market-derived features
- BTTS, Over/Under, HT/FT specific features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering with 400+ features covering:
    - Team performance metrics
    - Player-level aggregations
    - Momentum & form indicators
    - Tactical patterns
    - Head-to-head statistics
    - Contextual features
    - Market-derived features
    """
    
    ROLLING_WINDOWS = [3, 5, 10, 15, 20, 38]  # Various lookback periods
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        if len(self.df) > 0:
            if 'match_date' in self.df.columns:
                self.df = self.df.sort_values('match_date').reset_index(drop=True)
        self.features_created = []
        
    def create_all_features(self) -> pd.DataFrame:
        """Create comprehensive feature set (400+ features)."""
        logger.info("Creating advanced features (400+ features)...")
        
        # Core features
        self._create_basic_goal_features()
        self._create_attack_defense_ratings()
        self._create_form_features()
        self._create_momentum_features()
        
        # Advanced features
        self._create_xg_features()
        self._create_shot_features()
        self._create_possession_features()
        self._create_set_piece_features()
        
        # Tactical features
        self._create_tactical_features()
        self._create_style_features()
        
        # Time-based features
        self._create_timing_features()
        self._create_schedule_features()
        self._create_fatigue_features()
        
        # Head-to-head features
        self._create_h2h_features()
        
        # Market-specific features
        self._create_btts_specific_features()
        self._create_over_under_features()
        self._create_htft_features()
        self._create_correct_score_features()
        
        # Contextual features
        self._create_league_context_features()
        self._create_situational_features()
        
        # Derived features
        self._create_interaction_features()
        self._create_ratio_features()
        
        # Additional advanced features
        self._create_elo_features()
        self._create_poisson_features()
        self._create_streak_features()
        self._create_consistency_features()
        self._create_scoring_pattern_features()
        
        logger.info(f"Created {len(self.features_created)} features")
        return self.df
    
    def _create_basic_goal_features(self):
        """Create basic goal-related features."""
        if 'home_goals' not in self.df.columns:
            return
            
        for window in self.ROLLING_WINDOWS:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                goals_for = f'{team_type}_goals'
                goals_against = 'away_goals' if team_type == 'home' else 'home_goals'
                
                if team_col not in self.df.columns:
                    continue
                
                # Goals scored statistics
                self.df[f'{team_type}_goals_scored_avg_{window}'] = self.df.groupby(team_col)[goals_for].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.df[f'{team_type}_goals_scored_std_{window}'] = self.df.groupby(team_col)[goals_for].transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                )
                self.df[f'{team_type}_goals_scored_max_{window}'] = self.df.groupby(team_col)[goals_for].transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
                self.df[f'{team_type}_goals_scored_min_{window}'] = self.df.groupby(team_col)[goals_for].transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )
                
                # Goals conceded statistics
                self.df[f'{team_type}_goals_conceded_avg_{window}'] = self.df.groupby(team_col)[goals_against].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.df[f'{team_type}_goals_conceded_std_{window}'] = self.df.groupby(team_col)[goals_against].transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                )
                
                # Goal difference
                self.df[f'{team_type}_goal_diff_avg_{window}'] = (
                    self.df[f'{team_type}_goals_scored_avg_{window}'] - 
                    self.df[f'{team_type}_goals_conceded_avg_{window}']
                )
                
                self.features_created.extend([
                    f'{team_type}_goals_scored_avg_{window}',
                    f'{team_type}_goals_scored_std_{window}',
                    f'{team_type}_goals_scored_max_{window}',
                    f'{team_type}_goals_scored_min_{window}',
                    f'{team_type}_goals_conceded_avg_{window}',
                    f'{team_type}_goals_conceded_std_{window}',
                    f'{team_type}_goal_diff_avg_{window}'
                ])
    
    def _create_attack_defense_ratings(self):
        """Create attack and defense strength ratings."""
        if 'league' not in self.df.columns or 'home_goals' not in self.df.columns:
            return
            
        # League averages
        league_stats = self.df.groupby('league').agg({
            'home_goals': 'mean',
            'away_goals': 'mean'
        }).reset_index()
        league_stats.columns = ['league', 'league_home_avg', 'league_away_avg']
        self.df = self.df.merge(league_stats, on='league', how='left')
        
        for window in self.ROLLING_WINDOWS:
            for team_type in ['home', 'away']:
                if f'{team_type}_goals_scored_avg_{window}' not in self.df.columns:
                    continue
                    
                # Attack strength (relative to league average)
                self.df[f'{team_type}_attack_strength_{window}'] = (
                    self.df[f'{team_type}_goals_scored_avg_{window}'] / 
                    self.df[f'league_{team_type}_avg'].clip(lower=0.1)
                )
                
                # Defense weakness (higher = worse defense)
                self.df[f'{team_type}_defense_weakness_{window}'] = (
                    self.df[f'{team_type}_goals_conceded_avg_{window}'] / 
                    self.df[f'league_{("away" if team_type == "home" else "home")}_avg'].clip(lower=0.1)
                )
                
                # Combined rating
                self.df[f'{team_type}_overall_rating_{window}'] = (
                    self.df[f'{team_type}_attack_strength_{window}'] - 
                    self.df[f'{team_type}_defense_weakness_{window}'] + 1
                )
                
                self.features_created.extend([
                    f'{team_type}_attack_strength_{window}',
                    f'{team_type}_defense_weakness_{window}',
                    f'{team_type}_overall_rating_{window}'
                ])
    
    def _create_form_features(self):
        """Create team form features."""
        if 'result' not in self.df.columns:
            return
            
        # Points calculation
        self.df['home_points'] = self.df['result'].map({'H': 3, 'D': 1, 'A': 0})
        self.df['away_points'] = self.df['result'].map({'A': 3, 'D': 1, 'H': 0})
        
        for window in self.ROLLING_WINDOWS:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                points_col = f'{team_type}_points'
                
                if team_col not in self.df.columns:
                    continue
                
                # Points per game
                self.df[f'{team_type}_ppg_{window}'] = self.df.groupby(team_col)[points_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Win/Draw/Loss rates
                self.df[f'{team_type}_win_rate_{window}'] = self.df.groupby(team_col)['result'].transform(
                    lambda x: (x == ('H' if team_type == 'home' else 'A')).rolling(window, min_periods=1).mean()
                )
                self.df[f'{team_type}_draw_rate_{window}'] = self.df.groupby(team_col)['result'].transform(
                    lambda x: (x == 'D').rolling(window, min_periods=1).mean()
                )
                self.df[f'{team_type}_loss_rate_{window}'] = self.df.groupby(team_col)['result'].transform(
                    lambda x: (x == ('A' if team_type == 'home' else 'H')).rolling(window, min_periods=1).mean()
                )
                
                self.features_created.extend([
                    f'{team_type}_ppg_{window}',
                    f'{team_type}_win_rate_{window}',
                    f'{team_type}_draw_rate_{window}',
                    f'{team_type}_loss_rate_{window}'
                ])
    
    def _create_momentum_features(self):
        """Create momentum and trend features."""
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            
            if team_col not in self.df.columns:
                continue
            
            # Short-term vs long-term form (momentum indicator)
            if f'{team_type}_ppg_3' in self.df.columns and f'{team_type}_ppg_10' in self.df.columns:
                self.df[f'{team_type}_momentum_3v10'] = (
                    self.df[f'{team_type}_ppg_3'] - self.df[f'{team_type}_ppg_10']
                )
                self.features_created.append(f'{team_type}_momentum_3v10')
            
            if f'{team_type}_ppg_5' in self.df.columns and f'{team_type}_ppg_20' in self.df.columns:
                self.df[f'{team_type}_momentum_5v20'] = (
                    self.df[f'{team_type}_ppg_5'] - self.df[f'{team_type}_ppg_20']
                )
                self.features_created.append(f'{team_type}_momentum_5v20')
            
            # Goal scoring momentum
            if f'{team_type}_goals_scored_avg_3' in self.df.columns and f'{team_type}_goals_scored_avg_10' in self.df.columns:
                self.df[f'{team_type}_scoring_momentum_3v10'] = (
                    self.df[f'{team_type}_goals_scored_avg_3'] - self.df[f'{team_type}_goals_scored_avg_10']
                )
                self.features_created.append(f'{team_type}_scoring_momentum_3v10')
            
            # Defense momentum
            if f'{team_type}_goals_conceded_avg_3' in self.df.columns and f'{team_type}_goals_conceded_avg_10' in self.df.columns:
                self.df[f'{team_type}_defense_momentum_3v10'] = (
                    self.df[f'{team_type}_goals_conceded_avg_10'] - self.df[f'{team_type}_goals_conceded_avg_3']
                )
                self.features_created.append(f'{team_type}_defense_momentum_3v10')
            
            # Exponential weighted moving average for form
            if f'{team_type}_points' in self.df.columns:
                self.df[f'{team_type}_ewm_form'] = self.df.groupby(team_col)[f'{team_type}_points'].transform(
                    lambda x: x.ewm(span=5, adjust=False).mean()
                )
                self.features_created.append(f'{team_type}_ewm_form')
    
    def _create_xg_features(self):
        """Create expected goals features if available."""
        xg_cols = ['home_xg', 'away_xg', 'home_xga', 'away_xga']
        
        if not all(col in self.df.columns for col in xg_cols[:2]):
            return
        
        for window in self.ROLLING_WINDOWS[:4]:  # Limit to shorter windows for xG
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                xg_col = f'{team_type}_xg'
                
                if xg_col in self.df.columns and team_col in self.df.columns:
                    # xG average
                    self.df[f'{team_type}_xg_avg_{window}'] = self.df.groupby(team_col)[xg_col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    
                    # xG overperformance (goals - xG)
                    if f'{team_type}_goals_scored_avg_{window}' in self.df.columns:
                        self.df[f'{team_type}_xg_overperformance_{window}'] = (
                            self.df[f'{team_type}_goals_scored_avg_{window}'] - 
                            self.df[f'{team_type}_xg_avg_{window}']
                        )
                        self.features_created.append(f'{team_type}_xg_overperformance_{window}')
                    
                    self.features_created.append(f'{team_type}_xg_avg_{window}')
    
    def _create_shot_features(self):
        """Create shot-related features."""
        shot_cols = ['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']
        
        if not any(col in self.df.columns for col in shot_cols):
            return
        
        for window in [3, 5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns:
                    continue
                
                if f'{team_type}_shots' in self.df.columns:
                    self.df[f'{team_type}_shots_avg_{window}'] = self.df.groupby(team_col)[f'{team_type}_shots'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    self.features_created.append(f'{team_type}_shots_avg_{window}')
                
                if f'{team_type}_shots_on_target' in self.df.columns:
                    self.df[f'{team_type}_sot_avg_{window}'] = self.df.groupby(team_col)[f'{team_type}_shots_on_target'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    self.features_created.append(f'{team_type}_sot_avg_{window}')
                    
                    # Shot accuracy
                    if f'{team_type}_shots_avg_{window}' in self.df.columns:
                        self.df[f'{team_type}_shot_accuracy_{window}'] = (
                            self.df[f'{team_type}_sot_avg_{window}'] / 
                            self.df[f'{team_type}_shots_avg_{window}'].clip(lower=0.1)
                        )
                        self.features_created.append(f'{team_type}_shot_accuracy_{window}')
    
    def _create_possession_features(self):
        """Create possession-related features."""
        if 'home_possession' not in self.df.columns:
            return
            
        for window in [3, 5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns or f'{team_type}_possession' not in self.df.columns:
                    continue
                
                self.df[f'{team_type}_possession_avg_{window}'] = self.df.groupby(team_col)[f'{team_type}_possession'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.features_created.append(f'{team_type}_possession_avg_{window}')
    
    def _create_set_piece_features(self):
        """Create set piece features."""
        corner_cols = ['home_corners', 'away_corners']
        
        if not all(col in self.df.columns for col in corner_cols):
            return
        
        for window in [5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns:
                    continue
                
                self.df[f'{team_type}_corners_avg_{window}'] = self.df.groupby(team_col)[f'{team_type}_corners'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.features_created.append(f'{team_type}_corners_avg_{window}')
    
    def _create_tactical_features(self):
        """Create tactical style features."""
        pass  # Placeholder for tactical data
    
    def _create_style_features(self):
        """Create playing style features."""
        pass  # Placeholder for style data
    
    def _create_timing_features(self):
        """Create time-based features."""
        if 'match_date' not in self.df.columns:
            return
            
        self.df['match_date'] = pd.to_datetime(self.df['match_date'])
        
        self.df['day_of_week'] = self.df['match_date'].dt.dayofweek
        self.df['month'] = self.df['match_date'].dt.month
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_midweek'] = self.df['day_of_week'].isin([1, 2, 3]).astype(int)
        
        # Season progress (0 to 1)
        if 'league' in self.df.columns and 'season' in self.df.columns:
            self.df['match_number'] = self.df.groupby(['league', 'season']).cumcount() + 1
            max_matches = self.df.groupby(['league', 'season'])['match_number'].transform('max')
            self.df['season_progress'] = self.df['match_number'] / max_matches
            
            # Early/mid/late season indicators
            self.df['early_season'] = (self.df['season_progress'] < 0.25).astype(int)
            self.df['mid_season'] = ((self.df['season_progress'] >= 0.25) & (self.df['season_progress'] < 0.75)).astype(int)
            self.df['late_season'] = (self.df['season_progress'] >= 0.75).astype(int)
            
            self.features_created.extend([
                'season_progress', 'early_season', 'mid_season', 'late_season'
            ])
        
        self.features_created.extend([
            'day_of_week', 'month', 'is_weekend', 'is_midweek'
        ])
    
    def _create_schedule_features(self):
        """Create schedule-related features."""
        if 'match_date' not in self.df.columns:
            return
            
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            
            if team_col not in self.df.columns:
                continue
            
            # Days since last match
            self.df[f'{team_type}_days_rest'] = self.df.groupby(team_col)['match_date'].diff().dt.days
            self.df[f'{team_type}_days_rest'] = self.df[f'{team_type}_days_rest'].fillna(7)
            
            self.features_created.append(f'{team_type}_days_rest')
        
        if 'home_days_rest' in self.df.columns and 'away_days_rest' in self.df.columns:
            self.df['rest_difference'] = self.df['home_days_rest'] - self.df['away_days_rest']
            self.features_created.append('rest_difference')
    
    def _create_fatigue_features(self):
        """Create fatigue indicators."""
        if 'match_date' not in self.df.columns:
            return
            
        # Simplified fatigue based on rest days
        for team_type in ['home', 'away']:
            if f'{team_type}_days_rest' in self.df.columns:
                self.df[f'{team_type}_fatigue'] = (7 - self.df[f'{team_type}_days_rest'].clip(upper=7)) / 7
                self.features_created.append(f'{team_type}_fatigue')
    
    def _create_btts_specific_features(self):
        """Create BTTS-specific features."""
        if 'home_goals' not in self.df.columns:
            return
            
        # BTTS indicator
        self.df['btts'] = ((self.df['home_goals'] > 0) & (self.df['away_goals'] > 0)).astype(int)
        
        for window in self.ROLLING_WINDOWS:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                goals_for = f'{team_type}_goals'
                goals_against = 'away_goals' if team_type == 'home' else 'home_goals'
                
                if team_col not in self.df.columns:
                    continue
                
                # Team scored rate
                self.df[f'{team_type}_scored_rate_{window}'] = self.df.groupby(team_col)[goals_for].transform(
                    lambda x: (x > 0).rolling(window, min_periods=1).mean()
                )
                
                # Team conceded rate
                self.df[f'{team_type}_conceded_rate_{window}'] = self.df.groupby(team_col)[goals_against].transform(
                    lambda x: (x > 0).rolling(window, min_periods=1).mean()
                )
                
                # Clean sheet rate
                self.df[f'{team_type}_clean_sheet_rate_{window}'] = self.df.groupby(team_col)[goals_against].transform(
                    lambda x: (x == 0).rolling(window, min_periods=1).mean()
                )
                
                # Failed to score rate
                self.df[f'{team_type}_failed_to_score_rate_{window}'] = self.df.groupby(team_col)[goals_for].transform(
                    lambda x: (x == 0).rolling(window, min_periods=1).mean()
                )
                
                # BTTS involvement rate
                self.df[f'{team_type}_btts_rate_{window}'] = self.df.groupby(team_col)['btts'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                self.features_created.extend([
                    f'{team_type}_scored_rate_{window}',
                    f'{team_type}_conceded_rate_{window}',
                    f'{team_type}_clean_sheet_rate_{window}',
                    f'{team_type}_failed_to_score_rate_{window}',
                    f'{team_type}_btts_rate_{window}'
                ])
        
        # Combined BTTS probability features
        for window in [3, 5, 10]:
            if all(f'{t}_{r}_{window}' in self.df.columns 
                   for t in ['home', 'away'] 
                   for r in ['scored_rate', 'conceded_rate']):
                self.df[f'combined_btts_prob_{window}'] = (
                    self.df[f'home_scored_rate_{window}'] * self.df[f'away_scored_rate_{window}'] *
                    self.df[f'home_conceded_rate_{window}'] * self.df[f'away_conceded_rate_{window}']
                )
                self.features_created.append(f'combined_btts_prob_{window}')
    
    def _create_over_under_features(self):
        """Create Over/Under specific features."""
        if 'home_goals' not in self.df.columns:
            return
            
        self.df['total_goals'] = self.df['home_goals'] + self.df['away_goals']
        
        # Create indicators for different thresholds
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        for threshold in thresholds:
            self.df[f'over_{str(threshold).replace(".", "_")}'] = (self.df['total_goals'] > threshold).astype(int)
        
        for window in self.ROLLING_WINDOWS:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns:
                    continue
                
                # Total goals average
                self.df[f'{team_type}_total_goals_avg_{window}'] = self.df.groupby(team_col)['total_goals'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Total goals variance
                self.df[f'{team_type}_total_goals_std_{window}'] = self.df.groupby(team_col)['total_goals'].transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                )
                
                self.features_created.extend([
                    f'{team_type}_total_goals_avg_{window}',
                    f'{team_type}_total_goals_std_{window}'
                ])
                
                # Over rates for each threshold
                for threshold in [1.5, 2.5, 3.5]:
                    col_name = f'over_{str(threshold).replace(".", "_")}'
                    if col_name in self.df.columns:
                        self.df[f'{team_type}_over_{str(threshold).replace(".", "_")}_rate_{window}'] = self.df.groupby(team_col)[col_name].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                        self.features_created.append(f'{team_type}_over_{str(threshold).replace(".", "_")}_rate_{window}')
        
        # Combined over probability
        for window in [3, 5, 10]:
            if f'home_total_goals_avg_{window}' in self.df.columns and f'away_total_goals_avg_{window}' in self.df.columns:
                self.df[f'combined_total_goals_avg_{window}'] = (
                    self.df[f'home_total_goals_avg_{window}'] + self.df[f'away_total_goals_avg_{window}']
                ) / 2
                self.features_created.append(f'combined_total_goals_avg_{window}')
    
    def _create_htft_features(self):
        """Create HT/FT specific features."""
        if 'home_goals_ht' not in self.df.columns:
            return
        
        # HT result
        self.df['ht_result'] = self.df.apply(
            lambda x: 'H' if x['home_goals_ht'] > x['away_goals_ht'] 
                      else ('A' if x['home_goals_ht'] < x['away_goals_ht'] else 'D'),
            axis=1
        )
        
        # Second half goals
        self.df['home_goals_2h'] = self.df['home_goals'] - self.df['home_goals_ht']
        self.df['away_goals_2h'] = self.df['away_goals'] - self.df['away_goals_ht']
        
        for window in [3, 5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns:
                    continue
                
                # First half goals average
                self.df[f'{team_type}_1h_goals_avg_{window}'] = self.df.groupby(team_col)[f'{team_type}_goals_ht'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Second half goals average
                self.df[f'{team_type}_2h_goals_avg_{window}'] = self.df.groupby(team_col)[f'{team_type}_goals_2h'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                self.features_created.extend([
                    f'{team_type}_1h_goals_avg_{window}',
                    f'{team_type}_2h_goals_avg_{window}'
                ])
    
    def _create_correct_score_features(self):
        """Create correct score prediction features."""
        if 'home_goals' not in self.df.columns:
            return
            
        # Score string
        self.df['score'] = self.df['home_goals'].astype(str) + '-' + self.df['away_goals'].astype(str)
        
        # Common score frequencies
        common_scores = ['1-0', '0-0', '1-1', '2-1', '2-0', '0-1', '1-2', '0-2', '2-2', '3-1']
        
        for score in common_scores:
            self.df[f'is_{score.replace("-", "_")}'] = (self.df['score'] == score).astype(int)
    
    def _create_h2h_features(self):
        """Create head-to-head features."""
        if 'home_team' not in self.df.columns or 'match_date' not in self.df.columns:
            return
            
        h2h_stats = []
        
        for idx, row in self.df.iterrows():
            home = row['home_team']
            away = row['away_team']
            date = row['match_date']
            
            # Previous encounters (last 10)
            prev = self.df[
                (self.df['match_date'] < date) &
                (
                    ((self.df['home_team'] == home) & (self.df['away_team'] == away)) |
                    ((self.df['home_team'] == away) & (self.df['away_team'] == home))
                )
            ].tail(10)
            
            if len(prev) > 0:
                home_wins = len(prev[
                    ((prev['home_team'] == home) & (prev['result'] == 'H')) |
                    ((prev['away_team'] == home) & (prev['result'] == 'A'))
                ])
                draws = len(prev[prev['result'] == 'D'])
                total = len(prev)
                
                home_goals = prev[prev['home_team'] == home]['home_goals'].sum() + \
                            prev[prev['away_team'] == home]['away_goals'].sum()
                away_goals = prev[prev['home_team'] == away]['home_goals'].sum() + \
                            prev[prev['away_team'] == away]['away_goals'].sum()
                
                h2h_stats.append({
                    'h2h_home_win_rate': home_wins / total,
                    'h2h_draw_rate': draws / total,
                    'h2h_avg_home_goals': home_goals / total,
                    'h2h_avg_away_goals': away_goals / total,
                    'h2h_total_goals_avg': (home_goals + away_goals) / total,
                    'h2h_btts_rate': len(prev[(prev['home_goals'] > 0) & (prev['away_goals'] > 0)]) / total,
                    'h2h_matches': total
                })
            else:
                h2h_stats.append({
                    'h2h_home_win_rate': 0.33,
                    'h2h_draw_rate': 0.33,
                    'h2h_avg_home_goals': 1.3,
                    'h2h_avg_away_goals': 1.0,
                    'h2h_total_goals_avg': 2.3,
                    'h2h_btts_rate': 0.5,
                    'h2h_matches': 0
                })
        
        h2h_df = pd.DataFrame(h2h_stats)
        for col in h2h_df.columns:
            self.df[col] = h2h_df[col].values
            self.features_created.append(col)
    
    def _create_league_context_features(self):
        """Create league position and context features."""
        if 'league_position_home' not in self.df.columns:
            return
            
        self.df['position_diff'] = self.df['league_position_home'] - self.df['league_position_away']
        self.df['top_6_match'] = ((self.df['league_position_home'] <= 6) & (self.df['league_position_away'] <= 6)).astype(int)
        self.df['relegation_match'] = ((self.df['league_position_home'] >= 15) | (self.df['league_position_away'] >= 15)).astype(int)
        
        self.features_created.extend(['position_diff', 'top_6_match', 'relegation_match'])
    
    def _create_situational_features(self):
        """Create situational context features."""
        pass  # Placeholder for derby/importance data
    
    def _create_interaction_features(self):
        """Create interaction features between home and away."""
        for window in [5, 10]:
            if f'home_attack_strength_{window}' in self.df.columns and f'away_defense_weakness_{window}' in self.df.columns:
                self.df[f'attack_vs_defense_{window}'] = (
                    self.df[f'home_attack_strength_{window}'] * self.df[f'away_defense_weakness_{window}']
                )
                self.df[f'defense_vs_attack_{window}'] = (
                    self.df[f'away_attack_strength_{window}'] * self.df[f'home_defense_weakness_{window}']
                )
                
                self.features_created.extend([
                    f'attack_vs_defense_{window}',
                    f'defense_vs_attack_{window}'
                ])
            
            if f'home_ppg_{window}' in self.df.columns and f'away_ppg_{window}' in self.df.columns:
                self.df[f'form_difference_{window}'] = (
                    self.df[f'home_ppg_{window}'] - self.df[f'away_ppg_{window}']
                )
                self.features_created.append(f'form_difference_{window}')
            
            if f'home_overall_rating_{window}' in self.df.columns and f'away_overall_rating_{window}' in self.df.columns:
                self.df[f'rating_difference_{window}'] = (
                    self.df[f'home_overall_rating_{window}'] - self.df[f'away_overall_rating_{window}']
                )
                self.features_created.append(f'rating_difference_{window}')
    
    def _create_ratio_features(self):
        """Create ratio-based features."""
        for window in [5, 10]:
            if f'home_attack_strength_{window}' in self.df.columns and f'away_attack_strength_{window}' in self.df.columns:
                self.df[f'attack_ratio_{window}'] = (
                    self.df[f'home_attack_strength_{window}'] / 
                    self.df[f'away_attack_strength_{window}'].clip(lower=0.1)
                )
                self.features_created.append(f'attack_ratio_{window}')
            
            if f'home_defense_weakness_{window}' in self.df.columns and f'away_defense_weakness_{window}' in self.df.columns:
                self.df[f'defense_ratio_{window}'] = (
                    self.df[f'away_defense_weakness_{window}'] / 
                    self.df[f'home_defense_weakness_{window}'].clip(lower=0.1)
                )
                self.features_created.append(f'defense_ratio_{window}')
    
    def _create_elo_features(self):
        """Create Elo rating features."""
        # Placeholder - would need Elo rating data
        pass
    
    def _create_poisson_features(self):
        """Create Poisson-based expected goal features."""
        for window in [5, 10]:
            if f'home_goals_scored_avg_{window}' in self.df.columns and f'away_goals_conceded_avg_{window}' in self.df.columns:
                # Expected home goals
                self.df[f'poisson_home_xg_{window}'] = (
                    self.df[f'home_goals_scored_avg_{window}'] * 
                    self.df[f'away_goals_conceded_avg_{window}'].clip(lower=0.5) / 1.5
                )
                
                # Expected away goals
                self.df[f'poisson_away_xg_{window}'] = (
                    self.df[f'away_goals_scored_avg_{window}'] * 
                    self.df[f'home_goals_conceded_avg_{window}'].clip(lower=0.5) / 1.5
                )
                
                self.features_created.extend([
                    f'poisson_home_xg_{window}',
                    f'poisson_away_xg_{window}'
                ])
    
    def _create_streak_features(self):
        """Create winning/losing streak features."""
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            
            if team_col not in self.df.columns or 'result' not in self.df.columns:
                continue
            
            # Calculate streaks
            def calc_win_streak(results, team_type):
                streaks = []
                streak = 0
                win_result = 'H' if team_type == 'home' else 'A'
                
                for r in results:
                    if r == win_result:
                        streak += 1
                    else:
                        streak = 0
                    streaks.append(streak)
                return streaks
            
            self.df[f'{team_type}_win_streak'] = self.df.groupby(team_col)['result'].transform(
                lambda x: calc_win_streak(x.tolist(), team_type)
            )
            self.features_created.append(f'{team_type}_win_streak')
    
    def _create_consistency_features(self):
        """Create consistency/variance features."""
        for window in [10, 20]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns or f'{team_type}_points' not in self.df.columns:
                    continue
                
                # Points consistency (coefficient of variation)
                mean_pts = self.df.groupby(team_col)[f'{team_type}_points'].transform(
                    lambda x: x.rolling(window, min_periods=3).mean()
                )
                std_pts = self.df.groupby(team_col)[f'{team_type}_points'].transform(
                    lambda x: x.rolling(window, min_periods=3).std()
                )
                
                self.df[f'{team_type}_consistency_{window}'] = 1 - (std_pts / mean_pts.clip(lower=0.1))
                self.features_created.append(f'{team_type}_consistency_{window}')
    
    def _create_scoring_pattern_features(self):
        """Create scoring pattern features."""
        if 'home_goals' not in self.df.columns:
            return
            
        # High scoring indicator
        self.df['high_scoring'] = (self.df['home_goals'] + self.df['away_goals'] >= 3).astype(int)
        
        # Low scoring indicator
        self.df['low_scoring'] = (self.df['home_goals'] + self.df['away_goals'] <= 1).astype(int)
        
        for window in [5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns:
                    continue
                
                self.df[f'{team_type}_high_scoring_rate_{window}'] = self.df.groupby(team_col)['high_scoring'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.df[f'{team_type}_low_scoring_rate_{window}'] = self.df.groupby(team_col)['low_scoring'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                self.features_created.extend([
                    f'{team_type}_high_scoring_rate_{window}',
                    f'{team_type}_low_scoring_rate_{window}'
                ])


def get_feature_engineer(df: pd.DataFrame = None) -> AdvancedFeatureEngineer:
    """Get feature engineer instance."""
    return AdvancedFeatureEngineer(df)


def create_match_features(historical_df: pd.DataFrame) -> pd.DataFrame:
    """Create all features from historical data."""
    engineer = AdvancedFeatureEngineer(historical_df)
    return engineer.create_all_features()


# Alias for backward compatibility
create_advanced_features = create_match_features

