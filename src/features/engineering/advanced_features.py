"""
Advanced Feature Engineering Module V3.0
Creates 400+ features for comprehensive match prediction

Features cover:
- Goal statistics (scored, conceded, difference)
- Attack/defense ratings (relative to league)
- Form features (PPG, win/draw/loss rates, streaks)
- Momentum indicators (short vs long term)
- xG features and overperformance
- BTTS-specific features
- Over/Under features
- HT/FT features
- Correct score features
- H2H features
- Timing and schedule features
- Fatigue indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering with 400+ features covering:
    - Team performance metrics
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
        print("🔧 Creating advanced features...")
        
        # Core features
        self._create_basic_goal_features()
        self._create_attack_defense_ratings()
        self._create_form_features()
        self._create_momentum_features()
        
        # Advanced features
        self._create_xg_features()
        self._create_shot_features()
        self._create_possession_features()
        
        # Market-specific features
        self._create_btts_specific_features()
        self._create_over_under_features()
        self._create_htft_features()
        self._create_correct_score_features()
        
        # Context features
        self._create_timing_features()
        self._create_schedule_features()
        self._create_h2h_features()
        
        # Derived features
        self._create_interaction_features()
        self._create_ratio_features()
        
        print(f"✅ Created {len(self.features_created)} features")
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
                    lambda x: x.rolling(window, min_periods=2).std().fillna(0)
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
                    lambda x: x.rolling(window, min_periods=2).std().fillna(0)
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
        
        # Fill defaults
        self.df['league_home_avg'] = self.df['league_home_avg'].fillna(1.5)
        self.df['league_away_avg'] = self.df['league_away_avg'].fillna(1.2)
        
        for window in self.ROLLING_WINDOWS:
            for team_type in ['home', 'away']:
                # Attack strength (relative to league average)
                scored_col = f'{team_type}_goals_scored_avg_{window}'
                if scored_col in self.df.columns:
                    self.df[f'{team_type}_attack_strength_{window}'] = (
                        self.df[scored_col] / 
                        self.df[f'league_{team_type}_avg'].clip(lower=0.1)
                    )
                
                # Defense weakness (higher = worse defense)
                conceded_col = f'{team_type}_goals_conceded_avg_{window}'
                if conceded_col in self.df.columns:
                    opp_type = 'away' if team_type == 'home' else 'home'
                    self.df[f'{team_type}_defense_weakness_{window}'] = (
                        self.df[conceded_col] / 
                        self.df[f'league_{opp_type}_avg'].clip(lower=0.1)
                    )
                
                # Combined rating
                if f'{team_type}_attack_strength_{window}' in self.df.columns:
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
        self.df['home_points'] = self.df['result'].map({'H': 3, 'D': 1, 'A': 0}).fillna(0)
        self.df['away_points'] = self.df['result'].map({'A': 3, 'D': 1, 'H': 0}).fillna(0)
        
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
                
                # Win rate
                win_result = 'H' if team_type == 'home' else 'A'
                self.df[f'{team_type}_win_rate_{window}'] = self.df.groupby(team_col)['result'].transform(
                    lambda x: (x == win_result).rolling(window, min_periods=1).mean()
                )
                
                # Draw rate
                self.df[f'{team_type}_draw_rate_{window}'] = self.df.groupby(team_col)['result'].transform(
                    lambda x: (x == 'D').rolling(window, min_periods=1).mean()
                )
                
                # Loss rate
                loss_result = 'A' if team_type == 'home' else 'H'
                self.df[f'{team_type}_loss_rate_{window}'] = self.df.groupby(team_col)['result'].transform(
                    lambda x: (x == loss_result).rolling(window, min_periods=1).mean()
                )
                
                self.features_created.extend([
                    f'{team_type}_ppg_{window}',
                    f'{team_type}_win_rate_{window}',
                    f'{team_type}_draw_rate_{window}',
                    f'{team_type}_loss_rate_{window}'
                ])
        
        # Streaks
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            if team_col not in self.df.columns:
                continue
                
            win_result = 'H' if team_type == 'home' else 'A'
            
            # Winning streak
            self.df[f'{team_type}_winning_streak'] = self.df.groupby(team_col)['result'].transform(
                lambda x: self._calculate_streak(x, win_result)
            )
            
            # Unbeaten streak
            self.df[f'{team_type}_unbeaten_streak'] = self.df.groupby(team_col)['result'].transform(
                lambda x: self._calculate_streak(x, win_result, 'D')
            )
            
            self.features_created.extend([
                f'{team_type}_winning_streak',
                f'{team_type}_unbeaten_streak'
            ])
    
    def _create_momentum_features(self):
        """Create momentum and trend features."""
        for team_type in ['home', 'away']:
            # Check if required features exist
            if f'{team_type}_ppg_3' not in self.df.columns:
                continue
                
            # Short-term vs long-term form (momentum indicator)
            if f'{team_type}_ppg_10' in self.df.columns:
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
            team_col = f'{team_type}_team'
            points_col = f'{team_type}_points'
            if team_col in self.df.columns and points_col in self.df.columns:
                self.df[f'{team_type}_ewm_form'] = self.df.groupby(team_col)[points_col].transform(
                    lambda x: x.ewm(span=5, adjust=False).mean()
                )
                self.features_created.append(f'{team_type}_ewm_form')
    
    def _create_xg_features(self):
        """Create expected goals features if available."""
        xg_cols = ['home_xg', 'away_xg']
        
        if not all(col in self.df.columns for col in xg_cols):
            return
        
        for window in self.ROLLING_WINDOWS[:4]:  # Limit to shorter windows for xG
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                xg_col = f'{team_type}_xg'
                
                if xg_col not in self.df.columns or team_col not in self.df.columns:
                    continue
                    
                # xG average
                self.df[f'{team_type}_xg_avg_{window}'] = self.df.groupby(team_col)[xg_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # xG overperformance (goals - xG)
                scored_col = f'{team_type}_goals_scored_avg_{window}'
                if scored_col in self.df.columns:
                    self.df[f'{team_type}_xg_overperformance_{window}'] = (
                        self.df[scored_col] - 
                        self.df[f'{team_type}_xg_avg_{window}']
                    )
                    
                self.features_created.extend([
                    f'{team_type}_xg_avg_{window}',
                    f'{team_type}_xg_overperformance_{window}'
                ])
    
    def _create_shot_features(self):
        """Create shot-related features if available."""
        shot_cols = ['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']
        
        if not any(col in self.df.columns for col in shot_cols):
            return
            
        for window in [3, 5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                
                if team_col not in self.df.columns:
                    continue
                
                # Shots
                shots_col = f'{team_type}_shots'
                if shots_col in self.df.columns:
                    self.df[f'{team_type}_shots_avg_{window}'] = self.df.groupby(team_col)[shots_col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    self.features_created.append(f'{team_type}_shots_avg_{window}')
                
                # Shots on target
                sot_col = f'{team_type}_shots_on_target'
                if sot_col in self.df.columns:
                    self.df[f'{team_type}_sot_avg_{window}'] = self.df.groupby(team_col)[sot_col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    self.features_created.append(f'{team_type}_sot_avg_{window}')
                    
                    # Shot accuracy
                    if shots_col in self.df.columns:
                        self.df[f'{team_type}_shot_accuracy_{window}'] = (
                            self.df[f'{team_type}_sot_avg_{window}'] /
                            self.df[f'{team_type}_shots_avg_{window}'].clip(lower=1)
                        )
                        self.features_created.append(f'{team_type}_shot_accuracy_{window}')
    
    def _create_possession_features(self):
        """Create possession-related features if available."""
        poss_cols = ['home_possession', 'away_possession']
        
        if not all(col in self.df.columns for col in poss_cols):
            return
            
        for window in [3, 5, 10]:
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team'
                poss_col = f'{team_type}_possession'
                
                if team_col not in self.df.columns:
                    continue
                    
                self.df[f'{team_type}_possession_avg_{window}'] = self.df.groupby(team_col)[poss_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.features_created.append(f'{team_type}_possession_avg_{window}')
    
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
            if all(f'{t}_scored_rate_{window}' in self.df.columns for t in ['home', 'away']):
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
            self.df[f'over_{threshold}'] = (self.df['total_goals'] > threshold).astype(int)
        
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
                    lambda x: x.rolling(window, min_periods=2).std().fillna(0)
                )
                
                # Over rates for each threshold
                for threshold in [1.5, 2.5, 3.5]:
                    self.df[f'{team_type}_over_{threshold}_rate_{window}'] = self.df.groupby(team_col)[f'over_{threshold}'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    self.features_created.append(f'{team_type}_over_{threshold}_rate_{window}')
                
                self.features_created.extend([
                    f'{team_type}_total_goals_avg_{window}',
                    f'{team_type}_total_goals_std_{window}'
                ])
        
        # Combined over probability
        for window in [3, 5, 10]:
            if f'home_total_goals_avg_{window}' in self.df.columns:
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
                
                # First half win rate
                ht_win = 'H' if team_type == 'home' else 'A'
                self.df[f'{team_type}_1h_win_rate_{window}'] = self.df.groupby(team_col)['ht_result'].transform(
                    lambda x: (x == ht_win).rolling(window, min_periods=1).mean()
                )
                
                # First half draw rate
                self.df[f'{team_type}_1h_draw_rate_{window}'] = self.df.groupby(team_col)['ht_result'].transform(
                    lambda x: (x == 'D').rolling(window, min_periods=1).mean()
                )
                
                # Goal ratio 1H vs 2H
                self.df[f'{team_type}_1h_2h_ratio_{window}'] = (
                    self.df[f'{team_type}_1h_goals_avg_{window}'] / 
                    self.df[f'{team_type}_2h_goals_avg_{window}'].clip(lower=0.1)
                )
                
                self.features_created.extend([
                    f'{team_type}_1h_goals_avg_{window}',
                    f'{team_type}_2h_goals_avg_{window}',
                    f'{team_type}_1h_win_rate_{window}',
                    f'{team_type}_1h_draw_rate_{window}',
                    f'{team_type}_1h_2h_ratio_{window}'
                ])
    
    def _create_correct_score_features(self):
        """Create correct score prediction features."""
        if 'home_goals' not in self.df.columns:
            return
            
        # Score string
        self.df['score'] = self.df['home_goals'].astype(str) + '-' + self.df['away_goals'].astype(str)
        
        # Common score patterns
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            goals_col = f'{team_type}_goals'
            
            if team_col not in self.df.columns:
                continue
            
            for window in [10, 20]:
                # Nil scorer rate
                self.df[f'{team_type}_nil_scorer_rate_{window}'] = self.df.groupby(team_col)[goals_col].transform(
                    lambda x: (x == 0).rolling(window, min_periods=5).mean()
                )
                
                # One goal scorer rate
                self.df[f'{team_type}_one_goal_rate_{window}'] = self.df.groupby(team_col)[goals_col].transform(
                    lambda x: (x == 1).rolling(window, min_periods=5).mean()
                )
                
                # Two goals scorer rate
                self.df[f'{team_type}_two_goals_rate_{window}'] = self.df.groupby(team_col)[goals_col].transform(
                    lambda x: (x == 2).rolling(window, min_periods=5).mean()
                )
                
                # Three+ goals scorer rate
                self.df[f'{team_type}_three_plus_goals_rate_{window}'] = self.df.groupby(team_col)[goals_col].transform(
                    lambda x: (x >= 3).rolling(window, min_periods=5).mean()
                )
                
                self.features_created.extend([
                    f'{team_type}_nil_scorer_rate_{window}',
                    f'{team_type}_one_goal_rate_{window}',
                    f'{team_type}_two_goals_rate_{window}',
                    f'{team_type}_three_plus_goals_rate_{window}'
                ])
    
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
        if 'season' in self.df.columns and 'league' in self.df.columns:
            self.df['match_number'] = self.df.groupby(['league', 'season']).cumcount() + 1
            max_matches = self.df.groupby(['league', 'season'])['match_number'].transform('max')
            self.df['season_progress'] = self.df['match_number'] / max_matches
            
            # Early/mid/late season indicators
            self.df['early_season'] = (self.df['season_progress'] < 0.25).astype(int)
            self.df['mid_season'] = ((self.df['season_progress'] >= 0.25) & (self.df['season_progress'] < 0.75)).astype(int)
            self.df['late_season'] = (self.df['season_progress'] >= 0.75).astype(int)
            
            self.features_created.extend([
                'match_number', 'season_progress', 'early_season', 'mid_season', 'late_season'
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
            self.df[f'{team_type}_days_rest'] = self.df.groupby(team_col)['match_date'].diff().dt.days.fillna(7)
            
            self.features_created.append(f'{team_type}_days_rest')
        
        if 'home_days_rest' in self.df.columns and 'away_days_rest' in self.df.columns:
            self.df['rest_difference'] = self.df['home_days_rest'] - self.df['away_days_rest']
            self.features_created.append('rest_difference')
    
    def _create_h2h_features(self):
        """Create head-to-head features (simplified for performance)."""
        if 'home_team' not in self.df.columns or 'away_team' not in self.df.columns:
            return
            
        # Create matchup key
        self.df['matchup'] = self.df.apply(
            lambda x: tuple(sorted([x['home_team'], x['away_team']])), axis=1
        )
        
        # H2H total goals average
        self.df['h2h_total_goals_avg'] = self.df.groupby('matchup')['total_goals'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        ) if 'total_goals' in self.df.columns else 2.5
        
        # H2H BTTS rate
        if 'btts' in self.df.columns:
            self.df['h2h_btts_rate'] = self.df.groupby('matchup')['btts'].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )
            self.features_created.append('h2h_btts_rate')
        
        self.features_created.append('h2h_total_goals_avg')
    
    def _create_interaction_features(self):
        """Create interaction features between home and away."""
        for window in [5, 10]:
            # Check if required features exist
            if f'home_attack_strength_{window}' not in self.df.columns:
                continue
                
            # Attack vs Defense matchups
            self.df[f'attack_vs_defense_{window}'] = (
                self.df[f'home_attack_strength_{window}'] * self.df[f'away_defense_weakness_{window}']
            )
            self.df[f'defense_vs_attack_{window}'] = (
                self.df[f'away_attack_strength_{window}'] * self.df[f'home_defense_weakness_{window}']
            )
            
            # Form difference
            if f'home_ppg_{window}' in self.df.columns:
                self.df[f'form_difference_{window}'] = (
                    self.df[f'home_ppg_{window}'] - self.df[f'away_ppg_{window}']
                )
                self.features_created.append(f'form_difference_{window}')
            
            # Rating difference
            self.df[f'rating_difference_{window}'] = (
                self.df[f'home_overall_rating_{window}'] - self.df[f'away_overall_rating_{window}']
            )
            
            self.features_created.extend([
                f'attack_vs_defense_{window}',
                f'defense_vs_attack_{window}',
                f'rating_difference_{window}'
            ])
    
    def _create_ratio_features(self):
        """Create ratio-based features."""
        for window in [5, 10]:
            if f'home_attack_strength_{window}' not in self.df.columns:
                continue
                
            # Attack ratio
            self.df[f'attack_ratio_{window}'] = (
                self.df[f'home_attack_strength_{window}'] / 
                self.df[f'away_attack_strength_{window}'].clip(lower=0.1)
            )
            
            # Defense ratio
            self.df[f'defense_ratio_{window}'] = (
                self.df[f'away_defense_weakness_{window}'] / 
                self.df[f'home_defense_weakness_{window}'].clip(lower=0.1)
            )
            
            self.features_created.extend([
                f'attack_ratio_{window}',
                f'defense_ratio_{window}'
            ])
    
    @staticmethod
    def _calculate_streak(series, *winning_values):
        """Calculate current streak of winning/unbeaten."""
        streak = 0
        streaks = []
        for val in series:
            if val in winning_values:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        return pd.Series(streaks, index=series.index)
    
    def get_feature_importance(self, target_col: str = 'result') -> Dict[str, float]:
        """Get feature importance using correlation analysis."""
        if target_col not in self.df.columns:
            return {}
        
        # Encode target
        if self.df[target_col].dtype == 'object':
            target = self.df[target_col].map({'H': 1, 'D': 0.5, 'A': 0}).fillna(0.5)
        else:
            target = self.df[target_col]
        
        importance = {}
        for feature in self.features_created:
            if feature in self.df.columns:
                corr = self.df[feature].corr(target)
                if not np.isnan(corr):
                    importance[feature] = abs(corr)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# Convenience function
def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all advanced features for a DataFrame."""
    engineer = AdvancedFeatureEngineer(df)
    return engineer.create_all_features()
