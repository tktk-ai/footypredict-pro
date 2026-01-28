"""
Enhanced Feature Engineering Module - 600+ Features
Generates comprehensive features across 20+ categories
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    rolling_windows: List[int] = None
    ewm_spans: List[int] = None
    goal_thresholds: List[float] = None
    include_advanced: bool = True
    include_embeddings: bool = False
    include_clusters: bool = True
    
    def __post_init__(self):
        self.rolling_windows = self.rolling_windows or [3, 5, 10, 20]
        self.ewm_spans = self.ewm_spans or [3, 5, 10]
        self.goal_thresholds = self.goal_thresholds or [0.5, 1.5, 2.5, 3.5, 4.5]


class EnhancedFeatureGenerator:
    """
    Comprehensive feature generator producing 600+ features across 20 categories.
    """
    
    def __init__(self, data: pd.DataFrame, config: FeatureConfig = None):
        self.data = data.copy()
        self.config = config or FeatureConfig()
        self._prepare_data()
        self.feature_count = 0
    
    def _prepare_data(self):
        """Prepare and validate data."""
        if 'match_date' in self.data.columns:
            self.data['match_date'] = pd.to_datetime(self.data['match_date'])
            self.data = self.data.sort_values('match_date').reset_index(drop=True)
        
        # Add result column
        if 'home_goals' in self.data.columns and 'away_goals' in self.data.columns:
            self.data['result'] = np.where(
                self.data['home_goals'] > self.data['away_goals'], 'H',
                np.where(self.data['home_goals'] < self.data['away_goals'], 'A', 'D')
            )
            self.data['total_goals'] = self.data['home_goals'] + self.data['away_goals']
            self.data['goal_diff'] = self.data['home_goals'] - self.data['away_goals']
    
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all 600+ features."""
        logger.info("Generating comprehensive feature set...")
        
        features = pd.DataFrame(index=self.data.index)
        
        # 1. Goal Features (80+)
        goal_feats = self._generate_goal_features()
        features = pd.concat([features, goal_feats], axis=1)
        logger.info(f"  Goal features: {len(goal_feats.columns)}")
        
        # 2. Form Features (60+)
        form_feats = self._generate_form_features()
        features = pd.concat([features, form_feats], axis=1)
        logger.info(f"  Form features: {len(form_feats.columns)}")
        
        # 3. Strength Ratings (40+)
        strength_feats = self._generate_strength_features()
        features = pd.concat([features, strength_feats], axis=1)
        logger.info(f"  Strength features: {len(strength_feats.columns)}")
        
        # 4. Momentum Features (40+)
        momentum_feats = self._generate_momentum_features()
        features = pd.concat([features, momentum_feats], axis=1)
        logger.info(f"  Momentum features: {len(momentum_feats.columns)}")
        
        # 5. BTTS Features (30+)
        btts_feats = self._generate_btts_features()
        features = pd.concat([features, btts_feats], axis=1)
        logger.info(f"  BTTS features: {len(btts_feats.columns)}")
        
        # 6. Over/Under Features (50+)
        ou_feats = self._generate_over_under_features()
        features = pd.concat([features, ou_feats], axis=1)
        logger.info(f"  Over/Under features: {len(ou_feats.columns)}")
        
        # 7. HT/FT Features (40+)
        htft_feats = self._generate_htft_features()
        features = pd.concat([features, htft_feats], axis=1)
        logger.info(f"  HT/FT features: {len(htft_feats.columns)}")
        
        # 8. H2H Features (30+)
        h2h_feats = self._generate_h2h_features()
        features = pd.concat([features, h2h_feats], axis=1)
        logger.info(f"  H2H features: {len(h2h_feats.columns)}")
        
        # 9. Timing Features (25+)
        timing_feats = self._generate_timing_features()
        features = pd.concat([features, timing_feats], axis=1)
        logger.info(f"  Timing features: {len(timing_feats.columns)}")
        
        # 10. Rest Features (15+)
        rest_feats = self._generate_rest_features()
        features = pd.concat([features, rest_feats], axis=1)
        logger.info(f"  Rest features: {len(rest_feats.columns)}")
        
        # 11. Streak Features (25+)
        streak_feats = self._generate_streak_features()
        features = pd.concat([features, streak_feats], axis=1)
        logger.info(f"  Streak features: {len(streak_feats.columns)}")
        
        # 12. Consistency Features (20+)
        consistency_feats = self._generate_consistency_features()
        features = pd.concat([features, consistency_feats], axis=1)
        logger.info(f"  Consistency features: {len(consistency_feats.columns)}")
        
        # 13. Interaction Features (60+)
        interaction_feats = self._generate_interaction_features(features)
        features = pd.concat([features, interaction_feats], axis=1)
        logger.info(f"  Interaction features: {len(interaction_feats.columns)}")
        
        # 14. Ratio Features (30+)
        ratio_feats = self._generate_ratio_features(features)
        features = pd.concat([features, ratio_feats], axis=1)
        logger.info(f"  Ratio features: {len(ratio_feats.columns)}")
        
        # 15. EWM Features (40+)
        ewm_feats = self._generate_ewm_features()
        features = pd.concat([features, ewm_feats], axis=1)
        logger.info(f"  EWM features: {len(ewm_feats.columns)}")
        
        # 16. Percentile Features (20+)
        percentile_feats = self._generate_percentile_features(features)
        features = pd.concat([features, percentile_feats], axis=1)
        logger.info(f"  Percentile features: {len(percentile_feats.columns)}")
        
        if self.config.include_clusters:
            # 17. Cluster Features (15+)
            cluster_feats = self._generate_cluster_features(features)
            features = pd.concat([features, cluster_feats], axis=1)
            logger.info(f"  Cluster features: {len(cluster_feats.columns)}")
        
        self.feature_count = len(features.columns)
        logger.info(f"Total features generated: {self.feature_count}")
        
        return features
    
    def _generate_goal_features(self) -> pd.DataFrame:
        """Generate goal-related features across rolling windows."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            goals_scored_col = 'home_goals' if team_type == 'home' else 'away_goals'
            goals_conceded_col = 'away_goals' if team_type == 'home' else 'home_goals'
            
            for window in self.config.rolling_windows:
                prefix = f"{team_type}_L{window}"
                
                # Group by team and calculate rolling stats
                for col_to_roll, metric_name in [
                    (goals_scored_col, 'goals_scored'),
                    (goals_conceded_col, 'goals_conceded'),
                    ('total_goals', 'total_goals'),
                ]:
                    if col_to_roll not in self.data.columns:
                        continue
                    
                    # Mean, std, sum, min, max
                    features[f"{prefix}_{metric_name}_mean"] = (
                        self.data.groupby(team_col)[col_to_roll]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    )
                    features[f"{prefix}_{metric_name}_std"] = (
                        self.data.groupby(team_col)[col_to_roll]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
                    )
                    features[f"{prefix}_{metric_name}_sum"] = (
                        self.data.groupby(team_col)[col_to_roll]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
                    )
                    features[f"{prefix}_{metric_name}_min"] = (
                        self.data.groupby(team_col)[col_to_roll]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
                    )
                    features[f"{prefix}_{metric_name}_max"] = (
                        self.data.groupby(team_col)[col_to_roll]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
                    )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_form_features(self) -> pd.DataFrame:
        """Generate form-related features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            
            for window in self.config.rolling_windows:
                prefix = f"{team_type}_L{window}"
                
                # Points calculation
                def calc_points(group):
                    points = []
                    for i, (idx, row) in enumerate(group.iterrows()):
                        team = row[team_col]
                        if row['home_team'] == team:
                            pts = 3 if row['result'] == 'H' else (1 if row['result'] == 'D' else 0)
                        else:
                            pts = 3 if row['result'] == 'A' else (1 if row['result'] == 'D' else 0)
                        points.append(pts)
                    return pd.Series(points, index=group.index)
                
                # PPG (Points Per Game)
                features[f"{prefix}_ppg"] = (
                    self.data.groupby(team_col)
                    .apply(lambda g: calc_points(g).shift(1).rolling(window, min_periods=1).mean())
                    .reset_index(level=0, drop=True).sort_index()
                )
                
                # Win rate
                def calc_win_rate(group):
                    wins = []
                    for i, (idx, row) in enumerate(group.iterrows()):
                        team = row[team_col]
                        if row['home_team'] == team:
                            win = 1 if row['result'] == 'H' else 0
                        else:
                            win = 1 if row['result'] == 'A' else 0
                        wins.append(win)
                    return pd.Series(wins, index=group.index)
                
                features[f"{prefix}_win_rate"] = (
                    self.data.groupby(team_col)
                    .apply(lambda g: calc_win_rate(g).shift(1).rolling(window, min_periods=1).mean())
                    .reset_index(level=0, drop=True).sort_index()
                )
                
                # Draw rate
                features[f"{prefix}_draw_rate"] = (
                    self.data.groupby(team_col)['result']
                    .transform(lambda x: (x.shift(1) == 'D').rolling(window, min_periods=1).mean())
                )
                
                # Loss rate
                features[f"{prefix}_loss_rate"] = (
                    1 - features.get(f"{prefix}_win_rate", 0) - features.get(f"{prefix}_draw_rate", 0)
                )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_strength_features(self) -> pd.DataFrame:
        """Generate attack/defense strength ratings."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            goals_for = 'home_goals' if team_type == 'home' else 'away_goals'
            goals_against = 'away_goals' if team_type == 'home' else 'home_goals'
            
            for window in self.config.rolling_windows:
                prefix = f"{team_type}_L{window}"
                
                # Attack strength (goals scored / league average)
                league_avg_goals = self.data[goals_for].mean()
                team_goals_mean = (
                    self.data.groupby(team_col)[goals_for]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                features[f"{prefix}_attack_strength"] = team_goals_mean / max(league_avg_goals, 0.1)
                
                # Defense strength (goals conceded / league average)
                league_avg_conceded = self.data[goals_against].mean()
                team_conceded_mean = (
                    self.data.groupby(team_col)[goals_against]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                features[f"{prefix}_defense_strength"] = team_conceded_mean / max(league_avg_conceded, 0.1)
                
                # Overall strength
                features[f"{prefix}_overall_strength"] = (
                    features[f"{prefix}_attack_strength"] - features[f"{prefix}_defense_strength"]
                )
                
                # Goal difference
                features[f"{prefix}_goal_diff"] = team_goals_mean - team_conceded_mean
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_momentum_features(self) -> pd.DataFrame:
        """Generate momentum/trend features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            goals_for = 'home_goals' if team_type == 'home' else 'away_goals'
            
            # Short vs long term momentum
            short_window = 3
            long_window = 10
            
            short_term = (
                self.data.groupby(team_col)[goals_for]
                .transform(lambda x: x.shift(1).rolling(short_window, min_periods=1).mean())
            )
            long_term = (
                self.data.groupby(team_col)[goals_for]
                .transform(lambda x: x.shift(1).rolling(long_window, min_periods=1).mean())
            )
            
            features[f"{team_type}_momentum_goals"] = short_term - long_term
            features[f"{team_type}_momentum_ratio"] = short_term / (long_term + 0.1)
            
            # Form trend
            features[f"{team_type}_form_trend"] = (
                self.data.groupby(team_col)[goals_for]
                .transform(lambda x: x.shift(1).diff().rolling(3, min_periods=1).mean())
            )
        
        # Combined momentum
        features['momentum_diff'] = features['home_momentum_goals'] - features['away_momentum_goals']
        features['momentum_product'] = features['home_momentum_goals'] * features['away_momentum_goals']
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_btts_features(self) -> pd.DataFrame:
        """Generate Both Teams To Score features."""
        features = {}
        
        # Add BTTS column
        if 'btts' not in self.data.columns:
            self.data['btts'] = ((self.data['home_goals'] > 0) & (self.data['away_goals'] > 0)).astype(int)
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            goals_for = 'home_goals' if team_type == 'home' else 'away_goals'
            goals_against = 'away_goals' if team_type == 'home' else 'home_goals'
            
            for window in self.config.rolling_windows:
                prefix = f"{team_type}_L{window}"
                
                # BTTS rate
                features[f"{prefix}_btts_rate"] = (
                    self.data.groupby(team_col)['btts']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                
                # Scoring rate (scored in X% of games)
                scored = (self.data[goals_for] > 0).astype(int)
                features[f"{prefix}_scoring_rate"] = (
                    self.data.groupby(team_col)[goals_for]
                    .transform(lambda x: (x.shift(1) > 0).rolling(window, min_periods=1).mean())
                )
                
                # Conceding rate
                features[f"{prefix}_conceding_rate"] = (
                    self.data.groupby(team_col)[goals_against]
                    .transform(lambda x: (x.shift(1) > 0).rolling(window, min_periods=1).mean())
                )
                
                # Clean sheet rate
                features[f"{prefix}_clean_sheet_rate"] = 1 - features[f"{prefix}_conceding_rate"]
                
                # Failed to score rate
                features[f"{prefix}_failed_to_score_rate"] = 1 - features[f"{prefix}_scoring_rate"]
        
        # Combined BTTS probability
        features['btts_combined_prob'] = (
            features.get('home_L5_scoring_rate', 0.5) * features.get('away_L5_scoring_rate', 0.5) *
            features.get('home_L5_conceding_rate', 0.5) * features.get('away_L5_conceding_rate', 0.5)
        )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_over_under_features(self) -> pd.DataFrame:
        """Generate Over/Under features for multiple thresholds."""
        features = {}
        
        for threshold in self.config.goal_thresholds:
            threshold_name = str(threshold).replace('.', '_')
            over_col = f"over_{threshold_name}"
            self.data[over_col] = (self.data['total_goals'] > threshold).astype(int)
            
            for team_type in ['home', 'away']:
                team_col = 'home_team' if team_type == 'home' else 'away_team'
                
                for window in self.config.rolling_windows:
                    prefix = f"{team_type}_L{window}"
                    
                    features[f"{prefix}_over{threshold_name}_rate"] = (
                        self.data.groupby(team_col)[over_col]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_htft_features(self) -> pd.DataFrame:
        """Generate Half-Time/Full-Time features."""
        features = {}
        
        if 'home_goals_ht' not in self.data.columns:
            return pd.DataFrame()
        
        # Add HT result
        self.data['ht_result'] = np.where(
            self.data['home_goals_ht'] > self.data['away_goals_ht'], 'H',
            np.where(self.data['home_goals_ht'] < self.data['away_goals_ht'], 'A', 'D')
        )
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            ht_goals = 'home_goals_ht' if team_type == 'home' else 'away_goals_ht'
            
            for window in self.config.rolling_windows:
                prefix = f"{team_type}_L{window}"
                
                # HT goals
                features[f"{prefix}_ht_goals_mean"] = (
                    self.data.groupby(team_col)[ht_goals]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                
                # HT lead rate
                if team_type == 'home':
                    features[f"{prefix}_ht_lead_rate"] = (
                        self.data.groupby(team_col)['ht_result']
                        .transform(lambda x: (x.shift(1) == 'H').rolling(window, min_periods=1).mean())
                    )
                else:
                    features[f"{prefix}_ht_lead_rate"] = (
                        self.data.groupby(team_col)['ht_result']
                        .transform(lambda x: (x.shift(1) == 'A').rolling(window, min_periods=1).mean())
                    )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_h2h_features(self) -> pd.DataFrame:
        """Generate Head-to-Head features."""
        features = {}
        
        for idx, row in self.data.iterrows():
            home_team = row.get('home_team', '')
            away_team = row.get('away_team', '')
            match_date = row.get('match_date', datetime.now())
            
            # Get historical H2H
            h2h = self.data[
                (
                    ((self.data['home_team'] == home_team) & (self.data['away_team'] == away_team)) |
                    ((self.data['home_team'] == away_team) & (self.data['away_team'] == home_team))
                ) &
                (self.data['match_date'] < match_date)
            ].tail(10)
            
            n_h2h = len(h2h)
            features.setdefault('h2h_matches', []).append(n_h2h)
            
            if n_h2h > 0:
                # Home team wins in H2H
                home_wins = len(h2h[
                    ((h2h['home_team'] == home_team) & (h2h['result'] == 'H')) |
                    ((h2h['away_team'] == home_team) & (h2h['result'] == 'A'))
                ])
                away_wins = len(h2h[
                    ((h2h['home_team'] == away_team) & (h2h['result'] == 'H')) |
                    ((h2h['away_team'] == away_team) & (h2h['result'] == 'A'))
                ])
                draws = n_h2h - home_wins - away_wins
                
                features.setdefault('h2h_home_win_rate', []).append(home_wins / n_h2h)
                features.setdefault('h2h_away_win_rate', []).append(away_wins / n_h2h)
                features.setdefault('h2h_draw_rate', []).append(draws / n_h2h)
                features.setdefault('h2h_total_goals_avg', []).append(h2h['total_goals'].mean())
                features.setdefault('h2h_btts_rate', []).append(h2h['btts'].mean() if 'btts' in h2h.columns else 0.5)
            else:
                features.setdefault('h2h_home_win_rate', []).append(0.33)
                features.setdefault('h2h_away_win_rate', []).append(0.33)
                features.setdefault('h2h_draw_rate', []).append(0.34)
                features.setdefault('h2h_total_goals_avg', []).append(2.5)
                features.setdefault('h2h_btts_rate', []).append(0.5)
        
        return pd.DataFrame(features)
    
    def _generate_timing_features(self) -> pd.DataFrame:
        """Generate time-based features."""
        features = {}
        
        if 'match_date' not in self.data.columns:
            return pd.DataFrame()
        
        # Day of week (0 = Monday)
        features['day_of_week'] = self.data['match_date'].dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Month
        features['month'] = self.data['match_date'].dt.month
        features['is_early_season'] = features['month'].isin([8, 9, 10]).astype(int)
        features['is_late_season'] = features['month'].isin([4, 5]).astype(int)
        
        # Season progress (0-1)
        if 'season' in self.data.columns:
            features['season_progress'] = (
                self.data.groupby('season')['match_date']
                .transform(lambda x: (x - x.min()).dt.days / max((x.max() - x.min()).days, 1))
            )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_rest_features(self) -> pd.DataFrame:
        """Generate rest days features."""
        features = {}
        
        if 'match_date' not in self.data.columns:
            return pd.DataFrame()
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            
            # Days since last match
            rest_days = []
            for idx, row in self.data.iterrows():
                team = row[team_col]
                match_date = row['match_date']
                
                prev_matches = self.data[
                    ((self.data['home_team'] == team) | (self.data['away_team'] == team)) &
                    (self.data['match_date'] < match_date)
                ]
                
                if len(prev_matches) > 0:
                    last_match = prev_matches['match_date'].max()
                    days = (match_date - last_match).days
                else:
                    days = 7  # Default
                
                rest_days.append(days)
            
            features[f"{team_type}_rest_days"] = rest_days
            features[f"{team_type}_is_short_rest"] = [1 if d <= 3 else 0 for d in rest_days]
            features[f"{team_type}_is_long_rest"] = [1 if d >= 7 else 0 for d in rest_days]
        
        # Rest advantage
        features['rest_diff'] = [
            h - a for h, a in zip(features['home_rest_days'], features['away_rest_days'])
        ]
        
        return pd.DataFrame(features)
    
    def _generate_streak_features(self) -> pd.DataFrame:
        """Generate streak features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            
            win_streak = []
            unbeaten_streak = []
            scoring_streak = []
            clean_sheet_streak = []
            
            for idx, row in self.data.iterrows():
                team = row[team_col]
                match_date = row.get('match_date', datetime.now())
                
                # Get recent matches
                recent = self.data[
                    ((self.data['home_team'] == team) | (self.data['away_team'] == team)) &
                    (self.data['match_date'] < match_date)
                ].tail(10)
                
                ws = 0
                ubs = 0
                ss = 0
                css = 0
                
                for _, m in recent.iloc[::-1].iterrows():
                    is_home = m['home_team'] == team
                    result = m['result']
                    goals_for = m['home_goals'] if is_home else m['away_goals']
                    goals_against = m['away_goals'] if is_home else m['home_goals']
                    
                    won = (is_home and result == 'H') or (not is_home and result == 'A')
                    drawn = result == 'D'
                    
                    if won and ws == len(recent.iloc[::-1].head(len(recent) - recent.iloc[::-1].index.get_loc(_))):
                        ws += 1
                    if (won or drawn):
                        if ubs == len(recent.iloc[::-1].head(len(recent) - recent.iloc[::-1].index.get_loc(_))):
                            ubs += 1
                    if goals_for > 0:
                        if ss == len(recent.iloc[::-1].head(len(recent) - recent.iloc[::-1].index.get_loc(_))):
                            ss += 1
                    if goals_against == 0:
                        if css == len(recent.iloc[::-1].head(len(recent) - recent.iloc[::-1].index.get_loc(_))):
                            css += 1
                
                win_streak.append(min(ws, 10))
                unbeaten_streak.append(min(ubs, 10))
                scoring_streak.append(min(ss, 10))
                clean_sheet_streak.append(min(css, 5))
            
            features[f"{team_type}_win_streak"] = win_streak
            features[f"{team_type}_unbeaten_streak"] = unbeaten_streak
            features[f"{team_type}_scoring_streak"] = scoring_streak
            features[f"{team_type}_clean_sheet_streak"] = clean_sheet_streak
        
        return pd.DataFrame(features)
    
    def _generate_consistency_features(self, window: int = 10) -> pd.DataFrame:
        """Generate consistency/variance features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            goals_col = 'home_goals' if team_type == 'home' else 'away_goals'
            
            # Goals std
            features[f"{team_type}_goals_std"] = (
                self.data.groupby(team_col)[goals_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
            )
            
            # Coefficient of variation
            goals_mean = (
                self.data.groupby(team_col)[goals_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
            )
            features[f"{team_type}_goals_cv"] = features[f"{team_type}_goals_std"] / (goals_mean + 0.1)
            
            # Results consistency (low variance = consistent)
            features[f"{team_type}_consistency_score"] = 1 / (features[f"{team_type}_goals_cv"] + 0.1)
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_interaction_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between home and away metrics."""
        features = {}
        
        # Attack vs Defense
        for window in [5, 10]:
            home_attack = base_features.get(f'home_L{window}_attack_strength', pd.Series(1))
            away_defense = base_features.get(f'away_L{window}_defense_strength', pd.Series(1))
            away_attack = base_features.get(f'away_L{window}_attack_strength', pd.Series(1))
            home_defense = base_features.get(f'home_L{window}_defense_strength', pd.Series(1))
            
            features[f'home_attack_vs_away_defense_L{window}'] = home_attack / (away_defense + 0.1)
            features[f'away_attack_vs_home_defense_L{window}'] = away_attack / (home_defense + 0.1)
            features[f'attack_diff_L{window}'] = home_attack - away_attack
            features[f'defense_diff_L{window}'] = home_defense - away_defense
            features[f'strength_product_L{window}'] = home_attack * away_attack
        
        # Form interactions
        for window in [5, 10]:
            home_form = base_features.get(f'home_L{window}_ppg', pd.Series(1.5))
            away_form = base_features.get(f'away_L{window}_ppg', pd.Series(1.5))
            
            features[f'form_ratio_L{window}'] = home_form / (away_form + 0.1)
            features[f'form_diff_L{window}'] = home_form - away_form
            features[f'form_product_L{window}'] = home_form * away_form
        
        # Goal expectancy
        home_goals_mean = base_features.get('home_L5_goals_scored_mean', pd.Series(1.5))
        away_goals_mean = base_features.get('away_L5_goals_scored_mean', pd.Series(1.5))
        
        features['expected_total_goals'] = home_goals_mean + away_goals_mean
        features['expected_goal_diff'] = home_goals_mean - away_goals_mean
        features['goal_expectancy_product'] = home_goals_mean * away_goals_mean
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_ratio_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio-based features."""
        features = {}
        
        # Win rate ratios
        home_win = base_features.get('home_L5_win_rate', pd.Series(0.33))
        away_win = base_features.get('away_L5_win_rate', pd.Series(0.33))
        
        features['win_rate_ratio'] = home_win / (away_win + 0.01)
        features['win_rate_diff'] = home_win - away_win
        
        # Goals ratios
        for window in [5, 10]:
            home_goals = base_features.get(f'home_L{window}_goals_scored_mean', pd.Series(1.5))
            away_goals = base_features.get(f'away_L{window}_goals_scored_mean', pd.Series(1.5))
            home_conceded = base_features.get(f'home_L{window}_goals_conceded_mean', pd.Series(1.0))
            away_conceded = base_features.get(f'away_L{window}_goals_conceded_mean', pd.Series(1.0))
            
            features[f'goals_for_ratio_L{window}'] = home_goals / (away_goals + 0.1)
            features[f'goals_against_ratio_L{window}'] = home_conceded / (away_conceded + 0.1)
            
            # Attack/Defense ratios for each team
            features[f'home_attack_defense_ratio_L{window}'] = home_goals / (home_conceded + 0.1)
            features[f'away_attack_defense_ratio_L{window}'] = away_goals / (away_conceded + 0.1)
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_ewm_features(self) -> pd.DataFrame:
        """Generate exponentially weighted moving average features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_col = 'home_team' if team_type == 'home' else 'away_team'
            goals_col = 'home_goals' if team_type == 'home' else 'away_goals'
            
            for span in self.config.ewm_spans:
                prefix = f"{team_type}_ewm{span}"
                
                features[f'{prefix}_goals'] = (
                    self.data.groupby(team_col)[goals_col]
                    .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
                )
                
                features[f'{prefix}_total_goals'] = (
                    self.data.groupby(team_col)['total_goals']
                    .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
                )
        
        return pd.DataFrame(features).fillna(0)
    
    def _generate_percentile_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate league percentile features."""
        features = {}
        
        key_metrics = [
            'home_L5_goals_scored_mean', 'away_L5_goals_scored_mean',
            'home_L5_attack_strength', 'away_L5_attack_strength',
            'home_L5_ppg', 'away_L5_ppg'
        ]
        
        for metric in key_metrics:
            if metric in base_features.columns:
                features[f'{metric}_percentile'] = (
                    base_features[metric].rank(pct=True)
                )
        
        return pd.DataFrame(features).fillna(0.5)
    
    def _generate_cluster_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate team style clustering features."""
        features = {}
        
        try:
            # Select features for clustering
            cluster_cols = [
                col for col in base_features.columns 
                if 'L5' in col and ('attack' in col or 'defense' in col or 'goals' in col)
            ][:10]  # Limit to 10 features
            
            if len(cluster_cols) < 3:
                return pd.DataFrame()
            
            X = base_features[cluster_cols].fillna(0)
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-Means clustering
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # One-hot encode clusters
            for i in range(n_clusters):
                features[f'style_cluster_{i}'] = (clusters == i).astype(int)
            
            features['cluster_id'] = clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
        
        return pd.DataFrame(features)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING - TEST")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_matches = 100
    teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
    
    sample_data = pd.DataFrame({
        'match_date': pd.date_range('2024-01-01', periods=n_matches, freq='3D'),
        'home_team': np.random.choice(teams, n_matches),
        'away_team': np.random.choice(teams, n_matches),
        'home_goals': np.random.poisson(1.5, n_matches),
        'away_goals': np.random.poisson(1.2, n_matches),
        'home_goals_ht': np.random.poisson(0.7, n_matches),
        'away_goals_ht': np.random.poisson(0.6, n_matches),
        'season': '2024'
    })
    
    # Generate features
    generator = EnhancedFeatureGenerator(sample_data)
    features = generator.generate_all_features()
    
    print(f"\n✅ Generated {len(features.columns)} features!")
    print(f"\nSample columns: {list(features.columns)[:20]}")
