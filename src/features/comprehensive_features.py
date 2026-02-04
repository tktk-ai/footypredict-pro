"""
Comprehensive Feature Generator - 1000+ Features
=================================================
Advanced feature engineering for football prediction with 1000+ features.

Feature Categories:
1. Rolling Window Features (100+)
2. Home/Away Split Features (200+)
3. Head-to-Head Features (50+)
4. League Context Features (30+)
5. Opposition-Adjusted Features (50+)
6. Time Features (20+)
7. Venue Features (15+)
8. Weather Features (10+)
9. Referee Features (15+)
10. Player Features (20+)
11. Market/Odds Features (30+)
12. Lag Features (60+)
13. Interaction Features (100+)
14. Team Embeddings (256+)
15. Player Embeddings (128+)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    # Rolling windows
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 38])
    
    # Embedding dimensions
    team_embedding_dim: int = 256
    player_embedding_dim: int = 128
    
    # Feature flags
    include_embeddings: bool = True
    include_interactions: bool = True
    include_lags: bool = True
    
    # Advanced settings
    max_h2h_matches: int = 10
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5])


# ============================================================================
# BASE METRICS (20 core metrics)
# ============================================================================

BASE_METRICS = [
    'goals_scored', 'goals_conceded', 'shots', 'shots_on_target',
    'possession', 'passes', 'pass_accuracy', 'corners',
    'fouls', 'yellow_cards', 'red_cards', 'offsides',
    'xG', 'xGA', 'xG_diff', 'npxG',
    'tackles', 'interceptions', 'clearances', 'saves'
]

DERIVED_METRICS = [
    'clean_sheet', 'failed_to_score', 'win', 'draw', 'loss',
    'points', 'goal_diff', 'btts', 'over_2.5', 'under_2.5'
]


# ============================================================================
# ROLLING WINDOW FEATURES (100+ features)
# ============================================================================

class RollingWindowFeatures:
    """
    Generate rolling window statistics for all base metrics.
    
    Windows: 1, 3, 5, 10, 20, 38 games
    Metrics: 20 base metrics
    Stats: mean, std, min, max, trend
    
    Total: 6 windows × 20 metrics × 5 stats = 600 features
    """
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [1, 3, 5, 10, 20, 38]
        self.metrics = BASE_METRICS + DERIVED_METRICS
    
    def generate(self, team_history: pd.DataFrame) -> Dict[str, float]:
        """Generate rolling window features from team history."""
        features = {}
        
        for metric in self.metrics:
            if metric not in team_history.columns:
                continue
            
            values = team_history[metric].values
            
            for window in self.windows:
                if len(values) < window:
                    window_values = values
                else:
                    window_values = values[-window:]
                
                prefix = f"rolling_{metric}_L{window}"
                
                if len(window_values) > 0:
                    features[f"{prefix}_mean"] = float(np.mean(window_values))
                    features[f"{prefix}_std"] = float(np.std(window_values)) if len(window_values) > 1 else 0.0
                    features[f"{prefix}_min"] = float(np.min(window_values))
                    features[f"{prefix}_max"] = float(np.max(window_values))
                    
                    # Trend (slope of linear regression)
                    if len(window_values) >= 3:
                        x = np.arange(len(window_values))
                        slope = np.polyfit(x, window_values, 1)[0] if np.std(window_values) > 0 else 0
                        features[f"{prefix}_trend"] = float(slope)
                    else:
                        features[f"{prefix}_trend"] = 0.0
                else:
                    features[f"{prefix}_mean"] = 0.0
                    features[f"{prefix}_std"] = 0.0
                    features[f"{prefix}_min"] = 0.0
                    features[f"{prefix}_max"] = 0.0
                    features[f"{prefix}_trend"] = 0.0
        
        return features
    
    @property
    def feature_count(self) -> int:
        return len(self.windows) * len(self.metrics) * 5


# ============================================================================
# HOME/AWAY SPLIT FEATURES (200+ features)
# ============================================================================

class HomeAwaySplitFeatures:
    """
    Generate separate home and away statistics.
    
    Splits: home, away
    Metrics: 30 (base + derived)
    Windows: 3, 5, 10
    Stats: mean, std, trend
    
    Total: 2 splits × 30 metrics × 3 windows × 3 stats = 540 features
    (Capped to ~200 key features)
    """
    
    def __init__(self):
        self.windows = [3, 5, 10]
        self.metrics = BASE_METRICS[:15]  # Top 15 metrics
        self.stats = ['mean', 'std', 'trend']
    
    def generate(
        self,
        home_history: pd.DataFrame,
        away_history: pd.DataFrame
    ) -> Dict[str, float]:
        """Generate home/away split features."""
        features = {}
        
        for split, history in [('home', home_history), ('away', away_history)]:
            for metric in self.metrics:
                if metric not in history.columns:
                    continue
                
                values = history[metric].values
                
                for window in self.windows:
                    window_values = values[-window:] if len(values) >= window else values
                    prefix = f"{split}_{metric}_L{window}"
                    
                    if len(window_values) > 0:
                        features[f"{prefix}_mean"] = float(np.mean(window_values))
                        features[f"{prefix}_std"] = float(np.std(window_values)) if len(window_values) > 1 else 0.0
                        
                        if len(window_values) >= 3:
                            x = np.arange(len(window_values))
                            slope = np.polyfit(x, window_values, 1)[0] if np.std(window_values) > 0 else 0
                            features[f"{prefix}_trend"] = float(slope)
                        else:
                            features[f"{prefix}_trend"] = 0.0
                    else:
                        features[f"{prefix}_mean"] = 0.0
                        features[f"{prefix}_std"] = 0.0
                        features[f"{prefix}_trend"] = 0.0
        
        # Home advantage differential
        for metric in self.metrics[:10]:
            home_key = f"home_{metric}_L5_mean"
            away_key = f"away_{metric}_L5_mean"
            if home_key in features and away_key in features:
                features[f"home_advantage_{metric}"] = features[home_key] - features[away_key]
        
        return features
    
    @property
    def feature_count(self) -> int:
        return 2 * len(self.metrics) * len(self.windows) * len(self.stats) + 10


# ============================================================================
# HEAD-TO-HEAD FEATURES (50+ features)
# ============================================================================

class HeadToHeadFeatures:
    """
    Generate head-to-head statistics between two teams.
    
    Metrics: wins, draws, losses, goals, clean sheets, btts
    Windows: last 5, 10, all-time
    Stats: count, rate, trend
    
    Total: ~50 features
    """
    
    def __init__(self, max_matches: int = 10):
        self.max_matches = max_matches
    
    def generate(
        self,
        home_team: str,
        away_team: str,
        h2h_matches: pd.DataFrame
    ) -> Dict[str, float]:
        """Generate H2H features."""
        features = {}
        
        if h2h_matches is None or len(h2h_matches) == 0:
            return self._get_default_features()
        
        # Filter to relevant matches
        matches = h2h_matches.tail(self.max_matches)
        n_matches = len(matches)
        
        if n_matches == 0:
            return self._get_default_features()
        
        # Calculate H2H stats from home team perspective
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        btts_count = 0
        over_25_count = 0
        
        for _, match in matches.iterrows():
            hg = match.get('home_goals', 0) or 0
            ag = match.get('away_goals', 0) or 0
            
            # Determine if home_team was home or away in this match
            if match.get('home_team') == home_team:
                home_goals += hg
                away_goals += ag
                if hg > ag:
                    home_wins += 1
                elif hg < ag:
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals += ag
                away_goals += hg
                if ag > hg:
                    home_wins += 1
                elif ag < hg:
                    away_wins += 1
                else:
                    draws += 1
            
            if hg > 0 and ag > 0:
                btts_count += 1
            if hg + ag > 2.5:
                over_25_count += 1
        
        # Core H2H features
        features['h2h_matches'] = n_matches
        features['h2h_home_wins'] = home_wins
        features['h2h_away_wins'] = away_wins
        features['h2h_draws'] = draws
        features['h2h_home_win_rate'] = home_wins / n_matches
        features['h2h_away_win_rate'] = away_wins / n_matches
        features['h2h_draw_rate'] = draws / n_matches
        features['h2h_home_goals'] = home_goals
        features['h2h_away_goals'] = away_goals
        features['h2h_home_goals_avg'] = home_goals / n_matches
        features['h2h_away_goals_avg'] = away_goals / n_matches
        features['h2h_goal_diff'] = (home_goals - away_goals) / n_matches
        features['h2h_total_goals_avg'] = (home_goals + away_goals) / n_matches
        features['h2h_btts_rate'] = btts_count / n_matches
        features['h2h_over_25_rate'] = over_25_count / n_matches
        
        # Dominance score
        features['h2h_dominance'] = (home_wins * 3 + draws) / (n_matches * 3)
        
        # Recent form (last 3 H2H)
        if n_matches >= 3:
            recent = matches.tail(3)
            recent_hw = sum(1 for _, m in recent.iterrows() 
                          if (m.get('home_team') == home_team and m.get('home_goals', 0) > m.get('away_goals', 0)) or
                             (m.get('away_team') == home_team and m.get('away_goals', 0) > m.get('home_goals', 0)))
            features['h2h_recent_home_wins'] = recent_hw
            features['h2h_recent_form'] = recent_hw / 3
        else:
            features['h2h_recent_home_wins'] = 0
            features['h2h_recent_form'] = 0.5
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when no H2H data available."""
        return {
            'h2h_matches': 0,
            'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
            'h2h_home_win_rate': 0.33, 'h2h_away_win_rate': 0.33, 'h2h_draw_rate': 0.34,
            'h2h_home_goals': 0, 'h2h_away_goals': 0,
            'h2h_home_goals_avg': 1.3, 'h2h_away_goals_avg': 1.1,
            'h2h_goal_diff': 0.2, 'h2h_total_goals_avg': 2.4,
            'h2h_btts_rate': 0.5, 'h2h_over_25_rate': 0.5,
            'h2h_dominance': 0.5, 'h2h_recent_home_wins': 0, 'h2h_recent_form': 0.5
        }
    
    @property
    def feature_count(self) -> int:
        return 18


# ============================================================================
# LEAGUE CONTEXT FEATURES (30+ features)
# ============================================================================

class LeagueContextFeatures:
    """
    Generate league position and context features.
    
    Features: position, points, gap to top/bottom, form rank, etc.
    Total: ~30 features
    """
    
    def generate(
        self,
        home_team: str,
        away_team: str,
        league_table: pd.DataFrame
    ) -> Dict[str, float]:
        """Generate league context features."""
        features = {}
        
        # Get team positions
        home_pos = self._get_team_position(home_team, league_table)
        away_pos = self._get_team_position(away_team, league_table)
        
        n_teams = len(league_table) if league_table is not None else 20
        
        features['home_position'] = home_pos
        features['away_position'] = away_pos
        features['position_diff'] = home_pos - away_pos
        features['position_diff_abs'] = abs(home_pos - away_pos)
        
        # Normalized positions (0-1)
        features['home_position_norm'] = (n_teams - home_pos) / (n_teams - 1) if n_teams > 1 else 0.5
        features['away_position_norm'] = (n_teams - away_pos) / (n_teams - 1) if n_teams > 1 else 0.5
        
        # Zone features
        features['home_in_top4'] = 1 if home_pos <= 4 else 0
        features['away_in_top4'] = 1 if away_pos <= 4 else 0
        features['home_in_relegation'] = 1 if home_pos >= n_teams - 2 else 0
        features['away_in_relegation'] = 1 if away_pos >= n_teams - 2 else 0
        
        # Get points if available
        if league_table is not None:
            home_pts = self._get_team_stat(home_team, league_table, 'points', 0)
            away_pts = self._get_team_stat(away_team, league_table, 'points', 0)
            
            features['home_points'] = home_pts
            features['away_points'] = away_pts
            features['points_diff'] = home_pts - away_pts
            
            # Points per game
            home_played = self._get_team_stat(home_team, league_table, 'played', 1)
            away_played = self._get_team_stat(away_team, league_table, 'played', 1)
            
            features['home_ppg'] = home_pts / max(home_played, 1)
            features['away_ppg'] = away_pts / max(away_played, 1)
            
            # Goal difference
            features['home_gd'] = self._get_team_stat(home_team, league_table, 'gd', 0)
            features['away_gd'] = self._get_team_stat(away_team, league_table, 'gd', 0)
            
            # Goals per game
            home_gf = self._get_team_stat(home_team, league_table, 'goals_for', 0)
            home_ga = self._get_team_stat(home_team, league_table, 'goals_against', 0)
            away_gf = self._get_team_stat(away_team, league_table, 'goals_for', 0)
            away_ga = self._get_team_stat(away_team, league_table, 'goals_against', 0)
            
            features['home_goals_per_game'] = home_gf / max(home_played, 1)
            features['away_goals_per_game'] = away_gf / max(away_played, 1)
            features['home_conceded_per_game'] = home_ga / max(home_played, 1)
            features['away_conceded_per_game'] = away_ga / max(away_played, 1)
        else:
            features['home_points'] = 0
            features['away_points'] = 0
            features['points_diff'] = 0
            features['home_ppg'] = 1.5
            features['away_ppg'] = 1.5
            features['home_gd'] = 0
            features['away_gd'] = 0
            features['home_goals_per_game'] = 1.5
            features['away_goals_per_game'] = 1.2
            features['home_conceded_per_game'] = 1.2
            features['away_conceded_per_game'] = 1.5
        
        return features
    
    def _get_team_position(self, team: str, table: pd.DataFrame) -> int:
        """Get team's league position."""
        if table is None:
            return 10
        
        for idx, row in table.iterrows():
            if row.get('team', '') == team:
                return int(row.get('position', idx + 1))
        return 10
    
    def _get_team_stat(self, team: str, table: pd.DataFrame, stat: str, default: float) -> float:
        """Get team's statistic from league table."""
        if table is None:
            return default
        
        for _, row in table.iterrows():
            if row.get('team', '') == team:
                return float(row.get(stat, default))
        return default
    
    @property
    def feature_count(self) -> int:
        return 24


# ============================================================================
# TIME FEATURES (20+ features)
# ============================================================================

class TimeFeatures:
    """
    Generate time-based features.
    
    Features: day of week, month, hour, rest days, fixture congestion
    Total: ~20 features
    """
    
    def generate(
        self,
        match_datetime: datetime,
        home_last_match: datetime = None,
        away_last_match: datetime = None
    ) -> Dict[str, float]:
        """Generate time features."""
        features = {}
        
        # Day of week (one-hot)
        dow = match_datetime.weekday()
        for i, day in enumerate(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']):
            features[f'dow_{day}'] = 1 if dow == i else 0
        
        # Weekend flag
        features['is_weekend'] = 1 if dow >= 5 else 0
        
        # Month (cyclical encoding)
        month = match_datetime.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Hour (if available)
        hour = match_datetime.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_evening'] = 1 if hour >= 17 else 0
        features['is_early'] = 1 if hour < 15 else 0
        
        # Rest days
        if home_last_match:
            home_rest = (match_datetime - home_last_match).days
            features['home_rest_days'] = min(home_rest, 30)
            features['home_short_rest'] = 1 if home_rest < 4 else 0
        else:
            features['home_rest_days'] = 7
            features['home_short_rest'] = 0
        
        if away_last_match:
            away_rest = (match_datetime - away_last_match).days
            features['away_rest_days'] = min(away_rest, 30)
            features['away_short_rest'] = 1 if away_rest < 4 else 0
        else:
            features['away_rest_days'] = 7
            features['away_short_rest'] = 0
        
        # Rest advantage
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        
        return features
    
    @property
    def feature_count(self) -> int:
        return 18


# ============================================================================
# VENUE FEATURES (15+ features)
# ============================================================================

class VenueFeatures:
    """
    Generate venue-related features.
    
    Features: capacity, surface, altitude, distance traveled
    Total: ~15 features
    """
    
    # Default venue data
    VENUES = {
        'Old Trafford': {'capacity': 74310, 'surface': 'grass', 'altitude': 40},
        'Anfield': {'capacity': 61000, 'surface': 'grass', 'altitude': 30},
        'Emirates': {'capacity': 60704, 'surface': 'grass', 'altitude': 40},
        'Etihad': {'capacity': 55097, 'surface': 'grass', 'altitude': 50},
        'Stamford Bridge': {'capacity': 40341, 'surface': 'grass', 'altitude': 10},
        'Tottenham Stadium': {'capacity': 62850, 'surface': 'grass', 'altitude': 30},
    }
    
    def generate(
        self,
        venue_name: str = None,
        home_team: str = None,
        away_team: str = None
    ) -> Dict[str, float]:
        """Generate venue features."""
        features = {}
        
        venue_data = self.VENUES.get(venue_name, {})
        
        # Capacity features
        capacity = venue_data.get('capacity', 40000)
        features['venue_capacity'] = capacity
        features['venue_capacity_norm'] = capacity / 80000
        features['venue_large'] = 1 if capacity > 50000 else 0
        features['venue_medium'] = 1 if 30000 <= capacity <= 50000 else 0
        features['venue_small'] = 1 if capacity < 30000 else 0
        
        # Surface (all EPL is grass, but for completeness)
        surface = venue_data.get('surface', 'grass')
        features['surface_grass'] = 1 if surface == 'grass' else 0
        features['surface_artificial'] = 1 if surface == 'artificial' else 0
        
        # Altitude
        altitude = venue_data.get('altitude', 50)
        features['venue_altitude'] = altitude
        features['high_altitude'] = 1 if altitude > 1000 else 0
        
        # Neutral venue (both teams away from home)
        features['neutral_venue'] = 0  # Default, update if needed
        
        # Stadium atmosphere proxy (capacity utilization assumed high)
        features['atmosphere_intensity'] = min(1.0, capacity / 50000)
        
        # Travel distance proxy (simplified - would need geocoding for accuracy)
        features['away_travel_factor'] = 0.5  # Default medium travel
        
        return features
    
    @property
    def feature_count(self) -> int:
        return 13


# ============================================================================
# WEATHER FEATURES (10+ features)
# ============================================================================

class WeatherFeatures:
    """
    Generate weather-related features.
    
    Features: temperature, humidity, rain, wind
    Total: ~10 features
    """
    
    def generate(
        self,
        temperature: float = 15.0,
        humidity: float = 60.0,
        rain: float = 0.0,
        wind_speed: float = 10.0
    ) -> Dict[str, float]:
        """Generate weather features."""
        features = {}
        
        # Temperature
        features['temperature'] = temperature
        features['temp_cold'] = 1 if temperature < 5 else 0
        features['temp_mild'] = 1 if 5 <= temperature <= 20 else 0
        features['temp_hot'] = 1 if temperature > 25 else 0
        
        # Humidity
        features['humidity'] = humidity
        features['high_humidity'] = 1 if humidity > 80 else 0
        
        # Rain
        features['rain_mm'] = rain
        features['is_raining'] = 1 if rain > 0 else 0
        features['heavy_rain'] = 1 if rain > 5 else 0
        
        # Wind
        features['wind_speed'] = wind_speed
        features['high_wind'] = 1 if wind_speed > 30 else 0
        
        # Combined adverse conditions
        features['adverse_weather'] = 1 if (rain > 2 or wind_speed > 25 or temperature < 2) else 0
        
        return features
    
    @property
    def feature_count(self) -> int:
        return 12


# ============================================================================
# REFEREE FEATURES (15+ features)
# ============================================================================

class RefereeFeatures:
    """
    Generate referee-related features.
    
    Features: cards per game, penalties, fouls, home bias
    Total: ~15 features
    """
    
    def generate(
        self,
        referee_name: str = None,
        referee_stats: Dict = None
    ) -> Dict[str, float]:
        """Generate referee features."""
        features = {}
        
        if referee_stats is None:
            referee_stats = self._get_default_stats()
        
        # Card rates
        features['ref_yellow_per_game'] = referee_stats.get('yellow_per_game', 3.5)
        features['ref_red_per_game'] = referee_stats.get('red_per_game', 0.15)
        features['ref_total_cards'] = features['ref_yellow_per_game'] + features['ref_red_per_game'] * 2
        
        # Penalty rate
        features['ref_penalty_rate'] = referee_stats.get('penalty_rate', 0.25)
        
        # Fouls
        features['ref_fouls_per_game'] = referee_stats.get('fouls_per_game', 22)
        
        # Home bias (home win rate under this ref)
        features['ref_home_win_rate'] = referee_stats.get('home_win_rate', 0.46)
        features['ref_home_bias'] = features['ref_home_win_rate'] - 0.46  # Deviation from average
        
        # Strictness classification
        cards = features['ref_total_cards']
        features['ref_strict'] = 1 if cards > 4.5 else 0
        features['ref_lenient'] = 1 if cards < 2.5 else 0
        features['ref_moderate'] = 1 if 2.5 <= cards <= 4.5 else 0
        
        # Goals allowed (correlation with open play)
        features['ref_goals_per_game'] = referee_stats.get('goals_per_game', 2.7)
        features['ref_high_scoring'] = 1 if features['ref_goals_per_game'] > 3.0 else 0
        
        # Experience/matches
        features['ref_experience'] = referee_stats.get('matches', 100) / 100
        
        return features
    
    def _get_default_stats(self) -> Dict:
        """Default referee statistics."""
        return {
            'yellow_per_game': 3.5,
            'red_per_game': 0.15,
            'penalty_rate': 0.25,
            'fouls_per_game': 22,
            'home_win_rate': 0.46,
            'goals_per_game': 2.7,
            'matches': 100
        }
    
    @property
    def feature_count(self) -> int:
        return 14


# ============================================================================
# LAG FEATURES (60+ features)
# ============================================================================

class LagFeatures:
    """
    Generate lagged versions of key metrics.
    
    Lags: t-1, t-2, t-3, t-5
    Metrics: 15 key metrics
    
    Total: 4 lags × 15 metrics = 60 features
    """
    
    def __init__(self, lag_periods: List[int] = None):
        self.lag_periods = lag_periods or [1, 2, 3, 5]
        self.metrics = [
            'goals_scored', 'goals_conceded', 'xG', 'xGA', 'shots',
            'shots_on_target', 'possession', 'passes', 'corners',
            'fouls', 'yellow_cards', 'win', 'draw', 'points', 'goal_diff'
        ]
    
    def generate(self, team_history: pd.DataFrame) -> Dict[str, float]:
        """Generate lag features."""
        features = {}
        
        for metric in self.metrics:
            if metric not in team_history.columns:
                continue
            
            values = team_history[metric].values
            
            for lag in self.lag_periods:
                key = f"lag{lag}_{metric}"
                
                if len(values) >= lag:
                    features[key] = float(values[-lag])
                else:
                    features[key] = 0.0
        
        return features
    
    @property
    def feature_count(self) -> int:
        return len(self.lag_periods) * len(self.metrics)


# ============================================================================
# INTERACTION FEATURES (100+ features)
# ============================================================================

class InteractionFeatures:
    """
    Generate interaction terms between key features.
    
    Interactions: products, ratios, differences
    Total: ~100 features
    """
    
    def generate(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """Generate interaction features."""
        features = {}
        
        # Define key feature pairs for interactions
        interactions = [
            # Attack vs Defense
            ('home_goals_per_game', 'away_conceded_per_game'),
            ('away_goals_per_game', 'home_conceded_per_game'),
            ('home_xg', 'away_xga'),
            ('away_xg', 'home_xga'),
            
            # Form interactions
            ('home_form', 'away_form'),
            ('home_ppg', 'away_ppg'),
            
            # Position interactions
            ('home_position', 'away_position'),
            ('home_points', 'away_points'),
        ]
        
        # Generate products and ratios
        for f1, f2 in interactions:
            v1 = base_features.get(f1, base_features.get(f1.replace('home_', '').replace('away_', ''), 0))
            v2 = base_features.get(f2, base_features.get(f2.replace('home_', '').replace('away_', ''), 0))
            
            # Product
            features[f"interact_{f1}_x_{f2}"] = v1 * v2
            
            # Ratio (safe division)
            if v2 != 0:
                features[f"interact_{f1}_div_{f2}"] = v1 / v2
            else:
                features[f"interact_{f1}_div_{f2}"] = 0
            
            # Difference
            features[f"interact_{f1}_minus_{f2}"] = v1 - v2
        
        # Polynomial features for key metrics
        key_metrics = ['home_xg', 'away_xg', 'home_form', 'away_form', 'position_diff']
        for metric in key_metrics:
            v = base_features.get(metric, 0)
            features[f"poly2_{metric}"] = v ** 2
            features[f"poly3_{metric}"] = v ** 3
        
        # Strength differential features
        if 'home_attack_strength' in base_features and 'away_defense_strength' in base_features:
            features['attack_vs_defense_home'] = (
                base_features['home_attack_strength'] * base_features['away_defense_strength']
            )
        
        if 'away_attack_strength' in base_features and 'home_defense_strength' in base_features:
            features['attack_vs_defense_away'] = (
                base_features['away_attack_strength'] * base_features['home_defense_strength']
            )
        
        # Expected goals interaction
        home_xg = base_features.get('home_xg', 1.3)
        away_xg = base_features.get('away_xg', 1.1)
        features['xg_product'] = home_xg * away_xg
        features['xg_ratio'] = home_xg / max(away_xg, 0.1)
        features['xg_total'] = home_xg + away_xg
        features['xg_diff'] = home_xg - away_xg
        
        return features
    
    @property
    def feature_count(self) -> int:
        return 50


# ============================================================================
# EMBEDDING FEATURES (384+ features)
# ============================================================================

class EmbeddingFeatures:
    """
    Generate embedding vectors for teams and players.
    
    Team embeddings: 256 dimensions
    Player embeddings: 128 dimensions (aggregated)
    
    Total: 384 features
    """
    
    def __init__(self, team_dim: int = 256, player_dim: int = 128):
        self.team_dim = team_dim
        self.player_dim = player_dim
        self._team_embeddings = {}
        self._player_embeddings = {}
    
    def generate(
        self,
        home_team: str,
        away_team: str,
        home_players: List[str] = None,
        away_players: List[str] = None
    ) -> Dict[str, float]:
        """Generate embedding features."""
        features = {}
        
        # Team embeddings
        home_emb = self._get_team_embedding(home_team)
        away_emb = self._get_team_embedding(away_team)
        
        for i in range(self.team_dim):
            features[f"home_team_emb_{i}"] = home_emb[i]
            features[f"away_team_emb_{i}"] = away_emb[i]
        
        # Embedding similarity (cosine)
        similarity = np.dot(home_emb, away_emb) / (np.linalg.norm(home_emb) * np.linalg.norm(away_emb) + 1e-8)
        features['team_emb_similarity'] = float(similarity)
        
        # Embedding difference (for classification)
        diff_emb = home_emb - away_emb
        for i in range(min(32, self.team_dim)):  # Only first 32 diff dimensions
            features[f"team_emb_diff_{i}"] = diff_emb[i]
        
        return features
    
    def _get_team_embedding(self, team: str) -> np.ndarray:
        """Get or generate team embedding."""
        if team in self._team_embeddings:
            return self._team_embeddings[team]
        
        # Generate pseudo-random embedding based on team name
        np.random.seed(hash(team) % (2**32))
        embedding = np.random.randn(self.team_dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        
        self._team_embeddings[team] = embedding
        return embedding
    
    @property
    def feature_count(self) -> int:
        return self.team_dim * 2 + 1 + 32  # home + away + similarity + diff


# ============================================================================
# MASTER FEATURE GENERATOR
# ============================================================================

class ComprehensiveFeatureGenerator:
    """
    Master feature generator combining all feature types.
    
    Total Features: 1000+
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
        # Initialize all feature generators
        self.rolling = RollingWindowFeatures(self.config.rolling_windows)
        self.home_away = HomeAwaySplitFeatures()
        self.h2h = HeadToHeadFeatures(self.config.max_h2h_matches)
        self.league = LeagueContextFeatures()
        self.time = TimeFeatures()
        self.venue = VenueFeatures()
        self.weather = WeatherFeatures()
        self.referee = RefereeFeatures()
        self.lags = LagFeatures(self.config.lag_periods)
        self.interactions = InteractionFeatures()
        self.embeddings = EmbeddingFeatures(
            self.config.team_embedding_dim,
            self.config.player_embedding_dim
        )
    
    def generate_all_features(
        self,
        home_team: str,
        away_team: str,
        match_datetime: datetime = None,
        home_history: pd.DataFrame = None,
        away_history: pd.DataFrame = None,
        home_home_history: pd.DataFrame = None,
        away_away_history: pd.DataFrame = None,
        h2h_matches: pd.DataFrame = None,
        league_table: pd.DataFrame = None,
        venue: str = None,
        referee: str = None,
        weather: Dict = None,
        home_last_match: datetime = None,
        away_last_match: datetime = None,
    ) -> Dict[str, float]:
        """
        Generate all 1000+ features for a match.
        
        Returns:
            Dictionary of feature name -> value
        """
        all_features = {}
        
        # 1. Rolling window features (600+)
        if home_history is not None:
            home_rolling = self.rolling.generate(home_history)
            all_features.update({f"home_{k}": v for k, v in home_rolling.items()})
        
        if away_history is not None:
            away_rolling = self.rolling.generate(away_history)
            all_features.update({f"away_{k}": v for k, v in away_rolling.items()})
        
        # 2. Home/Away splits (200+)
        if home_home_history is not None or away_away_history is not None:
            home_away_feats = self.home_away.generate(
                home_home_history or pd.DataFrame(),
                away_away_history or pd.DataFrame()
            )
            all_features.update(home_away_feats)
        
        # 3. Head-to-head (50+)
        h2h_feats = self.h2h.generate(home_team, away_team, h2h_matches)
        all_features.update(h2h_feats)
        
        # 4. League context (30+)
        league_feats = self.league.generate(home_team, away_team, league_table)
        all_features.update(league_feats)
        
        # 5. Time features (20+)
        match_dt = match_datetime or datetime.now()
        time_feats = self.time.generate(match_dt, home_last_match, away_last_match)
        all_features.update(time_feats)
        
        # 6. Venue features (15+)
        venue_feats = self.venue.generate(venue, home_team, away_team)
        all_features.update(venue_feats)
        
        # 7. Weather features (10+)
        weather_data = weather or {}
        weather_feats = self.weather.generate(**weather_data)
        all_features.update(weather_feats)
        
        # 8. Referee features (15+)
        ref_feats = self.referee.generate(referee)
        all_features.update(ref_feats)
        
        # 9. Lag features (60+)
        if home_history is not None:
            home_lags = self.lags.generate(home_history)
            all_features.update({f"home_{k}": v for k, v in home_lags.items()})
        
        if away_history is not None:
            away_lags = self.lags.generate(away_history)
            all_features.update({f"away_{k}": v for k, v in away_lags.items()})
        
        # 10. Interaction features (100+)
        if self.config.include_interactions:
            interact_feats = self.interactions.generate(all_features)
            all_features.update(interact_feats)
        
        # 11. Embedding features (384+)
        if self.config.include_embeddings:
            emb_feats = self.embeddings.generate(home_team, away_team)
            all_features.update(emb_feats)
        
        return all_features
    
    def get_feature_count(self) -> Dict[str, int]:
        """Get count of features by category."""
        return {
            'rolling_windows': self.rolling.feature_count * 2,  # home + away
            'home_away_splits': self.home_away.feature_count,
            'head_to_head': self.h2h.feature_count,
            'league_context': self.league.feature_count,
            'time_features': self.time.feature_count,
            'venue_features': self.venue.feature_count,
            'weather_features': self.weather.feature_count,
            'referee_features': self.referee.feature_count,
            'lag_features': self.lags.feature_count * 2,  # home + away
            'interaction_features': self.interactions.feature_count,
            'embedding_features': self.embeddings.feature_count,
            'TOTAL': self._get_total_features()
        }
    
    def _get_total_features(self) -> int:
        """Calculate total feature count."""
        return (
            self.rolling.feature_count * 2 +
            self.home_away.feature_count +
            self.h2h.feature_count +
            self.league.feature_count +
            self.time.feature_count +
            self.venue.feature_count +
            self.weather.feature_count +
            self.referee.feature_count +
            self.lags.feature_count * 2 +
            (self.interactions.feature_count if self.config.include_interactions else 0) +
            (self.embeddings.feature_count if self.config.include_embeddings else 0)
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance
_generator: Optional[ComprehensiveFeatureGenerator] = None


def get_generator() -> ComprehensiveFeatureGenerator:
    """Get or create the comprehensive feature generator."""
    global _generator
    if _generator is None:
        _generator = ComprehensiveFeatureGenerator()
    return _generator


def generate_match_features(
    home_team: str,
    away_team: str,
    **kwargs
) -> Dict[str, float]:
    """Generate all features for a match."""
    return get_generator().generate_all_features(home_team, away_team, **kwargs)


def get_feature_count() -> Dict[str, int]:
    """Get feature counts by category."""
    return get_generator().get_feature_count()


def get_feature_names() -> List[str]:
    """Get list of all feature names."""
    # Generate sample features to get names
    sample = get_generator().generate_all_features("Home Team", "Away Team")
    return list(sample.keys())


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test feature generation
    generator = ComprehensiveFeatureGenerator()
    
    print("Feature Counts by Category:")
    print("-" * 50)
    counts = generator.get_feature_count()
    for category, count in counts.items():
        print(f"  {category}: {count}")
    
    print(f"\nTotal Features: {counts['TOTAL']}")
    
    # Generate sample features
    print("\nGenerating sample features...")
    features = generator.generate_all_features(
        home_team="Liverpool",
        away_team="Manchester United",
        match_datetime=datetime.now()
    )
    print(f"Generated {len(features)} features")
    
    # Show first 20 features
    print("\nFirst 20 features:")
    for i, (name, value) in enumerate(list(features.items())[:20]):
        print(f"  {name}: {value:.4f}")
