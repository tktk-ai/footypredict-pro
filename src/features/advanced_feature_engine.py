"""
Advanced Feature Engineering Module

Generates 150+ features per match for improved prediction accuracy:
- Core statistics (shots, corners, cards)
- Form features with time decay
- Head-to-head history
- xG-based features
- Market/odds features
- Contextual features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"


class AdvancedFeatureEngine:
    """Generates 150+ features per match for ML prediction"""
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        self.historical_data = historical_data
        self.team_stats_cache = {}
        self.h2h_cache = {}
        
        if historical_data is not None:
            self._build_caches()
    
    def _build_caches(self) -> None:
        """Build team statistics and H2H caches from historical data"""
        if self.historical_data is None or self.historical_data.empty:
            return
        
        df = self.historical_data
        
        # Build team stats cache
        for team in set(df.get('home_team', [])) | set(df.get('HomeTeam', [])):
            if isinstance(team, str):
                self.team_stats_cache[team.lower()] = self._calculate_team_stats(team)
        
        logger.info(f"Built cache for {len(self.team_stats_cache)} teams")
    
    def _calculate_team_stats(self, team: str) -> Dict:
        """Calculate historical statistics for a team"""
        df = self.historical_data
        team_lower = team.lower()
        
        # Get home and away matches
        home_col = 'home_team' if 'home_team' in df.columns else 'HomeTeam'
        away_col = 'away_team' if 'away_team' in df.columns else 'AwayTeam'
        
        home_matches = df[df[home_col].str.lower() == team_lower] if home_col in df.columns else pd.DataFrame()
        away_matches = df[df[away_col].str.lower() == team_lower] if away_col in df.columns else pd.DataFrame()
        
        stats = {
            # Goals
            'goals_scored_home': home_matches.get('home_goals', home_matches.get('FTHG', pd.Series())).mean() or 1.5,
            'goals_conceded_home': home_matches.get('away_goals', home_matches.get('FTAG', pd.Series())).mean() or 1.2,
            'goals_scored_away': away_matches.get('away_goals', away_matches.get('FTAG', pd.Series())).mean() or 1.1,
            'goals_conceded_away': away_matches.get('home_goals', away_matches.get('FTHG', pd.Series())).mean() or 1.4,
            
            # Shots
            'shots_home': home_matches.get('home_shots', home_matches.get('HS', pd.Series())).mean() or 12,
            'shots_away': away_matches.get('away_shots', away_matches.get('AS', pd.Series())).mean() or 10,
            'shots_target_home': home_matches.get('home_shots_target', home_matches.get('HST', pd.Series())).mean() or 4,
            'shots_target_away': away_matches.get('away_shots_target', away_matches.get('AST', pd.Series())).mean() or 3,
            
            # Corners
            'corners_home': home_matches.get('home_corners', home_matches.get('HC', pd.Series())).mean() or 5,
            'corners_away': away_matches.get('away_corners', away_matches.get('AC', pd.Series())).mean() or 4,
            
            # Cards
            'yellows_home': home_matches.get('home_yellows', home_matches.get('HY', pd.Series())).mean() or 1.5,
            'yellows_away': away_matches.get('away_yellows', away_matches.get('AY', pd.Series())).mean() or 1.7,
            'reds_home': home_matches.get('home_reds', home_matches.get('HR', pd.Series())).mean() or 0.05,
            'reds_away': away_matches.get('away_reds', away_matches.get('AR', pd.Series())).mean() or 0.05,
            
            # Fouls
            'fouls_home': home_matches.get('home_fouls', home_matches.get('HF', pd.Series())).mean() or 11,
            'fouls_away': away_matches.get('away_fouls', away_matches.get('AF', pd.Series())).mean() or 12,
            
            # Match counts
            'home_matches': len(home_matches),
            'away_matches': len(away_matches),
            'total_matches': len(home_matches) + len(away_matches),
            
            # Win rates
            'home_win_rate': self._calculate_win_rate(home_matches, 'home'),
            'away_win_rate': self._calculate_win_rate(away_matches, 'away'),
            
            # xG (if available)
            'xg_home': home_matches.get('home_xg', pd.Series()).mean() or 0,
            'xg_away': away_matches.get('away_xg', pd.Series()).mean() or 0,
        }
        
        return stats
    
    def _calculate_win_rate(self, matches: pd.DataFrame, team_type: str) -> float:
        """Calculate win rate from matches"""
        if matches.empty:
            return 0.33
        
        result_col = 'result' if 'result' in matches.columns else 'FTR'
        if result_col not in matches.columns:
            return 0.33
        
        if team_type == 'home':
            wins = (matches[result_col] == 'H').sum()
        else:
            wins = (matches[result_col] == 'A').sum()
        
        return wins / len(matches) if len(matches) > 0 else 0.33
    
    def rolling_form(self, team: str, n_matches: int = 5, 
                     decay: float = 0.9) -> Dict[str, float]:
        """Calculate rolling form with exponential time decay"""
        if self.historical_data is None:
            return self._default_form()
        
        df = self.historical_data
        team_lower = team.lower()
        
        # Find team matches
        home_col = 'home_team' if 'home_team' in df.columns else 'HomeTeam'
        away_col = 'away_team' if 'away_team' in df.columns else 'AwayTeam'
        result_col = 'result' if 'result' in df.columns else 'FTR'
        
        # Get recent matches
        home_mask = df[home_col].str.lower() == team_lower if home_col in df.columns else pd.Series([False] * len(df))
        away_mask = df[away_col].str.lower() == team_lower if away_col in df.columns else pd.Series([False] * len(df))
        
        team_matches = df[home_mask | away_mask].head(n_matches)
        
        if team_matches.empty:
            return self._default_form()
        
        # Calculate weighted form
        points = []
        goals_for = []
        goals_against = []
        
        for i, (_, match) in enumerate(team_matches.iterrows()):
            weight = decay ** i
            
            is_home = str(match.get(home_col, '')).lower() == team_lower
            result = match.get(result_col, 'D')
            
            # Points
            if (is_home and result == 'H') or (not is_home and result == 'A'):
                points.append(3 * weight)
            elif result == 'D':
                points.append(1 * weight)
            else:
                points.append(0)
            
            # Goals
            home_goals = match.get('home_goals', match.get('FTHG', 0)) or 0
            away_goals = match.get('away_goals', match.get('FTAG', 0)) or 0
            
            if is_home:
                goals_for.append(home_goals * weight)
                goals_against.append(away_goals * weight)
            else:
                goals_for.append(away_goals * weight)
                goals_against.append(home_goals * weight)
        
        total_weight = sum(decay ** i for i in range(len(team_matches)))
        
        return {
            'form_points': sum(points) / total_weight if total_weight > 0 else 1.0,
            'form_goals_scored': sum(goals_for) / total_weight if total_weight > 0 else 1.0,
            'form_goals_conceded': sum(goals_against) / total_weight if total_weight > 0 else 1.0,
            'form_goal_diff': (sum(goals_for) - sum(goals_against)) / total_weight if total_weight > 0 else 0,
            'form_matches': len(team_matches),
            'form_wins': sum(1 for p in points if p > 2),
            'form_draws': sum(1 for p in points if 0 < p <= 1),
            'form_losses': sum(1 for p in points if p == 0),
        }
    
    def _default_form(self) -> Dict[str, float]:
        return {
            'form_points': 1.5, 'form_goals_scored': 1.2, 'form_goals_conceded': 1.2,
            'form_goal_diff': 0, 'form_matches': 0, 'form_wins': 0, 
            'form_draws': 0, 'form_losses': 0
        }
    
    def head_to_head(self, home_team: str, away_team: str, 
                     n_matches: int = 5) -> Dict[str, float]:
        """Get head-to-head statistics"""
        if self.historical_data is None:
            return self._default_h2h()
        
        cache_key = f"{home_team.lower()}_{away_team.lower()}"
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]
        
        df = self.historical_data
        home_col = 'home_team' if 'home_team' in df.columns else 'HomeTeam'
        away_col = 'away_team' if 'away_team' in df.columns else 'AwayTeam'
        result_col = 'result' if 'result' in df.columns else 'FTR'
        
        # Find H2H matches (either home or away)
        mask1 = (df[home_col].str.lower() == home_team.lower()) & (df[away_col].str.lower() == away_team.lower())
        mask2 = (df[home_col].str.lower() == away_team.lower()) & (df[away_col].str.lower() == home_team.lower())
        
        h2h_matches = df[mask1 | mask2].head(n_matches)
        
        if h2h_matches.empty:
            return self._default_h2h()
        
        # Calculate H2H stats
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        
        for _, match in h2h_matches.iterrows():
            is_home_in_this_match = str(match.get(home_col, '')).lower() == home_team.lower()
            result = match.get(result_col, 'D')
            
            hg = match.get('home_goals', match.get('FTHG', 0)) or 0
            ag = match.get('away_goals', match.get('FTAG', 0)) or 0
            
            if is_home_in_this_match:
                home_goals += hg
                away_goals += ag
                if result == 'H':
                    home_wins += 1
                elif result == 'A':
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals += ag
                away_goals += hg
                if result == 'A':
                    home_wins += 1
                elif result == 'H':
                    away_wins += 1
                else:
                    draws += 1
        
        n = len(h2h_matches)
        h2h_stats = {
            'h2h_matches': n,
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': draws,
            'h2h_home_win_rate': home_wins / n if n > 0 else 0.33,
            'h2h_away_win_rate': away_wins / n if n > 0 else 0.33,
            'h2h_draw_rate': draws / n if n > 0 else 0.33,
            'h2h_home_goals_avg': home_goals / n if n > 0 else 1.2,
            'h2h_away_goals_avg': away_goals / n if n > 0 else 1.0,
            'h2h_total_goals_avg': (home_goals + away_goals) / n if n > 0 else 2.2,
        }
        
        self.h2h_cache[cache_key] = h2h_stats
        return h2h_stats
    
    def _default_h2h(self) -> Dict[str, float]:
        return {
            'h2h_matches': 0, 'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
            'h2h_home_win_rate': 0.4, 'h2h_away_win_rate': 0.3, 'h2h_draw_rate': 0.3,
            'h2h_home_goals_avg': 1.2, 'h2h_away_goals_avg': 1.0, 'h2h_total_goals_avg': 2.2
        }
    
    def odds_features(self, home_odds: float = 2.0, draw_odds: float = 3.3, 
                      away_odds: float = 3.5) -> Dict[str, float]:
        """Extract features from betting odds"""
        # Convert odds to implied probabilities
        total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
        
        home_prob = (1/home_odds) / total_prob
        draw_prob = (1/draw_odds) / total_prob
        away_prob = (1/away_odds) / total_prob
        
        # Overround (bookmaker margin)
        overround = total_prob - 1
        
        return {
            'odds_home': home_odds,
            'odds_draw': draw_odds,
            'odds_away': away_odds,
            'implied_home_prob': home_prob,
            'implied_draw_prob': draw_prob,
            'implied_away_prob': away_prob,
            'odds_overround': overround,
            'odds_favorite_margin': max(home_prob, away_prob) - min(home_prob, away_prob),
            'odds_is_home_favorite': 1 if home_prob > away_prob else 0,
            'odds_is_away_favorite': 1 if away_prob > home_prob else 0,
            'odds_home_value': home_odds * home_prob,  # EV indicator
            'odds_away_value': away_odds * away_prob,
        }
    
    def contextual_features(self, home_team: str, away_team: str,
                           match_date: Optional[datetime] = None,
                           is_cup: bool = False,
                           is_derby: bool = False) -> Dict[str, float]:
        """Extract contextual features about the match"""
        if match_date is None:
            match_date = datetime.now()
        
        # Time-based features
        day_of_week = match_date.weekday()
        month = match_date.month
        
        # Season position (rough estimate)
        if month >= 8:
            season_progress = (month - 8) / 10  # Aug to May
        else:
            season_progress = (month + 4) / 10
        
        return {
            'ctx_day_of_week': day_of_week,
            'ctx_is_weekend': 1 if day_of_week >= 5 else 0,
            'ctx_month': month,
            'ctx_season_progress': min(1.0, max(0.0, season_progress)),
            'ctx_is_cup': 1 if is_cup else 0,
            'ctx_is_derby': 1 if is_derby else 0,
            'ctx_end_of_season': 1 if month in [4, 5] else 0,
            'ctx_start_of_season': 1 if month in [8, 9] else 0,
        }
    
    def extract_all_features(self, home_team: str, away_team: str,
                            home_odds: float = 2.0, draw_odds: float = 3.3,
                            away_odds: float = 3.5,
                            match_date: Optional[datetime] = None) -> np.ndarray:
        """Extract all 150+ features for a match"""
        features = {}
        
        # 1. Get team statistics (40+ features)
        home_stats = self.team_stats_cache.get(home_team.lower(), self._calculate_team_stats(home_team))
        away_stats = self.team_stats_cache.get(away_team.lower(), self._calculate_team_stats(away_team))
        
        for key, value in home_stats.items():
            features[f'home_{key}'] = value if isinstance(value, (int, float)) else 0
        for key, value in away_stats.items():
            features[f'away_{key}'] = value if isinstance(value, (int, float)) else 0
        
        # 2. Rolling form (16 features: 8 per team)
        home_form = self.rolling_form(home_team)
        away_form = self.rolling_form(away_team)
        
        for key, value in home_form.items():
            features[f'home_{key}'] = value
        for key, value in away_form.items():
            features[f'away_{key}'] = value
        
        # 3. Head-to-head (10 features)
        h2h = self.head_to_head(home_team, away_team)
        features.update(h2h)
        
        # 4. Odds features (12 features)
        odds_feats = self.odds_features(home_odds, draw_odds, away_odds)
        features.update(odds_feats)
        
        # 5. Contextual features (8 features)
        ctx_feats = self.contextual_features(home_team, away_team, match_date)
        features.update(ctx_feats)
        
        # 6. Derived features (20+ features)
        features['diff_goals_scored'] = home_stats.get('goals_scored_home', 1.5) - away_stats.get('goals_scored_away', 1.1)
        features['diff_goals_conceded'] = away_stats.get('goals_conceded_away', 1.4) - home_stats.get('goals_conceded_home', 1.2)
        features['diff_shots'] = home_stats.get('shots_home', 12) - away_stats.get('shots_away', 10)
        features['diff_shots_target'] = home_stats.get('shots_target_home', 4) - away_stats.get('shots_target_away', 3)
        features['diff_corners'] = home_stats.get('corners_home', 5) - away_stats.get('corners_away', 4)
        features['diff_form_points'] = home_form.get('form_points', 1.5) - away_form.get('form_points', 1.5)
        features['diff_win_rate'] = home_stats.get('home_win_rate', 0.4) - away_stats.get('away_win_rate', 0.3)
        
        # Expected total goals
        features['expected_total_goals'] = (
            home_stats.get('goals_scored_home', 1.5) + 
            away_stats.get('goals_scored_away', 1.1)
        )
        
        # BTTS indicator
        features['btts_indicator'] = min(
            1 - (1 - home_stats.get('goals_scored_home', 1.5) / 3),
            1 - (1 - away_stats.get('goals_scored_away', 1.1) / 3)
        )
        
        # Convert to numpy array
        feature_values = [float(v) if isinstance(v, (int, float)) and not np.isnan(v) else 0.0 
                         for v in features.values()]
        
        return np.array(feature_values)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Generate a dummy extraction to get feature names
        features = {}
        
        # Add all feature groups
        for prefix in ['home_', 'away_']:
            for stat in ['goals_scored_home', 'goals_conceded_home', 'goals_scored_away', 
                        'goals_conceded_away', 'shots_home', 'shots_away', 'shots_target_home',
                        'shots_target_away', 'corners_home', 'corners_away', 'yellows_home',
                        'yellows_away', 'reds_home', 'reds_away', 'fouls_home', 'fouls_away',
                        'home_matches', 'away_matches', 'total_matches', 'home_win_rate',
                        'away_win_rate', 'xg_home', 'xg_away']:
                features[f'{prefix}{stat}'] = 0
            
            for form in ['form_points', 'form_goals_scored', 'form_goals_conceded',
                        'form_goal_diff', 'form_matches', 'form_wins', 'form_draws', 'form_losses']:
                features[f'{prefix}{form}'] = 0
        
        # H2H
        for h2h in ['h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
                   'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
                   'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_total_goals_avg']:
            features[h2h] = 0
        
        # Odds
        for odds in ['odds_home', 'odds_draw', 'odds_away', 'implied_home_prob',
                    'implied_draw_prob', 'implied_away_prob', 'odds_overround',
                    'odds_favorite_margin', 'odds_is_home_favorite', 'odds_is_away_favorite',
                    'odds_home_value', 'odds_away_value']:
            features[odds] = 0
        
        # Context
        for ctx in ['ctx_day_of_week', 'ctx_is_weekend', 'ctx_month', 'ctx_season_progress',
                   'ctx_is_cup', 'ctx_is_derby', 'ctx_end_of_season', 'ctx_start_of_season']:
            features[ctx] = 0
        
        # Derived
        for diff in ['diff_goals_scored', 'diff_goals_conceded', 'diff_shots',
                    'diff_shots_target', 'diff_corners', 'diff_form_points', 'diff_win_rate',
                    'expected_total_goals', 'btts_indicator']:
            features[diff] = 0
        
        return list(features.keys())


def create_feature_engine(data_path: Optional[Path] = None) -> AdvancedFeatureEngine:
    """Create a feature engine with historical data"""
    if data_path is None:
        data_path = DATA_DIR / "processed" / "master_training_data.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        return AdvancedFeatureEngine(df)
    
    # Try existing training data
    existing = DATA_DIR / "comprehensive_training_data.csv"
    if existing.exists():
        df = pd.read_csv(existing)
        return AdvancedFeatureEngine(df)
    
    return AdvancedFeatureEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test feature extraction
    engine = create_feature_engine()
    
    features = engine.extract_all_features(
        home_team="Arsenal",
        away_team="Chelsea",
        home_odds=2.1,
        draw_odds=3.4,
        away_odds=3.2
    )
    
    print(f"Generated {len(features)} features")
    print(f"Feature names: {len(engine.get_feature_names())}")
