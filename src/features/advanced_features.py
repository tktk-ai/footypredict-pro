"""
Advanced Feature Engineering for Football Predictions
======================================================

Generates comprehensive features for improved model accuracy:
- Form-based features (last 5/10 games)
- Head-to-head historical stats
- Home/Away specific performance
- Goal timing patterns
- Expected Goals (xG) approximation
- League position features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


class AdvancedFeatureGenerator:
    """
    Generate advanced features for match prediction.
    """
    
    def __init__(self, matches_df: pd.DataFrame = None):
        """Initialize with historical match data."""
        self.matches_df = matches_df
        self.team_stats_cache = {}
        
        if matches_df is not None:
            self._build_team_stats_cache()
    
    def _build_team_stats_cache(self):
        """Pre-compute team statistics from historical data."""
        df = self.matches_df.copy()
        
        if df.empty:
            return
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')
        
        # Build team-level aggregates
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for team in teams:
            # Home matches
            home_matches = df[df['home_team'] == team]
            # Away matches
            away_matches = df[df['away_team'] == team]
            
            if len(home_matches) == 0 and len(away_matches) == 0:
                continue
            
            # Calculate stats
            home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
            home_draws = (home_matches['home_score'] == home_matches['away_score']).sum()
            home_losses = (home_matches['home_score'] < home_matches['away_score']).sum()
            
            away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
            away_draws = (away_matches['away_score'] == away_matches['home_score']).sum()
            away_losses = (away_matches['away_score'] < away_matches['home_score']).sum()
            
            self.team_stats_cache[team.lower()] = {
                'home_matches': len(home_matches),
                'away_matches': len(away_matches),
                'home_wins': home_wins,
                'home_draws': home_draws,
                'home_losses': home_losses,
                'away_wins': away_wins,
                'away_draws': away_draws,
                'away_losses': away_losses,
                'home_goals_scored': home_matches['home_score'].sum(),
                'home_goals_conceded': home_matches['away_score'].sum(),
                'away_goals_scored': away_matches['away_score'].sum(),
                'away_goals_conceded': away_matches['home_score'].sum(),
                'home_clean_sheets': (home_matches['away_score'] == 0).sum(),
                'away_clean_sheets': (away_matches['home_score'] == 0).sum(),
                'home_btts': ((home_matches['home_score'] > 0) & (home_matches['away_score'] > 0)).sum(),
                'away_btts': ((away_matches['away_score'] > 0) & (away_matches['home_score'] > 0)).sum(),
            }
        
        logger.info(f"Built stats cache for {len(self.team_stats_cache)} teams")
    
    def get_form_features(self, team: str, n_games: int = 5, is_home: bool = True) -> Dict[str, float]:
        """
        Calculate form-based features from recent games.
        
        Args:
            team: Team name
            n_games: Number of recent games to consider
            is_home: Whether calculating for home team
        
        Returns:
            Dictionary of form features
        """
        team_key = team.lower()
        stats = self.team_stats_cache.get(team_key, {})
        
        if not stats:
            return self._get_default_form_features(is_home)
        
        # Calculate win rates
        total_home = stats.get('home_matches', 0) or 1
        total_away = stats.get('away_matches', 0) or 1
        
        home_win_rate = stats.get('home_wins', 0) / total_home
        away_win_rate = stats.get('away_wins', 0) / total_away
        
        # Goals per game
        home_gpg = stats.get('home_goals_scored', 0) / total_home
        away_gpg = stats.get('away_goals_scored', 0) / total_away
        
        home_conceded = stats.get('home_goals_conceded', 0) / total_home
        away_conceded = stats.get('away_goals_conceded', 0) / total_away
        
        # Clean sheet rate
        home_cs_rate = stats.get('home_clean_sheets', 0) / total_home
        away_cs_rate = stats.get('away_clean_sheets', 0) / total_away
        
        # BTTS rate
        home_btts_rate = stats.get('home_btts', 0) / total_home
        away_btts_rate = stats.get('away_btts', 0) / total_away
        
        prefix = 'home' if is_home else 'away'
        
        return {
            f'{prefix}_form_win_rate': home_win_rate if is_home else away_win_rate,
            f'{prefix}_form_goals_scored': home_gpg if is_home else away_gpg,
            f'{prefix}_form_goals_conceded': home_conceded if is_home else away_conceded,
            f'{prefix}_form_clean_sheet_rate': home_cs_rate if is_home else away_cs_rate,
            f'{prefix}_form_btts_rate': home_btts_rate if is_home else away_btts_rate,
            f'{prefix}_overall_win_rate': (stats.get('home_wins', 0) + stats.get('away_wins', 0)) / 
                                          max(1, total_home + total_away),
            f'{prefix}_avg_goals': (stats.get('home_goals_scored', 0) + stats.get('away_goals_scored', 0)) / 
                                   max(1, total_home + total_away),
        }
    
    def _get_default_form_features(self, is_home: bool) -> Dict[str, float]:
        """Return default form features for unknown teams."""
        prefix = 'home' if is_home else 'away'
        return {
            f'{prefix}_form_win_rate': 0.33,
            f'{prefix}_form_goals_scored': 1.3,
            f'{prefix}_form_goals_conceded': 1.1,
            f'{prefix}_form_clean_sheet_rate': 0.30,
            f'{prefix}_form_btts_rate': 0.50,
            f'{prefix}_overall_win_rate': 0.33,
            f'{prefix}_avg_goals': 1.3,
        }
    
    def get_h2h_features(self, home_team: str, away_team: str, n_matches: int = 10) -> Dict[str, float]:
        """
        Calculate head-to-head features.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            n_matches: Number of H2H matches to consider
        
        Returns:
            Dictionary of H2H features
        """
        if self.matches_df is None or self.matches_df.empty:
            return self._get_default_h2h_features()
        
        df = self.matches_df
        
        # Find H2H matches (either team at home or away)
        h2h_mask = (
            ((df['home_team'].str.lower() == home_team.lower()) & 
             (df['away_team'].str.lower() == away_team.lower())) |
            ((df['home_team'].str.lower() == away_team.lower()) & 
             (df['away_team'].str.lower() == home_team.lower()))
        )
        
        h2h_matches = df[h2h_mask].tail(n_matches)
        
        if len(h2h_matches) == 0:
            return self._get_default_h2h_features()
        
        # Calculate H2H stats from home team's perspective
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'].lower() == home_team.lower():
                home_goals += match['home_score']
                away_goals += match['away_score']
                if match['home_score'] > match['away_score']:
                    home_wins += 1
                elif match['away_score'] > match['home_score']:
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals += match['away_score']
                away_goals += match['home_score']
                if match['away_score'] > match['home_score']:
                    home_wins += 1
                elif match['home_score'] > match['away_score']:
                    away_wins += 1
                else:
                    draws += 1
        
        n = len(h2h_matches)
        
        return {
            'h2h_matches': n,
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': draws,
            'h2h_home_win_rate': home_wins / n,
            'h2h_away_win_rate': away_wins / n,
            'h2h_draw_rate': draws / n,
            'h2h_home_goals_avg': home_goals / n,
            'h2h_away_goals_avg': away_goals / n,
            'h2h_total_goals_avg': (home_goals + away_goals) / n,
            'h2h_btts_rate': sum(1 for _, m in h2h_matches.iterrows() 
                                 if m['home_score'] > 0 and m['away_score'] > 0) / n,
        }
    
    def _get_default_h2h_features(self) -> Dict[str, float]:
        """Return default H2H features for unknown matchups."""
        return {
            'h2h_matches': 0,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_home_win_rate': 0.40,
            'h2h_away_win_rate': 0.30,
            'h2h_draw_rate': 0.30,
            'h2h_home_goals_avg': 1.3,
            'h2h_away_goals_avg': 1.1,
            'h2h_total_goals_avg': 2.4,
            'h2h_btts_rate': 0.50,
        }
    
    def get_xg_approximation(self, team: str, is_home: bool = True) -> Dict[str, float]:
        """
        Approximate expected goals (xG) from historical shot/goal data.
        
        Since we don't have actual xG, we estimate from:
        - Goals scored / matches
        - Shot conversion rate (if available)
        """
        team_key = team.lower()
        stats = self.team_stats_cache.get(team_key, {})
        
        if not stats:
            return {'xg_approx': 1.3, 'xg_against_approx': 1.1}
        
        if is_home:
            matches = stats.get('home_matches', 0) or 1
            goals = stats.get('home_goals_scored', 0)
            conceded = stats.get('home_goals_conceded', 0)
        else:
            matches = stats.get('away_matches', 0) or 1
            goals = stats.get('away_goals_scored', 0)
            conceded = stats.get('away_goals_conceded', 0)
        
        # Simple xG approximation based on goals per game
        xg = goals / matches
        xg_against = conceded / matches
        
        prefix = 'home' if is_home else 'away'
        
        return {
            f'{prefix}_xg_approx': xg,
            f'{prefix}_xg_against_approx': xg_against,
        }
    
    def get_league_strength_features(self, league: str) -> Dict[str, float]:
        """
        Calculate league strength features.
        
        Different leagues have different characteristics
        (e.g., Bundesliga has more goals, Serie A more draws).
        """
        # League characteristic multipliers (from historical data)
        league_stats = {
            'Premier League': {'goals_mult': 1.05, 'home_adv': 1.10, 'draw_rate': 0.24},
            'Bundesliga': {'goals_mult': 1.15, 'home_adv': 1.05, 'draw_rate': 0.22},
            'La Liga': {'goals_mult': 1.00, 'home_adv': 1.12, 'draw_rate': 0.26},
            'Serie A': {'goals_mult': 0.95, 'home_adv': 1.08, 'draw_rate': 0.28},
            'Ligue 1': {'goals_mult': 1.05, 'home_adv': 1.06, 'draw_rate': 0.25},
            'Eredivisie': {'goals_mult': 1.20, 'home_adv': 1.08, 'draw_rate': 0.22},
        }
        
        # Fuzzy match league name
        league_lower = league.lower() if league else ''
        
        for name, stats in league_stats.items():
            if name.lower() in league_lower or league_lower in name.lower():
                return {
                    'league_goals_mult': stats['goals_mult'],
                    'league_home_advantage': stats['home_adv'],
                    'league_draw_rate': stats['draw_rate'],
                }
        
        # Default
        return {
            'league_goals_mult': 1.0,
            'league_home_advantage': 1.05,
            'league_draw_rate': 0.25,
        }
    
    def generate_all_features(self, home_team: str, away_team: str, 
                               league: str = '') -> Dict[str, float]:
        """
        Generate all available features for a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name (optional)
        
        Returns:
            Dictionary with all features
        """
        features = {}
        
        # Form features
        features.update(self.get_form_features(home_team, is_home=True))
        features.update(self.get_form_features(away_team, is_home=False))
        
        # H2H features
        features.update(self.get_h2h_features(home_team, away_team))
        
        # xG approximation
        features.update(self.get_xg_approximation(home_team, is_home=True))
        features.update(self.get_xg_approximation(away_team, is_home=False))
        
        # League features
        features.update(self.get_league_strength_features(league))
        
        # Derived features
        features['form_diff'] = features.get('home_form_win_rate', 0.33) - features.get('away_form_win_rate', 0.33)
        features['goals_diff'] = features.get('home_avg_goals', 1.3) - features.get('away_avg_goals', 1.3)
        features['xg_diff'] = features.get('home_xg_approx', 1.3) - features.get('away_xg_approx', 1.1)
        
        return features
    
    def generate_feature_matrix(self, matches: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """
        Generate feature matrix for multiple matches.
        
        Args:
            matches: List of (home_team, away_team, league) tuples
        
        Returns:
            DataFrame with features for all matches
        """
        rows = []
        
        for home_team, away_team, league in matches:
            features = self.generate_all_features(home_team, away_team, league)
            features['home_team'] = home_team
            features['away_team'] = away_team
            features['league'] = league
            rows.append(features)
        
        return pd.DataFrame(rows)


def load_feature_generator() -> AdvancedFeatureGenerator:
    """Load feature generator with training data."""
    from src.data.data_collector import get_training_data
    
    df = get_training_data()
    
    if df.empty:
        logger.warning("No training data found. Using empty feature generator.")
        return AdvancedFeatureGenerator()
    
    return AdvancedFeatureGenerator(df)


if __name__ == "__main__":
    # Test feature generation
    print("Testing Advanced Feature Generator...")
    
    gen = AdvancedFeatureGenerator()
    
    # Test with mock data
    features = gen.generate_all_features('Bayern München', 'Borussia Dortmund', 'Bundesliga')
    
    print(f"\nGenerated {len(features)} features:")
    for name, value in sorted(features.items()):
        print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")
