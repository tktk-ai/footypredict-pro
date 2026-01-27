"""
Momentum Features Module
Calculates form and momentum indicators.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MomentumFeatureGenerator:
    """
    Generates momentum and form-based features.
    
    Features include:
    - Streaks (wins, losses, unbeaten)
    - Form trajectory
    - Momentum scores
    - Confidence indicators
    """
    
    def __init__(self, matches_df: pd.DataFrame = None):
        self.matches = matches_df
        self.team_forms = {}
    
    def set_matches(self, matches_df: pd.DataFrame):
        """Set match data."""
        self.matches = matches_df.copy()
        if 'match_date' in self.matches.columns:
            self.matches = self.matches.sort_values('match_date')
        self._compute_all_forms()
    
    def _compute_all_forms(self):
        """Compute form for all teams."""
        if self.matches is None:
            return
        
        teams = set(self.matches['home_team'].unique()) | set(self.matches['away_team'].unique())
        for team in teams:
            self.team_forms[team] = self._compute_team_form(team)
    
    def _compute_team_form(self, team: str) -> Dict:
        """Compute form metrics for a team."""
        # Get matches
        home_matches = self.matches[self.matches['home_team'] == team].copy()
        away_matches = self.matches[self.matches['away_team'] == team].copy()
        
        home_matches['result_for_team'] = home_matches.apply(
            lambda r: 'W' if r['home_goals'] > r['away_goals']
                     else ('D' if r['home_goals'] == r['away_goals'] else 'L'),
            axis=1
        )
        away_matches['result_for_team'] = away_matches.apply(
            lambda r: 'W' if r['away_goals'] > r['home_goals']
                     else ('D' if r['away_goals'] == r['home_goals'] else 'L'),
            axis=1
        )
        
        all_matches = pd.concat([home_matches, away_matches]).sort_values('match_date')
        
        if len(all_matches) == 0:
            return {}
        
        results = all_matches['result_for_team'].tolist()
        
        form = {
            'total_matches': len(results),
        }
        
        # Current streak
        if results:
            streak_type = results[-1]
            streak_length = 0
            for r in reversed(results):
                if r == streak_type:
                    streak_length += 1
                else:
                    break
            
            form['current_streak'] = streak_length
            form['streak_type'] = streak_type
            
            # Unbeaten streak
            unbeaten = 0
            for r in reversed(results):
                if r != 'L':
                    unbeaten += 1
                else:
                    break
            form['unbeaten_streak'] = unbeaten
            
            # Without win streak
            without_win = 0
            for r in reversed(results):
                if r != 'W':
                    without_win += 1
                else:
                    break
            form['without_win_streak'] = without_win
        
        # Form strings
        for n in [3, 5, 10]:
            if len(results) >= n:
                recent = results[-n:]
                form[f'form_{n}'] = ''.join(recent)
                form[f'wins_last_{n}'] = recent.count('W')
                form[f'draws_last_{n}'] = recent.count('D')
                form[f'losses_last_{n}'] = recent.count('L')
                form[f'points_last_{n}'] = recent.count('W') * 3 + recent.count('D')
        
        # Momentum score (weighted recent results)
        if len(results) >= 5:
            weights = [0.35, 0.25, 0.20, 0.12, 0.08]
            recent_5 = results[-5:]
            points_map = {'W': 3, 'D': 1, 'L': 0}
            
            momentum = sum(
                points_map[r] * w 
                for r, w in zip(recent_5, weights)
            )
            form['momentum_score'] = momentum
        
        # Form trajectory (is form improving or declining?)
        if len(results) >= 10:
            first_5 = results[-10:-5]
            last_5 = results[-5:]
            
            first_5_pts = first_5.count('W') * 3 + first_5.count('D')
            last_5_pts = last_5.count('W') * 3 + last_5.count('D')
            
            form['form_trajectory'] = last_5_pts - first_5_pts
            form['is_improving'] = last_5_pts > first_5_pts
        
        return form
    
    def get_team_momentum(self, team: str) -> Dict:
        """Get momentum features for a team."""
        return self.team_forms.get(team, {})
    
    def get_match_features(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Get momentum features for a match."""
        home_form = self.get_team_momentum(home_team)
        away_form = self.get_team_momentum(away_team)
        
        features = {}
        
        # Add individual team features
        for key, value in home_form.items():
            if isinstance(value, (int, float)):
                features[f'home_{key}'] = value
        for key, value in away_form.items():
            if isinstance(value, (int, float)):
                features[f'away_{key}'] = value
        
        # Comparative features
        if 'momentum_score' in home_form and 'momentum_score' in away_form:
            features['momentum_diff'] = home_form['momentum_score'] - away_form['momentum_score']
        
        if 'points_last_5' in home_form and 'points_last_5' in away_form:
            features['recent_form_diff'] = home_form['points_last_5'] - away_form['points_last_5']
        
        # Streak comparisons
        if 'unbeaten_streak' in home_form and 'unbeaten_streak' in away_form:
            features['unbeaten_diff'] = home_form['unbeaten_streak'] - away_form['unbeaten_streak']
        
        # Form trajectory comparison
        if 'is_improving' in home_form:
            features['home_improving'] = 1 if home_form['is_improving'] else 0
        if 'is_improving' in away_form:
            features['away_improving'] = 1 if away_form['is_improving'] else 0
        
        return features
    
    def get_form_summary(self, team: str) -> str:
        """Get human-readable form summary."""
        form = self.get_team_momentum(team)
        
        if not form:
            return "No form data available"
        
        parts = []
        
        if 'form_5' in form:
            parts.append(f"Last 5: {form['form_5']}")
        
        if 'streak_type' in form and 'current_streak' in form:
            streak_map = {'W': 'wins', 'D': 'draws', 'L': 'losses'}
            parts.append(f"Streak: {form['current_streak']} {streak_map.get(form['streak_type'], 'matches')}")
        
        if 'momentum_score' in form:
            parts.append(f"Momentum: {form['momentum_score']:.2f}")
        
        return " | ".join(parts)


# Global instance
_generator: Optional[MomentumFeatureGenerator] = None


def get_generator(matches_df: pd.DataFrame = None) -> MomentumFeatureGenerator:
    """Get or create momentum feature generator."""
    global _generator
    if _generator is None:
        _generator = MomentumFeatureGenerator()
    if matches_df is not None:
        _generator.set_matches(matches_df)
    return _generator


def get_team_form(team: str, matches_df: pd.DataFrame) -> Dict:
    """Quick function to get team form."""
    generator = get_generator(matches_df)
    return generator.get_team_momentum(team)
