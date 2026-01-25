"""
Advanced Feature Engineering

Adds critical features for improved predictions:
- Team form (last 5 home/away)
- Head-to-head history
- Goal-scoring/conceding trends
- Home advantage factors
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_CACHE = DATA_DIR / "features_cache.json"


class FormCalculator:
    """Calculate team form from recent matches"""
    
    def __init__(self, matches_data: List[Dict] = None):
        self.matches = matches_data or []
        self.team_matches: Dict[str, List[Dict]] = defaultdict(list)
        self.home_matches: Dict[str, List[Dict]] = defaultdict(list)
        self.away_matches: Dict[str, List[Dict]] = defaultdict(list)
        
        if matches_data:
            self._index_matches()
    
    def _index_matches(self):
        """Index matches by team"""
        for match in self.matches:
            home = match.get('home_team')
            away = match.get('away_team')
            
            if home:
                self.team_matches[home].append(match)
                self.home_matches[home].append(match)
            if away:
                self.team_matches[away].append(match)
                self.away_matches[away].append(match)
    
    def get_form(self, team: str, n: int = 5) -> Dict:
        """Get team's last N results as form string and points"""
        matches = sorted(self.team_matches.get(team, []), 
                        key=lambda x: x.get('date', ''), reverse=True)[:n]
        
        if not matches:
            return {'form': '', 'points': 0, 'avg_points': 0, 'games': 0}
        
        form = []
        points = 0
        goals_for = 0
        goals_against = 0
        
        for m in matches:
            home = m.get('home_team')
            h_score = m.get('home_score', 0)
            a_score = m.get('away_score', 0)
            
            if team == home:
                gf, ga = h_score, a_score
                if h_score > a_score:
                    form.append('W')
                    points += 3
                elif h_score < a_score:
                    form.append('L')
                else:
                    form.append('D')
                    points += 1
            else:
                gf, ga = a_score, h_score
                if a_score > h_score:
                    form.append('W')
                    points += 3
                elif a_score < h_score:
                    form.append('L')
                else:
                    form.append('D')
                    points += 1
            
            goals_for += gf
            goals_against += ga
        
        games = len(matches)
        return {
            'form': ''.join(form),
            'points': points,
            'avg_points': points / games if games else 0,
            'games': games,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_diff': goals_for - goals_against,
            'avg_goals_for': goals_for / games if games else 0,
            'avg_goals_against': goals_against / games if games else 0
        }
    
    def get_home_form(self, team: str, n: int = 5) -> Dict:
        """Get form from last N home matches"""
        matches = sorted(self.home_matches.get(team, []),
                        key=lambda x: x.get('date', ''), reverse=True)[:n]
        return self._calculate_form_from_matches(team, matches, is_home=True)
    
    def get_away_form(self, team: str, n: int = 5) -> Dict:
        """Get form from last N away matches"""
        matches = sorted(self.away_matches.get(team, []),
                        key=lambda x: x.get('date', ''), reverse=True)[:n]
        return self._calculate_form_from_matches(team, matches, is_home=False)
    
    def _calculate_form_from_matches(self, team: str, matches: List[Dict], is_home: bool) -> Dict:
        if not matches:
            return {'form': '', 'points': 0, 'avg_points': 0, 'games': 0}
        
        form = []
        points = 0
        goals_for = 0
        goals_against = 0
        
        for m in matches:
            h_score = m.get('home_score', 0)
            a_score = m.get('away_score', 0)
            
            if is_home:
                gf, ga = h_score, a_score
                if h_score > a_score:
                    form.append('W')
                    points += 3
                elif h_score < a_score:
                    form.append('L')
                else:
                    form.append('D')
                    points += 1
            else:
                gf, ga = a_score, h_score
                if a_score > h_score:
                    form.append('W')
                    points += 3
                elif a_score < h_score:
                    form.append('L')
                else:
                    form.append('D')
                    points += 1
            
            goals_for += gf
            goals_against += ga
        
        games = len(matches)
        return {
            'form': ''.join(form),
            'points': points,
            'avg_points': points / games if games else 0,
            'games': games,
            'avg_goals': goals_for / games if games else 0
        }


class HeadToHeadAnalyzer:
    """Analyze head-to-head history between teams"""
    
    def __init__(self, matches_data: List[Dict] = None):
        self.matches = matches_data or []
        self.h2h_index: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        
        if matches_data:
            self._build_index()
    
    def _build_index(self):
        """Build H2H lookup index"""
        for match in self.matches:
            home = match.get('home_team')
            away = match.get('away_team')
            if home and away:
                # Store both directions
                self.h2h_index[(home, away)].append(match)
                self.h2h_index[(away, home)].append(match)
    
    def get_h2h(self, team1: str, team2: str, n: int = 10) -> Dict:
        """Get head-to-head stats between two teams"""
        # Try both orderings
        matches = self.h2h_index.get((team1, team2), [])
        if not matches:
            matches = self.h2h_index.get((team2, team1), [])
        
        matches = sorted(matches, key=lambda x: x.get('date', ''), reverse=True)[:n]
        
        if not matches:
            return {
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'team1_win_pct': 0.5,
                'team2_win_pct': 0.5,
                'avg_goals': 0
            }
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        total_goals = 0
        
        for m in matches:
            home = m.get('home_team')
            h_score = m.get('home_score', 0)
            a_score = m.get('away_score', 0)
            total_goals += h_score + a_score
            
            if home == team1:
                if h_score > a_score:
                    team1_wins += 1
                elif h_score < a_score:
                    team2_wins += 1
                else:
                    draws += 1
            else:
                if a_score > h_score:
                    team1_wins += 1
                elif a_score < h_score:
                    team2_wins += 1
                else:
                    draws += 1
        
        total = len(matches)
        return {
            'total_matches': total,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'draws': draws,
            'team1_win_pct': team1_wins / total if total else 0.5,
            'team2_win_pct': team2_wins / total if total else 0.5,
            'draw_pct': draws / total if total else 0.25,
            'avg_goals': total_goals / total if total else 2.5,
            'last_result': self._get_last_result(matches[0], team1) if matches else None
        }
    
    def _get_last_result(self, match: Dict, team: str) -> str:
        """Get result of last match from team's perspective"""
        home = match.get('home_team')
        h_score = match.get('home_score', 0)
        a_score = match.get('away_score', 0)
        
        if team == home:
            if h_score > a_score: return 'W'
            elif h_score < a_score: return 'L'
            else: return 'D'
        else:
            if a_score > h_score: return 'W'
            elif a_score < h_score: return 'L'
            else: return 'D'


class AdvancedFeatureBuilder:
    """Build all advanced features for a match"""
    
    def __init__(self):
        self.form_calc: Optional[FormCalculator] = None
        self.h2h_analyzer: Optional[HeadToHeadAnalyzer] = None
        self._load_data()
    
    def _load_data(self):
        """Load historical data for feature calculation"""
        try:
            # Try to load cached data
            data_file = DATA_DIR / "training_data.csv"
            if data_file.exists():
                import pandas as pd
                df = pd.read_csv(data_file)
                matches = df.to_dict('records')
                self.form_calc = FormCalculator(matches)
                self.h2h_analyzer = HeadToHeadAnalyzer(matches)
                logger.info(f"Loaded {len(matches)} matches for features")
            else:
                self.form_calc = FormCalculator([])
                self.h2h_analyzer = HeadToHeadAnalyzer([])
        except Exception as e:
            logger.warning(f"Could not load data: {e}")
            self.form_calc = FormCalculator([])
            self.h2h_analyzer = HeadToHeadAnalyzer([])
    
    def build_features(self, home_team: str, away_team: str) -> Dict:
        """Build all advanced features for a match"""
        features = {
            'home_team': home_team,
            'away_team': away_team
        }
        
        # Overall form
        home_form = self.form_calc.get_form(home_team, 5)
        away_form = self.form_calc.get_form(away_team, 5)
        
        features['home_form'] = home_form
        features['away_form'] = away_form
        features['form_diff'] = home_form['avg_points'] - away_form['avg_points']
        
        # Home/away specific form
        features['home_home_form'] = self.form_calc.get_home_form(home_team, 5)
        features['away_away_form'] = self.form_calc.get_away_form(away_team, 5)
        
        # Head-to-head
        h2h = self.h2h_analyzer.get_h2h(home_team, away_team, 10)
        features['h2h'] = h2h
        features['h2h_home_advantage'] = h2h['team1_win_pct'] - h2h['team2_win_pct']
        
        # Goal trends
        features['home_scoring_rate'] = home_form.get('avg_goals_for', 1.2)
        features['home_conceding_rate'] = home_form.get('avg_goals_against', 1.0)
        features['away_scoring_rate'] = away_form.get('avg_goals_for', 1.0)
        features['away_conceding_rate'] = away_form.get('avg_goals_against', 1.2)
        
        # Momentum (positive = improving, negative = declining)
        features['home_momentum'] = self._calculate_momentum(home_form.get('form', ''))
        features['away_momentum'] = self._calculate_momentum(away_form.get('form', ''))
        
        return features
    
    def _calculate_momentum(self, form_str: str) -> float:
        """Calculate momentum from form string (recent results weighted more)"""
        if not form_str:
            return 0
        
        weights = [1.5, 1.3, 1.1, 0.9, 0.7]  # Most recent = highest weight
        points = {'W': 3, 'D': 1, 'L': 0}
        
        score = 0
        for i, result in enumerate(form_str):
            if i < len(weights):
                score += points.get(result, 0) * weights[i]
        
        # Normalize to -1 to 1 (3*sum(weights) = max possible)
        max_score = 3 * sum(weights[:len(form_str)])
        if max_score > 0:
            return (score / max_score) * 2 - 1
        return 0


# Global instance
_builder: Optional[AdvancedFeatureBuilder] = None

def get_feature_builder() -> AdvancedFeatureBuilder:
    global _builder
    if _builder is None:
        _builder = AdvancedFeatureBuilder()
    return _builder

def get_match_features(home: str, away: str) -> Dict:
    """Get all advanced features for a match"""
    return get_feature_builder().build_features(home, away)

def get_team_form(team: str) -> Dict:
    """Get team's current form"""
    return get_feature_builder().form_calc.get_form(team, 5)

def get_h2h_stats(team1: str, team2: str) -> Dict:
    """Get head-to-head stats"""
    return get_feature_builder().h2h_analyzer.get_h2h(team1, team2, 10)
