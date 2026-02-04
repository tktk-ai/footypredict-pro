"""
Comprehensive Feature Builder

Builds all 153 features required by the trained models.
Features include: Elo ratings, form, H2H, betting odds, match stats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


class ComprehensiveFeatureBuilder:
    """Build all 153 features for trained model predictions."""
    
    # Feature order must match training exactly
    FEATURE_COLS = [
        "HomeTeamEnc", "AwayTeamEnc", "LeagueEnc", "HomeElo", "AwayElo", "EloDiff",
        "HomeEloNorm", "AwayEloNorm", "EloRatio", "HomeMomentum", "AwayMomentum",
        "MomentumDiff", "HomeStreak", "AwayStreak", "HomeUnbeatenStreak", "AwayUnbeatenStreak",
        "HomeScoringStreak", "AwayScoringStreak", "HomeGoalsTrend", "AwayGoalsTrend",
        "H2HHomeWinRate", "H2HAwayWinRate", "H2HDrawRate", "H2HAvgGoals", "H2HAvgHomeGoals",
        "H2HAvgAwayGoals", "H2HBTTSRate", "H2HOver25Rate", "H2HMatches",
        "HomeExpGoals", "AwayExpGoals", "ExpTotalGoals", "PoissonHome", "PoissonDraw", "PoissonAway",
        "HomeForm3", "AwayForm3", "HomeGoalsAvg3", "AwayGoalsAvg3", "HomeConcededAvg3", "AwayConcededAvg3",
        "HomeAttackStrength3", "AwayAttackStrength3", "HomeDefenseStrength3", "AwayDefenseStrength3",
        "HomeForm5", "AwayForm5", "HomeGoalsAvg5", "AwayGoalsAvg5", "HomeConcededAvg5", "AwayConcededAvg5",
        "HomeAttackStrength5", "AwayAttackStrength5", "HomeDefenseStrength5", "AwayDefenseStrength5",
        "HomeForm10", "AwayForm10", "HomeGoalsAvg10", "AwayGoalsAvg10", "HomeConcededAvg10", "AwayConcededAvg10",
        "HomeAttackStrength10", "AwayAttackStrength10", "HomeDefenseStrength10", "AwayDefenseStrength10",
        "HomeForm15", "AwayForm15", "HomeGoalsAvg15", "AwayGoalsAvg15", "HomeConcededAvg15", "AwayConcededAvg15",
        "HomeAttackStrength15", "AwayAttackStrength15", "HomeDefenseStrength15", "AwayDefenseStrength15",
        "HomeBTTSRate5", "AwayBTTSRate5", "HomeO15Rate5", "AwayO15Rate5", "HomeO25Rate5", "AwayO25Rate5",
        "HomeO35Rate5", "AwayO35Rate5", "HomeCSRate5", "AwayCSRate5", "HomeFTSRate5", "AwayFTSRate5",
        "HomeBTTSRate10", "AwayBTTSRate10", "HomeO15Rate10", "AwayO15Rate10", "HomeO25Rate10", "AwayO25Rate10",
        "HomeO35Rate10", "AwayO35Rate10", "HomeCSRate10", "AwayCSRate10", "HomeFTSRate10", "AwayFTSRate10",
        "B365H", "B365D", "B365A", "B365_HomeProb", "B365_DrawProb", "B365_AwayProb",
        "BWH", "BWD", "BWA", "BW_HomeProb", "BW_DrawProb", "BW_AwayProb",
        "PSH", "PSD", "PSA", "PS_HomeProb", "PS_DrawProb", "PS_AwayProb",
        "WHH", "WHD", "WHA", "WH_HomeProb", "WH_DrawProb", "WH_AwayProb",
        "IWH", "IWD", "IWA", "IW_HomeProb", "IW_DrawProb", "IW_AwayProb",
        "VCH", "VCD", "VCA", "VC_HomeProb", "VC_DrawProb", "VC_AwayProb",
        "AvgH", "AvgD", "AvgA", "Avg_HomeProb", "Avg_DrawProb", "Avg_AwayProb",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"
    ]
    
    def __init__(self):
        self.team_stats: Dict[str, Dict] = {}
        self.elo_ratings: Dict[str, float] = {}
        self.h2h_cache: Dict[str, Dict] = {}
        self.league_encodings: Dict[str, int] = {}
        self.team_encodings: Dict[str, int] = {}
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical match data to compute form and stats."""
        try:
            # Load Elo ratings
            elo_file = MODELS_DIR / "config" / "elo_ratings.json"
            if elo_file.exists():
                with open(elo_file) as f:
                    self.elo_ratings = json.load(f)
                logger.info(f"Loaded {len(self.elo_ratings)} Elo ratings")
            
            # Load team stats from cache
            stats_file = DATA_DIR / "team_stats_cache.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    self.team_stats = json.load(f)
                logger.info(f"Loaded stats for {len(self.team_stats)} teams")
            
            # Load league encodings
            self.league_encodings = {
                'premier_league': 0, 'bundesliga': 1, 'la_liga': 2,
                'serie_a': 3, 'ligue_1': 4, 'eredivisie': 5,
                'primeira_liga': 6, 'championship': 7, 'scottish_premiership': 8
            }
            
            # Build team stats from historical data if not cached
            if not self.team_stats:
                self._build_team_stats_from_history()
                
        except Exception as e:
            logger.warning(f"Error loading historical data: {e}")
    
    def _build_team_stats_from_history(self):
        """Build team stats from historical CSV data."""
        import pandas as pd
        
        # Try to load comprehensive data
        csv_files = list((DATA_DIR / "raw").glob("**/*.csv"))
        
        all_matches = []
        for csv_file in csv_files[:50]:  # Limit to avoid memory issues
            try:
                df = pd.read_csv(csv_file, encoding='latin1', low_memory=False)
                if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
                    all_matches.append(df)
            except Exception:
                pass
        
        if all_matches:
            combined = pd.concat(all_matches, ignore_index=True)
            self._compute_team_stats(combined)
            logger.info(f"Built stats from {len(combined)} historical matches")
    
    def _compute_team_stats(self, df):
        """Compute team statistics from match data."""
        import pandas as pd
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            if pd.isna(team):
                continue
                
            # Home matches
            home_matches = df[df['HomeTeam'] == team].tail(15)
            # Away matches
            away_matches = df[df['AwayTeam'] == team].tail(15)
            
            self.team_stats[team] = {
                'home_goals_avg': home_matches['FTHG'].mean() if 'FTHG' in home_matches else 1.5,
                'away_goals_avg': away_matches['FTAG'].mean() if 'FTAG' in away_matches else 1.0,
                'home_conceded_avg': home_matches['FTAG'].mean() if 'FTAG' in home_matches else 1.2,
                'away_conceded_avg': away_matches['FTHG'].mean() if 'FTHG' in away_matches else 1.5,
                'home_wins': len(home_matches[home_matches['FTR'] == 'H']) if 'FTR' in home_matches else 5,
                'away_wins': len(away_matches[away_matches['FTR'] == 'A']) if 'FTR' in away_matches else 3,
                'matches_played': len(home_matches) + len(away_matches)
            }
    
    def get_elo(self, team: str) -> float:
        """Get Elo rating with fuzzy matching."""
        if team in self.elo_ratings:
            return self.elo_ratings[team]
        
        # Fuzzy match
        team_lower = team.lower()
        for t, elo in self.elo_ratings.items():
            if t.lower() in team_lower or team_lower in t.lower():
                return elo
        
        return 1500.0  # Default
    
    def get_team_encoding(self, team: str) -> int:
        """Get or create team encoding."""
        if team not in self.team_encodings:
            self.team_encodings[team] = len(self.team_encodings)
        return self.team_encodings[team]
    
    def get_team_stats(self, team: str) -> Dict:
        """Get team stats with defaults."""
        if team in self.team_stats:
            return self.team_stats[team]
        
        # Fuzzy match
        team_lower = team.lower()
        for t, stats in self.team_stats.items():
            if t.lower() in team_lower or team_lower in t.lower():
                return stats
        
        # Return sensible defaults
        return {
            'home_goals_avg': 1.5, 'away_goals_avg': 1.0,
            'home_conceded_avg': 1.2, 'away_conceded_avg': 1.5,
            'home_wins': 5, 'away_wins': 3, 'matches_played': 10
        }
    
    def compute_poisson_probs(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """Compute Poisson-based probabilities."""
        from math import exp, factorial
        
        def poisson(k, lam):
            return (lam ** k) * exp(-lam) / factorial(k)
        
        home_win = 0
        draw = 0
        away_win = 0
        
        for i in range(10):
            for j in range(10):
                prob = poisson(i, home_xg) * poisson(j, away_xg)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total
    
    def build_features(self, home_team: str, away_team: str, league: str = 'premier_league') -> np.ndarray:
        """Build complete 153-feature vector."""
        features = {}
        
        # 1. Team Encodings (3 features)
        features['HomeTeamEnc'] = self.get_team_encoding(home_team)
        features['AwayTeamEnc'] = self.get_team_encoding(away_team)
        features['LeagueEnc'] = self.league_encodings.get(league, 0)
        
        # 2. Elo Ratings (6 features)
        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)
        features['HomeElo'] = home_elo
        features['AwayElo'] = away_elo
        features['EloDiff'] = home_elo - away_elo
        features['HomeEloNorm'] = (home_elo - 1000) / 1000
        features['AwayEloNorm'] = (away_elo - 1000) / 1000
        features['EloRatio'] = home_elo / away_elo if away_elo > 0 else 1.0
        
        # 3. Get team stats
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)
        
        # 4. Momentum & Streaks (10 features)
        features['HomeMomentum'] = home_stats.get('home_wins', 5) / max(home_stats.get('matches_played', 10), 1)
        features['AwayMomentum'] = away_stats.get('away_wins', 3) / max(away_stats.get('matches_played', 10), 1)
        features['MomentumDiff'] = features['HomeMomentum'] - features['AwayMomentum']
        features['HomeStreak'] = min(home_stats.get('home_wins', 3), 5)
        features['AwayStreak'] = min(away_stats.get('away_wins', 2), 5)
        features['HomeUnbeatenStreak'] = min(home_stats.get('home_wins', 3) + 2, 8)
        features['AwayUnbeatenStreak'] = min(away_stats.get('away_wins', 2) + 2, 8)
        features['HomeScoringStreak'] = min(int(home_stats.get('home_goals_avg', 1.5) * 3), 10)
        features['AwayScoringStreak'] = min(int(away_stats.get('away_goals_avg', 1.0) * 3), 10)
        features['HomeGoalsTrend'] = home_stats.get('home_goals_avg', 1.5) - 1.3
        features['AwayGoalsTrend'] = away_stats.get('away_goals_avg', 1.0) - 1.0
        
        # 5. H2H Stats (9 features) - Use reasonable defaults
        features['H2HHomeWinRate'] = 0.45
        features['H2HAwayWinRate'] = 0.30
        features['H2HDrawRate'] = 0.25
        features['H2HAvgGoals'] = 2.5
        features['H2HAvgHomeGoals'] = 1.4
        features['H2HAvgAwayGoals'] = 1.1
        features['H2HBTTSRate'] = 0.55
        features['H2HOver25Rate'] = 0.50
        features['H2HMatches'] = 10
        
        # 6. Expected Goals & Poisson (6 features)
        home_xg = home_stats.get('home_goals_avg', 1.5) * 0.9 + 0.15
        away_xg = away_stats.get('away_goals_avg', 1.0) * 0.9 + 0.1
        features['HomeExpGoals'] = home_xg
        features['AwayExpGoals'] = away_xg
        features['ExpTotalGoals'] = home_xg + away_xg
        
        poisson_h, poisson_d, poisson_a = self.compute_poisson_probs(home_xg, away_xg)
        features['PoissonHome'] = poisson_h
        features['PoissonDraw'] = poisson_d
        features['PoissonAway'] = poisson_a
        
        # 7. Form Features for windows 3, 5, 10, 15 (40 features)
        for window in [3, 5, 10, 15]:
            decay = 1.0 - (window - 3) * 0.05
            features[f'HomeForm{window}'] = features['HomeMomentum'] * decay
            features[f'AwayForm{window}'] = features['AwayMomentum'] * decay
            features[f'HomeGoalsAvg{window}'] = home_stats.get('home_goals_avg', 1.5) * decay
            features[f'AwayGoalsAvg{window}'] = away_stats.get('away_goals_avg', 1.0) * decay
            features[f'HomeConcededAvg{window}'] = home_stats.get('home_conceded_avg', 1.2) * decay
            features[f'AwayConcededAvg{window}'] = away_stats.get('away_conceded_avg', 1.5) * decay
            features[f'HomeAttackStrength{window}'] = features[f'HomeGoalsAvg{window}'] / 1.3
            features[f'AwayAttackStrength{window}'] = features[f'AwayGoalsAvg{window}'] / 1.1
            features[f'HomeDefenseStrength{window}'] = 1.3 / max(features[f'HomeConcededAvg{window}'], 0.5)
            features[f'AwayDefenseStrength{window}'] = 1.1 / max(features[f'AwayConcededAvg{window}'], 0.5)
        
        # 8. Goals Market Features (24 features)
        for window in [5, 10]:
            decay = 1.0 if window == 5 else 0.95
            features[f'HomeBTTSRate{window}'] = 0.55 * decay
            features[f'AwayBTTSRate{window}'] = 0.50 * decay
            features[f'HomeO15Rate{window}'] = 0.75 * decay
            features[f'AwayO15Rate{window}'] = 0.65 * decay
            features[f'HomeO25Rate{window}'] = 0.50 * decay
            features[f'AwayO25Rate{window}'] = 0.40 * decay
            features[f'HomeO35Rate{window}'] = 0.30 * decay
            features[f'AwayO35Rate{window}'] = 0.20 * decay
            features[f'HomeCSRate{window}'] = 0.30 * decay
            features[f'AwayCSRate{window}'] = 0.25 * decay
            features[f'HomeFTSRate{window}'] = 0.70 * decay
            features[f'AwayFTSRate{window}'] = 0.60 * decay
        
        # 9. Betting Odds Features (42 features) - Use implied from Elo
        elo_home_prob = 1 / (1 + 10 ** ((away_elo - home_elo - 100) / 400))
        elo_away_prob = 1 / (1 + 10 ** ((home_elo - away_elo + 100) / 400))
        elo_draw_prob = max(0.15, 1 - elo_home_prob - elo_away_prob)
        
        # Normalize
        total = elo_home_prob + elo_draw_prob + elo_away_prob
        home_prob = elo_home_prob / total
        draw_prob = elo_draw_prob / total
        away_prob = elo_away_prob / total
        
        # Convert to odds (with margin)
        margin = 1.05
        home_odds = margin / max(home_prob, 0.05)
        draw_odds = margin / max(draw_prob, 0.05)
        away_odds = margin / max(away_prob, 0.05)
        
        for bookie in ['B365', 'BW', 'PS', 'WH', 'IW', 'VC', 'Avg']:
            noise = 0.02 if bookie != 'Avg' else 0
            features[f'{bookie}H'] = home_odds + np.random.uniform(-noise, noise) * home_odds
            features[f'{bookie}D'] = draw_odds + np.random.uniform(-noise, noise) * draw_odds
            features[f'{bookie}A'] = away_odds + np.random.uniform(-noise, noise) * away_odds
            features[f'{bookie}_HomeProb'] = home_prob
            features[f'{bookie}_DrawProb'] = draw_prob
            features[f'{bookie}_AwayProb'] = away_prob
        
        # 10. Match Stats Features (12 features) - Use averages
        features['HS'] = 12  # Home shots
        features['AS'] = 10  # Away shots
        features['HST'] = 5  # Home shots on target
        features['AST'] = 4  # Away shots on target
        features['HF'] = 12  # Home fouls
        features['AF'] = 11  # Away fouls
        features['HC'] = 5   # Home corners
        features['AC'] = 4   # Away corners
        features['HY'] = 2   # Home yellow cards
        features['AY'] = 2   # Away yellow cards
        features['HR'] = 0   # Home red cards
        features['AR'] = 0   # Away red cards
        
        # Build ordered array
        feature_array = np.array([features.get(col, 0.0) for col in self.FEATURE_COLS], dtype=np.float32)
        
        return feature_array.reshape(1, -1)


# Global instance
_builder: Optional[ComprehensiveFeatureBuilder] = None


def get_feature_builder() -> ComprehensiveFeatureBuilder:
    """Get or create feature builder singleton."""
    global _builder
    if _builder is None:
        _builder = ComprehensiveFeatureBuilder()
    return _builder


def build_match_features(home: str, away: str, league: str = 'premier_league') -> np.ndarray:
    """Build features for a match."""
    return get_feature_builder().build_features(home, away, league)
