"""
Advanced Goal Prediction Module

Predicts:
- Total goals (Over/Under)
- Both teams to score (BTTS)
- Correct score probabilities
- First team to score
- Expected goals (xG) for each team
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GoalPrediction:
    """Complete goal prediction output"""
    # Expected goals
    home_xg: float
    away_xg: float
    total_xg: float
    
    # Over/Under probabilities
    over_0_5: float
    over_1_5: float
    over_2_5: float
    over_3_5: float
    over_4_5: float
    
    # Both teams to score
    btts_yes: float
    btts_no: float
    
    # First to score
    home_first: float
    away_first: float
    no_goals: float
    
    # Clean sheet
    home_clean_sheet: float
    away_clean_sheet: float
    
    # Most likely scores
    likely_scores: List[Tuple[str, float]]
    
    def to_dict(self) -> Dict:
        return {
            'expected_goals': {
                'home': round(self.home_xg, 2),
                'away': round(self.away_xg, 2),
                'total': round(self.total_xg, 2)
            },
            'over_under': {
                'over_0.5': round(self.over_0_5, 3),
                'over_1.5': round(self.over_1_5, 3),
                'over_2.5': round(self.over_2_5, 3),
                'over_3.5': round(self.over_3_5, 3),
                'over_4.5': round(self.over_4_5, 3)
            },
            'btts': {
                'yes': round(self.btts_yes, 3),
                'no': round(self.btts_no, 3)
            },
            'first_to_score': {
                'home': round(self.home_first, 3),
                'away': round(self.away_first, 3),
                'no_goals': round(self.no_goals, 3)
            },
            'clean_sheet': {
                'home': round(self.home_clean_sheet, 3),
                'away': round(self.away_clean_sheet, 3)
            },
            'likely_scores': [
                {'score': score, 'probability': round(prob, 3)} 
                for score, prob in self.likely_scores[:10]
            ]
        }


class PoissonGoalPredictor:
    """
    Goal prediction using Poisson distribution model
    
    Based on expected goals (xG) for each team, calculates
    probabilities for various goal-related outcomes.
    """
    
    # League average goals per game
    LEAGUE_AVERAGES = {
        'bundesliga': 3.18,      # Highest scoring
        'premier_league': 2.85,
        'la_liga': 2.55,
        'serie_a': 2.65,
        'ligue_1': 2.75,
        'champions_league': 2.90,
        'eredivisie': 3.10,
        'default': 2.75
    }
    
    # Home advantage factor for goals
    HOME_GOAL_FACTOR = 1.35
    AWAY_GOAL_FACTOR = 0.85
    
    def __init__(self):
        # Team attacking/defending strengths (would be loaded from historical data)
        self.attack_strength = {}
        self.defense_strength = {}
        self._init_team_strengths()
    
    def _init_team_strengths(self):
        """Initialize team attack/defense ratings"""
        # Attack strength: >1 = scores more than average, <1 = less
        # Defense strength: <1 = concedes less than average, >1 = more
        
        team_data = {
            # Premier League
            'Manchester City': (1.45, 0.65),
            'Liverpool': (1.40, 0.70),
            'Arsenal': (1.30, 0.75),
            'Chelsea': (1.15, 0.85),
            'Manchester United': (1.10, 0.95),
            'Tottenham Hotspur': (1.20, 0.90),
            'Newcastle United': (1.15, 0.80),
            'Aston Villa': (1.10, 0.85),
            'Brighton': (1.05, 0.90),
            'West Ham': (0.95, 1.00),
            
            # Bundesliga
            'FC Bayern München': (1.55, 0.60),
            'Bayern': (1.55, 0.60),
            'Bayer 04 Leverkusen': (1.40, 0.70),
            'Leverkusen': (1.40, 0.70),
            'Borussia Dortmund': (1.35, 0.80),
            'Dortmund': (1.35, 0.80),
            'RB Leipzig': (1.25, 0.75),
            'Leipzig': (1.25, 0.75),
            'VfB Stuttgart': (1.20, 0.85),
            'Stuttgart': (1.20, 0.85),
            'Eintracht Frankfurt': (1.15, 0.90),
            'Frankfurt': (1.15, 0.90),
            'SC Freiburg': (1.05, 0.85),
            'Freiburg': (1.05, 0.85),
            'FC Augsburg': (0.85, 1.10),
            'Augsburg': (0.85, 1.10),
            'SV Werder Bremen': (1.00, 1.00),
            'Bremen': (1.00, 1.00),
            'VfL Wolfsburg': (0.95, 0.95),
            'Wolfsburg': (0.95, 0.95),
            '1. FSV Mainz 05': (0.90, 1.05),
            'Mainz': (0.90, 1.05),
            'FC St. Pauli': (0.85, 1.00),
            'St. Pauli': (0.85, 1.00),
            'Hamburger SV': (1.00, 0.95),
            'HSV': (1.00, 0.95),
            'Borussia Mönchengladbach': (1.00, 1.00),
            'Gladbach': (1.00, 1.00),
            'TSG Hoffenheim': (1.05, 1.00),
            'Hoffenheim': (1.05, 1.00),
            '1. FC Union Berlin': (0.85, 0.90),
            'Union Berlin': (0.85, 0.90),
            '1. FC Heidenheim 1846': (0.80, 1.05),
            'Heidenheim': (0.80, 1.05),
            '1. FC Köln': (0.90, 1.10),
            'Köln': (0.90, 1.10),
            
            # La Liga
            'Real Madrid': (1.50, 0.65),
            'Barcelona': (1.45, 0.70),
            'Atlético Madrid': (1.10, 0.70),
            
            # Serie A
            'Inter Milan': (1.35, 0.65),
            'AC Milan': (1.20, 0.80),
            'Juventus': (1.15, 0.75),
            'Napoli': (1.25, 0.75),
        }
        
        for team, (attack, defense) in team_data.items():
            self.attack_strength[team] = attack
            self.defense_strength[team] = defense
    
    def get_attack_strength(self, team: str) -> float:
        """Get attack strength for a team"""
        # Try direct match
        if team in self.attack_strength:
            return self.attack_strength[team]
        
        # Try partial match
        team_lower = team.lower()
        for name, strength in self.attack_strength.items():
            if team_lower in name.lower() or name.lower() in team_lower:
                return strength
        
        return 1.0  # Average
    
    def get_defense_strength(self, team: str) -> float:
        """Get defense strength for a team"""
        if team in self.defense_strength:
            return self.defense_strength[team]
        
        team_lower = team.lower()
        for name, strength in self.defense_strength.items():
            if team_lower in name.lower() or name.lower() in team_lower:
                return strength
        
        return 1.0  # Average
    
    def poisson_prob(self, lam: float, k: int) -> float:
        """Calculate Poisson probability P(X=k) given lambda"""
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    def calculate_xg(
        self,
        home_team: str,
        away_team: str,
        league: str = 'default'
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for each team
        
        xG_home = league_avg * home_attack * away_defense * home_factor
        xG_away = league_avg * away_attack * home_defense * away_factor
        """
        league_avg = self.LEAGUE_AVERAGES.get(league.lower(), 2.75) / 2
        
        home_attack = self.get_attack_strength(home_team)
        home_defense = self.get_defense_strength(home_team)
        away_attack = self.get_attack_strength(away_team)
        away_defense = self.get_defense_strength(away_team)
        
        # Home team xG
        home_xg = league_avg * home_attack * away_defense * self.HOME_GOAL_FACTOR
        
        # Away team xG
        away_xg = league_avg * away_attack * home_defense * self.AWAY_GOAL_FACTOR
        
        return home_xg, away_xg
    
    def predict_goals(
        self,
        home_team: str,
        away_team: str,
        league: str = 'default'
    ) -> GoalPrediction:
        """Generate complete goal prediction"""
        
        home_xg, away_xg = self.calculate_xg(home_team, away_team, league)
        total_xg = home_xg + away_xg
        
        # Calculate score probabilities using Poisson
        max_goals = 8
        score_probs = {}
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                prob = self.poisson_prob(home_xg, home_goals) * \
                       self.poisson_prob(away_xg, away_goals)
                score_probs[f"{home_goals}-{away_goals}"] = prob
        
        # Over/Under calculations
        over_0_5 = 1 - score_probs["0-0"]
        over_1_5 = 1 - score_probs["0-0"] - score_probs["1-0"] - score_probs["0-1"]
        over_2_5 = sum(p for s, p in score_probs.items() 
                      if int(s.split("-")[0]) + int(s.split("-")[1]) > 2)
        over_3_5 = sum(p for s, p in score_probs.items() 
                      if int(s.split("-")[0]) + int(s.split("-")[1]) > 3)
        over_4_5 = sum(p for s, p in score_probs.items() 
                      if int(s.split("-")[0]) + int(s.split("-")[1]) > 4)
        
        # BTTS (Both Teams To Score)
        btts_yes = sum(p for s, p in score_probs.items() 
                       if int(s.split("-")[0]) > 0 and int(s.split("-")[1]) > 0)
        btts_no = 1 - btts_yes
        
        # Clean sheets
        home_clean = sum(p for s, p in score_probs.items() 
                        if int(s.split("-")[1]) == 0)
        away_clean = sum(p for s, p in score_probs.items() 
                        if int(s.split("-")[0]) == 0)
        
        # First to score (approximation based on xG)
        total_intensity = home_xg + away_xg
        if total_intensity > 0:
            home_first = (home_xg / total_intensity) * over_0_5
            away_first = (away_xg / total_intensity) * over_0_5
        else:
            home_first = 0.4
            away_first = 0.4
        no_goals = score_probs["0-0"]
        
        # Sort likely scores
        sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
        
        return GoalPrediction(
            home_xg=home_xg,
            away_xg=away_xg,
            total_xg=total_xg,
            over_0_5=over_0_5,
            over_1_5=over_1_5,
            over_2_5=over_2_5,
            over_3_5=over_3_5,
            over_4_5=over_4_5,
            btts_yes=btts_yes,
            btts_no=btts_no,
            home_first=home_first,
            away_first=away_first,
            no_goals=no_goals,
            home_clean_sheet=home_clean,
            away_clean_sheet=away_clean,
            likely_scores=sorted_scores
        )


# Global instance
goal_predictor = PoissonGoalPredictor()


def predict_goals(home_team: str, away_team: str, league: str = 'default') -> GoalPrediction:
    """Convenience function for goal predictions"""
    return goal_predictor.predict_goals(home_team, away_team, league)
