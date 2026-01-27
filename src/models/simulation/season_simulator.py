"""
Season Simulator
Simulates full season outcomes using Monte Carlo.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SeasonSimulator:
    """
    Simulates full season outcomes.
    
    Features:
    - Monte Carlo simulation
    - League table projections
    - Title/relegation probabilities
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        home_advantage: float = 0.3
    ):
        self.n_simulations = n_simulations
        self.home_advantage = home_advantage
        self.baseline_goals = 1.35
    
    def simulate_match(
        self,
        home_strength: float,
        away_strength: float,
        home_defense: float = 1.0,
        away_defense: float = 1.0
    ) -> Tuple[int, int]:
        """Simulate a single match."""
        home_lambda = (
            home_strength * away_defense * 
            self.baseline_goals * 
            np.exp(self.home_advantage)
        )
        away_lambda = away_strength * home_defense * self.baseline_goals
        
        home_goals = np.random.poisson(home_lambda)
        away_goals = np.random.poisson(away_lambda)
        
        return home_goals, away_goals
    
    def simulate_season(
        self,
        teams: List[str],
        team_strengths: Dict[str, float],
        team_defenses: Dict[str, float] = None,
        completed_matches: List[Dict] = None
    ) -> Dict:
        """
        Simulate a full season.
        
        Args:
            teams: List of team names
            team_strengths: Attack strength per team
            team_defenses: Defense ratings per team
            completed_matches: Already played matches
        """
        if team_defenses is None:
            team_defenses = {t: 1.0 for t in teams}
        
        # Initialize base points from completed matches
        base_points = defaultdict(int)
        base_gd = defaultdict(int)
        base_gf = defaultdict(int)
        completed_fixtures = set()
        
        if completed_matches:
            for match in completed_matches:
                home = match['home_team']
                away = match['away_team']
                hg = match['home_goals']
                ag = match['away_goals']
                
                if hg > ag:
                    base_points[home] += 3
                elif hg == ag:
                    base_points[home] += 1
                    base_points[away] += 1
                else:
                    base_points[away] += 3
                
                base_gf[home] += hg
                base_gf[away] += ag
                base_gd[home] += hg - ag
                base_gd[away] += ag - hg
                
                completed_fixtures.add((home, away))
        
        # Generate remaining fixtures
        remaining_fixtures = []
        for home in teams:
            for away in teams:
                if home != away and (home, away) not in completed_fixtures:
                    remaining_fixtures.append((home, away))
        
        # Simulation results
        final_positions = defaultdict(list)
        title_count = defaultdict(int)
        relegation_count = defaultdict(int)
        top_4_count = defaultdict(int)
        
        for _ in range(self.n_simulations):
            points = dict(base_points)
            gd = dict(base_gd)
            gf = dict(base_gf)
            
            for t in teams:
                if t not in points:
                    points[t] = 0
                    gd[t] = 0
                    gf[t] = 0
            
            # Simulate remaining matches
            for home, away in remaining_fixtures:
                hg, ag = self.simulate_match(
                    team_strengths.get(home, 1.0),
                    team_strengths.get(away, 1.0),
                    team_defenses.get(home, 1.0),
                    team_defenses.get(away, 1.0)
                )
                
                if hg > ag:
                    points[home] += 3
                elif hg == ag:
                    points[home] += 1
                    points[away] += 1
                else:
                    points[away] += 3
                
                gf[home] += hg
                gf[away] += ag
                gd[home] += hg - ag
                gd[away] += ag - hg
            
            # Sort by points, then GD, then GF
            standings = sorted(
                teams,
                key=lambda t: (points[t], gd[t], gf[t]),
                reverse=True
            )
            
            for pos, team in enumerate(standings, 1):
                final_positions[team].append(pos)
                
                if pos == 1:
                    title_count[team] += 1
                if pos <= 4:
                    top_4_count[team] += 1
                if pos >= len(teams) - 2:  # Bottom 3
                    relegation_count[team] += 1
        
        # Calculate probabilities
        results = {
            'teams': {},
            'simulation_count': self.n_simulations
        }
        
        for team in teams:
            positions = final_positions[team]
            results['teams'][team] = {
                'avg_position': round(np.mean(positions), 2),
                'title_probability': round(title_count[team] / self.n_simulations, 4),
                'top_4_probability': round(top_4_count[team] / self.n_simulations, 4),
                'relegation_probability': round(relegation_count[team] / self.n_simulations, 4),
                'best_position': min(positions),
                'worst_position': max(positions)
            }
        
        # Sort by title probability
        sorted_teams = sorted(
            results['teams'].items(),
            key=lambda x: x[1]['title_probability'],
            reverse=True
        )
        
        results['title_race'] = [
            {'team': t, 'probability': r['title_probability']}
            for t, r in sorted_teams[:5]
        ]
        
        return results
    
    def simulate_match_outcomes(
        self,
        home_team: str,
        away_team: str,
        home_strength: float,
        away_strength: float
    ) -> Dict:
        """Simulate various outcomes for a single match."""
        results = {
            'home_wins': 0,
            'draws': 0,
            'away_wins': 0,
            'score_counts': defaultdict(int),
            'total_goals': [],
            'btts': 0
        }
        
        for _ in range(self.n_simulations):
            hg, ag = self.simulate_match(home_strength, away_strength)
            
            if hg > ag:
                results['home_wins'] += 1
            elif hg == ag:
                results['draws'] += 1
            else:
                results['away_wins'] += 1
            
            results['score_counts'][(hg, ag)] += 1
            results['total_goals'].append(hg + ag)
            
            if hg > 0 and ag > 0:
                results['btts'] += 1
        
        n = self.n_simulations
        
        return {
            '1x2': {
                'home': round(results['home_wins'] / n, 4),
                'draw': round(results['draws'] / n, 4),
                'away': round(results['away_wins'] / n, 4)
            },
            'btts': round(results['btts'] / n, 4),
            'over_2.5': round(sum(1 for g in results['total_goals'] if g > 2.5) / n, 4),
            'avg_goals': round(np.mean(results['total_goals']), 2),
            'top_scores': sorted(
                [(f"{s[0]}-{s[1]}", c/n) for s, c in results['score_counts'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


_simulator: Optional[SeasonSimulator] = None

def get_simulator(n_sims: int = 10000) -> SeasonSimulator:
    global _simulator
    if _simulator is None:
        _simulator = SeasonSimulator(n_sims)
    return _simulator
