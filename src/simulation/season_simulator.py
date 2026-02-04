"""
Season Simulator
================
Monte Carlo simulation for full season outcomes, league standings, and title races.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from scipy.stats import poisson

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Team statistics for simulation."""
    name: str
    attack_strength: float = 1.0
    defense_strength: float = 1.0
    home_advantage: float = 0.25
    current_points: int = 0
    current_gd: int = 0
    current_position: int = 0


@dataclass
class SimulationConfig:
    """Configuration for season simulation."""
    n_simulations: int = 10000
    home_advantage: float = 0.25
    avg_goals: float = 2.75
    random_seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Result of season simulation."""
    team: str
    avg_points: float
    avg_position: float
    title_prob: float
    top4_prob: float
    top6_prob: float
    relegation_prob: float
    points_range: Tuple[int, int]
    position_distribution: Dict[int, float]


class SeasonSimulator:
    """
    Monte Carlo season simulator for predicting league outcomes.
    
    Simulates remaining fixtures multiple times to estimate:
    - Title win probability
    - Champions League qualification odds
    - Relegation probability
    - Expected final points and position
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.teams: Dict[str, TeamStats] = {}
        self.remaining_fixtures: List[Tuple[str, str]] = []
        self._rng = np.random.RandomState(self.config.random_seed)
        
    def add_team(self, team: TeamStats) -> None:
        """Add a team to the simulation."""
        self.teams[team.name] = team
        
    def set_remaining_fixtures(self, fixtures: List[Tuple[str, str]]) -> None:
        """Set remaining fixtures (home, away) tuples."""
        self.remaining_fixtures = fixtures
        
    def simulate_match(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[int, int]:
        """
        Simulate a single match using Poisson distribution.
        
        Returns:
            Tuple of (home_goals, away_goals)
        """
        home_stats = self.teams.get(home_team)
        away_stats = self.teams.get(away_team)
        
        if not home_stats or not away_stats:
            # Default simulation if teams not found
            home_lambda = 1.5
            away_lambda = 1.2
        else:
            # Calculate expected goals
            avg_goals = self.config.avg_goals / 2
            
            home_lambda = (
                avg_goals * 
                home_stats.attack_strength * 
                away_stats.defense_strength *
                (1 + self.config.home_advantage)
            )
            
            away_lambda = (
                avg_goals * 
                away_stats.attack_strength * 
                home_stats.defense_strength
            )
        
        # Simulate goals
        home_goals = self._rng.poisson(home_lambda)
        away_goals = self._rng.poisson(away_lambda)
        
        return int(home_goals), int(away_goals)
    
    def simulate_season(self) -> Dict[str, Dict]:
        """
        Simulate one full remaining season.
        
        Returns:
            Final standings with points and goal difference
        """
        # Initialize standings from current positions
        standings = {}
        for name, team in self.teams.items():
            standings[name] = {
                'points': team.current_points,
                'gd': team.current_gd,
                'wins': 0,
                'draws': 0,
                'losses': 0
            }
        
        # Simulate each remaining fixture
        for home_team, away_team in self.remaining_fixtures:
            if home_team not in standings:
                standings[home_team] = {'points': 0, 'gd': 0, 'wins': 0, 'draws': 0, 'losses': 0}
            if away_team not in standings:
                standings[away_team] = {'points': 0, 'gd': 0, 'wins': 0, 'draws': 0, 'losses': 0}
            
            home_goals, away_goals = self.simulate_match(home_team, away_team)
            
            # Update goal difference
            standings[home_team]['gd'] += home_goals - away_goals
            standings[away_team]['gd'] += away_goals - home_goals
            
            # Update points
            if home_goals > away_goals:
                standings[home_team]['points'] += 3
                standings[home_team]['wins'] += 1
                standings[away_team]['losses'] += 1
            elif away_goals > home_goals:
                standings[away_team]['points'] += 3
                standings[away_team]['wins'] += 1
                standings[home_team]['losses'] += 1
            else:
                standings[home_team]['points'] += 1
                standings[away_team]['points'] += 1
                standings[home_team]['draws'] += 1
                standings[away_team]['draws'] += 1
        
        return standings
    
    def get_final_positions(self, standings: Dict) -> Dict[str, int]:
        """Get final league positions from standings."""
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (x[1]['points'], x[1]['gd']),
            reverse=True
        )
        
        return {team: pos + 1 for pos, (team, _) in enumerate(sorted_teams)}
    
    def run_simulations(self) -> Dict[str, SimulationResult]:
        """
        Run full Monte Carlo simulation.
        
        Returns:
            Results for each team
        """
        if not self.teams:
            logger.warning("No teams added to simulation")
            return {}
        
        logger.info(f"Running {self.config.n_simulations} season simulations")
        
        # Track results for each team
        team_results = defaultdict(lambda: {
            'points': [],
            'positions': [],
            'title_wins': 0,
            'top4': 0,
            'top6': 0,
            'relegated': 0
        })
        
        n_teams = len(self.teams)
        relegation_zone = max(1, n_teams - 2)  # Bottom 3 for most leagues
        
        # Run simulations
        for sim in range(self.config.n_simulations):
            standings = self.simulate_season()
            positions = self.get_final_positions(standings)
            
            for team, pos in positions.items():
                team_results[team]['points'].append(standings[team]['points'])
                team_results[team]['positions'].append(pos)
                
                if pos == 1:
                    team_results[team]['title_wins'] += 1
                if pos <= 4:
                    team_results[team]['top4'] += 1
                if pos <= 6:
                    team_results[team]['top6'] += 1
                if pos >= relegation_zone:
                    team_results[team]['relegated'] += 1
        
        # Compile results
        results = {}
        n_sims = self.config.n_simulations
        
        for team, data in team_results.items():
            points_arr = np.array(data['points'])
            pos_arr = np.array(data['positions'])
            
            # Position distribution
            pos_counts = defaultdict(int)
            for p in pos_arr:
                pos_counts[p] += 1
            pos_dist = {k: v / n_sims for k, v in pos_counts.items()}
            
            results[team] = SimulationResult(
                team=team,
                avg_points=float(points_arr.mean()),
                avg_position=float(pos_arr.mean()),
                title_prob=data['title_wins'] / n_sims,
                top4_prob=data['top4'] / n_sims,
                top6_prob=data['top6'] / n_sims,
                relegation_prob=data['relegated'] / n_sims,
                points_range=(int(points_arr.min()), int(points_arr.max())),
                position_distribution=dict(pos_dist)
            )
        
        logger.info(f"Simulation complete for {len(results)} teams")
        return results
    
    def get_title_race(self) -> List[Dict]:
        """Get title race probabilities."""
        results = self.run_simulations()
        
        title_race = []
        for team, result in sorted(results.items(), key=lambda x: x[1].title_prob, reverse=True):
            title_race.append({
                'team': team,
                'title_prob': round(result.title_prob * 100, 1),
                'avg_points': round(result.avg_points, 1),
                'current_points': self.teams.get(team, TeamStats(team)).current_points
            })
        
        return title_race[:10]  # Top 10 contenders
    
    def predict_final_standings(self) -> pd.DataFrame:
        """Predict final league standings."""
        results = self.run_simulations()
        
        standings_data = []
        for team, result in results.items():
            current_stats = self.teams.get(team, TeamStats(team))
            standings_data.append({
                'team': team,
                'current_points': current_stats.current_points,
                'predicted_points': round(result.avg_points, 1),
                'predicted_position': round(result.avg_position, 1),
                'title_prob': round(result.title_prob * 100, 1),
                'top4_prob': round(result.top4_prob * 100, 1),
                'relegation_prob': round(result.relegation_prob * 100, 1)
            })
        
        df = pd.DataFrame(standings_data)
        df = df.sort_values('predicted_position')
        
        return df


# Global instance
_simulator: Optional[SeasonSimulator] = None


def get_simulator() -> SeasonSimulator:
    """Get or create season simulator."""
    global _simulator
    if _simulator is None:
        _simulator = SeasonSimulator()
    return _simulator


def simulate_premier_league(
    current_standings: Dict[str, Dict],
    remaining_fixtures: List[Tuple[str, str]],
    n_simulations: int = 10000
) -> Dict[str, SimulationResult]:
    """
    Convenience function to simulate Premier League season.
    
    Args:
        current_standings: Dict mapping team name to {'points': int, 'gd': int}
        remaining_fixtures: List of (home, away) fixture tuples
        n_simulations: Number of simulations to run
        
    Returns:
        Simulation results for each team
    """
    simulator = SeasonSimulator(SimulationConfig(n_simulations=n_simulations))
    
    # Add teams with current standings
    for i, (team, stats) in enumerate(current_standings.items()):
        simulator.add_team(TeamStats(
            name=team,
            current_points=stats.get('points', 0),
            current_gd=stats.get('gd', 0),
            current_position=i + 1,
            attack_strength=stats.get('attack', 1.0),
            defense_strength=stats.get('defense', 1.0)
        ))
    
    simulator.set_remaining_fixtures(remaining_fixtures)
    
    return simulator.run_simulations()
