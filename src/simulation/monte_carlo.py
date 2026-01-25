"""
Advanced Monte Carlo Simulation Engine V3.0
For match outcome and season simulation

Features:
- 100,000 iteration match simulation
- Shot-by-shot xG simulation
- HT/FT simulation with time-segmented goals
- Season/tournament simulation
- Parallel processing support
- Negative binomial for overdispersion
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import poisson, nbinom
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Structure for simulation results."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    score_matrix: np.ndarray
    expected_home_goals: float
    expected_away_goals: float
    over_under_probs: Dict[str, float]
    btts_prob: float
    correct_score_probs: Dict[str, float]
    htft_probs: Optional[Dict[str, float]] = None
    asian_handicap_probs: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            '1x2': {
                'home_win': round(self.home_win_prob, 4),
                'draw': round(self.draw_prob, 4),
                'away_win': round(self.away_win_prob, 4)
            },
            'expected_goals': {
                'home': round(self.expected_home_goals, 2),
                'away': round(self.expected_away_goals, 2),
                'total': round(self.expected_home_goals + self.expected_away_goals, 2)
            },
            'over_under': {k: round(v, 4) for k, v in self.over_under_probs.items()},
            'btts': {
                'yes': round(self.btts_prob, 4),
                'no': round(1 - self.btts_prob, 4)
            },
            'correct_scores': {k: round(v, 4) for k, v in self.correct_score_probs.items()}
        }
        
        if self.htft_probs:
            result['htft'] = {k: round(v, 4) for k, v in self.htft_probs.items()}
            
        if self.asian_handicap_probs:
            result['asian_handicap'] = {k: round(v, 4) for k, v in self.asian_handicap_probs.items()}
            
        return result


class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulation for football matches.
    
    Supports:
    - Match outcome simulation (100k iterations)
    - Shot-by-shot xG simulation
    - HT/FT simulation with time-segmented goals
    - Season/tournament simulation
    - Parallel processing for batch predictions
    """
    
    def __init__(
        self,
        n_simulations: int = 100000,
        max_goals: int = 10,
        use_negative_binomial: bool = False,
        include_overtime: bool = False
    ):
        """
        Initialize simulator.
        
        Args:
            n_simulations: Number of Monte Carlo iterations (default 100k)
            max_goals: Maximum goals to consider per team
            use_negative_binomial: Use NB distribution for overdispersion
            include_overtime: Include extra time for knockout matches
        """
        self.n_simulations = n_simulations
        self.max_goals = max_goals
        self.use_negative_binomial = use_negative_binomial
        self.include_overtime = include_overtime
        
    def simulate_match(
        self,
        home_xg: float,
        away_xg: float,
        home_xg_std: float = None,
        away_xg_std: float = None,
        correlation: float = 0.0,
        rho: float = -0.03
    ) -> SimulationResult:
        """
        Simulate a single match using Monte Carlo.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            home_xg_std: Standard deviation of home xG (for uncertainty)
            away_xg_std: Standard deviation of away xG
            correlation: Correlation between team scores
            rho: Dixon-Coles rho parameter for low-score adjustment
        """
        # Generate xG samples with uncertainty
        if home_xg_std is not None and home_xg_std > 0:
            home_lambdas = np.maximum(0.1, np.random.normal(home_xg, home_xg_std, self.n_simulations))
        else:
            home_lambdas = np.full(self.n_simulations, home_xg)
            
        if away_xg_std is not None and away_xg_std > 0:
            away_lambdas = np.maximum(0.1, np.random.normal(away_xg, away_xg_std, self.n_simulations))
        else:
            away_lambdas = np.full(self.n_simulations, away_xg)
        
        # Generate goal samples
        if self.use_negative_binomial:
            home_goals = self._sample_negative_binomial(home_lambdas)
            away_goals = self._sample_negative_binomial(away_lambdas)
        else:
            home_goals = np.random.poisson(home_lambdas)
            away_goals = np.random.poisson(away_lambdas)
        
        # Apply correlation adjustment
        if correlation != 0:
            home_goals, away_goals = self._apply_correlation(
                home_goals, away_goals, correlation
            )
        
        # Apply Dixon-Coles rho correction for low scores
        if rho != 0:
            home_goals, away_goals = self._apply_rho_correction(
                home_goals, away_goals, rho
            )
        
        # Calculate outcomes
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        # Score matrix
        score_matrix = self._calculate_score_matrix(home_goals, away_goals)
        
        # Over/Under probabilities
        total_goals = home_goals + away_goals
        over_under_probs = {
            'over_0.5': float(np.mean(total_goals > 0.5)),
            'over_1.5': float(np.mean(total_goals > 1.5)),
            'over_2.5': float(np.mean(total_goals > 2.5)),
            'over_3.5': float(np.mean(total_goals > 3.5)),
            'over_4.5': float(np.mean(total_goals > 4.5)),
            'over_5.5': float(np.mean(total_goals > 5.5)),
            'under_0.5': float(np.mean(total_goals < 0.5)),
            'under_1.5': float(np.mean(total_goals < 1.5)),
            'under_2.5': float(np.mean(total_goals < 2.5)),
            'under_3.5': float(np.mean(total_goals < 3.5)),
        }
        
        # BTTS probability
        btts_prob = float(np.mean((home_goals > 0) & (away_goals > 0)))
        
        # Correct score probabilities (top 20)
        correct_score_probs = self._get_correct_score_probs(score_matrix)
        
        # Asian handicap probabilities
        asian_handicap_probs = self._calculate_asian_handicap_probs(home_goals, away_goals)
        
        return SimulationResult(
            home_win_prob=home_wins / self.n_simulations,
            draw_prob=draws / self.n_simulations,
            away_win_prob=away_wins / self.n_simulations,
            score_matrix=score_matrix,
            expected_home_goals=float(np.mean(home_goals)),
            expected_away_goals=float(np.mean(away_goals)),
            over_under_probs=over_under_probs,
            btts_prob=btts_prob,
            correct_score_probs=correct_score_probs,
            asian_handicap_probs=asian_handicap_probs
        )
    
    def simulate_match_with_xg_shots(
        self,
        home_shots: List[float],
        away_shots: List[float]
    ) -> SimulationResult:
        """
        Simulate match using individual shot xG values.
        
        This is more accurate as it considers the actual
        shot quality distribution rather than just totals.
        
        Args:
            home_shots: List of xG values for each home shot
            away_shots: List of xG values for each away shot
        """
        home_goals_all = np.zeros(self.n_simulations, dtype=int)
        away_goals_all = np.zeros(self.n_simulations, dtype=int)
        
        # Vectorized shot simulation
        for xg in home_shots:
            home_goals_all += (np.random.random(self.n_simulations) < xg).astype(int)
            
        for xg in away_shots:
            away_goals_all += (np.random.random(self.n_simulations) < xg).astype(int)
        
        home_goals = home_goals_all
        away_goals = away_goals_all
        
        # Calculate results
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        score_matrix = self._calculate_score_matrix(home_goals, away_goals)
        total_goals = home_goals + away_goals
        
        return SimulationResult(
            home_win_prob=home_wins / self.n_simulations,
            draw_prob=draws / self.n_simulations,
            away_win_prob=away_wins / self.n_simulations,
            score_matrix=score_matrix,
            expected_home_goals=float(np.mean(home_goals)),
            expected_away_goals=float(np.mean(away_goals)),
            over_under_probs={
                'over_0.5': float(np.mean(total_goals > 0.5)),
                'over_1.5': float(np.mean(total_goals > 1.5)),
                'over_2.5': float(np.mean(total_goals > 2.5)),
                'over_3.5': float(np.mean(total_goals > 3.5)),
                'over_4.5': float(np.mean(total_goals > 4.5)),
            },
            btts_prob=float(np.mean((home_goals > 0) & (away_goals > 0))),
            correct_score_probs=self._get_correct_score_probs(score_matrix)
        )
    
    def simulate_match_with_htft(
        self,
        home_xg_1h: float,
        away_xg_1h: float,
        home_xg_2h: float,
        away_xg_2h: float
    ) -> SimulationResult:
        """
        Simulate match with half-time/full-time breakdown.
        
        Uses time-segmented Poisson (typically 42% of goals in 1st half).
        """
        # First half
        home_goals_1h = np.random.poisson(home_xg_1h, self.n_simulations)
        away_goals_1h = np.random.poisson(away_xg_1h, self.n_simulations)
        
        # Second half
        home_goals_2h = np.random.poisson(home_xg_2h, self.n_simulations)
        away_goals_2h = np.random.poisson(away_xg_2h, self.n_simulations)
        
        # Full match
        home_goals = home_goals_1h + home_goals_2h
        away_goals = away_goals_1h + away_goals_2h
        
        # HT results
        ht_home_wins = home_goals_1h > away_goals_1h
        ht_draws = home_goals_1h == away_goals_1h
        ht_away_wins = home_goals_1h < away_goals_1h
        
        # FT results
        ft_home_wins = home_goals > away_goals
        ft_draws = home_goals == away_goals
        ft_away_wins = home_goals < away_goals
        
        # HT/FT combinations (9 outcomes)
        htft_probs = {
            'H/H': float(np.mean(ht_home_wins & ft_home_wins)),
            'H/D': float(np.mean(ht_home_wins & ft_draws)),
            'H/A': float(np.mean(ht_home_wins & ft_away_wins)),
            'D/H': float(np.mean(ht_draws & ft_home_wins)),
            'D/D': float(np.mean(ht_draws & ft_draws)),
            'D/A': float(np.mean(ht_draws & ft_away_wins)),
            'A/H': float(np.mean(ht_away_wins & ft_home_wins)),
            'A/D': float(np.mean(ht_away_wins & ft_draws)),
            'A/A': float(np.mean(ht_away_wins & ft_away_wins)),
        }
        
        score_matrix = self._calculate_score_matrix(home_goals, away_goals)
        total_goals = home_goals + away_goals
        
        return SimulationResult(
            home_win_prob=float(np.mean(ft_home_wins)),
            draw_prob=float(np.mean(ft_draws)),
            away_win_prob=float(np.mean(ft_away_wins)),
            score_matrix=score_matrix,
            expected_home_goals=float(np.mean(home_goals)),
            expected_away_goals=float(np.mean(away_goals)),
            over_under_probs={
                'over_1.5': float(np.mean(total_goals > 1.5)),
                'over_2.5': float(np.mean(total_goals > 2.5)),
                'over_3.5': float(np.mean(total_goals > 3.5)),
            },
            btts_prob=float(np.mean((home_goals > 0) & (away_goals > 0))),
            correct_score_probs=self._get_correct_score_probs(score_matrix),
            htft_probs=htft_probs
        )
    
    def simulate_season(
        self,
        fixtures: List[Dict],
        team_strengths: Dict[str, Dict],
        n_simulations: int = 10000
    ) -> Dict:
        """
        Simulate entire season to get final standings distribution.
        
        Args:
            fixtures: List of remaining fixtures
            team_strengths: Dictionary with team attack/defense ratings
            n_simulations: Number of season simulations
        """
        all_standings = []
        
        for sim in range(n_simulations):
            # Initialize points
            points = {team: 0 for team in team_strengths.keys()}
            goal_diff = {team: 0 for team in team_strengths.keys()}
            goals_scored = {team: 0 for team in team_strengths.keys()}
            
            for fixture in fixtures:
                home = fixture['home_team']
                away = fixture['away_team']
                
                if home not in team_strengths or away not in team_strengths:
                    continue
                
                # Calculate expected goals
                home_xg = (
                    team_strengths[home].get('attack', 1.0) * 
                    team_strengths[away].get('defense', 1.0) * 
                    1.35  # League average home goals
                )
                away_xg = (
                    team_strengths[away].get('attack', 1.0) * 
                    team_strengths[home].get('defense', 1.0) * 
                    1.10  # Away factor adjustment
                )
                
                # Simulate match
                home_goals = np.random.poisson(home_xg)
                away_goals = np.random.poisson(away_xg)
                
                # Update points
                if home_goals > away_goals:
                    points[home] += 3
                elif home_goals < away_goals:
                    points[away] += 3
                else:
                    points[home] += 1
                    points[away] += 1
                
                goal_diff[home] += home_goals - away_goals
                goal_diff[away] += away_goals - home_goals
                goals_scored[home] += home_goals
                goals_scored[away] += away_goals
            
            # Get final standings (sorted by points, GD, GS)
            standings = sorted(
                points.keys(),
                key=lambda x: (points[x], goal_diff[x], goals_scored[x]),
                reverse=True
            )
            
            all_standings.append({
                team: rank + 1 
                for rank, team in enumerate(standings)
            })
        
        # Calculate probabilities
        results = []
        for team in team_strengths.keys():
            ranks = [s.get(team, 20) for s in all_standings]
            results.append({
                'team': team,
                'avg_position': round(np.mean(ranks), 2),
                'std_position': round(np.std(ranks), 2),
                'title_prob': round(np.mean([r == 1 for r in ranks]), 4),
                'top4_prob': round(np.mean([r <= 4 for r in ranks]), 4),
                'top6_prob': round(np.mean([r <= 6 for r in ranks]), 4),
                'relegation_prob': round(np.mean([r >= 18 for r in ranks]), 4),
            })
        
        return {
            'standings': sorted(results, key=lambda x: x['avg_position']),
            'simulations': n_simulations
        }
    
    def _sample_negative_binomial(self, lambdas: np.ndarray, r: float = 5.0) -> np.ndarray:
        """Sample from negative binomial distribution for overdispersion."""
        p = r / (r + lambdas)
        return np.random.negative_binomial(r, p)
    
    def _apply_correlation(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        correlation: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply correlation adjustment using simple approach."""
        adjustment = np.random.binomial(
            1, 
            abs(correlation), 
            self.n_simulations
        )
        
        if correlation > 0:
            # Positive correlation - high scoring games
            home_goals = home_goals + adjustment
            away_goals = away_goals + adjustment
        else:
            # Negative correlation - one team dominates
            home_goals = home_goals + adjustment
            away_goals = np.maximum(0, away_goals - adjustment)
        
        return home_goals, away_goals
    
    def _apply_rho_correction(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        rho: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Dixon-Coles rho correction for low scores."""
        # For 0-0 draws, adjust based on rho
        zero_zero = (home_goals == 0) & (away_goals == 0)
        
        if rho < 0:
            # Fewer 0-0 draws than expected
            flip = np.random.random(self.n_simulations) < abs(rho)
            flip_to_one = zero_zero & flip
            home_goals[flip_to_one] = 1  # Convert some 0-0 to 1-0
        
        return home_goals, away_goals
    
    def _calculate_score_matrix(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray
    ) -> np.ndarray:
        """Calculate probability matrix for all scorelines."""
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for h, a in zip(home_goals, away_goals):
            if h <= self.max_goals and a <= self.max_goals:
                matrix[int(h), int(a)] += 1
        
        matrix /= self.n_simulations
        return matrix
    
    def _get_correct_score_probs(
        self,
        score_matrix: np.ndarray,
        top_n: int = 20
    ) -> Dict[str, float]:
        """Get top N most likely scores."""
        scores = {}
        for h in range(score_matrix.shape[0]):
            for a in range(score_matrix.shape[1]):
                if score_matrix[h, a] > 0.001:  # Only include meaningful probabilities
                    scores[f'{h}-{a}'] = float(score_matrix[h, a])
        
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    def _calculate_asian_handicap_probs(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate Asian handicap probabilities."""
        goal_diff = home_goals - away_goals
        
        return {
            'home_-2.5': float(np.mean(goal_diff > 2.5)),
            'home_-2': float(np.mean(goal_diff > 2)),
            'home_-1.5': float(np.mean(goal_diff > 1.5)),
            'home_-1': float(np.mean(goal_diff > 1)),
            'home_-0.5': float(np.mean(goal_diff > 0.5)),
            'home_0': float(np.mean(goal_diff > 0)),
            'home_+0.5': float(np.mean(goal_diff > -0.5)),
            'home_+1': float(np.mean(goal_diff > -1)),
            'home_+1.5': float(np.mean(goal_diff > -1.5)),
            'home_+2': float(np.mean(goal_diff > -2)),
            'home_+2.5': float(np.mean(goal_diff > -2.5)),
        }


class ParallelMonteCarloSimulator(MonteCarloSimulator):
    """
    Parallelized Monte Carlo simulator for batch predictions.
    """
    
    def __init__(self, n_simulations: int = 100000, n_workers: int = None):
        super().__init__(n_simulations)
        self.n_workers = n_workers or max(1, multiprocessing.cpu_count() - 1)
    
    def simulate_batch(
        self,
        matches: List[Dict]
    ) -> List[SimulationResult]:
        """
        Simulate multiple matches in parallel.
        
        Args:
            matches: List of dicts with home_xg, away_xg keys
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(
                self._simulate_single_match,
                matches
            ))
        
        return results
    
    def _simulate_single_match(self, match: Dict) -> SimulationResult:
        """Simulate a single match (for parallel execution)."""
        return self.simulate_match(
            home_xg=match['home_xg'],
            away_xg=match['away_xg'],
            home_xg_std=match.get('home_xg_std'),
            away_xg_std=match.get('away_xg_std'),
            correlation=match.get('correlation', 0.0),
            rho=match.get('rho', -0.03)
        )


# Convenience function
def run_monte_carlo(
    home_xg: float,
    away_xg: float,
    n_simulations: int = 100000,
    include_htft: bool = False
) -> Dict:
    """
    Run Monte Carlo simulation for a match.
    
    Args:
        home_xg: Expected goals for home team
        away_xg: Expected goals for away team
        n_simulations: Number of iterations
        include_htft: Include HT/FT probabilities
    """
    simulator = MonteCarloSimulator(n_simulations=n_simulations)
    
    if include_htft:
        # Assume 42% of goals in first half
        result = simulator.simulate_match_with_htft(
            home_xg_1h=home_xg * 0.42,
            away_xg_1h=away_xg * 0.42,
            home_xg_2h=home_xg * 0.58,
            away_xg_2h=away_xg * 0.58
        )
    else:
        result = simulator.simulate_match(home_xg, away_xg)
    
    return result.to_dict()
