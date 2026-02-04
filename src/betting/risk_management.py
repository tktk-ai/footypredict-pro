"""
Risk Management Module
Comprehensive risk management for betting.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages betting risk.
    
    Features:
    - Exposure limits
    - Correlation risk
    - VaR calculations
    - Risk scoring
    """
    
    def __init__(
        self,
        max_single_bet: float = 0.05,
        max_total_exposure: float = 0.4,
        max_correlated_exposure: float = 0.15
    ):
        self.max_single_bet = max_single_bet
        self.max_total_exposure = max_total_exposure
        self.max_correlated_exposure = max_correlated_exposure
        
        self.current_bets: List[Dict] = []
    
    def check_bet(
        self,
        bet: Dict,
        bankroll: float
    ) -> Dict:
        """
        Check if a bet passes risk controls.
        
        Args:
            bet: Bet to check (stake, odds, match, etc.)
            bankroll: Current bankroll
        """
        issues = []
        warnings = []
        
        stake = bet.get('stake', 0)
        stake_pct = stake / bankroll if bankroll > 0 else 1
        
        # Single bet limit
        if stake_pct > self.max_single_bet:
            issues.append({
                'type': 'single_bet_limit',
                'message': f"Stake {stake_pct:.1%} exceeds max {self.max_single_bet:.1%}",
                'max_allowed': bankroll * self.max_single_bet
            })
        
        # Total exposure
        current_exposure = sum(b.get('stake', 0) for b in self.current_bets)
        new_exposure = (current_exposure + stake) / bankroll
        
        if new_exposure > self.max_total_exposure:
            issues.append({
                'type': 'total_exposure_limit',
                'message': f"Total exposure {new_exposure:.1%} exceeds max {self.max_total_exposure:.1%}"
            })
        
        # Correlated bets (same match)
        match_id = bet.get('match_id', bet.get('match', ''))
        correlated = [b for b in self.current_bets if b.get('match_id', b.get('match', '')) == match_id]
        correlated_stake = sum(b.get('stake', 0) for b in correlated) + stake
        
        if correlated_stake / bankroll > self.max_correlated_exposure:
            issues.append({
                'type': 'correlated_exposure',
                'message': f"Correlated exposure on same match exceeds limit"
            })
        
        # Edge warning
        edge = bet.get('edge', 0)
        if edge < 0.03:
            warnings.append({
                'type': 'low_edge',
                'message': f"Edge {edge:.1%} is marginally positive"
            })
        
        return {
            'approved': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'adjusted_stake': min(stake, bankroll * self.max_single_bet)
        }
    
    def add_bet(self, bet: Dict):
        """Add bet to current tracking."""
        self.current_bets.append(bet)
    
    def remove_bet(self, bet_id: str):
        """Remove settled bet from tracking."""
        self.current_bets = [b for b in self.current_bets if b.get('id') != bet_id]
    
    def calculate_var(
        self,
        confidence: float = 0.95,
        n_simulations: int = 10000
    ) -> Dict:
        """
        Calculate Value at Risk.
        
        Uses Monte Carlo simulation.
        """
        if not self.current_bets:
            return {'var': 0, 'expected_return': 0}
        
        # Simulate outcomes
        outcomes = []
        
        for _ in range(n_simulations):
            result = 0
            for bet in self.current_bets:
                stake = bet.get('stake', 0)
                prob = bet.get('probability', 0.5)
                odds = bet.get('odds', 2.0)
                
                if np.random.random() < prob:
                    result += stake * (odds - 1)
                else:
                    result -= stake
            
            outcomes.append(result)
        
        outcomes = np.array(outcomes)
        var = np.percentile(outcomes, (1 - confidence) * 100)
        
        return {
            'var': round(float(abs(var)), 2),
            'var_confidence': confidence,
            'expected_return': round(float(np.mean(outcomes)), 2),
            'std_dev': round(float(np.std(outcomes)), 2),
            'worst_case': round(float(np.min(outcomes)), 2),
            'best_case': round(float(np.max(outcomes)), 2)
        }
    
    def get_risk_score(self) -> Dict:
        """Calculate overall risk score."""
        if not self.current_bets:
            return {'score': 0, 'level': 'none'}
        
        # Factors
        n_bets = len(self.current_bets)
        total_stake = sum(b.get('stake', 0) for b in self.current_bets)
        avg_odds = np.mean([b.get('odds', 2.0) for b in self.current_bets])
        
        # Score components
        concentration = total_stake / max(b.get('stake', 0) for b in self.current_bets) if self.current_bets else 0
        odds_risk = min((avg_odds - 1.5) / 3, 1)  # Higher odds = more risk
        
        score = (0.4 * (1 / max(concentration, 0.1)) + 0.6 * odds_risk) * 10
        score = min(max(score, 0), 10)
        
        level = 'low' if score < 3 else ('medium' if score < 6 else 'high')
        
        return {
            'score': round(score, 1),
            'level': level,
            'factors': {
                'number_of_bets': n_bets,
                'total_at_risk': round(total_stake, 2),
                'average_odds': round(avg_odds, 2)
            }
        }


_manager: Optional[RiskManager] = None

def get_manager() -> RiskManager:
    global _manager
    if _manager is None:
        _manager = RiskManager()
    return _manager
