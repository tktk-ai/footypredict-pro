"""
Portfolio Optimization Module
Optimizes betting portfolio allocation.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolio of bets.
    
    Methods:
    - Mean-variance optimization
    - Risk parity
    - Maximum Sharpe ratio
    """
    
    def __init__(
        self,
        max_exposure: float = 0.5,
        min_bet: float = 0.01,
        risk_free_rate: float = 0.0
    ):
        self.max_exposure = max_exposure
        self.min_bet = min_bet
        self.risk_free_rate = risk_free_rate
    
    def optimize_mean_variance(
        self,
        bets: List[Dict],
        bankroll: float,
        target_return: float = None
    ) -> Dict:
        """
        Mean-variance optimization.
        
        Args:
            bets: List of bets with expected_return and variance
            bankroll: Total bankroll
            target_return: Target portfolio return
        """
        if not bets:
            return {'allocations': [], 'total': 0}
        
        n = len(bets)
        
        # Extract returns and calculate covariance
        returns = np.array([b.get('expected_return', b.get('edge', 0.05)) for b in bets])
        variances = np.array([b.get('variance', 0.25) for b in bets])
        
        # Simple diagonal covariance (assume independence)
        cov_matrix = np.diag(variances)
        
        # Equal weights as starting point
        weights = np.ones(n) / n
        
        # Optimize for max Sharpe (simplified)
        for _ in range(100):
            gradient = returns - 2 * np.dot(cov_matrix, weights)
            weights += 0.01 * gradient
            
            # Project to simplex and apply constraints
            weights = np.maximum(weights, 0)
            if weights.sum() > self.max_exposure:
                weights = weights / weights.sum() * self.max_exposure
        
        # Calculate allocations
        allocations = []
        for i, bet in enumerate(bets):
            if weights[i] >= self.min_bet:
                allocations.append({
                    **bet,
                    'weight': round(float(weights[i]), 4),
                    'stake': round(bankroll * weights[i], 2)
                })
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0
        
        return {
            'allocations': allocations,
            'total_weight': round(float(weights.sum()), 4),
            'total_stake': round(bankroll * weights.sum(), 2),
            'expected_return': round(float(portfolio_return), 4),
            'portfolio_variance': round(float(portfolio_variance), 4),
            'sharpe_ratio': round(float(sharpe), 4)
        }
    
    def optimize_risk_parity(
        self,
        bets: List[Dict],
        bankroll: float
    ) -> Dict:
        """
        Risk parity allocation (equal risk contribution).
        """
        if not bets:
            return {'allocations': [], 'total': 0}
        
        n = len(bets)
        variances = np.array([b.get('variance', 0.25) for b in bets])
        
        # Inverse volatility weighting
        inv_vol = 1 / np.sqrt(variances)
        weights = inv_vol / inv_vol.sum()
        
        # Scale to max exposure
        if weights.sum() > self.max_exposure:
            weights = weights / weights.sum() * self.max_exposure
        
        allocations = []
        for i, bet in enumerate(bets):
            if weights[i] >= self.min_bet:
                allocations.append({
                    **bet,
                    'weight': round(float(weights[i]), 4),
                    'stake': round(bankroll * weights[i], 2),
                    'risk_contribution': round(float(weights[i] * np.sqrt(variances[i])), 4)
                })
        
        return {
            'allocations': allocations,
            'total_stake': round(bankroll * weights.sum(), 2),
            'method': 'risk_parity'
        }
    
    def diversification_check(
        self,
        bets: List[Dict]
    ) -> Dict:
        """Check portfolio diversification."""
        if not bets:
            return {'is_diversified': True, 'issues': []}
        
        issues = []
        
        # Check concentration
        stakes = [b.get('stake', b.get('weight', 0)) for b in bets]
        total = sum(stakes)
        
        if total > 0:
            max_concentration = max(stakes) / total
            if max_concentration > 0.5:
                issues.append(f"High concentration: {max_concentration:.0%} in single bet")
        
        # Check correlation (by match)
        matches = [b.get('match_id', b.get('match', '')) for b in bets]
        unique_matches = len(set(matches))
        if len(bets) > 3 and unique_matches < len(bets) * 0.5:
            issues.append("Low match diversification")
        
        return {
            'is_diversified': len(issues) == 0,
            'issues': issues,
            'unique_matches': unique_matches,
            'total_bets': len(bets)
        }


_optimizer: Optional[PortfolioOptimizer] = None

def get_optimizer() -> PortfolioOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
    return _optimizer
