"""
Kelly Criterion Module
Optimal stake sizing using Kelly criterion.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion stake sizing.
    
    Variants:
    - Full Kelly
    - Fractional Kelly
    - Kelly with constraints
    """
    
    def __init__(
        self,
        fraction: float = 0.25,
        max_stake: float = 0.10,
        min_edge: float = 0.03
    ):
        self.fraction = fraction  # Fractional Kelly
        self.max_stake = max_stake  # Max stake as fraction of bankroll
        self.min_edge = min_edge  # Min edge to place bet
    
    def calculate_stake(
        self,
        probability: float,
        odds: float,
        bankroll: float = 1000
    ) -> Dict:
        """
        Calculate optimal stake.
        
        Args:
            probability: Estimated probability
            odds: Decimal odds
            bankroll: Current bankroll
        """
        # Kelly formula
        b = odds - 1  # Net odds
        p = probability
        q = 1 - probability
        
        # Edge check
        edge = p * odds - 1
        if edge < self.min_edge:
            return {
                'stake': 0,
                'kelly_fraction': 0,
                'reason': 'insufficient_edge',
                'edge': round(edge, 4)
            }
        
        # Full Kelly
        full_kelly = (p * b - q) / b if b > 0 else 0
        
        # Fractional Kelly
        fraction_kelly = full_kelly * self.fraction
        
        # Apply constraints
        constrained_kelly = max(0, min(fraction_kelly, self.max_stake))
        
        # Calculate stake
        stake = bankroll * constrained_kelly
        
        return {
            'stake': round(stake, 2),
            'stake_percentage': round(constrained_kelly * 100, 2),
            'full_kelly': round(full_kelly, 4),
            'fractional_kelly': round(fraction_kelly, 4),
            'edge': round(edge, 4),
            'expected_return': round((probability * (odds - 1) - (1 - probability)) * stake, 2),
            'odds': odds,
            'probability': round(probability, 4)
        }
    
    def calculate_multi_bet(
        self,
        bets: List[Dict],
        bankroll: float = 1000
    ) -> List[Dict]:
        """Calculate stakes for multiple bets."""
        results = []
        total_allocation = 0
        
        for bet in bets:
            result = self.calculate_stake(
                bet['probability'],
                bet['odds'],
                bankroll
            )
            
            if result['stake'] > 0:
                # Adjust if total would exceed bankroll limits
                remaining = min(bankroll * 0.5 - total_allocation, result['stake'])
                if remaining > 0:
                    result['stake'] = round(remaining, 2)
                    total_allocation += remaining
                    results.append({**bet, **result})
        
        return results
    
    def optimal_portfolio(
        self,
        bets: List[Dict],
        bankroll: float = 1000,
        correlation_adjustment: bool = True
    ) -> Dict:
        """
        Calculate optimal portfolio allocation.
        
        Args:
            bets: List of bets with probability and odds
            bankroll: Total bankroll
            correlation_adjustment: Reduce stakes for correlated bets
        """
        if not bets:
            return {'bets': [], 'total_stake': 0}
        
        # Calculate individual Kelly stakes
        kelly_stakes = []
        for bet in bets:
            stake_info = self.calculate_stake(bet['probability'], bet['odds'], bankroll)
            if stake_info['stake'] > 0:
                kelly_stakes.append({
                    **bet,
                    'kelly_stake': stake_info['stake'],
                    'edge': stake_info['edge']
                })
        
        if not kelly_stakes:
            return {'bets': [], 'total_stake': 0}
        
        # Sort by edge
        kelly_stakes.sort(key=lambda x: x['edge'], reverse=True)
        
        # Allocate with decreasing priority
        allocated_bets = []
        total = 0
        max_total = bankroll * 0.4  # Max 40% of bankroll across all bets
        
        for bet in kelly_stakes:
            available = max_total - total
            if available <= 0:
                break
            
            stake = min(bet['kelly_stake'], available)
            
            # Correlation adjustment (reduce stakes for multiple bets in same match)
            if correlation_adjustment and len(allocated_bets) > 0:
                stake *= 0.7
            
            if stake > 10:  # Min stake
                allocated_bets.append({
                    **bet,
                    'allocated_stake': round(stake, 2)
                })
                total += stake
        
        return {
            'bets': allocated_bets,
            'total_stake': round(total, 2),
            'bankroll_percentage': round(total / bankroll * 100, 2),
            'expected_return': round(sum(
                b['edge'] * b['allocated_stake'] for b in allocated_bets
            ), 2)
        }


_kelly: Optional[KellyCriterion] = None

def get_kelly() -> KellyCriterion:
    global _kelly
    if _kelly is None:
        _kelly = KellyCriterion()
    return _kelly


def calculate_stake(prob: float, odds: float, bankroll: float = 1000) -> Dict:
    """Quick function to calculate stake."""
    return get_kelly().calculate_stake(prob, odds, bankroll)
