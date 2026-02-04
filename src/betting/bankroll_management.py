"""
Bankroll Management Module
Manages betting bankroll and position sizing.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BankrollManager:
    """
    Manages betting bankroll.
    
    Features:
    - Position sizing
    - Drawdown monitoring
    - Profit tracking
    - Unit sizing
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000,
        unit_size: float = None,
        max_daily_risk: float = 0.1,
        max_drawdown: float = 0.3
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.unit_size = unit_size or initial_bankroll * 0.01
        self.max_daily_risk = max_daily_risk
        self.max_drawdown = max_drawdown
        
        self.peak_bankroll = initial_bankroll
        self.daily_risk_used = 0
        self.last_reset_date = datetime.now().date()
        
        self.transaction_history: List[Dict] = []
    
    def get_unit_stake(self, units: float = 1) -> float:
        """Get stake for given number of units."""
        return min(units * self.unit_size, self.available_stake())
    
    def available_stake(self) -> float:
        """Get available stake considering daily limits."""
        self._check_daily_reset()
        
        # Max daily risk
        daily_limit = self.current_bankroll * self.max_daily_risk - self.daily_risk_used
        
        # Max single bet as portion of bankroll
        single_bet_limit = self.current_bankroll * 0.05
        
        return max(0, min(daily_limit, single_bet_limit))
    
    def _check_daily_reset(self):
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_risk_used = 0
            self.last_reset_date = today
    
    def place_bet(self, stake: float, odds: float, description: str = "") -> Dict:
        """Record placing a bet."""
        if stake > self.available_stake():
            return {
                'success': False,
                'reason': 'insufficient_stake_available',
                'available': self.available_stake()
            }
        
        self.current_bankroll -= stake
        self.daily_risk_used += stake
        
        transaction = {
            'type': 'bet_placed',
            'stake': stake,
            'odds': odds,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'bankroll_after': self.current_bankroll
        }
        self.transaction_history.append(transaction)
        
        return {'success': True, 'transaction': transaction}
    
    def settle_bet(
        self,
        stake: float,
        odds: float,
        won: bool,
        description: str = ""
    ) -> Dict:
        """Settle a bet result."""
        if won:
            profit = stake * (odds - 1)
            self.current_bankroll += stake + profit
        else:
            profit = -stake
        
        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
        
        transaction = {
            'type': 'bet_settled',
            'stake': stake,
            'odds': odds,
            'won': won,
            'profit': profit,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'bankroll_after': self.current_bankroll
        }
        self.transaction_history.append(transaction)
        
        return {
            'profit': profit,
            'new_bankroll': self.current_bankroll,
            'transaction': transaction
        }
    
    def get_drawdown(self) -> Dict:
        """Calculate current drawdown."""
        drawdown_amount = self.peak_bankroll - self.current_bankroll
        drawdown_pct = drawdown_amount / self.peak_bankroll if self.peak_bankroll > 0 else 0
        
        return {
            'current_bankroll': self.current_bankroll,
            'peak_bankroll': self.peak_bankroll,
            'drawdown_amount': round(drawdown_amount, 2),
            'drawdown_percentage': round(drawdown_pct * 100, 2),
            'max_allowed': self.max_drawdown * 100,
            'is_at_limit': drawdown_pct >= self.max_drawdown
        }
    
    def get_stats(self) -> Dict:
        """Get bankroll statistics."""
        bets = [t for t in self.transaction_history if t['type'] == 'bet_settled']
        
        if not bets:
            return {
                'total_bets': 0,
                'current_bankroll': self.current_bankroll,
                'roi': 0
            }
        
        wins = [b for b in bets if b['won']]
        total_staked = sum(b['stake'] for b in bets)
        total_profit = sum(b['profit'] for b in bets)
        
        return {
            'total_bets': len(bets),
            'wins': len(wins),
            'losses': len(bets) - len(wins),
            'win_rate': round(len(wins) / len(bets) * 100, 2),
            'total_staked': round(total_staked, 2),
            'total_profit': round(total_profit, 2),
            'roi': round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0,
            'current_bankroll': round(self.current_bankroll, 2),
            'initial_bankroll': self.initial_bankroll,
            'growth': round((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 2)
        }
    
    def should_stop(self) -> Dict:
        """Check if should stop betting."""
        drawdown = self.get_drawdown()
        
        reasons = []
        if drawdown['is_at_limit']:
            reasons.append('Max drawdown reached')
        if self.current_bankroll < self.initial_bankroll * 0.2:
            reasons.append('Bankroll critically low')
        
        return {
            'should_stop': len(reasons) > 0,
            'reasons': reasons,
            'drawdown': drawdown
        }


_manager: Optional[BankrollManager] = None

def get_manager(initial: float = 1000) -> BankrollManager:
    global _manager
    if _manager is None:
        _manager = BankrollManager(initial)
    return _manager
