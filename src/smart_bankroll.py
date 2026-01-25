"""
Smart Bankroll Management System

Advanced money management with:
- Kelly Criterion optimization
- Risk-adjusted stake sizing
- Drawdown protection
- Portfolio diversification
- Betting limits enforcement
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class BankrollState:
    """Current bankroll state"""
    initial: float
    current: float
    peak: float
    lowest: float
    total_staked: float
    total_profit: float
    bet_count: int
    win_count: int
    
    @property
    def roi(self) -> float:
        if self.total_staked == 0:
            return 0.0
        return (self.total_profit / self.total_staked) * 100
    
    @property
    def win_rate(self) -> float:
        if self.bet_count == 0:
            return 0.0
        return (self.win_count / self.bet_count) * 100
    
    @property
    def drawdown(self) -> float:
        if self.peak == 0:
            return 0.0
        return ((self.peak - self.current) / self.peak) * 100
    
    def to_dict(self) -> Dict:
        return {
            'initial': self.initial,
            'current': round(self.current, 2),
            'peak': round(self.peak, 2),
            'total_staked': round(self.total_staked, 2),
            'total_profit': round(self.total_profit, 2),
            'roi': round(self.roi, 2),
            'bet_count': self.bet_count,
            'win_count': self.win_count,
            'win_rate': round(self.win_rate, 1),
            'drawdown': round(self.drawdown, 1)
        }


@dataclass
class StakeRecommendation:
    """Recommended stake with reasoning"""
    amount: float
    percentage: float
    method: str
    reasoning: str
    risk_level: str
    max_allowed: float
    adjustments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'amount': round(self.amount, 2),
            'percentage': round(self.percentage, 2),
            'method': self.method,
            'reasoning': self.reasoning,
            'risk_level': self.risk_level,
            'max_allowed': round(self.max_allowed, 2),
            'adjustments': self.adjustments
        }


class KellyCriterion:
    """Advanced Kelly Criterion calculator"""
    
    @staticmethod
    def full_kelly(probability: float, odds: float) -> float:
        """Calculate full Kelly stake percentage"""
        if probability <= 0 or odds <= 1:
            return 0.0
        
        q = 1 - probability
        b = odds - 1  # Decimal odds to profit ratio
        
        kelly = (probability * b - q) / b
        return max(0.0, kelly)
    
    @staticmethod
    def fractional_kelly(probability: float, odds: float, fraction: float = 0.25) -> float:
        """Calculate fractional Kelly (safer)"""
        full = KellyCriterion.full_kelly(probability, odds)
        return full * fraction
    
    @staticmethod
    def half_kelly(probability: float, odds: float) -> float:
        """Half Kelly - common conservative approach"""
        return KellyCriterion.fractional_kelly(probability, odds, 0.5)
    
    @staticmethod
    def quarter_kelly(probability: float, odds: float) -> float:
        """Quarter Kelly - very conservative"""
        return KellyCriterion.fractional_kelly(probability, odds, 0.25)


class DrawdownProtection:
    """Protect against excessive drawdowns"""
    
    def __init__(
        self, 
        max_daily_loss_pct: float = 10.0,
        max_weekly_loss_pct: float = 20.0,
        max_drawdown_pct: float = 25.0
    ):
        self.max_daily_loss = max_daily_loss_pct
        self.max_weekly_loss = max_weekly_loss_pct
        self.max_drawdown = max_drawdown_pct
        self.daily_losses: Dict[str, float] = defaultdict(float)
        self.weekly_losses: Dict[str, float] = defaultdict(float)
        
    def record_loss(self, amount: float, bankroll: float):
        """Record a loss"""
        today = datetime.now().strftime('%Y-%m-%d')
        week = datetime.now().strftime('%Y-W%W')
        
        self.daily_losses[today] += amount
        self.weekly_losses[week] += amount
        
    def can_bet(self, stake: float, bankroll: float, current_drawdown: float) -> Tuple[bool, str]:
        """Check if betting is allowed"""
        today = datetime.now().strftime('%Y-%m-%d')
        week = datetime.now().strftime('%Y-W%W')
        
        # Check daily limit
        daily_loss = self.daily_losses.get(today, 0)
        if (daily_loss / bankroll * 100) >= self.max_daily_loss:
            return False, f"Daily loss limit reached ({self.max_daily_loss}%)"
        
        # Check weekly limit
        weekly_loss = self.weekly_losses.get(week, 0)
        if (weekly_loss / bankroll * 100) >= self.max_weekly_loss:
            return False, f"Weekly loss limit reached ({self.max_weekly_loss}%)"
        
        # Check drawdown
        if current_drawdown >= self.max_drawdown:
            return False, f"Max drawdown reached ({self.max_drawdown}%)"
        
        return True, "OK"
    
    def get_stake_reduction(self, current_drawdown: float) -> float:
        """Get stake reduction factor based on drawdown"""
        if current_drawdown < 5:
            return 1.0
        elif current_drawdown < 10:
            return 0.8
        elif current_drawdown < 15:
            return 0.6
        elif current_drawdown < 20:
            return 0.4
        else:
            return 0.2


class BankrollManager:
    """Complete bankroll management system"""
    
    def __init__(
        self,
        initial_bankroll: float,
        risk_level: RiskLevel = RiskLevel.MODERATE
    ):
        self.state = BankrollState(
            initial=initial_bankroll,
            current=initial_bankroll,
            peak=initial_bankroll,
            lowest=initial_bankroll,
            total_staked=0.0,
            total_profit=0.0,
            bet_count=0,
            win_count=0
        )
        
        self.risk_level = risk_level
        self.kelly = KellyCriterion()
        self.protection = DrawdownProtection()
        
        # Risk level settings
        self.settings = {
            RiskLevel.CONSERVATIVE: {
                'max_stake_pct': 2.0,
                'kelly_fraction': 0.25,
                'min_edge': 0.05,
                'max_odds': 4.0
            },
            RiskLevel.MODERATE: {
                'max_stake_pct': 5.0,
                'kelly_fraction': 0.5,
                'min_edge': 0.03,
                'max_odds': 8.0
            },
            RiskLevel.AGGRESSIVE: {
                'max_stake_pct': 10.0,
                'kelly_fraction': 0.75,
                'min_edge': 0.02,
                'max_odds': 15.0
            }
        }
    
    def get_settings(self) -> Dict:
        return self.settings[self.risk_level]
    
    def calculate_stake(
        self,
        probability: float,
        odds: float,
        is_value_bet: bool = False,
        is_sure_bet: bool = False
    ) -> StakeRecommendation:
        """Calculate optimal stake for a bet"""
        settings = self.get_settings()
        adjustments = []
        
        # Check if betting is allowed
        can_bet, reason = self.protection.can_bet(
            0, self.state.current, self.state.drawdown
        )
        
        if not can_bet:
            return StakeRecommendation(
                amount=0,
                percentage=0,
                method="BLOCKED",
                reasoning=reason,
                risk_level=self.risk_level.value,
                max_allowed=0,
                adjustments=[reason]
            )
        
        # Calculate Kelly stake
        kelly_pct = self.kelly.fractional_kelly(
            probability, odds, settings['kelly_fraction']
        ) * 100
        
        # Check edge
        implied_prob = 1 / odds
        edge = probability - implied_prob
        
        if edge < settings['min_edge']:
            return StakeRecommendation(
                amount=0,
                percentage=0,
                method="NO_VALUE",
                reasoning=f"Edge too small ({edge*100:.1f}% < {settings['min_edge']*100}%)",
                risk_level=self.risk_level.value,
                max_allowed=0,
                adjustments=["Insufficient edge"]
            )
        
        # Apply maximum stake limit
        max_stake_pct = settings['max_stake_pct']
        if kelly_pct > max_stake_pct:
            kelly_pct = max_stake_pct
            adjustments.append(f"Capped at {max_stake_pct}% max")
        
        # Drawdown adjustment
        drawdown_factor = self.protection.get_stake_reduction(self.state.drawdown)
        if drawdown_factor < 1.0:
            kelly_pct *= drawdown_factor
            adjustments.append(f"Reduced {int((1-drawdown_factor)*100)}% for drawdown protection")
        
        # Sure bet boost
        if is_sure_bet and probability >= 0.91:
            kelly_pct *= 1.2
            adjustments.append("Sure bet: +20% stake")
        
        # Value bet adjustment
        if is_value_bet and edge > 0.1:
            kelly_pct *= 1.1
            adjustments.append("High value: +10% stake")
        
        # Calculate final amount
        stake_amount = (kelly_pct / 100) * self.state.current
        max_allowed = (max_stake_pct / 100) * self.state.current
        
        # Determine reasoning
        if kelly_pct >= 4:
            reasoning = "Strong edge detected - optimal stake"
        elif kelly_pct >= 2:
            reasoning = "Good value - moderate stake recommended"
        elif kelly_pct >= 1:
            reasoning = "Acceptable edge - small stake"
        else:
            reasoning = "Marginal edge - minimal stake"
        
        return StakeRecommendation(
            amount=stake_amount,
            percentage=kelly_pct,
            method="kelly",
            reasoning=reasoning,
            risk_level=self.risk_level.value,
            max_allowed=max_allowed,
            adjustments=adjustments
        )
    
    def record_bet(self, stake: float, odds: float, won: bool):
        """Record a completed bet"""
        self.state.total_staked += stake
        self.state.bet_count += 1
        
        if won:
            profit = stake * (odds - 1)
            self.state.current += profit
            self.state.total_profit += profit
            self.state.win_count += 1
        else:
            self.state.current -= stake
            self.state.total_profit -= stake
            self.protection.record_loss(stake, self.state.current)
        
        # Update peak/lowest
        if self.state.current > self.state.peak:
            self.state.peak = self.state.current
        if self.state.current < self.state.lowest:
            self.state.lowest = self.state.current
    
    def get_status(self) -> Dict:
        """Get current bankroll status"""
        return {
            'bankroll': self.state.to_dict(),
            'risk_level': self.risk_level.value,
            'settings': self.get_settings(),
            'can_bet': self.protection.can_bet(0, self.state.current, self.state.drawdown)[0],
            'drawdown_factor': self.protection.get_stake_reduction(self.state.drawdown)
        }
    
    def get_portfolio_allocation(self, bets: List[Dict]) -> List[Dict]:
        """Optimize stake allocation across multiple bets"""
        if not bets:
            return []
        
        # Calculate individual Kelly for each bet
        total_kelly = 0
        bet_stakes = []
        
        for bet in bets:
            prob = bet.get('probability', 0.5)
            odds = bet.get('odds', 2.0)
            
            kelly = self.kelly.fractional_kelly(
                prob, odds, self.get_settings()['kelly_fraction']
            )
            total_kelly += kelly
            bet_stakes.append({
                'bet': bet,
                'kelly': kelly
            })
        
        # If total Kelly exceeds max, scale down proportionally
        max_total = self.get_settings()['max_stake_pct'] / 100 * 1.5  # Allow 50% over for accas
        
        if total_kelly > max_total:
            scale = max_total / total_kelly
            for bs in bet_stakes:
                bs['kelly'] *= scale
        
        # Calculate actual amounts
        result = []
        for bs in bet_stakes:
            amount = bs['kelly'] * self.state.current
            result.append({
                **bs['bet'],
                'recommended_stake': round(amount, 2),
                'stake_percentage': round(bs['kelly'] * 100, 2)
            })
        
        return result


class BettingLimits:
    """Enforce betting limits and responsible gambling"""
    
    def __init__(self):
        self.daily_limits: Dict[str, float] = {}
        self.weekly_limits: Dict[str, float] = {}
        self.monthly_limits: Dict[str, float] = {}
        self.bet_counts: Dict[str, int] = defaultdict(int)
        self.cooldowns: Dict[str, datetime] = {}
    
    def set_daily_limit(self, user_id: str, amount: float):
        """Set daily betting limit"""
        self.daily_limits[user_id] = amount
    
    def set_weekly_limit(self, user_id: str, amount: float):
        """Set weekly betting limit"""
        self.weekly_limits[user_id] = amount
    
    def check_limit(self, user_id: str, amount: float) -> Tuple[bool, str]:
        """Check if bet is within limits"""
        today = datetime.now().strftime('%Y-%m-%d')
        week = datetime.now().strftime('%Y-W%W')
        
        # Check cooldown
        if user_id in self.cooldowns:
            if datetime.now() < self.cooldowns[user_id]:
                return False, f"Cooldown active until {self.cooldowns[user_id]}"
        
        # Check bet count (max 20 per day)
        if self.bet_counts.get(f"{user_id}_{today}", 0) >= 20:
            return False, "Daily bet count limit reached (20)"
        
        return True, "OK"
    
    def record_bet(self, user_id: str):
        """Record a bet for limit tracking"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.bet_counts[f"{user_id}_{today}"] += 1
    
    def set_cooldown(self, user_id: str, hours: int):
        """Set betting cooldown"""
        self.cooldowns[user_id] = datetime.now() + timedelta(hours=hours)


# Global instances
default_bankroll = BankrollManager(1000.0, RiskLevel.MODERATE)
betting_limits = BettingLimits()


def calculate_optimal_stake(
    probability: float,
    odds: float,
    bankroll: float = None,
    risk_level: str = "moderate"
) -> Dict:
    """Calculate optimal stake for a bet"""
    if bankroll:
        manager = BankrollManager(bankroll, RiskLevel[risk_level.upper()])
    else:
        manager = default_bankroll
    
    recommendation = manager.calculate_stake(probability, odds)
    return recommendation.to_dict()


def get_bankroll_status() -> Dict:
    """Get current bankroll status"""
    return default_bankroll.get_status()


def record_bet_result(stake: float, odds: float, won: bool):
    """Record a bet result"""
    default_bankroll.record_bet(stake, odds, won)
