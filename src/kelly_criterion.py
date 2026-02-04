"""
Kelly Criterion and Value Betting System

Research-based implementation of optimal bet sizing using fractional Kelly.

Key findings:
- Full Kelly leads to bankruptcy in 100% of realistic scenarios
- Partial Kelly (0.25-0.50) with 10% edge threshold is most profitable
- Multiple Kelly optimizes portfolio across simultaneous bets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValueBet:
    """Identified value betting opportunity"""
    market: str
    selection: str
    home_team: str
    away_team: str
    our_probability: float
    implied_probability: float
    decimal_odds: float
    edge: float
    expected_value: float
    kelly_stake: float
    recommended_stake: float
    confidence: str
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BetResult:
    """Result of a placed bet"""
    market: str
    selection: str
    stake: float
    odds: float
    won: bool
    profit: float
    bankroll_after: float


class KellyCriterion:
    """
    Kelly Criterion calculator for optimal bet sizing.
    
    The Kelly Criterion maximizes expected log wealth, but is too aggressive
    for real-world use. This implementation uses fractional Kelly.
    
    Research: "Partial Kelly with coefficient 0.50 and 10% threshold
    is the most profitable strategy."
    """
    
    def __init__(self,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.05,
                 max_stake_pct: float = 5.0):
        """
        Initialize Kelly calculator.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly)
            min_edge: Minimum edge required to bet (5% = 0.05)
            max_stake_pct: Maximum stake as % of bankroll (caps risk)
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_stake_pct = max_stake_pct
    
    def calculate_edge(self, our_probability: float, decimal_odds: float) -> float:
        """
        Calculate edge over bookmaker.
        
        Edge = (our_probability × odds) - 1
        
        Positive edge = expected profit per unit staked
        """
        return (our_probability * decimal_odds) - 1
    
    def implied_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if decimal_odds <= 1:
            return 1.0
        return 1 / decimal_odds
    
    def calculate_ev(self, our_probability: float, decimal_odds: float) -> float:
        """
        Calculate expected value per unit stake.
        
        EV = P(win) × (odds - 1) - P(lose)
        """
        return our_probability * (decimal_odds - 1) - (1 - our_probability)
    
    def calculate_kelly(self, our_probability: float, decimal_odds: float) -> float:
        """
        Calculate full Kelly stake.
        
        Formula: f* = (bp - q) / b
        where:
            b = decimal_odds - 1 (net odds)
            p = probability of winning
            q = 1 - p (probability of losing)
        """
        if our_probability <= 0 or our_probability >= 1:
            return 0.0
        if decimal_odds <= 1:
            return 0.0
        
        b = decimal_odds - 1
        p = our_probability
        q = 1 - p
        
        full_kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction
        
        # Cap at maximum stake
        return max(0, min(fractional_kelly * 100, self.max_stake_pct))
    
    def evaluate_bet(self,
                     market: str,
                     selection: str,
                     home_team: str,
                     away_team: str,
                     our_probability: float,
                     decimal_odds: float) -> Optional[ValueBet]:
        """
        Evaluate a potential bet.
        
        Returns ValueBet if it meets minimum edge threshold, else None.
        """
        edge = self.calculate_edge(our_probability, decimal_odds)
        
        # Check minimum edge
        if edge < self.min_edge:
            return None
        
        implied = self.implied_probability(decimal_odds)
        ev = self.calculate_ev(our_probability, decimal_odds)
        kelly = self.calculate_kelly(our_probability, decimal_odds)
        
        # Determine confidence
        if edge >= 0.15:
            confidence = 'HIGH'
        elif edge >= 0.10:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Build reasoning
        reasoning = (
            f"Our model: {our_probability*100:.1f}% vs "
            f"Market implied: {implied*100:.1f}% → "
            f"Edge: {edge*100:.1f}%"
        )
        
        return ValueBet(
            market=market,
            selection=selection,
            home_team=home_team,
            away_team=away_team,
            our_probability=round(our_probability, 4),
            implied_probability=round(implied, 4),
            decimal_odds=decimal_odds,
            edge=round(edge, 4),
            expected_value=round(ev, 4),
            kelly_stake=round(kelly, 2),
            recommended_stake=round(kelly, 2),
            confidence=confidence,
            reasoning=reasoning
        )


class ValueBettingSystem:
    """
    Complete value betting system with bankroll management.
    
    Integrates:
    - Kelly Criterion for stake sizing
    - Value bet identification across all markets
    - Bankroll tracking and statistics
    """
    
    def __init__(self,
                 bankroll: float = 1000.0,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.05):
        """
        Initialize value betting system.
        
        Args:
            bankroll: Starting bankroll
            kelly_fraction: Fractional Kelly to use
            min_edge: Minimum edge to consider a bet
        """
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.kelly = KellyCriterion(
            kelly_fraction=kelly_fraction,
            min_edge=min_edge
        )
        self.bet_history: List[BetResult] = []
        self.value_bets_found: List[ValueBet] = []
    
    def find_value_bets(self,
                        home_team: str,
                        away_team: str,
                        predictions: Dict,
                        odds: Dict) -> List[ValueBet]:
        """
        Find all value bets for a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            predictions: Dict with our probability assessments
            odds: Dict with bookmaker odds
        """
        value_bets = []
        
        # 1X2 Market
        market_checks = [
            ('1X2', 'Home Win', 'home_win', 'odds_home'),
            ('1X2', 'Draw', 'draw', 'odds_draw'),
            ('1X2', 'Away Win', 'away_win', 'odds_away'),
        ]
        
        for market, selection, prob_key, odds_key in market_checks:
            if prob_key in predictions and odds_key in odds:
                bet = self.kelly.evaluate_bet(
                    market=market,
                    selection=selection,
                    home_team=home_team,
                    away_team=away_team,
                    our_probability=predictions[prob_key],
                    decimal_odds=odds[odds_key]
                )
                if bet:
                    value_bets.append(bet)
        
        # BTTS Market
        if 'btts_yes' in predictions and 'odds_btts_yes' in odds:
            bet = self.kelly.evaluate_bet(
                market='BTTS',
                selection='Yes',
                home_team=home_team,
                away_team=away_team,
                our_probability=predictions['btts_yes'],
                decimal_odds=odds['odds_btts_yes']
            )
            if bet:
                value_bets.append(bet)
        
        if 'btts_no' in predictions and 'odds_btts_no' in odds:
            bet = self.kelly.evaluate_bet(
                market='BTTS',
                selection='No',
                home_team=home_team,
                away_team=away_team,
                our_probability=predictions['btts_no'],
                decimal_odds=odds['odds_btts_no']
            )
            if bet:
                value_bets.append(bet)
        
        # Over/Under Markets
        for threshold in ['1.5', '2.5', '3.5']:
            over_key = f'over_{threshold}'
            odds_over_key = f'odds_over_{threshold}'
            odds_under_key = f'odds_under_{threshold}'
            
            if over_key in predictions and odds_over_key in odds:
                bet = self.kelly.evaluate_bet(
                    market=f'Goals',
                    selection=f'Over {threshold}',
                    home_team=home_team,
                    away_team=away_team,
                    our_probability=predictions[over_key],
                    decimal_odds=odds[odds_over_key]
                )
                if bet:
                    value_bets.append(bet)
            
            # Under
            if over_key in predictions and odds_under_key in odds:
                under_prob = 1 - predictions[over_key]
                bet = self.kelly.evaluate_bet(
                    market=f'Goals',
                    selection=f'Under {threshold}',
                    home_team=home_team,
                    away_team=away_team,
                    our_probability=under_prob,
                    decimal_odds=odds[odds_under_key]
                )
                if bet:
                    value_bets.append(bet)
        
        # Correct Score (if provided)
        if 'correct_scores' in predictions:
            for score, prob in predictions['correct_scores'].items():
                odds_key = f'odds_cs_{score}'
                if odds_key in odds and prob >= 0.03:  # Only if at least 3% probability
                    bet = self.kelly.evaluate_bet(
                        market='Correct Score',
                        selection=score,
                        home_team=home_team,
                        away_team=away_team,
                        our_probability=prob,
                        decimal_odds=odds[odds_key]
                    )
                    if bet:
                        value_bets.append(bet)
        
        # Double Chance
        dc_checks = [
            ('dc_1x', 'odds_dc_1x', '1X'),
            ('dc_12', 'odds_dc_12', '12'),
            ('dc_x2', 'odds_dc_x2', 'X2'),
        ]
        
        for prob_key, odds_key, selection in dc_checks:
            if prob_key in predictions and odds_key in odds:
                bet = self.kelly.evaluate_bet(
                    market='Double Chance',
                    selection=selection,
                    home_team=home_team,
                    away_team=away_team,
                    our_probability=predictions[prob_key],
                    decimal_odds=odds[odds_key]
                )
                if bet:
                    value_bets.append(bet)
        
        # Sort by expected value
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)
        
        self.value_bets_found.extend(value_bets)
        return value_bets
    
    def place_bet(self, bet: ValueBet, won: bool) -> BetResult:
        """
        Record a placed bet and update bankroll.
        """
        stake = self.bankroll * (bet.recommended_stake / 100)
        
        if won:
            profit = stake * (bet.decimal_odds - 1)
        else:
            profit = -stake
        
        self.bankroll += profit
        
        result = BetResult(
            market=bet.market,
            selection=bet.selection,
            stake=stake,
            odds=bet.decimal_odds,
            won=won,
            profit=profit,
            bankroll_after=self.bankroll
        )
        
        self.bet_history.append(result)
        return result
    
    def get_statistics(self) -> Dict:
        """Get betting statistics."""
        if not self.bet_history:
            return {
                'total_bets': 0,
                'current_bankroll': self.bankroll,
                'roi': 0
            }
        
        total_bets = len(self.bet_history)
        wins = sum(1 for b in self.bet_history if b.won)
        total_staked = sum(b.stake for b in self.bet_history)
        total_profit = sum(b.profit for b in self.bet_history)
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': round(wins / total_bets, 4) if total_bets > 0 else 0,
            'total_staked': round(total_staked, 2),
            'total_profit': round(total_profit, 2),
            'roi': round((total_profit / total_staked) * 100, 2) if total_staked > 0 else 0,
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': round(self.bankroll, 2),
            'bankroll_growth': round((self.bankroll / self.initial_bankroll - 1) * 100, 2)
        }
    
    def get_best_value_bet(self, value_bets: List[ValueBet]) -> Optional[ValueBet]:
        """Get the single best value bet from a list."""
        if not value_bets:
            return None
        return max(value_bets, key=lambda x: x.expected_value)


class MultipleKelly:
    """
    Generalized Kelly for multiple simultaneous bets.
    
    When betting on multiple outcomes, the optimal strategy considers
    how bets hedge against each other.
    """
    
    def __init__(self, kelly_fraction: float = 0.25, max_total_stake: float = 25.0):
        """
        Initialize multiple Kelly optimizer.
        
        Args:
            kelly_fraction: Fraction to apply to optimal stakes
            max_total_stake: Maximum total stake across all bets (% of bankroll)
        """
        self.kelly_fraction = kelly_fraction
        self.max_total_stake = max_total_stake
    
    def optimize_portfolio(self, bets: List[Dict]) -> List[Dict]:
        """
        Optimize stake allocation across multiple bets.
        
        Args:
            bets: List of dicts with 'probability' and 'odds' keys
            
        Returns:
            List of dicts with added 'optimal_stake' key
        """
        if not bets:
            return []
        
        # Calculate individual Kelly stakes
        for bet in bets:
            p = bet['probability']
            odds = bet['odds']
            
            if p > 0 and p < 1 and odds > 1:
                b = odds - 1
                kelly = (b * p - (1 - p)) / b
                bet['kelly_stake'] = max(0, kelly * self.kelly_fraction * 100)
            else:
                bet['kelly_stake'] = 0
        
        # Scale down if total exceeds maximum
        total_kelly = sum(b['kelly_stake'] for b in bets)
        
        if total_kelly > self.max_total_stake:
            scale = self.max_total_stake / total_kelly
            for bet in bets:
                bet['optimal_stake'] = round(bet['kelly_stake'] * scale, 2)
        else:
            for bet in bets:
                bet['optimal_stake'] = round(bet['kelly_stake'], 2)
        
        return bets
    
    def create_accumulator_stakes(self, 
                                   legs: List[Dict],
                                   bankroll: float = 1000.0) -> Dict:
        """
        Calculate optimal stake for an accumulator bet.
        
        Accumulators require special handling as one loss = total loss.
        """
        if not legs:
            return {'stake': 0, 'potential_return': 0}
        
        # Calculate combined probability
        combined_prob = 1.0
        combined_odds = 1.0
        
        for leg in legs:
            combined_prob *= leg.get('probability', 0.5)
            combined_odds *= leg.get('odds', 2.0)
        
        # Calculate Kelly for the acca
        if combined_prob > 0 and combined_odds > 1:
            b = combined_odds - 1
            kelly = (b * combined_prob - (1 - combined_prob)) / b
            kelly_stake = max(0, kelly * self.kelly_fraction * 100)
        else:
            kelly_stake = 0
        
        # More conservative for accas (reduce by half)
        acca_stake = kelly_stake * 0.5
        stake_amount = bankroll * (acca_stake / 100)
        
        return {
            'combined_probability': round(combined_prob, 4),
            'combined_odds': round(combined_odds, 2),
            'kelly_stake_pct': round(acca_stake, 2),
            'stake_amount': round(stake_amount, 2),
            'potential_return': round(stake_amount * combined_odds, 2),
            'expected_value': round((combined_prob * combined_odds) - 1, 4)
        }


# Global instances
kelly_calculator = KellyCriterion()
value_betting_system = ValueBettingSystem()
multiple_kelly = MultipleKelly()


def calculate_optimal_stake(our_probability: float, decimal_odds: float) -> Dict:
    """Calculate optimal stake using Kelly Criterion."""
    edge = kelly_calculator.calculate_edge(our_probability, decimal_odds)
    ev = kelly_calculator.calculate_ev(our_probability, decimal_odds)
    kelly = kelly_calculator.calculate_kelly(our_probability, decimal_odds)
    
    return {
        'edge': round(edge * 100, 2),
        'expected_value': round(ev * 100, 2),
        'kelly_stake_pct': round(kelly, 2),
        'is_value': edge >= kelly_calculator.min_edge,
        'recommendation': 'BET' if edge >= kelly_calculator.min_edge else 'SKIP'
    }


def find_all_value_bets(home_team: str, away_team: str,
                        predictions: Dict, odds: Dict) -> List[Dict]:
    """Find all value bets for a match."""
    value_bets = value_betting_system.find_value_bets(
        home_team, away_team, predictions, odds
    )
    return [vb.to_dict() for vb in value_bets]
