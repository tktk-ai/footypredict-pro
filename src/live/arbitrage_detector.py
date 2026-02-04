"""
Arbitrage Detection Module
Detects arbitrage opportunities across bookmakers.

Based on the blueprint for value betting.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage betting opportunity."""
    match: str
    market: str
    total_margin: float
    profit_pct: float
    best_odds: Dict[str, Tuple[str, float]]  # outcome -> (bookmaker, odds)
    stakes: Dict[str, float]
    investment: float
    guaranteed_return: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'match': self.match,
            'market': self.market,
            'total_margin': round(self.total_margin * 100, 2),
            'profit_pct': round(self.profit_pct * 100, 2),
            'best_odds': {k: {'bookmaker': v[0], 'odds': v[1]} for k, v in self.best_odds.items()},
            'stakes': {k: round(v, 2) for k, v in self.stakes.items()},
            'investment': round(self.investment, 2),
            'guaranteed_return': round(self.guaranteed_return, 2),
            'timestamp': self.timestamp.isoformat()
        }


class ArbitrageDetector:
    """
    Detects arbitrage opportunities across multiple bookmakers.
    
    Arbitrage exists when:
    sum(1/best_odds_per_outcome) < 1
    """
    
    def __init__(
        self,
        default_stake: float = 100.0,
        min_profit_pct: float = 0.01,  # 1% minimum profit
        max_odds_age_seconds: int = 300  # 5 minutes
    ):
        self.default_stake = default_stake
        self.min_profit_pct = min_profit_pct
        self.max_odds_age = max_odds_age_seconds
        
    def detect_arb(
        self,
        match: str,
        market: str,
        odds_by_bookmaker: Dict[str, Dict[str, float]],
        stake: float = None
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunity for a specific market.
        
        Args:
            match: Match name
            market: Market type (e.g., '1X2', 'Over/Under 2.5')
            odds_by_bookmaker: {bookmaker: {outcome: odds}}
            stake: Total investment amount
            
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        stake = stake or self.default_stake
        
        # Find best odds for each outcome
        best_odds = self._find_best_odds(odds_by_bookmaker)
        
        if not best_odds:
            return None
        
        # Calculate total implied probability
        total_margin = sum(1 / odds for _, odds in best_odds.values())
        
        # Check for arbitrage (margin < 1)
        if total_margin >= 1:
            return None
        
        # Calculate profit percentage
        profit_pct = (1 / total_margin) - 1
        
        if profit_pct < self.min_profit_pct:
            return None
        
        # Calculate optimal stakes
        stakes = self._calculate_stakes(best_odds, stake)
        total_investment = sum(stakes.values())
        guaranteed_return = stake / total_margin
        
        return ArbitrageOpportunity(
            match=match,
            market=market,
            total_margin=total_margin,
            profit_pct=profit_pct,
            best_odds=best_odds,
            stakes=stakes,
            investment=total_investment,
            guaranteed_return=guaranteed_return,
            timestamp=datetime.now()
        )
    
    def detect_all_arbs(
        self,
        matches_odds: List[Dict],
        stake: float = None
    ) -> List[ArbitrageOpportunity]:
        """
        Detect all arbitrage opportunities across multiple matches and markets.
        
        Args:
            matches_odds: List of {
                'match': str,
                'market': str,
                'odds_by_bookmaker': Dict[str, Dict[str, float]]
            }
        """
        opportunities = []
        
        for match_data in matches_odds:
            arb = self.detect_arb(
                match=match_data['match'],
                market=match_data['market'],
                odds_by_bookmaker=match_data['odds_by_bookmaker'],
                stake=stake
            )
            
            if arb:
                opportunities.append(arb)
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        return opportunities
    
    def _find_best_odds(
        self,
        odds_by_bookmaker: Dict[str, Dict[str, float]]
    ) -> Dict[str, Tuple[str, float]]:
        """Find best odds for each outcome across bookmakers."""
        best_odds = {}
        
        # Get all possible outcomes
        all_outcomes = set()
        for bookmaker_odds in odds_by_bookmaker.values():
            all_outcomes.update(bookmaker_odds.keys())
        
        # Find best for each
        for outcome in all_outcomes:
            best_bookmaker = None
            best_value = 0
            
            for bookmaker, odds in odds_by_bookmaker.items():
                if outcome in odds and odds[outcome] > best_value:
                    best_value = odds[outcome]
                    best_bookmaker = bookmaker
            
            if best_bookmaker:
                best_odds[outcome] = (best_bookmaker, best_value)
        
        return best_odds
    
    def _calculate_stakes(
        self,
        best_odds: Dict[str, Tuple[str, float]],
        total_stake: float
    ) -> Dict[str, float]:
        """Calculate optimal stake distribution for each outcome."""
        stakes = {}
        total_margin = sum(1 / odds for _, odds in best_odds.values())
        
        for outcome, (_, odds) in best_odds.items():
            stakes[outcome] = (total_stake / total_margin) / odds
        
        return stakes
    
    def check_margin(
        self,
        odds: Dict[str, float]
    ) -> Dict:
        """
        Check bookmaker margin for a market.
        
        Args:
            odds: {outcome: odds} for a single bookmaker
        """
        if not odds:
            return {'margin': 0, 'overround': 0}
        
        total_implied = sum(1 / o for o in odds.values())
        margin = total_implied - 1
        overround = margin * 100
        
        return {
            'margin': margin,
            'overround': overround,
            'implied_probabilities': {k: 1/v for k, v in odds.items()},
            'true_probabilities': {k: (1/v) / total_implied for k, v in odds.items()}
        }


class SurebetScanner:
    """
    Real-time scanner for surebets across bookmakers.
    """
    
    def __init__(
        self,
        min_profit_pct: float = 0.5,  # 0.5% minimum
        bookmakers: List[str] = None
    ):
        self.detector = ArbitrageDetector(min_profit_pct=min_profit_pct / 100)
        self.bookmakers = bookmakers or [
            'bet365', 'pinnacle', 'betfair', 'unibet', 
            'william_hill', '1xbet', 'betway'
        ]
        self.active_arbs = []
        
    def scan(self, odds_data: List[Dict]) -> List[ArbitrageOpportunity]:
        """
        Scan current odds for surebets.
        
        Args:
            odds_data: Current odds from all bookmakers
        """
        self.active_arbs = self.detector.detect_all_arbs(odds_data)
        
        if self.active_arbs:
            logger.info(f"Found {len(self.active_arbs)} surebet opportunities!")
            for arb in self.active_arbs[:3]:  # Log top 3
                logger.info(
                    f"  {arb.match} ({arb.market}): "
                    f"{arb.profit_pct*100:.2f}% profit"
                )
        
        return self.active_arbs
    
    def get_active_arbs(self) -> List[Dict]:
        """Get current arbitrage opportunities as dicts."""
        return [arb.to_dict() for arb in self.active_arbs]


class ValueBetDetector:
    """
    Detects value bets where model probability > implied probability.
    """
    
    def __init__(
        self,
        min_edge: float = 0.05,  # 5% minimum edge
        kelly_fraction: float = 0.25  # Quarter Kelly
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
    
    def find_value_bets(
        self,
        model_probs: Dict[str, float],
        bookmaker_odds: Dict[str, float]
    ) -> List[Dict]:
        """
        Find value betting opportunities.
        
        Args:
            model_probs: {outcome: probability} from our model
            bookmaker_odds: {outcome: odds} from bookmaker
        """
        value_bets = []
        
        for outcome, model_prob in model_probs.items():
            if outcome not in bookmaker_odds:
                continue
            
            odds = bookmaker_odds[outcome]
            implied_prob = 1 / odds
            edge = model_prob - implied_prob
            
            if edge >= self.min_edge:
                # Calculate Kelly stake
                kelly_stake = self._kelly_criterion(model_prob, odds)
                
                value_bets.append({
                    'outcome': outcome,
                    'model_probability': round(model_prob, 4),
                    'implied_probability': round(implied_prob, 4),
                    'edge': round(edge, 4),
                    'edge_pct': round(edge * 100, 2),
                    'odds': odds,
                    'kelly_stake_pct': round(kelly_stake * 100, 2),
                    'recommended_stake_pct': round(kelly_stake * self.kelly_fraction * 100, 2),
                    'expected_value': round(edge * odds, 4)
                })
        
        # Sort by edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return value_bets
    
    def _kelly_criterion(self, prob: float, odds: float) -> float:
        """Calculate Kelly criterion stake."""
        q = 1 - prob
        b = odds - 1
        
        if b <= 0:
            return 0
        
        kelly = (prob * b - q) / b
        return max(0, kelly)


# Global instances
_arb_detector: Optional[ArbitrageDetector] = None
_surebet_scanner: Optional[SurebetScanner] = None
_value_detector: Optional[ValueBetDetector] = None


def get_arb_detector() -> ArbitrageDetector:
    """Get arbitrage detector instance."""
    global _arb_detector
    if _arb_detector is None:
        _arb_detector = ArbitrageDetector()
    return _arb_detector


def get_surebet_scanner() -> SurebetScanner:
    """Get surebet scanner instance."""
    global _surebet_scanner
    if _surebet_scanner is None:
        _surebet_scanner = SurebetScanner()
    return _surebet_scanner


def get_value_detector() -> ValueBetDetector:
    """Get value bet detector instance."""
    global _value_detector
    if _value_detector is None:
        _value_detector = ValueBetDetector()
    return _value_detector


def detect_arbitrage(
    match: str,
    odds_by_bookmaker: Dict[str, Dict[str, float]],
    market: str = '1X2'
) -> Optional[Dict]:
    """Quick function to detect arbitrage."""
    detector = get_arb_detector()
    arb = detector.detect_arb(match, market, odds_by_bookmaker)
    return arb.to_dict() if arb else None


def find_value_bets(
    model_probs: Dict[str, float],
    bookmaker_odds: Dict[str, float]
) -> List[Dict]:
    """Quick function to find value bets."""
    return get_value_detector().find_value_bets(model_probs, bookmaker_odds)
