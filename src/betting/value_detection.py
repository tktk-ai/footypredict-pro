"""
Value Detection Module
Identifies value bets by comparing probabilities to odds.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ValueDetector:
    """
    Detects value bets by comparing model probabilities to bookmaker odds.
    
    Methods:
    - Expected value calculation
    - Kelly criterion sizing
    - Edge detection
    """
    
    def __init__(
        self,
        min_edge: float = 0.05,
        max_kelly_fraction: float = 0.25
    ):
        self.min_edge = min_edge
        self.max_kelly_fraction = max_kelly_fraction
    
    def calculate_edge(
        self,
        probability: float,
        odds: float
    ) -> float:
        """Calculate edge over bookmaker."""
        implied_prob = 1 / odds
        return probability - implied_prob
    
    def calculate_expected_value(
        self,
        probability: float,
        odds: float,
        stake: float = 1.0
    ) -> float:
        """Calculate expected value of a bet."""
        ev = (probability * (odds - 1) * stake) - ((1 - probability) * stake)
        return ev
    
    def calculate_kelly(
        self,
        probability: float,
        odds: float
    ) -> float:
        """Calculate Kelly criterion stake fraction."""
        # Kelly formula: (p*b - q) / b
        # where p = probability, b = decimal odds - 1, q = 1 - p
        b = odds - 1
        q = 1 - probability
        
        kelly = (probability * b - q) / b if b > 0 else 0
        
        # Limit max stake
        return max(0, min(kelly, self.max_kelly_fraction))
    
    def detect_value(
        self,
        predictions: Dict,
        odds: Dict,
        market: str = '1x2'
    ) -> List[Dict]:
        """
        Detect value bets in a market.
        
        Args:
            predictions: Model predictions with probabilities
            odds: Bookmaker odds
            market: Market type
        """
        value_bets = []
        
        if market == '1x2':
            outcomes = ['home', 'draw', 'away']
            probs = predictions.get('1x2', {})
            
            for outcome in outcomes:
                prob = probs.get(outcome, 0)
                odd = odds.get(outcome, 0)
                
                if prob > 0 and odd > 0:
                    edge = self.calculate_edge(prob, odd)
                    ev = self.calculate_expected_value(prob, odd)
                    kelly = self.calculate_kelly(prob, odd)
                    
                    if edge >= self.min_edge:
                        value_bets.append({
                            'market': market,
                            'outcome': outcome,
                            'probability': round(prob, 4),
                            'odds': odd,
                            'implied_prob': round(1/odd, 4),
                            'edge': round(edge, 4),
                            'expected_value': round(ev, 4),
                            'kelly_stake': round(kelly, 4),
                            'is_value': True
                        })
        
        elif market == 'btts':
            for outcome in ['yes', 'no']:
                prob = predictions.get(f'btts_{outcome}', 0)
                odd = odds.get(f'btts_{outcome}', 0)
                
                if prob > 0 and odd > 0:
                    edge = self.calculate_edge(prob, odd)
                    
                    if edge >= self.min_edge:
                        value_bets.append({
                            'market': 'btts',
                            'outcome': outcome,
                            'probability': round(prob, 4),
                            'odds': odd,
                            'edge': round(edge, 4),
                            'kelly_stake': round(self.calculate_kelly(prob, odd), 4),
                            'is_value': True
                        })
        
        elif market == 'over_under':
            for line in ['2.5', '1.5', '3.5']:
                for direction in ['over', 'under']:
                    key = f'{direction}_{line}'
                    prob = predictions.get('lines', {}).get(key, 0)
                    odd = odds.get(key, 0)
                    
                    if prob > 0 and odd > 0:
                        edge = self.calculate_edge(prob, odd)
                        
                        if edge >= self.min_edge:
                            value_bets.append({
                                'market': f'{direction}_{line}',
                                'probability': round(prob, 4),
                                'odds': odd,
                                'edge': round(edge, 4),
                                'kelly_stake': round(self.calculate_kelly(prob, odd), 4),
                                'is_value': True
                            })
        
        # Sort by edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return value_bets
    
    def get_best_value_bet(
        self,
        value_bets: List[Dict]
    ) -> Optional[Dict]:
        """Get the best value bet from a list."""
        if not value_bets:
            return None
        return value_bets[0]


_detector: Optional[ValueDetector] = None

def get_detector() -> ValueDetector:
    global _detector
    if _detector is None:
        _detector = ValueDetector()
    return _detector


def find_value_bets(predictions: Dict, odds: Dict) -> List[Dict]:
    """Quick function to find value bets."""
    return get_detector().detect_value(predictions, odds)
