"""
Odds Integration Module
Integrates odds from multiple bookmakers.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class OddsIntegration:
    """
    Integrates odds from multiple bookmakers.
    
    Features:
    - Multi-bookmaker aggregation
    - Best odds finding
    - Odds movement tracking
    - Implied probability calculation
    """
    
    def __init__(self):
        self.bookmakers = {}
        self.odds_history: Dict[str, List[Dict]] = {}
        self.supported_markets = ['1x2', 'btts', 'over_under', 'asian_handicap']
    
    def register_bookmaker(
        self,
        name: str,
        api_key: str = None,
        base_url: str = None
    ):
        """Register a bookmaker source."""
        self.bookmakers[name] = {
            'api_key': api_key,
            'base_url': base_url,
            'last_update': None
        }
        logger.info(f"Registered bookmaker: {name}")
    
    def add_odds(
        self,
        match_id: str,
        bookmaker: str,
        market: str,
        odds: Dict
    ):
        """Add odds for a match."""
        if match_id not in self.odds_history:
            self.odds_history[match_id] = []
        
        self.odds_history[match_id].append({
            'bookmaker': bookmaker,
            'market': market,
            'odds': odds,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_best_odds(
        self,
        match_id: str,
        market: str = '1x2'
    ) -> Dict:
        """Get best odds across all bookmakers."""
        if match_id not in self.odds_history:
            return {}
        
        match_odds = [
            o for o in self.odds_history[match_id]
            if o['market'] == market
        ]
        
        if not match_odds:
            return {}
        
        if market == '1x2':
            best = {'home': 0, 'draw': 0, 'away': 0}
            best_bookmaker = {'home': '', 'draw': '', 'away': ''}
            
            for entry in match_odds:
                odds = entry['odds']
                for outcome in ['home', 'draw', 'away']:
                    if odds.get(outcome, 0) > best[outcome]:
                        best[outcome] = odds[outcome]
                        best_bookmaker[outcome] = entry['bookmaker']
            
            return {
                'best_odds': best,
                'bookmakers': best_bookmaker,
                'margin': self.calculate_margin(best)
            }
        
        return {}
    
    def calculate_margin(self, odds: Dict) -> float:
        """Calculate bookmaker margin."""
        if not odds:
            return 0
        
        implied_prob = sum(1/o for o in odds.values() if o > 0)
        margin = (implied_prob - 1) * 100
        return round(margin, 2)
    
    def get_implied_probabilities(
        self,
        odds: Dict,
        remove_margin: bool = True
    ) -> Dict:
        """Convert odds to implied probabilities."""
        if not odds:
            return {}
        
        total_implied = sum(1/o for o in odds.values() if o > 0)
        
        if remove_margin and total_implied > 0:
            return {
                k: round((1/v) / total_implied, 4)
                for k, v in odds.items() if v > 0
            }
        else:
            return {
                k: round(1/v, 4)
                for k, v in odds.items() if v > 0
            }
    
    def detect_odds_movement(
        self,
        match_id: str,
        market: str = '1x2',
        threshold: float = 0.1
    ) -> Dict:
        """Detect significant odds movements."""
        if match_id not in self.odds_history:
            return {}
        
        match_odds = sorted(
            [o for o in self.odds_history[match_id] if o['market'] == market],
            key=lambda x: x['timestamp']
        )
        
        if len(match_odds) < 2:
            return {'movement_detected': False}
        
        first = match_odds[0]['odds']
        last = match_odds[-1]['odds']
        
        movements = {}
        for outcome in first.keys():
            if outcome in last:
                change = (last[outcome] - first[outcome]) / first[outcome]
                movements[outcome] = {
                    'start': first[outcome],
                    'current': last[outcome],
                    'change_pct': round(change * 100, 2),
                    'significant': abs(change) >= threshold
                }
        
        return {
            'movement_detected': any(m['significant'] for m in movements.values()),
            'movements': movements,
            'time_span': {
                'start': match_odds[0]['timestamp'],
                'end': match_odds[-1]['timestamp']
            }
        }
    
    def get_consensus_probability(
        self,
        match_id: str,
        market: str = '1x2'
    ) -> Dict:
        """Get consensus probability from all bookmakers."""
        if match_id not in self.odds_history:
            return {}
        
        match_odds = [
            o for o in self.odds_history[match_id]
            if o['market'] == market
        ]
        
        if not match_odds:
            return {}
        
        all_probs = []
        for entry in match_odds:
            probs = self.get_implied_probabilities(entry['odds'])
            if probs:
                all_probs.append(probs)
        
        if not all_probs:
            return {}
        
        # Average across bookmakers
        consensus = {}
        keys = all_probs[0].keys()
        for key in keys:
            values = [p[key] for p in all_probs if key in p]
            consensus[key] = round(np.mean(values), 4)
        
        return {
            'consensus': consensus,
            'bookmaker_count': len(all_probs)
        }


_integration: Optional[OddsIntegration] = None

def get_integration() -> OddsIntegration:
    global _integration
    if _integration is None:
        _integration = OddsIntegration()
    return _integration
