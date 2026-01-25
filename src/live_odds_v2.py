"""
Live Odds Integration V2 - Multi-Source API Client

Integrates with multiple odds providers:
- The Odds API (free tier)
- Betfair Exchange API
- Pinnacle API
- Odds comparison with value detection
"""

import os
import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MultiOddsClient:
    """
    Unified client for multiple odds APIs.
    Supports: The Odds API, Betfair, Pinnacle
    """
    
    def __init__(self):
        self.odds_api_key = os.getenv('ODDS_API_KEY', '')
        self.betfair_key = os.getenv('BETFAIR_API_KEY', '')
        self.betfair_session = os.getenv('BETFAIR_SESSION_TOKEN', '')
        self.pinnacle_key = os.getenv('PINNACLE_API_KEY', '')
        
        self.base_urls = {
            'odds_api': 'https://api.the-odds-api.com/v4',
            'betfair': 'https://api.betfair.com/exchange/betting/json-rpc/v1',
            'pinnacle': 'https://api.pinnacle.com/v1'
        }
        
        self.cached_odds = {}
        self.cache_expiry = {}
        
    async def get_odds_from_all_sources(
        self,
        sport: str = 'soccer',
        league: str = None
    ) -> Dict:
        """
        Fetch odds from all available sources.
        """
        results = {}
        
        # The Odds API (always try if key available)
        if self.odds_api_key:
            try:
                results['odds_api'] = await self._fetch_odds_api(sport, league)
            except Exception as e:
                logger.error(f"Odds API error: {e}")
        
        # Betfair
        if self.betfair_key and self.betfair_session:
            try:
                results['betfair'] = await self._fetch_betfair(sport)
            except Exception as e:
                logger.error(f"Betfair error: {e}")
        
        # Pinnacle
        if self.pinnacle_key:
            try:
                results['pinnacle'] = await self._fetch_pinnacle(sport)
            except Exception as e:
                logger.error(f"Pinnacle error: {e}")
        
        return results
    
    async def _fetch_odds_api(self, sport: str, league: str = None) -> List[Dict]:
        """Fetch from The Odds API."""
        sport_key = 'soccer_epl' if league in ['EPL', 'premier_league', None] else f'soccer_{league.lower()}'
        
        url = f"{self.base_urls['odds_api']}/sports/{sport_key}/odds"
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'uk,eu',
            'markets': 'h2h,totals,btts',
            'oddsFormat': 'decimal'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    async def _fetch_betfair(self, sport: str) -> List[Dict]:
        """Fetch from Betfair Exchange."""
        # Betfair requires session authentication
        headers = {
            'X-Application': self.betfair_key,
            'X-Authentication': self.betfair_session,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'jsonrpc': '2.0',
            'method': 'SportsAPING/v1.0/listEvents',
            'params': {
                'filter': {
                    'eventTypeIds': ['1'],  # Soccer
                    'marketTypeCodes': ['MATCH_ODDS']
                }
            },
            'id': 1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_urls['betfair'],
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    async def _fetch_pinnacle(self, sport: str) -> List[Dict]:
        """Fetch from Pinnacle."""
        headers = {
            'Authorization': f'Basic {self.pinnacle_key}'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_urls['pinnacle']}/odds",
                headers=headers,
                params={'sportId': 29}  # Soccer
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    def get_odds_sync(self, sport: str = 'soccer', league: str = None) -> Dict:
        """
        Synchronous wrapper for getting odds.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_odds_from_all_sources(sport, league)
        )


class ValueBetDetector:
    """
    Detect value bets by comparing model predictions with market odds.
    """
    
    def __init__(self, min_edge: float = 0.03, max_stake_pct: float = 0.10):
        self.min_edge = min_edge
        self.max_stake_pct = max_stake_pct
    
    def find_value_bets(
        self,
        predictions: Dict[str, float],
        odds: Dict[str, float]
    ) -> List[Dict]:
        """
        Find value betting opportunities.
        
        Args:
            predictions: Model probabilities {'home_win': 0.5, 'draw': 0.25, 'away_win': 0.25}
            odds: Market odds {'home_win': 2.0, 'draw': 3.5, 'away_win': 4.0}
        """
        value_bets = []
        
        for market, prob in predictions.items():
            if market not in odds or prob <= 0:
                continue
            
            market_odds = odds[market]
            implied_prob = 1 / market_odds
            edge = prob - implied_prob
            
            if edge >= self.min_edge:
                # Kelly criterion
                kelly = (prob * market_odds - 1) / (market_odds - 1)
                kelly = max(0, min(kelly, self.max_stake_pct))
                
                # Expected value
                ev = (prob * (market_odds - 1)) - ((1 - prob))
                
                value_bets.append({
                    'market': market,
                    'probability': round(prob, 4),
                    'odds': market_odds,
                    'implied_probability': round(implied_prob, 4),
                    'edge': round(edge, 4),
                    'edge_pct': f"{edge * 100:.1f}%",
                    'expected_value': round(ev, 4),
                    'kelly_stake_pct': round(kelly * 100, 2),
                    'rating': self._rate_bet(edge, kelly)
                })
        
        # Sort by edge descending
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        return value_bets
    
    def _rate_bet(self, edge: float, kelly: float) -> str:
        """Rate a value bet."""
        if edge >= 0.10 and kelly >= 0.05:
            return "⭐⭐⭐ STRONG VALUE"
        elif edge >= 0.06 and kelly >= 0.03:
            return "⭐⭐ GOOD VALUE"
        elif edge >= 0.03:
            return "⭐ VALUE"
        return ""


class OddsComparisonEngine:
    """
    Compare odds across multiple bookmakers.
    """
    
    def __init__(self):
        self.odds_client = MultiOddsClient()
        self.value_detector = ValueBetDetector()
    
    def get_best_odds(self, match_odds: Dict[str, Dict]) -> Dict:
        """
        Find best odds across bookmakers.
        
        Args:
            match_odds: {'bookmaker1': {'home': 1.8, 'draw': 3.5, 'away': 4.0}, ...}
        """
        best_odds = {
            'home_win': {'odds': 0, 'bookmaker': None},
            'draw': {'odds': 0, 'bookmaker': None},
            'away_win': {'odds': 0, 'bookmaker': None}
        }
        
        for bookmaker, odds in match_odds.items():
            for market, market_odds in odds.items():
                key = market
                if market == 'home':
                    key = 'home_win'
                elif market == 'away':
                    key = 'away_win'
                
                if key in best_odds and market_odds > best_odds[key]['odds']:
                    best_odds[key]['odds'] = market_odds
                    best_odds[key]['bookmaker'] = bookmaker
        
        return best_odds
    
    def calculate_margin(self, odds: Dict) -> float:
        """Calculate bookmaker margin."""
        total_prob = sum(1 / o for o in odds.values() if o > 0)
        return max(0, total_prob - 1)
    
    def find_arbitrage(self, bookmaker_odds: Dict[str, Dict]) -> Optional[Dict]:
        """
        Find arbitrage opportunities across bookmakers.
        """
        best = self.get_best_odds(bookmaker_odds)
        
        if not all(best[m]['odds'] > 0 for m in ['home_win', 'draw', 'away_win']):
            return None
        
        # Calculate combined probability
        total_prob = sum(1 / best[m]['odds'] for m in ['home_win', 'draw', 'away_win'])
        
        if total_prob < 1:
            profit_margin = (1 - total_prob) * 100
            stakes = {}
            
            for market in ['home_win', 'draw', 'away_win']:
                stakes[market] = (1 / best[market]['odds']) / total_prob
            
            return {
                'type': 'Arbitrage Opportunity',
                'profit_margin': round(profit_margin, 2),
                'stakes': stakes,
                'best_odds': best
            }
        
        return None


# Convenience functions
def get_live_odds_v2(league: str = 'EPL') -> Dict:
    """Get live odds from multiple sources."""
    client = MultiOddsClient()
    return client.get_odds_sync('soccer', league)


def find_live_value_bets(
    predictions: Dict[str, float],
    league: str = 'EPL'
) -> List[Dict]:
    """Find value bets using live odds."""
    engine = OddsComparisonEngine()
    all_odds = engine.odds_client.get_odds_sync('soccer', league)
    
    # Combine odds from all sources
    combined_odds = {}
    for source, matches in all_odds.items():
        if isinstance(matches, list):
            for match in matches:
                if 'bookmakers' in match:
                    for bm in match['bookmakers']:
                        for market in bm.get('markets', []):
                            if market['key'] == 'h2h':
                                for outcome in market.get('outcomes', []):
                                    combined_odds[outcome['name']] = outcome['price']
    
    return engine.value_detector.find_value_bets(predictions, combined_odds)
