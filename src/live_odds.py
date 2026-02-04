"""
Live Odds Integration

Fetches real-time odds from free sources and provides comparison.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import aiohttp

logger = logging.getLogger(__name__)


class LiveOddsProvider:
    """Fetch live odds from free APIs"""
    
    # Free odds API (The Odds API has a free tier)
    ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "demo"  # Demo key for testing
        self.cache = {}
        self.cache_time = None
        self.cache_ttl = 300  # 5 minutes
    
    async def fetch_odds(self, sport: str = "soccer_epl") -> List[Dict]:
        """Fetch live odds for a sport"""
        # Check cache
        if self.cache.get(sport) and self.cache_time:
            age = (datetime.now() - self.cache_time).seconds
            if age < self.cache_ttl:
                return self.cache[sport]
        
        try:
            async with aiohttp.ClientSession() as session:
                url = self.ODDS_API_URL.format(sport=sport)
                params = {
                    'apiKey': self.api_key,
                    'regions': 'eu',
                    'markets': 'h2h',
                    'oddsFormat': 'decimal'
                }
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.cache[sport] = data
                        self.cache_time = datetime.now()
                        return data
                    else:
                        logger.warning(f"Odds API returned {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return []
    
    def get_sample_odds(self) -> List[Dict]:
        """Return sample odds data when API not available"""
        return [
            {
                'id': 'sample1',
                'sport_key': 'soccer_epl',
                'commence_time': datetime.now().isoformat(),
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'bookmakers': [
                    {
                        'key': 'bet365',
                        'title': 'Bet365',
                        'markets': [{
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Manchester United', 'price': 2.75},
                                {'name': 'Draw', 'price': 3.40},
                                {'name': 'Liverpool', 'price': 2.50}
                            ]
                        }]
                    },
                    {
                        'key': 'pinnacle',
                        'title': 'Pinnacle',
                        'markets': [{
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Manchester United', 'price': 2.80},
                                {'name': 'Draw', 'price': 3.35},
                                {'name': 'Liverpool', 'price': 2.55}
                            ]
                        }]
                    }
                ]
            },
            {
                'id': 'sample2',
                'sport_key': 'soccer_epl',
                'commence_time': datetime.now().isoformat(),
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'bookmakers': [
                    {
                        'key': 'bet365',
                        'title': 'Bet365',
                        'markets': [{
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Arsenal', 'price': 1.95},
                                {'name': 'Draw', 'price': 3.60},
                                {'name': 'Chelsea', 'price': 3.80}
                            ]
                        }]
                    }
                ]
            }
        ]
    
    def format_odds_comparison(self, odds_data: List[Dict]) -> List[Dict]:
        """Format odds for easy comparison"""
        formatted = []
        
        for match in odds_data:
            bookmaker_odds = {}
            
            for bookmaker in match.get('bookmakers', []):
                bk_name = bookmaker.get('title', bookmaker.get('key'))
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            team = outcome['name']
                            price = outcome['price']
                            if team not in bookmaker_odds:
                                bookmaker_odds[team] = {}
                            bookmaker_odds[team][bk_name] = price
            
            # Find best odds for each outcome
            best_odds = {}
            for team, odds in bookmaker_odds.items():
                if odds:
                    best_bk = max(odds.items(), key=lambda x: x[1])
                    best_odds[team] = {'price': best_bk[1], 'bookmaker': best_bk[0]}
            
            formatted.append({
                'id': match.get('id'),
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'commence_time': match.get('commence_time'),
                'bookmaker_odds': bookmaker_odds,
                'best_odds': best_odds
            })
        
        return formatted
    
    def find_value_bets(self, odds_data: List[Dict], predictions: Dict[str, Dict]) -> List[Dict]:
        """Find value bets where predicted probability beats odds"""
        value_bets = []
        
        for match in odds_data:
            match_key = f"{match.get('home_team')} vs {match.get('away_team')}"
            pred = predictions.get(match_key)
            
            if not pred:
                continue
            
            for bookmaker in match.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            price = outcome['price']
                            implied_prob = 1 / price
                            
                            # Map outcome to prediction
                            if outcome['name'] == match.get('home_team'):
                                our_prob = pred.get('home_prob', 0)
                            elif outcome['name'] == match.get('away_team'):
                                our_prob = pred.get('away_prob', 0)
                            else:
                                our_prob = pred.get('draw_prob', 0)
                            
                            edge = our_prob - implied_prob
                            
                            if edge > 0.05:  # Minimum 5% edge
                                value_bets.append({
                                    'match': match_key,
                                    'selection': outcome['name'],
                                    'odds': price,
                                    'bookmaker': bookmaker.get('title'),
                                    'implied_prob': round(implied_prob, 3),
                                    'our_prob': round(our_prob, 3),
                                    'edge': round(edge, 3),
                                    'kelly_stake': round(edge / (price - 1) * 100, 1) if price > 1 else 0
                                })
        
        return sorted(value_bets, key=lambda x: x['edge'], reverse=True)


# Global instance
_provider: Optional[LiveOddsProvider] = None

def get_odds_provider() -> LiveOddsProvider:
    global _provider
    if _provider is None:
        import os
        api_key = os.environ.get('ODDS_API_KEY')
        _provider = LiveOddsProvider(api_key)
    return _provider

def get_live_odds(sport: str = "soccer_epl") -> List[Dict]:
    """Get live odds (sync wrapper)"""
    provider = get_odds_provider()
    try:
        loop = asyncio.new_event_loop()
        odds = loop.run_until_complete(provider.fetch_odds(sport))
        loop.close()
        if not odds:
            odds = provider.get_sample_odds()
        return provider.format_odds_comparison(odds)
    except:
        return provider.format_odds_comparison(provider.get_sample_odds())

def get_sample_odds() -> List[Dict]:
    """Get sample odds data"""
    provider = get_odds_provider()
    return provider.format_odds_comparison(provider.get_sample_odds())
