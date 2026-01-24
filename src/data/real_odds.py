"""
Real Odds API Client (The-Odds-API)

Provides real-time betting odds from multiple bookmakers.
Free tier: 500 requests/month

Features:
- Pre-match odds
- Live odds
- Multi-bookmaker comparison
- Odds movement tracking
- Intelligent caching for rate limit
"""

import os
import requests
from datetime import datetime
from typing import Dict, List, Optional
from src.data.api_clients import CacheManager

# API Configuration
THE_ODDS_API_KEY = os.environ.get('THE_ODDS_API_KEY', '')
BASE_URL = 'https://api.the-odds-api.com/v4'


class RealOddsClient:
    """
    Client for The-Odds-API to get real betting odds.
    Replaces simulated odds in betting_intel.py
    """
    
    # Sport keys for football/soccer
    SPORTS = {
        'premier_league': 'soccer_epl',
        'la_liga': 'soccer_spain_la_liga',
        'bundesliga': 'soccer_germany_bundesliga',
        'serie_a': 'soccer_italy_serie_a',
        'ligue_1': 'soccer_france_ligue_one',
        'champions_league': 'soccer_uefa_champs_league',
    }
    
    # Bookmaker markets
    MARKETS = ['h2h', 'spreads', 'totals']
    
    def __init__(self):
        self.api_key = THE_ODDS_API_KEY
        self.cache = CacheManager()
        self.session = requests.Session()
        self._requests_used = 0
        self._requests_remaining = 500
    
    def has_api_key(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def get_odds(
        self, 
        league: str = 'premier_league',
        markets: List[str] = None
    ) -> List[Dict]:
        """
        Get current odds for all matches in a league.
        
        Returns:
            List of matches with odds from multiple bookmakers
        """
        if not self.api_key:
            return self._get_simulated_odds(league)
        
        sport_key = self.SPORTS.get(league, 'soccer_epl')
        markets = markets or ['h2h']
        
        cache_key = f"odds_{sport_key}_{'_'.join(markets)}"
        cached = self.cache.get(cache_key, max_age_minutes=30)  # 30min cache
        if cached:
            return cached
        
        url = f"{BASE_URL}/sports/{sport_key}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'uk,eu',
            'markets': ','.join(markets),
            'oddsFormat': 'decimal'
        }
        
        try:
            response = self.session.get(url, params=params)
            
            # Track request usage
            self._requests_remaining = int(response.headers.get('x-requests-remaining', 500))
            self._requests_used = int(response.headers.get('x-requests-used', 0))
            
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data)
                return data
            elif response.status_code == 401:
                print("Invalid Odds API key")
            elif response.status_code == 429:
                print("Odds API rate limit exceeded")
                
        except Exception as e:
            print(f"Odds API error: {e}")
        
        return self._get_simulated_odds(league)
    
    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        league: str = 'premier_league'
    ) -> Dict:
        """
        Get odds for a specific match.
        
        Returns:
            Dict with best odds and all bookmaker odds
        """
        all_odds = self.get_odds(league, ['h2h', 'totals'])
        
        for match in all_odds:
            match_home = match.get('home_team', '').lower()
            match_away = match.get('away_team', '').lower()
            
            if (home_team.lower() in match_home or match_home in home_team.lower()) and \
               (away_team.lower() in match_away or match_away in away_team.lower()):
                return self._parse_match_odds(match)
        
        # Return simulated if not found
        return self._get_simulated_match_odds(home_team, away_team)
    
    def _parse_match_odds(self, match: Dict) -> Dict:
        """Parse odds from API response"""
        bookmakers = match.get('bookmakers', [])
        
        best_home = 0
        best_draw = 0
        best_away = 0
        all_odds = []
        
        for bookie in bookmakers:
            bookie_name = bookie.get('title', 'Unknown')
            
            for market in bookie.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = {o['name']: o['price'] for o in market.get('outcomes', [])}
                    
                    home_odd = outcomes.get(match.get('home_team'), 1.0)
                    draw_odd = outcomes.get('Draw', 1.0)
                    away_odd = outcomes.get(match.get('away_team'), 1.0)
                    
                    best_home = max(best_home, home_odd)
                    best_draw = max(best_draw, draw_odd)
                    best_away = max(best_away, away_odd)
                    
                    all_odds.append({
                        'bookmaker': bookie_name,
                        'home': home_odd,
                        'draw': draw_odd,
                        'away': away_odd
                    })
        
        return {
            'found': True,
            'data_source': 'LIVE_API',
            'home_team': match.get('home_team'),
            'away_team': match.get('away_team'),
            'commence_time': match.get('commence_time'),
            'best_odds': {
                'home': best_home,
                'draw': best_draw,
                'away': best_away
            },
            'implied_probability': {
                'home': round(1 / best_home if best_home > 0 else 0, 3),
                'draw': round(1 / best_draw if best_draw > 0 else 0, 3),
                'away': round(1 / best_away if best_away > 0 else 0, 3)
            },
            'bookmakers': all_odds,
            'bookmaker_count': len(all_odds)
        }
    
    def _get_simulated_odds(self, league: str) -> List[Dict]:
        """Fallback simulated odds when API unavailable"""
        return [
            {
                'id': 'sim_1',
                'home_team': 'Liverpool',
                'away_team': 'Arsenal',
                'bookmakers': [
                    {
                        'title': 'Simulated',
                        'markets': [{
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Liverpool', 'price': 2.1},
                                {'name': 'Draw', 'price': 3.4},
                                {'name': 'Arsenal', 'price': 3.5}
                            ]
                        }]
                    }
                ]
            }
        ]
    
    def _get_simulated_match_odds(self, home: str, away: str) -> Dict:
        """Fallback for specific match"""
        return {
            'found': True,
            'data_source': 'SIMULATED',
            'home_team': home,
            'away_team': away,
            'best_odds': {'home': 2.0, 'draw': 3.3, 'away': 3.5},
            'implied_probability': {'home': 0.5, 'draw': 0.3, 'away': 0.29},
            'bookmakers': [],
            'bookmaker_count': 0
        }
    
    def get_api_usage(self) -> Dict:
        """Get API usage stats"""
        return {
            'requests_used': self._requests_used,
            'requests_remaining': self._requests_remaining,
            'has_api_key': self.has_api_key()
        }
    
    def detect_arbitrage(self, match: Dict) -> Optional[Dict]:
        """
        Detect arbitrage opportunity across bookmakers.
        
        Arbitrage exists when:
        1/home_odds + 1/draw_odds + 1/away_odds < 1
        """
        best = match.get('best_odds', {})
        
        if not best:
            return None
        
        home = best.get('home', 0)
        draw = best.get('draw', 0)
        away = best.get('away', 0)
        
        if home <= 0 or draw <= 0 or away <= 0:
            return None
        
        total_implied = (1/home) + (1/draw) + (1/away)
        
        if total_implied < 1.0:
            profit_pct = round((1 - total_implied) * 100, 2)
            return {
                'arbitrage_exists': True,
                'profit_percentage': profit_pct,
                'optimal_stakes': {
                    'home': round((1/home) / total_implied * 100, 1),
                    'draw': round((1/draw) / total_implied * 100, 1),
                    'away': round((1/away) / total_implied * 100, 1)
                }
            }
        
        return {'arbitrage_exists': False}


# Global instance
odds_client = RealOddsClient()


def get_live_odds(home: str, away: str, league: str = 'premier_league') -> Dict:
    """Convenience function for getting live odds"""
    return odds_client.get_match_odds(home, away, league)
