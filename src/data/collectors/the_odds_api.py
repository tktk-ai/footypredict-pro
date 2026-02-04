"""
The Odds API Client
Fetches real betting odds from The Odds API (free tier: 500 credits/month)

API Documentation: https://the-odds-api.com/
Free tier includes: All sports, most bookmakers, all betting markets
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

# Get API key from environment or use default (demo)
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', '')

# API Base URL
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Sport keys for soccer leagues
SOCCER_SPORT_KEYS = [
    'soccer_epl',           # English Premier League
    'soccer_spain_la_liga', # La Liga
    'soccer_germany_bundesliga',  # Bundesliga
    'soccer_italy_serie_a', # Serie A
    'soccer_france_ligue_one',  # Ligue 1
    'soccer_uefa_champs_league',  # Champions League
    'soccer_uefa_europa_league',  # Europa League
    'soccer_england_league1',  # EFL League One
    'soccer_england_efl_cup',  # EFL Cup
    'soccer_fa_cup',  # FA Cup
]


class TheOddsAPI:
    """Client for The Odds API to fetch real betting odds."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or THE_ODDS_API_KEY
        self.session = requests.Session()
        self.remaining_requests = None
        self.used_requests = None
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to The Odds API."""
        if not self.api_key:
            logger.warning("No API key configured for The Odds API")
            return None
            
        params = params or {}
        params['apiKey'] = self.api_key
        
        url = f"{THE_ODDS_API_BASE}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            # Track API usage from headers
            self.remaining_requests = response.headers.get('x-requests-remaining')
            self.used_requests = response.headers.get('x-requests-used')
            
            if response.status_code == 401:
                logger.error("Invalid API key for The Odds API")
                return None
            elif response.status_code == 429:
                logger.error("Rate limit exceeded for The Odds API")
                return None
                
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"The Odds API request failed: {e}")
            return None
    
    def get_sports(self) -> List[Dict]:
        """Get list of available sports and their active status."""
        return self._make_request("/sports") or []
    
    def get_odds(
        self,
        sport_key: str,
        regions: str = "uk,eu",
        markets: str = "h2h,totals",
        odds_format: str = "decimal"
    ) -> List[Dict]:
        """Get odds for upcoming games in a sport.
        
        Args:
            sport_key: Sport key (e.g., 'soccer_epl')
            regions: Comma-separated regions for bookmakers (uk, us, eu, au)
            markets: Comma-separated markets (h2h, spreads, totals)
            odds_format: 'american' or 'decimal'
            
        Returns:
            List of events with odds
        """
        params = {
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format
        }
        
        return self._make_request(f"/sports/{sport_key}/odds", params) or []
    
    def get_scores(self, sport_key: str, days_from: int = 1) -> List[Dict]:
        """Get scores for recently completed games.
        
        Args:
            sport_key: Sport key
            days_from: Number of days back to look for completed games
        """
        params = {'daysFrom': days_from}
        return self._make_request(f"/sports/{sport_key}/scores", params) or []
    
    def get_soccer_odds(self, leagues: List[str] = None) -> List[Dict]:
        """Get odds for soccer matches from specified leagues.
        
        Args:
            leagues: List of league keys (defaults to all supported)
            
        Returns:
            List of matches with normalized odds structure
        """
        leagues = leagues or SOCCER_SPORT_KEYS
        all_matches = []
        
        for sport_key in leagues:
            try:
                events = self.get_odds(sport_key, markets="h2h,totals")
                
                for event in events:
                    match = self._normalize_event(event, sport_key)
                    if match:
                        all_matches.append(match)
                        
            except Exception as e:
                logger.error(f"Error fetching odds for {sport_key}: {e}")
                continue
                
        return all_matches
    
    def _normalize_event(self, event: Dict, sport_key: str) -> Optional[Dict]:
        """Normalize The Odds API event to our standard format."""
        try:
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            commence_time = event.get('commence_time', '')
            
            # Parse commence time
            match_date = ''
            match_time = ''
            if commence_time:
                try:
                    dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    match_date = dt.strftime('%Y-%m-%d')
                    match_time = dt.strftime('%H:%M')
                except:
                    pass
            
            # Extract odds from bookmakers
            odds = self._extract_best_odds(event.get('bookmakers', []))
            
            return {
                'id': event.get('id', ''),
                'home_team': home_team,
                'away_team': away_team,
                'date': match_date,
                'time': match_time,
                'league': sport_key,
                'odds': odds
            }
        except Exception as e:
            logger.error(f"Error normalizing event: {e}")
            return None
    
    def _extract_best_odds(self, bookmakers: List[Dict]) -> Dict:
        """Extract best odds from all bookmakers."""
        odds = {
            'home': 0, 'draw': 0, 'away': 0,
            'over_05': 0, 'under_05': 0,
            'over_15': 0, 'under_15': 0,
            'over_25': 0, 'under_25': 0,
            'over_35': 0, 'under_35': 0,
            'btts_yes': 0, 'btts_no': 0,
        }
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                outcomes = market.get('outcomes', [])
                
                # 1X2 (h2h) market
                if market_key == 'h2h':
                    for outcome in outcomes:
                        name = outcome.get('name', '')
                        price = outcome.get('price', 0)
                        
                        if name == bookmaker.get('home_team', ''):
                            odds['home'] = max(odds['home'], price)
                        elif name == bookmaker.get('away_team', ''):
                            odds['away'] = max(odds['away'], price)
                        elif name.lower() == 'draw':
                            odds['draw'] = max(odds['draw'], price)
                
                # Totals market (Over/Under)
                elif market_key == 'totals':
                    for outcome in outcomes:
                        name = outcome.get('name', '').lower()
                        point = outcome.get('point', 0)
                        price = outcome.get('price', 0)
                        
                        if point == 0.5:
                            if 'over' in name:
                                odds['over_05'] = max(odds['over_05'], price)
                            else:
                                odds['under_05'] = max(odds['under_05'], price)
                        elif point == 1.5:
                            if 'over' in name:
                                odds['over_15'] = max(odds['over_15'], price)
                            else:
                                odds['under_15'] = max(odds['under_15'], price)
                        elif point == 2.5:
                            if 'over' in name:
                                odds['over_25'] = max(odds['over_25'], price)
                            else:
                                odds['under_25'] = max(odds['under_25'], price)
                        elif point == 3.5:
                            if 'over' in name:
                                odds['over_35'] = max(odds['over_35'], price)
                            else:
                                odds['under_35'] = max(odds['under_35'], price)
        
        return odds
    
    def get_api_usage(self) -> Dict:
        """Get current API usage stats."""
        return {
            'remaining': self.remaining_requests,
            'used': self.used_requests
        }


# Global instance
_api: Optional[TheOddsAPI] = None


def get_api(api_key: str = None) -> TheOddsAPI:
    """Get or create The Odds API client."""
    global _api
    if _api is None or api_key:
        _api = TheOddsAPI(api_key)
    return _api


def get_soccer_odds(leagues: List[str] = None) -> List[Dict]:
    """Get soccer odds from The Odds API."""
    api = get_api()
    return api.get_soccer_odds(leagues)


# Map our league names to The Odds API sport keys
LEAGUE_TO_SPORT_KEY = {
    'premier_league': 'soccer_epl',
    'la_liga': 'soccer_spain_la_liga',
    'bundesliga': 'soccer_germany_bundesliga',
    'serie_a': 'soccer_italy_serie_a',
    'ligue_1': 'soccer_france_ligue_one',
    'champions_league': 'soccer_uefa_champs_league',
    'europa_league': 'soccer_uefa_europa_league',
}


def get_odds_for_league(league_name: str) -> List[Dict]:
    """Get odds for a specific league by name."""
    sport_key = LEAGUE_TO_SPORT_KEY.get(league_name.lower().replace(' ', '_'))
    if sport_key:
        api = get_api()
        return api.get_odds(sport_key)
    return []
