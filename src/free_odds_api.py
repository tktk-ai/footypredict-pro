"""
Free Odds API Client

Provides access to real-time betting odds from multiple free APIs:
- The Odds API (limited free tier)
- API-Football (free plan, 100 requests/day)
- API-Sports (100 requests/day per sport)

Used for:
- Value betting calculations
- Market sentiment analysis
- Odds movement tracking
- Arbitrage detection
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchOdds:
    """Odds for a single match"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    commence_time: str
    
    # 1X2 Odds
    home_win: float
    draw: float
    away_win: float
    
    # Goals odds (optional)
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    
    # Bookmaker info
    bookmaker: str = "average"
    last_update: Optional[str] = None
    
    # Odds movement
    opening_home: Optional[float] = None
    opening_draw: Optional[float] = None
    opening_away: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def implied_home_prob(self) -> float:
        """Implied probability for home win"""
        return 1 / self.home_win if self.home_win > 1 else 0
    
    @property
    def implied_draw_prob(self) -> float:
        """Implied probability for draw"""
        return 1 / self.draw if self.draw > 1 else 0
    
    @property
    def implied_away_prob(self) -> float:
        """Implied probability for away win"""
        return 1 / self.away_win if self.away_win > 1 else 0
    
    @property
    def overround(self) -> float:
        """Total overround (bookmaker margin)"""
        return self.implied_home_prob + self.implied_draw_prob + self.implied_away_prob
    
    @property
    def margin(self) -> float:
        """Bookmaker margin percentage"""
        return (self.overround - 1) * 100
    
    @property
    def odds_movement(self) -> Dict[str, str]:
        """Check odds movement direction"""
        movements = {}
        if self.opening_home:
            if self.home_win < self.opening_home:
                movements['home'] = 'shortening'  # More favored
            elif self.home_win > self.opening_home:
                movements['home'] = 'drifting'  # Less favored
            else:
                movements['home'] = 'stable'
        return movements


class OddsAPIClient:
    """
    Client for The Odds API (https://the-odds-api.com)
    
    Free tier: 500 requests/month
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY', '')
        self.requests_used = 0
        self.requests_remaining = 500
    
    def get_sports(self) -> List[Dict]:
        """Get list of available sports"""
        if not self.api_key:
            return []
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports",
                params={'apiKey': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                self._update_usage(response.headers)
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return []
    
    def get_odds(
        self,
        sport: str = 'soccer_epl',
        regions: str = 'uk,eu',
        markets: str = 'h2h',
        odds_format: str = 'decimal'
    ) -> List[MatchOdds]:
        """
        Get odds for a sport.
        
        Sports examples:
        - soccer_epl: English Premier League
        - soccer_spain_la_liga: La Liga
        - soccer_germany_bundesliga: Bundesliga
        - soccer_italy_serie_a: Serie A
        """
        if not self.api_key:
            return self._get_sample_odds()
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport}/odds",
                params={
                    'apiKey': self.api_key,
                    'regions': regions,
                    'markets': markets,
                    'oddsFormat': odds_format
                },
                timeout=15
            )
            
            if response.status_code == 200:
                self._update_usage(response.headers)
                return self._parse_odds(response.json())
            else:
                logger.warning(f"Odds API returned {response.status_code}")
                return self._get_sample_odds()
        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return self._get_sample_odds()
    
    def _parse_odds(self, data: List[Dict]) -> List[MatchOdds]:
        """Parse API response into MatchOdds objects"""
        odds_list = []
        
        for event in data:
            try:
                # Get best odds across bookmakers
                home_odds = []
                draw_odds = []
                away_odds = []
                
                for bookmaker in event.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = {o['name']: o['price'] for o in market['outcomes']}
                            home_odds.append(outcomes.get(event['home_team'], 2.0))
                            draw_odds.append(outcomes.get('Draw', 3.5))
                            away_odds.append(outcomes.get(event['away_team'], 3.0))
                
                if home_odds:
                    odds_list.append(MatchOdds(
                        match_id=event['id'],
                        home_team=event['home_team'],
                        away_team=event['away_team'],
                        league=event.get('sport_key', 'unknown'),
                        commence_time=event['commence_time'],
                        home_win=max(home_odds),  # Best odds for bettor
                        draw=max(draw_odds),
                        away_win=max(away_odds),
                        bookmaker='best',
                        last_update=datetime.now().isoformat()
                    ))
            except Exception as e:
                logger.warning(f"Error parsing event: {e}")
                continue
        
        return odds_list
    
    def _update_usage(self, headers: Dict):
        """Update API usage from response headers"""
        self.requests_used = int(headers.get('x-requests-used', 0))
        self.requests_remaining = int(headers.get('x-requests-remaining', 500))
    
    def _get_sample_odds(self) -> List[MatchOdds]:
        """Return sample odds when API not available"""
        now = datetime.now()
        
        return [
            MatchOdds(
                match_id="sample_1",
                home_team="Manchester City",
                away_team="Arsenal",
                league="Premier League",
                commence_time=(now + timedelta(days=1)).isoformat(),
                home_win=1.75,
                draw=3.80,
                away_win=4.20,
                over_2_5=1.70,
                under_2_5=2.10,
                btts_yes=1.65,
                btts_no=2.20,
                bookmaker="sample"
            ),
            MatchOdds(
                match_id="sample_2",
                home_team="Liverpool",
                away_team="Chelsea",
                league="Premier League",
                commence_time=(now + timedelta(days=1, hours=3)).isoformat(),
                home_win=1.90,
                draw=3.50,
                away_win=4.00,
                over_2_5=1.80,
                under_2_5=2.00,
                btts_yes=1.75,
                btts_no=2.05,
                bookmaker="sample"
            ),
            MatchOdds(
                match_id="sample_3",
                home_team="Real Madrid",
                away_team="Barcelona",
                league="La Liga",
                commence_time=(now + timedelta(days=2)).isoformat(),
                home_win=2.20,
                draw=3.40,
                away_win=3.10,
                over_2_5=1.75,
                under_2_5=2.05,
                btts_yes=1.60,
                btts_no=2.25,
                bookmaker="sample"
            ),
            MatchOdds(
                match_id="sample_4",
                home_team="Bayern Munich",
                away_team="Borussia Dortmund",
                league="Bundesliga",
                commence_time=(now + timedelta(days=2, hours=2)).isoformat(),
                home_win=1.55,
                draw=4.20,
                away_win=5.50,
                over_2_5=1.50,
                under_2_5=2.50,
                btts_yes=1.55,
                btts_no=2.35,
                bookmaker="sample"
            ),
        ]


class APISportsClient:
    """
    Client for API-Sports (https://api-sports.io)
    
    Free tier: 100 requests/day per sport
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('API_SPORTS_KEY', '')
        self.daily_requests = 0
        self.daily_limit = 100
    
    def get_odds(
        self,
        fixture_id: Optional[int] = None,
        league: Optional[int] = None,
        season: int = 2025,
        bookmaker: int = 8  # Bet365
    ) -> List[MatchOdds]:
        """Get odds from API-Sports"""
        if not self.api_key:
            return []
        
        try:
            params = {'season': season, 'bookmaker': bookmaker}
            if fixture_id:
                params['fixture'] = fixture_id
            if league:
                params['league'] = league
            
            response = requests.get(
                f"{self.BASE_URL}/odds",
                headers={'x-apisports-key': self.api_key},
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                self.daily_requests = int(data.get('results', 0))
                return self._parse_response(data)
            return []
        except Exception as e:
            logger.error(f"API-Sports error: {e}")
            return []
    
    def get_predictions(
        self,
        fixture_id: int
    ) -> Optional[Dict]:
        """Get predictions from API-Sports (they have their own ML)"""
        if not self.api_key:
            return None
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/predictions",
                headers={'x-apisports-key': self.api_key},
                params={'fixture': fixture_id},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('response'):
                    return data['response'][0]
            return None
        except Exception as e:
            logger.error(f"API-Sports predictions error: {e}")
            return None
    
    def _parse_response(self, data: Dict) -> List[MatchOdds]:
        """Parse API-Sports odds response"""
        odds_list = []
        
        for entry in data.get('response', []):
            try:
                fixture = entry.get('fixture', {})
                league_info = entry.get('league', {})
                
                # Find 1X2 market
                for bookmaker in entry.get('bookmakers', []):
                    for bet in bookmaker.get('bets', []):
                        if bet.get('name') == 'Match Winner':
                            values = {v['value']: float(v['odd']) for v in bet.get('values', [])}
                            
                            odds_list.append(MatchOdds(
                                match_id=str(fixture.get('id', '')),
                                home_team=entry.get('teams', {}).get('home', {}).get('name', 'Home'),
                                away_team=entry.get('teams', {}).get('away', {}).get('name', 'Away'),
                                league=league_info.get('name', 'Unknown'),
                                commence_time=fixture.get('date', ''),
                                home_win=values.get('Home', 2.0),
                                draw=values.get('Draw', 3.5),
                                away_win=values.get('Away', 3.0),
                                bookmaker=bookmaker.get('name', 'unknown'),
                                last_update=datetime.now().isoformat()
                            ))
                            break  # Only need first bookmaker
            except Exception as e:
                logger.warning(f"Error parsing API-Sports entry: {e}")
                continue
        
        return odds_list


class UnifiedOddsClient:
    """
    Unified client that tries multiple odds sources.
    
    Falls back gracefully if APIs are unavailable.
    """
    
    def __init__(self):
        self.odds_api = OddsAPIClient()
        self.api_sports = APISportsClient()
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_odds(
        self,
        league: str = 'premier_league',
        force_refresh: bool = False
    ) -> List[MatchOdds]:
        """
        Get odds from best available source.
        
        Priority:
        1. The Odds API (if key available)
        2. API-Sports (if key available)
        3. Sample odds (always available)
        """
        cache_key = f"odds_{league}"
        
        # Check cache
        if not force_refresh and cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['odds']
        
        odds = []
        
        # Map league names to API formats
        league_mapping = {
            'premier_league': ('soccer_epl', 39),
            'la_liga': ('soccer_spain_la_liga', 140),
            'bundesliga': ('soccer_germany_bundesliga', 78),
            'serie_a': ('soccer_italy_serie_a', 135),
            'ligue_1': ('soccer_france_ligue_one', 61),
            'champions_league': ('soccer_uefa_champs_league', 2)
        }
        
        api_sport, api_sports_id = league_mapping.get(league, ('soccer_epl', 39))
        
        # Try The Odds API first
        if self.odds_api.api_key:
            odds = self.odds_api.get_odds(sport=api_sport)
        
        # Try API-Sports if no results
        if not odds and self.api_sports.api_key:
            odds = self.api_sports.get_odds(league=api_sports_id)
        
        # Fall back to sample odds
        if not odds:
            odds = self.odds_api._get_sample_odds()
        
        # Cache results
        self.cache[cache_key] = {
            'odds': odds,
            'timestamp': time.time()
        }
        
        return odds
    
    def get_match_odds(
        self,
        home_team: str,
        away_team: str
    ) -> Optional[MatchOdds]:
        """Find odds for a specific match"""
        # Check all cached leagues
        for league in ['premier_league', 'la_liga', 'bundesliga', 'serie_a']:
            odds = self.get_odds(league)
            
            for match in odds:
                if (home_team.lower() in match.home_team.lower() or 
                    match.home_team.lower() in home_team.lower()):
                    if (away_team.lower() in match.away_team.lower() or 
                        match.away_team.lower() in away_team.lower()):
                        return match
        
        return None
    
    def calculate_value(
        self,
        our_probability: float,
        bookmaker_odds: float
    ) -> Dict:
        """
        Calculate value bet metrics.
        
        Args:
            our_probability: Our predicted probability (0-1)
            bookmaker_odds: Decimal odds from bookmaker
            
        Returns:
            Dict with value metrics
        """
        implied_prob = 1 / bookmaker_odds if bookmaker_odds > 1 else 0
        edge = our_probability - implied_prob
        
        # Expected value per unit stake
        ev = (our_probability * (bookmaker_odds - 1)) - (1 - our_probability)
        
        # Kelly criterion stake recommendation
        if edge > 0 and bookmaker_odds > 1:
            kelly = edge / (bookmaker_odds - 1)
        else:
            kelly = 0
        
        return {
            'our_probability': round(our_probability * 100, 1),
            'implied_probability': round(implied_prob * 100, 1),
            'edge': round(edge * 100, 1),
            'expected_value': round(ev, 4),
            'kelly_stake': round(min(kelly * 100, 25), 2),  # Cap at 25%
            'is_value': edge > 0.05,  # 5% edge minimum
            'value_rating': 'high' if edge > 0.15 else 'medium' if edge > 0.10 else 'low' if edge > 0.05 else 'none'
        }


# Global instance
unified_odds = UnifiedOddsClient()


def fetch_live_odds(league: str = 'premier_league') -> List[Dict]:
    """Fetch live odds for a league"""
    odds = unified_odds.get_odds(league)
    return [o.to_dict() for o in odds]


def get_match_odds(home_team: str, away_team: str) -> Optional[Dict]:
    """Get odds for a specific match"""
    odds = unified_odds.get_match_odds(home_team, away_team)
    return odds.to_dict() if odds else None


def calculate_value_bet(our_prob: float, odds: float) -> Dict:
    """Calculate value bet metrics"""
    return unified_odds.calculate_value(our_prob, odds)
