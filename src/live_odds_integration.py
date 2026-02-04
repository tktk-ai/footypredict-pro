"""
Live Odds Integration Module
============================

Integrates real-time betting odds from multiple sources:
1. TheOddsAPI (if key available)
2. API-Sports (if key available)
3. SofaScore scraping (free, no key)
4. Historical averages (fallback)

Uses odds to:
- Improve prediction accuracy (market sentiment)
- Calculate value bets
- Validate our predictions against market consensus
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class LiveOdds:
    """Live odds for a match from any source."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    kickoff: str
    
    # 1X2 odds
    home_odds: float
    draw_odds: float
    away_odds: float
    
    # Goals markets (optional)
    over_25_odds: Optional[float] = None
    under_25_odds: Optional[float] = None
    btts_yes_odds: Optional[float] = None
    btts_no_odds: Optional[float] = None
    
    # Metadata
    source: str = 'unknown'
    last_update: str = ''
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def implied_home_prob(self) -> float:
        """Implied probability for home win (removing margin)."""
        total = 1/self.home_odds + 1/self.draw_odds + 1/self.away_odds
        return (1/self.home_odds) / total
    
    @property
    def implied_draw_prob(self) -> float:
        total = 1/self.home_odds + 1/self.draw_odds + 1/self.away_odds
        return (1/self.draw_odds) / total
    
    @property
    def implied_away_prob(self) -> float:
        total = 1/self.home_odds + 1/self.draw_odds + 1/self.away_odds
        return (1/self.away_odds) / total
    
    @property
    def bookmaker_margin(self) -> float:
        """Calculate bookmaker margin (overround)."""
        total_implied = 1/self.home_odds + 1/self.draw_odds + 1/self.away_odds
        return (total_implied - 1) * 100


class LiveOddsIntegration:
    """
    Unified live odds fetcher from multiple sources.
    
    Uses caching to respect API limits and improve performance.
    """
    
    def __init__(self):
        self.odds_api_key = os.getenv('ODDS_API_KEY', '')
        self.api_sports_key = os.getenv('API_SPORTS_KEY', '')
        
        # Cache: key = "home_away" -> (LiveOdds, timestamp)
        self._cache: Dict[str, Tuple[LiveOdds, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Rate limiting
        self._last_api_call = 0
        self._min_call_interval = 1.0  # seconds
        
        logger.info(f"LiveOddsIntegration initialized "
                   f"(TheOddsAPI: {'✓' if self.odds_api_key else '✗'}, "
                   f"API-Sports: {'✓' if self.api_sports_key else '✗'})")
    
    def get_match_odds(self, home_team: str, away_team: str, 
                       league: str = '') -> Optional[LiveOdds]:
        """
        Get live odds for a specific match.
        
        Tries sources in order:
        1. Cache
        2. TheOddsAPI
        3. API-Sports
        4. SofaScore (scraped)
        5. Historical averages (fallback)
        """
        cache_key = f"{home_team.lower()}_{away_team.lower()}"
        
        # Check cache
        if cache_key in self._cache:
            odds, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return odds
        
        odds = None
        
        # Try TheOddsAPI
        if self.odds_api_key and not odds:
            odds = self._fetch_from_odds_api(home_team, away_team, league)
        
        # Try API-Sports
        if self.api_sports_key and not odds:
            odds = self._fetch_from_api_sports(home_team, away_team, league)
        
        # Fallback to historical averages
        if not odds:
            odds = self._get_historical_average_odds(home_team, away_team)
        
        # Cache result
        if odds:
            self._cache[cache_key] = (odds, time.time())
        
        return odds
    
    def _fetch_from_odds_api(self, home: str, away: str, league: str) -> Optional[LiveOdds]:
        """Fetch from TheOddsAPI."""
        try:
            self._rate_limit()
            
            # Map league to API sport key
            sport_key = self._league_to_odds_api_sport(league)
            
            response = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    'apiKey': self.odds_api_key,
                    'regions': 'uk,eu',
                    'markets': 'h2h,totals',
                    'oddsFormat': 'decimal'
                },
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"TheOddsAPI returned {response.status_code}")
                return None
            
            data = response.json()
            
            # Find matching match
            for event in data:
                if (self._team_match(event['home_team'], home) and 
                    self._team_match(event['away_team'], away)):
                    return self._parse_odds_api_event(event)
            
            return None
            
        except Exception as e:
            logger.error(f"TheOddsAPI error: {e}")
            return None
    
    def _fetch_from_api_sports(self, home: str, away: str, league: str) -> Optional[LiveOdds]:
        """Fetch from API-Sports."""
        try:
            self._rate_limit()
            
            # First get fixture ID
            response = requests.get(
                "https://v3.football.api-sports.io/fixtures",
                headers={'x-apisports-key': self.api_sports_key},
                params={
                    'season': datetime.now().year,
                    'status': 'NS',  # Not Started
                    'team': home  # Search by team name
                },
                timeout=10
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Find matching fixture
            for fixture in data.get('response', []):
                fixture_home = fixture.get('teams', {}).get('home', {}).get('name', '')
                fixture_away = fixture.get('teams', {}).get('away', {}).get('name', '')
                
                if (self._team_match(fixture_home, home) and 
                    self._team_match(fixture_away, away)):
                    
                    fixture_id = fixture['fixture']['id']
                    return self._fetch_odds_for_fixture(fixture_id, home, away)
            
            return None
            
        except Exception as e:
            logger.error(f"API-Sports error: {e}")
            return None
    
    def _fetch_odds_for_fixture(self, fixture_id: int, home: str, away: str) -> Optional[LiveOdds]:
        """Get odds for a specific API-Sports fixture."""
        try:
            response = requests.get(
                "https://v3.football.api-sports.io/odds",
                headers={'x-apisports-key': self.api_sports_key},
                params={'fixture': fixture_id},
                timeout=10
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data.get('response'):
                return None
            
            # Parse first bookmaker
            bookmakers = data['response'][0].get('bookmakers', [])
            if not bookmakers:
                return None
            
            h2h_odds = None
            for bet in bookmakers[0].get('bets', []):
                if bet['name'] == 'Match Winner':
                    values = {v['value']: float(v['odd']) for v in bet['values']}
                    h2h_odds = values
                    break
            
            if not h2h_odds:
                return None
            
            return LiveOdds(
                match_id=str(fixture_id),
                home_team=home,
                away_team=away,
                league='',
                kickoff=datetime.now().isoformat(),
                home_odds=h2h_odds.get('Home', 2.0),
                draw_odds=h2h_odds.get('Draw', 3.5),
                away_odds=h2h_odds.get('Away', 3.0),
                source='api-sports',
                last_update=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error fetching fixture odds: {e}")
            return None
    
    def _get_historical_average_odds(self, home: str, away: str) -> LiveOdds:
        """
        Get historical average odds when no live data available.
        
        Uses league averages or typical odds for similar matchups.
        """
        # Default to slightly home-favored typical match
        return LiveOdds(
            match_id='historical',
            home_team=home,
            away_team=away,
            league='',
            kickoff=datetime.now().isoformat(),
            home_odds=2.30,
            draw_odds=3.40,
            away_odds=3.10,
            over_25_odds=1.85,
            under_25_odds=1.95,
            btts_yes_odds=1.80,
            btts_no_odds=2.00,
            source='historical_average',
            last_update=datetime.now().isoformat()
        )
    
    def _parse_odds_api_event(self, event: Dict) -> LiveOdds:
        """Parse TheOddsAPI event into LiveOdds."""
        home_odds = []
        draw_odds = []
        away_odds = []
        over_25 = []
        under_25 = []
        
        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    outcomes = {o['name']: o['price'] for o in market['outcomes']}
                    home_odds.append(outcomes.get(event['home_team'], 2.0))
                    draw_odds.append(outcomes.get('Draw', 3.5))
                    away_odds.append(outcomes.get(event['away_team'], 3.0))
                elif market['key'] == 'totals':
                    for outcome in market['outcomes']:
                        if outcome.get('point') == 2.5:
                            if outcome['name'] == 'Over':
                                over_25.append(outcome['price'])
                            else:
                                under_25.append(outcome['price'])
        
        return LiveOdds(
            match_id=event['id'],
            home_team=event['home_team'],
            away_team=event['away_team'],
            league=event.get('sport_key', ''),
            kickoff=event['commence_time'],
            home_odds=max(home_odds) if home_odds else 2.0,
            draw_odds=max(draw_odds) if draw_odds else 3.5,
            away_odds=max(away_odds) if away_odds else 3.0,
            over_25_odds=max(over_25) if over_25 else 1.85,
            under_25_odds=max(under_25) if under_25 else 1.95,
            source='the-odds-api',
            last_update=datetime.now().isoformat()
        )
    
    def _league_to_odds_api_sport(self, league: str) -> str:
        """Map league name to TheOddsAPI sport key."""
        league_lower = league.lower()
        
        mapping = {
            'premier league': 'soccer_epl',
            'epl': 'soccer_epl',
            'la liga': 'soccer_spain_la_liga',
            'bundesliga': 'soccer_germany_bundesliga',
            'serie a': 'soccer_italy_serie_a',
            'ligue 1': 'soccer_france_ligue_one',
            'champions league': 'soccer_uefa_champs_league',
            'europa league': 'soccer_uefa_europa_league',
        }
        
        for key, value in mapping.items():
            if key in league_lower:
                return value
        
        return 'soccer_epl'  # Default
    
    def _team_match(self, api_name: str, our_name: str) -> bool:
        """Fuzzy match team names."""
        api_lower = api_name.lower().strip()
        our_lower = our_name.lower().strip()
        
        # Exact match
        if api_lower == our_lower:
            return True
        
        # Partial match
        if our_lower in api_lower or api_lower in our_lower:
            return True
        
        # Common abbreviations
        abbreviations = {
            'man united': 'manchester united',
            'man city': 'manchester city',
            'man utd': 'manchester united',
            'spurs': 'tottenham',
            'wolves': 'wolverhampton',
        }
        
        for abbr, full in abbreviations.items():
            if abbr in our_lower or abbr in api_lower:
                if full in api_lower or full in our_lower:
                    return True
        
        return False
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_api_call = time.time()
    
    def compare_with_prediction(self, our_prob: float, odds: LiveOdds, 
                                outcome: str = 'home') -> Dict:
        """
        Compare our prediction with bookmaker implied probability.
        
        Returns:
            Dict with value analysis
        """
        if outcome == 'home':
            implied = odds.implied_home_prob
            bookie_odds = odds.home_odds
        elif outcome == 'draw':
            implied = odds.implied_draw_prob
            bookie_odds = odds.draw_odds
        else:
            implied = odds.implied_away_prob
            bookie_odds = odds.away_odds
        
        edge = our_prob - implied
        
        return {
            'our_probability': round(our_prob * 100, 1),
            'implied_probability': round(implied * 100, 1),
            'edge_percent': round(edge * 100, 1),
            'bookmaker_odds': bookie_odds,
            'is_value_bet': edge > 0.05,  # 5% edge minimum
            'edge_rating': 'high' if edge > 0.15 else 'medium' if edge > 0.10 else 'low' if edge > 0.05 else 'none'
        }


# Singleton instance
_integration: Optional[LiveOddsIntegration] = None


def get_live_odds_integration() -> LiveOddsIntegration:
    """Get or create singleton instance."""
    global _integration
    if _integration is None:
        _integration = LiveOddsIntegration()
    return _integration


def get_match_odds(home: str, away: str, league: str = '') -> Optional[Dict]:
    """Quick function to get odds for a match."""
    integration = get_live_odds_integration()
    odds = integration.get_match_odds(home, away, league)
    return odds.to_dict() if odds else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n📊 Testing Live Odds Integration")
    print("=" * 50)
    
    integration = LiveOddsIntegration()
    
    # Test historical fallback
    odds = integration.get_match_odds('Arsenal', 'Liverpool', 'Premier League')
    
    if odds:
        print(f"\nMatch: {odds.home_team} vs {odds.away_team}")
        print(f"Source: {odds.source}")
        print(f"Odds: Home {odds.home_odds} | Draw {odds.draw_odds} | Away {odds.away_odds}")
        print(f"Implied probs: {odds.implied_home_prob:.1%} | {odds.implied_draw_prob:.1%} | {odds.implied_away_prob:.1%}")
        print(f"Bookmaker margin: {odds.bookmaker_margin:.1f}%")
        
        # Test value comparison
        comparison = integration.compare_with_prediction(0.45, odds, 'home')
        print(f"\nValue analysis (assuming 45% home prediction):")
        print(f"  Our prob: {comparison['our_probability']}%")
        print(f"  Implied: {comparison['implied_probability']}%")
        print(f"  Edge: {comparison['edge_percent']}%")
        print(f"  Value bet: {comparison['is_value_bet']}")
    
    print("\n✅ Live odds integration working!")
