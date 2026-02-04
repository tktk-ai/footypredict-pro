"""
Oddspedia API Client
Fetches betting odds from Oddspedia's public JSON endpoints

Oddspedia provides comprehensive odds comparison from multiple bookmakers.
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

# Oddspedia API endpoints (discovered from browser network inspection)
ODDSPEDIA_API_BASE = "https://api.oddspedia.com/api/v1"


class OddspediaAPI:
    """Client for Oddspedia API to fetch betting odds."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://oddspedia.com/',
            'Origin': 'https://oddspedia.com'
        })
        
    def get_matches(
        self,
        sport: str = "football",
        date: str = None,
        competition: str = None
    ) -> List[Dict]:
        """Get matches with odds for a date.
        
        Args:
            sport: Sport name (football, basketball, etc.)
            date: Date in YYYY-MM-DD format (default: today)
            competition: Optional competition filter
            
        Returns:
            List of matches with odds
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        # Try the matches endpoint
        url = f"{ODDSPEDIA_API_BASE}/getMatches"
        
        params = {
            'sport': sport,
            'startDate': date,
            'endDate': date,
            'country': 'all',
            'status': 'upcoming',
            'sortBy': 'dateAsc'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 403:
                logger.warning("Oddspedia API returned 403 - trying alternative endpoint")
                return self._get_matches_alternative(sport, date)
                
            response.raise_for_status()
            data = response.json()
            
            matches = data.get('data', data.get('matches', []))
            return [self._normalize_match(m) for m in matches]
            
        except Exception as e:
            logger.error(f"Oddspedia API error: {e}")
            return self._get_matches_alternative(sport, date)
    
    def _get_matches_alternative(self, sport: str, date: str) -> List[Dict]:
        """Try alternative Oddspedia endpoint format."""
        try:
            # Try GraphQL-style endpoint
            url = "https://api.oddspedia.com/odds-feed/v1/football"
            
            params = {
                'status': 'prematch',
                'sort': 'startTime',
                'locale': 'en',
                'format': 'json'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Alternative endpoint returned {response.status_code}")
                return []
                
            data = response.json()
            events = data.get('events', data.get('data', []))
            
            return [self._normalize_match(e) for e in events]
            
        except Exception as e:
            logger.error(f"Oddspedia alternative endpoint error: {e}")
            return []
    
    def get_match_odds(self, match_id: str) -> Dict:
        """Get detailed odds for a specific match.
        
        Args:
            match_id: Oddspedia match ID
            
        Returns:
            Dictionary with all available odds from different bookmakers
        """
        url = f"{ODDSPEDIA_API_BASE}/getMatchOdds"
        
        params = {'matchId': match_id}
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            return self._extract_odds(data)
            
        except Exception as e:
            logger.error(f"Error fetching match odds: {e}")
            return {}
    
    def _normalize_match(self, match: Dict) -> Dict:
        """Normalize Oddspedia match to our standard format."""
        try:
            # Try different key formats
            home_team = (match.get('homeTeam', {}).get('name') or 
                        match.get('home_team') or 
                        match.get('homeTeamName', ''))
            away_team = (match.get('awayTeam', {}).get('name') or 
                        match.get('away_team') or 
                        match.get('awayTeamName', ''))
            
            # Parse time
            start_time = match.get('startTime', match.get('start_time', match.get('date', '')))
            match_date = ''
            match_time = ''
            
            if start_time:
                try:
                    if isinstance(start_time, (int, float)):
                        dt = datetime.fromtimestamp(start_time)
                    else:
                        dt = datetime.fromisoformat(str(start_time).replace('Z', '+00:00'))
                    match_date = dt.strftime('%Y-%m-%d')
                    match_time = dt.strftime('%H:%M')
                except:
                    pass
            
            # Extract odds if available in the match data
            odds = {}
            if 'odds' in match:
                odds = self._extract_odds(match['odds'])
            elif 'markets' in match:
                odds = self._extract_odds_from_markets(match['markets'])
            
            return {
                'id': match.get('id', match.get('matchId', '')),
                'home_team': home_team,
                'away_team': away_team,
                'date': match_date,
                'time': match_time,
                'league': match.get('competition', {}).get('name', match.get('league', '')),
                'odds': odds
            }
        except Exception as e:
            logger.error(f"Error normalizing match: {e}")
            return {}
    
    def _extract_odds(self, odds_data: Dict) -> Dict:
        """Extract standardized odds from Oddspedia format."""
        odds = {
            'home': 0, 'draw': 0, 'away': 0,
            'over_05': 0, 'under_05': 0,
            'over_15': 0, 'under_15': 0,
            'over_25': 0, 'under_25': 0,
            'over_35': 0, 'under_35': 0,
            'btts_yes': 0, 'btts_no': 0,
            'dc_1x': 0, 'dc_x2': 0, 'dc_12': 0,
        }
        
        if not odds_data:
            return odds
            
        # Try to extract 1X2 odds
        if '1x2' in odds_data:
            market = odds_data['1x2']
            odds['home'] = self._get_best_odd(market, '1')
            odds['draw'] = self._get_best_odd(market, 'x')
            odds['away'] = self._get_best_odd(market, '2')
        
        # Over/Under
        if 'totals' in odds_data:
            totals = odds_data['totals']
            for key, value in totals.items():
                if '0.5' in key:
                    if 'over' in key.lower():
                        odds['over_05'] = self._get_best_odd_value(value)
                    else:
                        odds['under_05'] = self._get_best_odd_value(value)
                elif '1.5' in key:
                    if 'over' in key.lower():
                        odds['over_15'] = self._get_best_odd_value(value)
                    else:
                        odds['under_15'] = self._get_best_odd_value(value)
                elif '2.5' in key:
                    if 'over' in key.lower():
                        odds['over_25'] = self._get_best_odd_value(value)
                    else:
                        odds['under_25'] = self._get_best_odd_value(value)
                elif '3.5' in key:
                    if 'over' in key.lower():
                        odds['over_35'] = self._get_best_odd_value(value)
                    else:
                        odds['under_35'] = self._get_best_odd_value(value)
        
        # BTTS
        if 'btts' in odds_data:
            btts = odds_data['btts']
            odds['btts_yes'] = self._get_best_odd(btts, 'yes')
            odds['btts_no'] = self._get_best_odd(btts, 'no')
        
        return odds
    
    def _extract_odds_from_markets(self, markets: List) -> Dict:
        """Extract odds from markets array format."""
        odds = {
            'home': 0, 'draw': 0, 'away': 0,
            'over_05': 0, 'under_05': 0,
            'over_15': 0, 'under_15': 0,
            'over_25': 0, 'under_25': 0,
            'over_35': 0, 'under_35': 0,
            'btts_yes': 0, 'btts_no': 0,
        }
        
        for market in markets:
            market_type = market.get('type', market.get('name', '')).lower()
            selections = market.get('selections', market.get('outcomes', []))
            
            if market_type in ['1x2', 'match_winner', 'fulltime']:
                for sel in selections:
                    name = sel.get('name', '').lower()
                    price = sel.get('price', sel.get('odds', 0))
                    
                    if '1' in name or 'home' in name:
                        odds['home'] = max(odds['home'], price)
                    elif 'x' in name or 'draw' in name:
                        odds['draw'] = max(odds['draw'], price)
                    elif '2' in name or 'away' in name:
                        odds['away'] = max(odds['away'], price)
        
        return odds
    
    def _get_best_odd(self, market_data: Dict, outcome: str) -> float:
        """Get the best (highest) odd for an outcome."""
        if isinstance(market_data, dict):
            value = market_data.get(outcome, 0)
            if isinstance(value, dict):
                # Multiple bookmakers
                return max(value.values()) if value else 0
            return float(value) if value else 0
        return 0
    
    def _get_best_odd_value(self, value) -> float:
        """Get best odd from value (could be number or dict)."""
        if isinstance(value, dict):
            return max(value.values()) if value else 0
        return float(value) if value else 0


# Global instance
_api: Optional[OddspediaAPI] = None


def get_api() -> OddspediaAPI:
    """Get or create Oddspedia API client."""
    global _api
    if _api is None:
        _api = OddspediaAPI()
    return _api


def get_football_matches(date: str = None) -> List[Dict]:
    """Get football matches with odds for a date."""
    api = get_api()
    return api.get_matches("football", date)


def get_match_odds(match_id: str) -> Dict:
    """Get detailed odds for a match."""
    api = get_api()
    return api.get_match_odds(match_id)
