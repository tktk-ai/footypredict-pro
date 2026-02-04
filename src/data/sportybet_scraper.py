"""
SportyBet Complete Fixture Scraper
===================================

Enhanced scraper for SportyBet's web API with:
- ALL market types (8 markets, 30+ outcomes)
- Multi-day fetching (7 days)
- Multi-country support (ng, ke, gh, tz)
- Historical results collection
- Detailed match data

Discovered API Endpoints:
- /api/ng/factsCenter/pcUpcomingEvents - List of fixtures with basic markets
- /api/ng/factsCenter/event - Detailed match with all markets
- /api/ng/factsCenter/matchResults - Past results

Market IDs:
- 1: 1X2 (Home/Draw/Away)
- 10: Double Chance
- 11: Draw No Bet
- 14: Asian Handicap
- 18: Over/Under Goals
- 26: Halftime/Fulltime
- 29: GG/NG (BTTS)
- 36: Correct Score
- 60100: Combo markets
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketOdds:
    """Complete odds structure for all SportyBet markets."""
    # 1X2
    home: float = 0.0
    draw: float = 0.0
    away: float = 0.0
    
    # Over/Under Goals
    over_05: float = 0.0
    under_05: float = 0.0
    over_15: float = 0.0
    under_15: float = 0.0
    over_25: float = 0.0
    under_25: float = 0.0
    over_35: float = 0.0
    under_35: float = 0.0
    over_45: float = 0.0
    under_45: float = 0.0
    
    # BTTS
    btts_yes: float = 0.0
    btts_no: float = 0.0
    
    # Double Chance
    dc_1x: float = 0.0
    dc_x2: float = 0.0
    dc_12: float = 0.0
    
    # Draw No Bet
    dnb_home: float = 0.0
    dnb_away: float = 0.0
    
    # Halftime/Fulltime (9 outcomes)
    htft_1_1: float = 0.0
    htft_1_x: float = 0.0
    htft_1_2: float = 0.0
    htft_x_1: float = 0.0
    htft_x_x: float = 0.0
    htft_x_2: float = 0.0
    htft_2_1: float = 0.0
    htft_2_x: float = 0.0
    htft_2_2: float = 0.0
    
    # Correct Score (common scores)
    cs_1_0: float = 0.0
    cs_2_0: float = 0.0
    cs_2_1: float = 0.0
    cs_3_0: float = 0.0
    cs_3_1: float = 0.0
    cs_3_2: float = 0.0
    cs_0_0: float = 0.0
    cs_1_1: float = 0.0
    cs_2_2: float = 0.0
    cs_0_1: float = 0.0
    cs_0_2: float = 0.0
    cs_1_2: float = 0.0
    cs_0_3: float = 0.0
    cs_1_3: float = 0.0
    cs_2_3: float = 0.0
    
    # Asian Handicap
    ah_home_minus_15: float = 0.0
    ah_away_plus_15: float = 0.0
    ah_home_minus_10: float = 0.0
    ah_away_plus_10: float = 0.0
    ah_home_minus_05: float = 0.0
    ah_away_plus_05: float = 0.0
    ah_home_plus_05: float = 0.0
    ah_away_minus_05: float = 0.0
    
    # First Half (derived)
    ht_over_05: float = 0.0
    ht_under_05: float = 0.0
    ht_btts_yes: float = 0.0
    ht_btts_no: float = 0.0


class SportyBetScraper:
    """Complete SportyBet scraper with all markets and multi-day support."""
    
    # Supported countries
    COUNTRIES = {
        'ng': 'Nigeria',
        'ke': 'Kenya', 
        'gh': 'Ghana',
        'tz': 'Tanzania'
    }
    
    # Market ID mappings
    MARKET_IDS = {
        '1x2': '1',
        'over_under': '18',
        'double_chance': '10',
        'btts': '29',
        'draw_no_bet': '11',
        'halftime_fulltime': '26',
        'correct_score': '36',
        'handicap': '14',
    }
    
    # All markets for comprehensive request
    ALL_MARKET_IDS = "1,18,10,29,11,26,36,14,60100"
    
    def __init__(self, country_code: str = "ng"):
        """
        Initialize scraper.
        
        Args:
            country_code: SportyBet country code (ng, ke, gh, tz)
        """
        if country_code not in self.COUNTRIES:
            logger.warning(f"Unknown country '{country_code}', using 'ng'")
            country_code = "ng"
            
        self.country_code = country_code
        self.base_url = f"https://www.sportybet.com/api/{country_code}/factsCenter"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': f'https://www.sportybet.com/{country_code}/'
        })
        
        # Data directory for saving
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "sportybet"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_fixtures(self, days: int = 7, page_size: int = 100) -> List[Dict]:
        """
        Get ALL fixtures for the next N days with ALL markets.
        
        Args:
            days: Number of days ahead to fetch (default 7)
            page_size: Results per page
            
        Returns:
            List of fixture dictionaries with complete odds
        """
        url = f"{self.base_url}/pcUpcomingEvents"
        params = {
            'sportId': 'sr:sport:1',
            'marketId': self.ALL_MARKET_IDS,
            'pageSize': page_size,
            'pageNum': 1,
            '_t': int(time.time() * 1000)
        }
        
        all_fixtures = []
        page = 1
        max_pages = 20  # Allow more pages for 7 days
        cutoff_time = datetime.now() + timedelta(days=days)
        
        logger.info(f"Fetching fixtures for next {days} days from SportyBet ({self.country_code})...")
        
        while page <= max_pages:
            params['pageNum'] = page
            
            try:
                response = self.session.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                
                # bizCode 10000 is success for SportyBet
                if data.get('bizCode') not in [0, 10000]:
                    logger.warning(f"API returned bizCode: {data.get('bizCode')}")
                    break
                
                tournaments = data.get('data', {}).get('tournaments', [])
                if not tournaments:
                    break
                
                events_found = 0
                for tournament in tournaments:
                    tournament_name = tournament.get('name', 'Unknown')
                    category = tournament.get('category', {}).get('name', '')
                    
                    for event in tournament.get('events', []):
                        # Check if within date range
                        start_time = event.get('estimateStartTime', 0)
                        if start_time:
                            event_time = datetime.fromtimestamp(start_time / 1000)
                            if event_time > cutoff_time:
                                continue
                        
                        # Add tournament info
                        event['tournament'] = {'name': tournament_name, 'country': category}
                        fixture = self._parse_fixture_complete(event)
                        if fixture:
                            all_fixtures.append(fixture)
                            events_found += 1
                
                logger.info(f"Page {page}: {events_found} fixtures from {len(tournaments)} leagues")
                
                # Check if done
                total = data.get('data', {}).get('totalNum', 0)
                if len(all_fixtures) >= total or events_found == 0:
                    break
                    
                page += 1
                time.sleep(0.3)  # Rate limiting
                
            except requests.RequestException as e:
                logger.error(f"Request failed on page {page}: {e}")
                break
        
        logger.info(f"Total: {len(all_fixtures)} fixtures for next {days} days")
        return all_fixtures
    
    def get_todays_fixtures(self, page_size: int = 100) -> List[Dict]:
        """Get all fixtures for today only."""
        url = f"{self.base_url}/pcUpcomingEvents"
        params = {
            'sportId': 'sr:sport:1',
            'marketId': self.ALL_MARKET_IDS,
            'pageSize': page_size,
            'pageNum': 1,
            'todayGames': 'true',
            '_t': int(time.time() * 1000)
        }
        
        all_fixtures = []
        page = 1
        
        while True:
            params['pageNum'] = page
            logger.info(f"Fetching today's fixtures, page {page}...")
            
            try:
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if data.get('bizCode') not in [0, 10000]:
                    break
                
                tournaments = data.get('data', {}).get('tournaments', [])
                if not tournaments:
                    break
                
                for tournament in tournaments:
                    tournament_name = tournament.get('name', 'Unknown')
                    for event in tournament.get('events', []):
                        event['tournament'] = {'name': tournament_name}
                        fixture = self._parse_fixture_complete(event)
                        if fixture:
                            all_fixtures.append(fixture)
                
                total = data.get('data', {}).get('totalNum', 0)
                if len(all_fixtures) >= total:
                    break
                    
                page += 1
                time.sleep(0.5)
                
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
        
        logger.info(f"Fetched {len(all_fixtures)} fixtures for today")
        return all_fixtures
    
    def get_match_details(self, event_id: str) -> Optional[Dict]:
        """
        Get detailed markets for a specific match (ALL markets).
        
        Args:
            event_id: SportyBet event ID
            
        Returns:
            Dictionary with all available markets and odds
        """
        url = f"{self.base_url}/event"
        params = {
            'eventId': event_id,
            'productId': 3,
            '_t': int(time.time() * 1000)
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('bizCode') not in [0, 10000]:
                return None
            
            event_data = data.get('data', {})
            return self._parse_detailed_match(event_data)
            
        except requests.RequestException as e:
            logger.error(f"Failed to get match details: {e}")
            return None
    
    def _parse_fixture_complete(self, event: Dict) -> Optional[Dict]:
        """Parse a fixture with ALL market odds."""
        try:
            home_team = event.get('homeTeamName', '')
            away_team = event.get('awayTeamName', '')
            
            if not home_team or not away_team:
                return None
            
            # Parse start time
            start_time = event.get('estimateStartTime', 0)
            if start_time:
                dt = datetime.fromtimestamp(start_time / 1000)
                date_str = dt.strftime('%Y-%m-%d')
                time_str = dt.strftime('%H:%M:%S')
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
                time_str = '00:00:00'
            
            # Parse tournament/league
            tournament = event.get('tournament', {})
            league_name = tournament.get('name', 'Unknown League')
            country = tournament.get('country', '')
            
            # Parse ALL markets
            markets = event.get('markets', [])
            odds = self._parse_all_markets(markets)
            
            return {
                'event_id': event.get('eventId', ''),
                'home_team': home_team,
                'away_team': away_team,
                'league': league_name,
                'country': country,
                'date': date_str,
                'time': time_str,
                'kickoff': f"{date_str}T{time_str}",
                'venue': event.get('venue', f"{home_team} Stadium"),
                'odds': odds,
                'source': 'sportybet',
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse fixture: {e}")
            return None
    
    def _parse_all_markets(self, markets: List[Dict]) -> Dict:
        """Parse ALL market types from API response."""
        odds = MarketOdds()
        
        for market in markets:
            market_id = market.get('id', '')
            outcomes = market.get('outcomes', [])
            specifier = market.get('specifier', '')
            
            if market_id == '1':  # 1X2
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    if o.get('id') == '1':
                        odds.home = val
                    elif o.get('id') == '2':
                        odds.draw = val
                    elif o.get('id') == '3':
                        odds.away = val
            
            elif market_id == '18':  # Over/Under
                total = self._extract_total(specifier)
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    is_over = o.get('id') == '12'
                    is_under = o.get('id') == '13'
                    
                    if total == 0.5:
                        if is_over: odds.over_05 = val
                        elif is_under: odds.under_05 = val
                    elif total == 1.5:
                        if is_over: odds.over_15 = val
                        elif is_under: odds.under_15 = val
                    elif total == 2.5:
                        if is_over: odds.over_25 = val
                        elif is_under: odds.under_25 = val
                    elif total == 3.5:
                        if is_over: odds.over_35 = val
                        elif is_under: odds.under_35 = val
                    elif total == 4.5:
                        if is_over: odds.over_45 = val
                        elif is_under: odds.under_45 = val
            
            elif market_id == '29':  # BTTS
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    if o.get('id') == '74':
                        odds.btts_yes = val
                    elif o.get('id') == '76':
                        odds.btts_no = val
            
            elif market_id == '10':  # Double Chance
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    if o.get('id') == '9':
                        odds.dc_1x = val
                    elif o.get('id') == '10':
                        odds.dc_12 = val
                    elif o.get('id') == '11':
                        odds.dc_x2 = val
            
            elif market_id == '11':  # Draw No Bet
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    if o.get('id') == '1714':
                        odds.dnb_home = val
                    elif o.get('id') == '1715':
                        odds.dnb_away = val
            
            elif market_id == '26':  # HT/FT
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    oid = o.get('id', '')
                    htft_map = {
                        '1': 'htft_1_1', '4': 'htft_1_x', '7': 'htft_1_2',
                        '2': 'htft_x_1', '5': 'htft_x_x', '8': 'htft_x_2',
                        '3': 'htft_2_1', '6': 'htft_2_x', '9': 'htft_2_2'
                    }
                    if oid in htft_map:
                        setattr(odds, htft_map[oid], val)
            
            elif market_id == '36':  # Correct Score
                score = self._extract_score(specifier)
                if score:
                    for o in outcomes:
                        val = self._parse_odd(o.get('odds'))
                        attr = f"cs_{score[0]}_{score[1]}"
                        if hasattr(odds, attr):
                            setattr(odds, attr, val)
            
            elif market_id == '14':  # Asian Handicap
                hcap = self._extract_handicap(specifier)
                for o in outcomes:
                    val = self._parse_odd(o.get('odds'))
                    oid = o.get('id', '')
                    if hcap == -1.5:
                        if oid == '1714': odds.ah_home_minus_15 = val
                        elif oid == '1715': odds.ah_away_plus_15 = val
                    elif hcap == -1.0:
                        if oid == '1714': odds.ah_home_minus_10 = val
                        elif oid == '1715': odds.ah_away_plus_10 = val
                    elif hcap == -0.5:
                        if oid == '1714': odds.ah_home_minus_05 = val
                        elif oid == '1715': odds.ah_away_plus_05 = val
                    elif hcap == 0.5:
                        if oid == '1714': odds.ah_home_plus_05 = val
                        elif oid == '1715': odds.ah_away_minus_05 = val
        
        return asdict(odds)
    
    def _parse_odd(self, odd_value: Any) -> float:
        """Parse odd value to float."""
        if not odd_value:
            return 0.0
        try:
            return float(odd_value)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_total(self, specifier: str) -> float:
        """Extract total goals from specifier like 'total=2.5'."""
        try:
            if 'total=' in specifier:
                return float(specifier.split('total=')[1].split('|')[0])
        except:
            pass
        return 0.0
    
    def _extract_score(self, specifier: str) -> Optional[tuple]:
        """Extract score from specifier like 'score=2:1'."""
        try:
            if 'score=' in specifier:
                score_str = specifier.split('score=')[1].split('|')[0]
                parts = score_str.replace(':', '-').split('-')
                return (int(parts[0]), int(parts[1]))
        except:
            pass
        return None
    
    def _extract_handicap(self, specifier: str) -> float:
        """Extract handicap from specifier like 'hcp=-1.5'."""
        try:
            if 'hcp=' in specifier:
                return float(specifier.split('hcp=')[1].split('|')[0])
        except:
            pass
        return 0.0
    
    def _parse_detailed_match(self, event_data: Dict) -> Dict:
        """Parse detailed match data with all markets."""
        basic_info = {
            'event_id': event_data.get('eventId', ''),
            'home_team': event_data.get('homeTeamName', ''),
            'away_team': event_data.get('awayTeamName', ''),
            'league': event_data.get('tournament', {}).get('name', ''),
        }
        
        all_markets = {}
        for market in event_data.get('markets', []):
            market_name = market.get('desc', 'Unknown')
            market_outcomes = []
            
            for outcome in market.get('outcomes', []):
                market_outcomes.append({
                    'name': outcome.get('desc', ''),
                    'odds': self._parse_odd(outcome.get('odds'))
                })
            
            all_markets[market_name] = market_outcomes
        
        basic_info['markets'] = all_markets
        basic_info['total_markets'] = len(all_markets)
        
        return basic_info
    
    def save_fixtures_to_csv(self, fixtures: List[Dict], filename: str = None) -> str:
        """Save fixtures to CSV for training."""
        if not filename:
            filename = f"sportybet_fixtures_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = self.data_dir / filename
        
        if not fixtures:
            logger.warning("No fixtures to save")
            return str(filepath)
        
        # Flatten odds dict
        rows = []
        for fix in fixtures:
            row = {k: v for k, v in fix.items() if k != 'odds'}
            if fix.get('odds'):
                for odds_key, odds_val in fix['odds'].items():
                    row[f'odds_{odds_key}'] = odds_val
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Saved {len(rows)} fixtures to {filepath}")
        
        return str(filepath)
    
    def scrape_all_countries(self, days: int = 7) -> Dict[str, List[Dict]]:
        """Scrape fixtures from all supported countries."""
        all_data = {}
        
        for code, name in self.COUNTRIES.items():
            logger.info(f"Scraping {name} ({code})...")
            scraper = SportyBetScraper(country_code=code)
            fixtures = scraper.get_all_fixtures(days=days)
            all_data[code] = fixtures
            time.sleep(1)  # Rate limit between countries
        
        return all_data


# Convenience functions
def get_sportybet_fixtures(days: int = 7) -> List[Dict]:
    """Get fixtures from SportyBet."""
    scraper = SportyBetScraper()
    return scraper.get_all_fixtures(days=days)


def get_sportybet_today() -> List[Dict]:
    """Get today's fixtures from SportyBet."""
    scraper = SportyBetScraper()
    return scraper.get_todays_fixtures()


def save_weekly_fixtures() -> str:
    """Save a week's worth of fixtures for training."""
    scraper = SportyBetScraper()
    fixtures = scraper.get_all_fixtures(days=7)
    return scraper.save_fixtures_to_csv(fixtures)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SportyBet Scraper')
    parser.add_argument('--days', type=int, default=7, help='Days to fetch')
    parser.add_argument('--country', type=str, default='ng', help='Country code')
    parser.add_argument('--save', action='store_true', help='Save to CSV')
    parser.add_argument('--all-countries', action='store_true', help='Scrape all countries')
    
    args = parser.parse_args()
    
    if args.all_countries:
        scraper = SportyBetScraper()
        all_data = scraper.scrape_all_countries(days=args.days)
        for code, fixtures in all_data.items():
            print(f"{code}: {len(fixtures)} fixtures")
    else:
        scraper = SportyBetScraper(country_code=args.country)
        fixtures = scraper.get_all_fixtures(days=args.days)
        
        print(f"\n{'='*60}")
        print(f"SportyBet Fixtures ({args.country.upper()}) - Next {args.days} days")
        print(f"{'='*60}")
        print(f"Total: {len(fixtures)} fixtures\n")
        
        # Sample output
        for i, fix in enumerate(fixtures[:5], 1):
            print(f"{i}. {fix['home_team']} vs {fix['away_team']}")
            print(f"   League: {fix['league']} | {fix['date']} {fix['time'][:5]}")
            odds = fix['odds']
            if odds.get('home', 0) > 0:
                print(f"   1X2: {odds['home']:.2f} / {odds['draw']:.2f} / {odds['away']:.2f}")
            if odds.get('over_25', 0) > 0:
                print(f"   O/U 2.5: {odds['over_25']:.2f} / {odds['under_25']:.2f}")
            if odds.get('btts_yes', 0) > 0:
                print(f"   BTTS: Yes {odds['btts_yes']:.2f} / No {odds['btts_no']:.2f}")
            if odds.get('dc_1x', 0) > 0:
                print(f"   DC: 1X {odds['dc_1x']:.2f} | X2 {odds['dc_x2']:.2f} | 12 {odds['dc_12']:.2f}")
            print()
        
        if len(fixtures) > 5:
            print(f"... and {len(fixtures) - 5} more fixtures")
        
        if args.save:
            filepath = scraper.save_fixtures_to_csv(fixtures)
            print(f"\nSaved to: {filepath}")
