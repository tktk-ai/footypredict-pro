"""
API Football Data Scraper
===========================

Alternative data source using API-Football.com which provides
reliable football data with a free tier.

Also includes scraping from publicly accessible football data APIs.
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent


class APIFootballScraper:
    """
    Scraper for API-Football (RapidAPI).
    
    Note: Requires API key. Free tier provides 100 requests/day.
    Get your key at: https://rapidapi.com/api-sports/api/api-football
    """
    
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    
    # Global League IDs for API-Football
    # Coverage: Europe, Americas, Asia, Oceania
    GLOBAL_LEAGUES = {
        # 🌍 EUROPE - Top Leagues
        'premier_league': {'id': 39, 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League', 'country': 'England'},
        'championship': {'id': 40, 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship', 'country': 'England'},
        'league_one': {'id': 41, 'name': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 League One', 'country': 'England'},
        'la_liga': {'id': 140, 'name': '🇪🇸 La Liga', 'country': 'Spain'},
        'la_liga_2': {'id': 141, 'name': '🇪🇸 La Liga 2', 'country': 'Spain'},
        'bundesliga': {'id': 78, 'name': '🇩🇪 Bundesliga', 'country': 'Germany'},
        'bundesliga_2': {'id': 79, 'name': '🇩🇪 2. Bundesliga', 'country': 'Germany'},
        'serie_a': {'id': 135, 'name': '🇮🇹 Serie A', 'country': 'Italy'},
        'serie_b': {'id': 136, 'name': '🇮🇹 Serie B', 'country': 'Italy'},
        'ligue_1': {'id': 61, 'name': '🇫🇷 Ligue 1', 'country': 'France'},
        'ligue_2': {'id': 62, 'name': '🇫🇷 Ligue 2', 'country': 'France'},
        'eredivisie': {'id': 88, 'name': '🇳🇱 Eredivisie', 'country': 'Netherlands'},
        'primeira_liga': {'id': 94, 'name': '🇵🇹 Primeira Liga', 'country': 'Portugal'},
        'belgian_pro_league': {'id': 144, 'name': '🇧🇪 Jupiler Pro League', 'country': 'Belgium'},
        'scottish_premiership': {'id': 179, 'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership', 'country': 'Scotland'},
        'super_lig': {'id': 203, 'name': '🇹🇷 Süper Lig', 'country': 'Turkey'},
        'super_league_greece': {'id': 197, 'name': '🇬🇷 Super League Greece', 'country': 'Greece'},
        
        # 🌍 EUROPE - Additional
        'russian_premier': {'id': 235, 'name': '🇷🇺 Russian Premier League', 'country': 'Russia'},
        'ukrainian_premier': {'id': 333, 'name': '🇺🇦 Ukrainian Premier League', 'country': 'Ukraine'},
        'austrian_bundesliga': {'id': 218, 'name': '🇦🇹 Austrian Bundesliga', 'country': 'Austria'},
        'swiss_super_league': {'id': 207, 'name': '🇨🇭 Swiss Super League', 'country': 'Switzerland'},
        'czech_first_league': {'id': 345, 'name': '🇨🇿 Czech First League', 'country': 'Czechia'},
        'polish_ekstraklasa': {'id': 106, 'name': '🇵🇱 Ekstraklasa', 'country': 'Poland'},
        'danish_superliga': {'id': 119, 'name': '🇩🇰 Danish Superliga', 'country': 'Denmark'},
        'norwegian_eliteserien': {'id': 103, 'name': '🇳🇴 Eliteserien', 'country': 'Norway'},
        'swedish_allsvenskan': {'id': 113, 'name': '🇸🇪 Allsvenskan', 'country': 'Sweden'},
        'serbian_superliga': {'id': 286, 'name': '🇷🇸 Serbian SuperLiga', 'country': 'Serbia'},
        'croatian_hnl': {'id': 210, 'name': '🇭🇷 Croatian HNL', 'country': 'Croatia'},
        
        # 🌎 AMERICAS - North
        'mls': {'id': 253, 'name': '🇺🇸 MLS', 'country': 'USA'},
        'usl_championship': {'id': 255, 'name': '🇺🇸 USL Championship', 'country': 'USA'},
        'liga_mx': {'id': 262, 'name': '🇲🇽 Liga MX', 'country': 'Mexico'},
        'liga_mx_clausura': {'id': 263, 'name': '🇲🇽 Liga MX Clausura', 'country': 'Mexico'},
        'cpl': {'id': 459, 'name': '🇨🇦 Canadian Premier League', 'country': 'Canada'},
        
        # 🌎 AMERICAS - South
        'brasileirao': {'id': 71, 'name': '🇧🇷 Brasileirão Série A', 'country': 'Brazil'},
        'brasileirao_b': {'id': 72, 'name': '🇧🇷 Brasileirão Série B', 'country': 'Brazil'},
        'argentine_liga': {'id': 128, 'name': '🇦🇷 Liga Profesional', 'country': 'Argentina'},
        'chilean_primera': {'id': 265, 'name': '🇨🇱 Primera División', 'country': 'Chile'},
        'colombian_liga': {'id': 239, 'name': '🇨🇴 Liga BetPlay', 'country': 'Colombia'},
        'peruvian_liga': {'id': 281, 'name': '🇵🇪 Liga 1', 'country': 'Peru'},
        'ecuadorian_liga': {'id': 242, 'name': '🇪🇨 LigaPro', 'country': 'Ecuador'},
        'uruguayan_primera': {'id': 268, 'name': '🇺🇾 Primera División', 'country': 'Uruguay'},
        'paraguayan_primera': {'id': 279, 'name': '🇵🇾 Primera División', 'country': 'Paraguay'},
        'bolivian_liga': {'id': 157, 'name': '🇧🇴 División Profesional', 'country': 'Bolivia'},
        'venezuelan_liga': {'id': 299, 'name': '🇻🇪 Liga FUTVE', 'country': 'Venezuela'},
        
        # 🌏 ASIA
        'j1_league': {'id': 98, 'name': '🇯🇵 J1 League', 'country': 'Japan'},
        'j2_league': {'id': 99, 'name': '🇯🇵 J2 League', 'country': 'Japan'},
        'k_league_1': {'id': 292, 'name': '🇰🇷 K League 1', 'country': 'South Korea'},
        'chinese_super': {'id': 169, 'name': '🇨🇳 Chinese Super League', 'country': 'China'},
        'saudi_pro': {'id': 307, 'name': '🇸🇦 Saudi Pro League', 'country': 'Saudi Arabia'},
        'uae_pro': {'id': 304, 'name': '🇦🇪 UAE Pro League', 'country': 'UAE'},
        'qatari_stars': {'id': 305, 'name': '🇶🇦 Qatar Stars League', 'country': 'Qatar'},
        'indian_super': {'id': 323, 'name': '🇮🇳 Indian Super League', 'country': 'India'},
        'thai_league': {'id': 296, 'name': '🇹🇭 Thai League 1', 'country': 'Thailand'},
        'malaysian_super': {'id': 302, 'name': '🇲🇾 Malaysia Super League', 'country': 'Malaysia'},
        'indonesian_liga': {'id': 274, 'name': '🇮🇩 Liga 1', 'country': 'Indonesia'},
        'vietnamese_vleague': {'id': 340, 'name': '🇻🇳 V.League 1', 'country': 'Vietnam'},
        
        # 🦘 OCEANIA / AUSTRALIA
        'a_league': {'id': 188, 'name': '🇦🇺 A-League', 'country': 'Australia'},
        'a_league_women': {'id': 189, 'name': '🇦🇺 A-League Women', 'country': 'Australia'},
        'nz_premiership': {'id': 288, 'name': '🇳🇿 NZ Premiership', 'country': 'New Zealand'},
        
        # 🌍 AFRICA (bonus)
        'egyptian_premier': {'id': 233, 'name': '🇪🇬 Egyptian Premier League', 'country': 'Egypt'},
        'south_african_psl': {'id': 288, 'name': '🇿🇦 Premier Soccer League', 'country': 'South Africa'},
        'moroccan_botola': {'id': 200, 'name': '🇲🇦 Botola Pro', 'country': 'Morocco'},
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
            })
    
    def get_fixtures(self, date: str = None, league_id: int = None) -> List[Dict]:
        """Get fixtures for a date and/or league."""
        if not self.api_key:
            logger.warning("API key not configured")
            return []
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        params = {"date": date}
        if league_id:
            params["league"] = league_id
        
        response = self.session.get(f"{self.BASE_URL}/fixtures", params=params)
        
        if response.status_code != 200:
            logger.error(f"API-Football request failed: {response.status_code}")
            return []
        
        data = response.json()
        return data.get("response", [])
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int = 2024) -> Dict:
        """Get detailed team statistics."""
        if not self.api_key:
            return {}
        
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        
        response = self.session.get(f"{self.BASE_URL}/teams/statistics", params=params)
        
        if response.status_code != 200:
            return {}
        
        data = response.json()
        return data.get("response", {})
    
    def get_predictions(self, fixture_id: int) -> Dict:
        """Get predictions for a fixture."""
        if not self.api_key:
            return {}
        
        params = {"fixture": fixture_id}
        
        response = self.session.get(f"{self.BASE_URL}/predictions", params=params)
        
        if response.status_code != 200:
            return {}
        
        data = response.json()
        predictions = data.get("response", [])
        if predictions:
            return predictions[0]
        return {}
    
    def get_league_fixtures(self, league_key: str, days: int = 7) -> List[Dict]:
        """
        Get fixtures for a specific league by key.
        
        Args:
            league_key: League key from GLOBAL_LEAGUES (e.g., 'mls', 'j1_league')
            days: Number of days ahead
            
        Returns:
            List of fixture dictionaries
        """
        if league_key not in self.GLOBAL_LEAGUES:
            logger.warning(f"Unknown league: {league_key}")
            return []
        
        league_id = self.GLOBAL_LEAGUES[league_key]['id']
        all_fixtures = []
        
        for day_offset in range(days):
            date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            fixtures = self.get_fixtures(date=date, league_id=league_id)
            all_fixtures.extend(fixtures)
        
        return all_fixtures
    
    def get_global_fixtures(self, leagues: List[str] = None, days: int = 3) -> List[Dict]:
        """
        Get fixtures from multiple global leagues.
        
        Args:
            leagues: List of league keys, or None for top global leagues
            days: Number of days ahead
            
        Returns:
            List of standardized fixture dictionaries
        """
        if not self.api_key:
            logger.warning("API key not configured for global fixtures")
            return []
        
        if leagues is None:
            # Default: top leagues from each region
            leagues = [
                # Europe
                'premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1',
                # Americas
                'mls', 'liga_mx', 'brasileirao', 'argentine_liga',
                # Asia
                'j1_league', 'k_league_1', 'saudi_pro',
                # Oceania
                'a_league',
            ]
        
        all_fixtures = []
        
        for league_key in leagues:
            if league_key not in self.GLOBAL_LEAGUES:
                continue
            
            league_info = self.GLOBAL_LEAGUES[league_key]
            fixtures = self.get_league_fixtures(league_key, days)
            
            for fix in fixtures:
                try:
                    fixture_data = fix.get('fixture', {})
                    teams = fix.get('teams', {})
                    league_data = fix.get('league', {})
                    
                    all_fixtures.append({
                        'id': fixture_data.get('id'),
                        'home_team': {'name': teams.get('home', {}).get('name', '')},
                        'away_team': {'name': teams.get('away', {}).get('name', '')},
                        'league': league_key,
                        'league_name': league_info['name'],
                        'country': league_info['country'],
                        'date': fixture_data.get('date', '')[:10],
                        'time': fixture_data.get('date', '')[11:16] if fixture_data.get('date') else '',
                        'status': fixture_data.get('status', {}).get('short', ''),
                        'source': 'api-football'
                    })
                except Exception as e:
                    logger.error(f"Error parsing fixture: {e}")
                    continue
        
        return all_fixtures
    
    @classmethod
    def get_available_leagues(cls) -> Dict:
        """Get all available global leagues."""
        return cls.GLOBAL_LEAGUES
    
    @classmethod
    def get_leagues_by_region(cls, region: str) -> Dict:
        """
        Get leagues filtered by region.
        
        Args:
            region: 'europe', 'americas', 'asia', 'oceania', 'africa'
        """
        region_countries = {
            'europe': ['England', 'Spain', 'Germany', 'Italy', 'France', 'Netherlands', 
                       'Portugal', 'Belgium', 'Scotland', 'Turkey', 'Greece', 'Russia',
                       'Ukraine', 'Austria', 'Switzerland', 'Czechia', 'Poland', 
                       'Denmark', 'Norway', 'Sweden', 'Serbia', 'Croatia'],
            'americas': ['USA', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Colombia',
                         'Peru', 'Ecuador', 'Uruguay', 'Paraguay', 'Bolivia', 'Venezuela', 'Canada'],
            'asia': ['Japan', 'South Korea', 'China', 'Saudi Arabia', 'UAE', 'Qatar',
                     'India', 'Thailand', 'Malaysia', 'Indonesia', 'Vietnam'],
            'oceania': ['Australia', 'New Zealand'],
            'africa': ['Egypt', 'South Africa', 'Morocco']
        }
        
        countries = region_countries.get(region.lower(), [])
        return {k: v for k, v in cls.GLOBAL_LEAGUES.items() if v['country'] in countries}


class OpenFootballData:
    """
    Scraper for open/free football data sources.
    
    Sources:
    - football-data.org (free API)
    - OpenLigaDB
    - GitHub datasets
    """
    
    FOOTBALL_DATA_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: str = None):
        """
        Initialize scraper.
        
        Args:
            api_key: Football-data.org API key (free registration)
        """
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                "X-Auth-Token": api_key
            })
    
    def get_matches(self, competition: str = "PL", days: int = 7) -> List[Dict]:
        """
        Get upcoming matches from football-data.org.
        
        Args:
            competition: Competition code (PL, BL1, SA, FL1, etc.)
            days: Number of days ahead
            
        Returns:
            List of match dictionaries
        """
        date_from = datetime.now().strftime("%Y-%m-%d")
        date_to = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = f"{self.FOOTBALL_DATA_URL}/competitions/{competition}/matches"
        params = {
            "dateFrom": date_from,
            "dateTo": date_to
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                
                # Convert to our format
                formatted = []
                for match in matches:
                    formatted.append({
                        'match_id': match.get('id'),
                        'home_team': match.get('homeTeam', {}).get('name', ''),
                        'away_team': match.get('awayTeam', {}).get('name', ''),
                        'competition': match.get('competition', {}).get('name', ''),
                        'date': match.get('utcDate', '')[:10],
                        'status': match.get('status', ''),
                        'source': 'football-data.org'
                    })
                
                logger.info(f"Fetched {len(formatted)} matches from football-data.org")
                return formatted
            else:
                logger.warning(f"Football-data.org returned {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Football-data.org request failed: {e}")
            return []
    
    def get_standings(self, competition: str = "PL") -> List[Dict]:
        """Get league standings."""
        url = f"{self.FOOTBALL_DATA_URL}/competitions/{competition}/standings"
        
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                standings = data.get("standings", [])
                if standings:
                    return standings[0].get("table", [])
            return []
            
        except Exception as e:
            logger.error(f"Football-data.org standings failed: {e}")
            return []
    
    def get_team_info(self, team_id: int) -> Dict:
        """Get team information."""
        url = f"{self.FOOTBALL_DATA_URL}/teams/{team_id}"
        
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            return {}
            
        except Exception as e:
            logger.error(f"Football-data.org team info failed: {e}")
            return {}


class GitHubFootballData:
    """
    Scraper for football datasets hosted on GitHub.
    
    Uses datasets like:
    - openfootball/football.json
    - footballcsv/england
    """
    
    OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
    
    def get_season_data(self, league: str = "en.1", season: str = "2024-25") -> List[Dict]:
        """
        Get season match data from openfootball.
        
        Args:
            league: League code (en.1 = Premier League, de.1 = Bundesliga)
            season: Season string
            
        Returns:
            List of match dictionaries
        """
        url = f"{self.OPENFOOTBALL_URL}/{season}/{league}.json"
        
        try:
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                matches = []
                for round_data in data.get("rounds", []):
                    round_name = round_data.get("name", "")
                    for match in round_data.get("matches", []):
                        matches.append({
                            'round': round_name,
                            'date': match.get('date', ''),
                            'home_team': match.get('team1', {}).get('name', ''),
                            'away_team': match.get('team2', {}).get('name', ''),
                            'home_goals': match.get('score', {}).get('ft', [None, None])[0],
                            'away_goals': match.get('score', {}).get('ft', [None, None])[1],
                            'source': 'openfootball'
                        })
                
                logger.info(f"Fetched {len(matches)} matches from GitHub")
                return matches
            else:
                logger.warning(f"GitHub data returned {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"GitHub football data failed: {e}")
            return []


class UnifiedDataCollector:
    """
    Unified collector that attempts multiple sources.
    """
    
    def __init__(self, api_football_key: str = None, football_data_key: str = None):
        self.api_football = APIFootballScraper(api_football_key)
        self.open_football = OpenFootballData(football_data_key)
        self.github_data = GitHubFootballData()
        self.data_dir = project_root / "data" / "collected"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Collect upcoming matches from all available sources."""
        all_matches = []
        seen = set()
        
        # Try football-data.org first (most reliable free source)
        for comp in ["PL", "BL1", "SA", "FL1", "PD"]:  # Various leagues
            matches = self.open_football.get_matches(comp, days)
            for m in matches:
                key = f"{m['home_team']}_{m['away_team']}"
                if key not in seen:
                    all_matches.append(m)
                    seen.add(key)
        
        logger.info(f"Collected {len(all_matches)} unique matches")
        return all_matches
    
    def collect_historical_data(self, leagues: List[str] = None, seasons: List[str] = None) -> List[Dict]:
        """Collect historical match data."""
        if leagues is None:
            leagues = ["en.1", "de.1", "es.1", "it.1", "fr.1"]
        if seasons is None:
            seasons = ["2023-24", "2022-23", "2021-22"]
        
        all_data = []
        for league in leagues:
            for season in seasons:
                matches = self.github_data.get_season_data(league, season)
                all_data.extend(matches)
        
        logger.info(f"Collected {len(all_data)} historical matches")
        return all_data
    
    def save_data(self, data: List[Dict], filename: str):
        """Save collected data."""
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath


def main():
    """CLI for API scrapers."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='API Football Data Scraper')
    parser.add_argument('--source', type=str, choices=['all', 'football-data', 'github'],
                       default='all', help='Data source')
    parser.add_argument('--days', type=int, default=7, help='Days ahead')
    parser.add_argument('--historical', action='store_true', help='Collect historical data')
    
    args = parser.parse_args()
    
    collector = UnifiedDataCollector()
    
    print("="*60)
    print("API Football Data Collector")
    print("="*60)
    
    if args.historical:
        data = collector.collect_historical_data()
        collector.save_data(data, f"historical_{datetime.now().strftime('%Y%m%d')}.json")
    else:
        matches = collector.collect_upcoming_matches(args.days)
        collector.save_data(matches, f"upcoming_{datetime.now().strftime('%Y%m%d')}.json")
        
        print(f"\nCollected {len(matches)} matches")
        for m in matches[:5]:
            print(f"  {m['home_team']} vs {m['away_team']} ({m.get('competition', 'Unknown')})")
    
    print("\n✅ Collection complete!")


if __name__ == "__main__":
    main()
