"""
Multi-Source Statistics Scraper
================================

Scrapes football statistics from multiple sources:
1. Predictz - Match predictions
2. SofaScore - Live stats, lineups, ratings
3. FootyStats - Team/league statistics
4. SoccerStats - Goal timing, H2H
5. WhoScored - Player/team ratings
6. FotMob - xG, momentum

Usage:
    from src.data.multi_source_scraper import MultiSourceScraper
    
    scraper = MultiSourceScraper()
    data = scraper.scrape_all("Manchester United", "Liverpool")
"""

import requests
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import quote, urljoin
import time
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Container for team statistics."""
    team_name: str
    # Form
    form_last_5: str = ""  # e.g., "WWDLW"
    form_points: int = 0
    # Goals
    goals_scored: float = 0.0
    goals_conceded: float = 0.0
    goals_per_game: float = 0.0
    # xG
    xg_for: float = 0.0
    xg_against: float = 0.0
    # Other
    clean_sheets: int = 0
    failed_to_score: int = 0
    btts_rate: float = 0.0
    over_25_rate: float = 0.0
    corners_avg: float = 0.0
    cards_avg: float = 0.0
    # Ratings
    attack_rating: float = 0.0
    defense_rating: float = 0.0
    overall_rating: float = 0.0


@dataclass
class MatchPrediction:
    """Container for match prediction data."""
    home_team: str
    away_team: str
    prediction: str = ""
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    over_25_prob: float = 0.0
    btts_prob: float = 0.0
    predicted_score: str = ""
    confidence: float = 0.0
    source: str = ""


class BaseScraper:
    """Base class for all scrapers."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.cache = {}
        self.rate_limit = 1.0  # seconds between requests
        self.last_request = 0
    
    def _rate_limit_wait(self):
        """Wait for rate limiting."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make GET request with rate limiting and error handling."""
        self._rate_limit_wait()
        try:
            response = self.session.get(url, timeout=15, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching."""
        name = name.lower().strip()
        # Remove common suffixes
        for suffix in [' fc', ' united', ' city', ' afc', ' sc']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name


# =============================================================================
# PREDICTZ SCRAPER
# =============================================================================

class PredictZScraper(BaseScraper):
    """Scraper for predictz.com predictions."""
    
    BASE_URL = "https://www.predictz.com"
    
    def get_predictions(self, league: str = "premier-league") -> List[Dict]:
        """Get match predictions from Predictz."""
        predictions = []
        
        url = f"{self.BASE_URL}/predictions/{league}/"
        response = self._get(url)
        
        if not response:
            logger.warning("Failed to fetch Predictz predictions")
            return predictions
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find prediction rows
            rows = soup.find_all('tr', class_='pttr')
            
            for row in rows:
                try:
                    teams = row.find_all('td', class_='pttd')
                    if len(teams) < 3:
                        continue
                    
                    home_team = teams[0].get_text(strip=True)
                    away_team = teams[2].get_text(strip=True)
                    
                    # Get prediction
                    pred_cell = row.find('td', class_='predict')
                    prediction = pred_cell.get_text(strip=True) if pred_cell else ""
                    
                    # Get score prediction
                    score_cell = row.find('td', class_='st')
                    predicted_score = score_cell.get_text(strip=True) if score_cell else ""
                    
                    predictions.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'prediction': prediction,
                        'predicted_score': predicted_score,
                        'source': 'predictz'
                    })
                except Exception as e:
                    continue
            
            logger.info(f"Fetched {len(predictions)} predictions from Predictz")
            
        except Exception as e:
            logger.error(f"Error parsing Predictz: {e}")
        
        return predictions
    
    def get_league_stats(self, league: str = "premier-league") -> Dict:
        """Get league statistics from Predictz."""
        stats = {}
        
        url = f"{self.BASE_URL}/stats/{league}/"
        response = self._get(url)
        
        if not response:
            return stats
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse tables for stats
            tables = soup.find_all('table', class_='leaguetable')
            
            for table in tables:
                # Extract team stats
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        team_name = cells[0].get_text(strip=True)
                        stats[team_name] = {
                            'played': int(cells[1].get_text(strip=True) or 0),
                            'won': int(cells[2].get_text(strip=True) or 0),
                            'drawn': int(cells[3].get_text(strip=True) or 0),
                            'lost': int(cells[4].get_text(strip=True) or 0)
                        }
            
        except Exception as e:
            logger.error(f"Error parsing Predictz stats: {e}")
        
        return stats


# =============================================================================
# SOFASCORE SCRAPER
# =============================================================================

class SofaScoreScraper(BaseScraper):
    """Scraper for SofaScore statistics via their API."""
    
    API_URL = "https://api.sofascore.com/api/v1"
    
    def __init__(self):
        super().__init__()
        self.session.headers.update({
            'Accept': 'application/json',
            'Cache-Control': 'no-cache'
        })
    
    def get_team_stats(self, team_id: int) -> Dict:
        """Get team statistics from SofaScore."""
        url = f"{self.API_URL}/team/{team_id}/unique-tournament/17/season/52186/statistics/overall"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            stats = data.get('statistics', {})
            
            return {
                'goals_scored': stats.get('goalsScored', 0),
                'goals_conceded': stats.get('goalsConceded', 0),
                'clean_sheets': stats.get('cleanSheets', 0),
                'xg': stats.get('expectedGoals', 0),
                'xga': stats.get('expectedGoalsAgainst', 0),
                'possession_avg': stats.get('averageBallPossession', 0),
                'shots_avg': stats.get('shotsPerGame', 0),
                'corners_avg': stats.get('cornersPerGame', 0),
                'source': 'sofascore'
            }
        except Exception as e:
            logger.error(f"Error parsing SofaScore: {e}")
            return {}
    
    def get_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Get upcoming matches from SofaScore."""
        matches = []
        
        for day_offset in range(days):
            date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            url = f"{self.API_URL}/sport/football/scheduled-events/{date}"
            
            response = self._get(url)
            if not response:
                continue
            
            try:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    home_team = event.get('homeTeam', {})
                    away_team = event.get('awayTeam', {})
                    
                    matches.append({
                        'event_id': event.get('id'),
                        'home_team': home_team.get('name', ''),
                        'home_team_id': home_team.get('id'),
                        'away_team': away_team.get('name', ''),
                        'away_team_id': away_team.get('id'),
                        'start_time': event.get('startTimestamp'),
                        'tournament': event.get('tournament', {}).get('name', ''),
                        'source': 'sofascore'
                    })
                    
            except Exception as e:
                logger.error(f"Error parsing SofaScore matches: {e}")
        
        logger.info(f"Fetched {len(matches)} matches from SofaScore")
        return matches
    
    def get_match_stats(self, event_id: int) -> Dict:
        """Get detailed match statistics."""
        url = f"{self.API_URL}/event/{event_id}/statistics"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            stats = {}
            
            for group in data.get('statistics', []):
                for item in group.get('groups', []):
                    for stat in item.get('statisticsItems', []):
                        name = stat.get('name', '').replace(' ', '_').lower()
                        stats[f"home_{name}"] = stat.get('home', '')
                        stats[f"away_{name}"] = stat.get('away', '')
            
            stats['source'] = 'sofascore'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing SofaScore match stats: {e}")
            return {}
    
    def get_team_form(self, team_id: int) -> Dict:
        """Get team's recent form."""
        url = f"{self.API_URL}/team/{team_id}/events/last/5"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            events = data.get('events', [])
            
            form = ""
            points = 0
            goals_scored = 0
            goals_conceded = 0
            
            for event in events:
                home_score = event.get('homeScore', {}).get('current', 0)
                away_score = event.get('awayScore', {}).get('current', 0)
                home_id = event.get('homeTeam', {}).get('id')
                
                is_home = home_id == team_id
                team_score = home_score if is_home else away_score
                opp_score = away_score if is_home else home_score
                
                goals_scored += team_score
                goals_conceded += opp_score
                
                if team_score > opp_score:
                    form += "W"
                    points += 3
                elif team_score == opp_score:
                    form += "D"
                    points += 1
                else:
                    form += "L"
            
            return {
                'form': form,
                'form_points': points,
                'goals_scored_5': goals_scored,
                'goals_conceded_5': goals_conceded,
                'source': 'sofascore'
            }
            
        except Exception as e:
            logger.error(f"Error parsing SofaScore form: {e}")
            return {}


# =============================================================================
# FOOTYSTATS SCRAPER
# =============================================================================

class FootyStatsScraper(BaseScraper):
    """Scraper for FootyStats.org statistics."""
    
    BASE_URL = "https://footystats.org"
    
    def get_league_stats(self, league_slug: str = "england/premier-league") -> Dict:
        """Get league statistics from FootyStats."""
        url = f"{self.BASE_URL}/leagues/{league_slug}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = {}
            
            # Find stat cards
            cards = soup.find_all('div', class_='stat-card')
            
            for card in cards:
                title = card.find('div', class_='stat-title')
                value = card.find('div', class_='stat-value')
                
                if title and value:
                    stat_name = title.get_text(strip=True).lower().replace(' ', '_')
                    stat_value = value.get_text(strip=True)
                    
                    # Try to convert to number
                    try:
                        if '%' in stat_value:
                            stats[stat_name] = float(stat_value.replace('%', '')) / 100
                        else:
                            stats[stat_name] = float(stat_value)
                    except:
                        stats[stat_name] = stat_value
            
            stats['source'] = 'footystats'
            logger.info(f"Fetched {len(stats)} stats from FootyStats")
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing FootyStats: {e}")
            return {}
    
    def get_team_stats(self, team_slug: str) -> Dict:
        """Get detailed team statistics."""
        url = f"{self.BASE_URL}/clubs/{team_slug}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = {}
            
            # Parse team overview stats
            overview = soup.find('div', class_='team-overview')
            if overview:
                stat_items = overview.find_all('div', class_='stat-item')
                for item in stat_items:
                    label = item.find('span', class_='label')
                    value = item.find('span', class_='value')
                    if label and value:
                        name = label.get_text(strip=True).lower().replace(' ', '_')
                        val = value.get_text(strip=True)
                        try:
                            stats[name] = float(val.replace('%', ''))
                        except:
                            stats[name] = val
            
            # Parse xG if available
            xg_section = soup.find('div', class_='xg-stats')
            if xg_section:
                xg_for = xg_section.find('span', class_='xg-for')
                xg_against = xg_section.find('span', class_='xg-against')
                if xg_for:
                    stats['xg_for'] = float(xg_for.get_text(strip=True))
                if xg_against:
                    stats['xg_against'] = float(xg_against.get_text(strip=True))
            
            stats['source'] = 'footystats'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing FootyStats team: {e}")
            return {}
    
    def get_h2h(self, team1_slug: str, team2_slug: str) -> Dict:
        """Get head-to-head statistics."""
        url = f"{self.BASE_URL}/head2head/{team1_slug}-vs-{team2_slug}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            h2h = {}
            
            # Parse H2H record
            record = soup.find('div', class_='h2h-record')
            if record:
                team1_wins = record.find('span', class_='team1-wins')
                draws = record.find('span', class_='draws')
                team2_wins = record.find('span', class_='team2-wins')
                
                if team1_wins:
                    h2h['team1_wins'] = int(team1_wins.get_text(strip=True))
                if draws:
                    h2h['draws'] = int(draws.get_text(strip=True))
                if team2_wins:
                    h2h['team2_wins'] = int(team2_wins.get_text(strip=True))
            
            # Parse goals
            goals = soup.find('div', class_='h2h-goals')
            if goals:
                avg_goals = goals.find('span', class_='avg-goals')
                if avg_goals:
                    h2h['avg_goals'] = float(avg_goals.get_text(strip=True))
            
            h2h['source'] = 'footystats'
            return h2h
            
        except Exception as e:
            logger.error(f"Error parsing FootyStats H2H: {e}")
            return {}


# =============================================================================
# SOCCERSTATS SCRAPER
# =============================================================================

class SoccerStatsScraper(BaseScraper):
    """Scraper for SoccerStats.com statistics."""
    
    BASE_URL = "https://www.soccerstats.com"
    
    def get_league_stats(self, league: str = "eng-premier-league") -> Dict:
        """Get league statistics from SoccerStats."""
        url = f"{self.BASE_URL}/l/{league}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = {}
            
            # Find teams table
            table = soup.find('table', {'id': 'btable'})
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                teams = {}
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 9:
                        team_name = cells[1].get_text(strip=True)
                        teams[team_name] = {
                            'played': int(cells[2].get_text(strip=True) or 0),
                            'won': int(cells[3].get_text(strip=True) or 0),
                            'drawn': int(cells[4].get_text(strip=True) or 0),
                            'lost': int(cells[5].get_text(strip=True) or 0),
                            'goals_for': int(cells[6].get_text(strip=True) or 0),
                            'goals_against': int(cells[7].get_text(strip=True) or 0),
                            'points': int(cells[8].get_text(strip=True) or 0)
                        }
                
                stats['teams'] = teams
            
            # Get goal timing stats
            timing_table = soup.find('table', {'class': 'trow2'})
            if timing_table:
                stats['goal_timing'] = self._parse_goal_timing(timing_table)
            
            stats['source'] = 'soccerstats'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing SoccerStats: {e}")
            return {}
    
    def _parse_goal_timing(self, table) -> Dict:
        """Parse goal timing statistics."""
        timing = {}
        
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                period = cells[0].get_text(strip=True)
                goals = cells[1].get_text(strip=True)
                try:
                    timing[period] = int(goals)
                except:
                    timing[period] = goals
        
        return timing
    
    def get_team_stats(self, team: str) -> Dict:
        """Get team statistics from SoccerStats."""
        # Normalize team name for URL
        team_slug = team.lower().replace(' ', '-')
        url = f"{self.BASE_URL}/team/{team_slug}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = {}
            
            # Parse team stats tables
            tables = soup.find_all('table', class_='stat')
            
            for table in tables:
                title = table.find_previous('h2')
                if title:
                    section_name = title.get_text(strip=True).lower().replace(' ', '_')
                    
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            stat_name = cells[0].get_text(strip=True)
                            stat_value = cells[1].get_text(strip=True)
                            
                            key = f"{section_name}_{stat_name}".lower().replace(' ', '_')
                            try:
                                stats[key] = float(stat_value.replace('%', ''))
                            except:
                                stats[key] = stat_value
            
            stats['source'] = 'soccerstats'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing SoccerStats team: {e}")
            return {}


# =============================================================================
# WHOSCORED SCRAPER
# =============================================================================

class WhoScoredScraper(BaseScraper):
    """Scraper for WhoScored.com statistics."""
    
    BASE_URL = "https://www.whoscored.com"
    
    def __init__(self):
        super().__init__()
        self.rate_limit = 2.0  # WhoScored is more restrictive
    
    def get_team_stats(self, team_url: str) -> Dict:
        """Get team statistics from WhoScored."""
        response = self._get(team_url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = {}
            
            # Parse team rating
            rating = soup.find('span', class_='team-rating')
            if rating:
                stats['overall_rating'] = float(rating.get_text(strip=True))
            
            # Parse stat categories
            stat_boxes = soup.find_all('div', class_='stat-box')
            for box in stat_boxes:
                label = box.find('span', class_='label')
                value = box.find('span', class_='value')
                if label and value:
                    stat_name = label.get_text(strip=True).lower().replace(' ', '_')
                    try:
                        stats[stat_name] = float(value.get_text(strip=True))
                    except:
                        stats[stat_name] = value.get_text(strip=True)
            
            stats['source'] = 'whoscored'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing WhoScored: {e}")
            return {}
    
    def get_league_statistics(self) -> Dict:
        """Get league-wide statistics."""
        url = f"{self.BASE_URL}/Statistics"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = {}
            
            # Look for team stat tables
            tables = soup.find_all('table', id='top-team-stats-summary')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        team = cells[1].get_text(strip=True)
                        rating = cells[2].get_text(strip=True)
                        try:
                            stats[team] = {'rating': float(rating)}
                        except:
                            stats[team] = {'rating': rating}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing WhoScored statistics: {e}")
            return {}


# =============================================================================
# FOTMOB SCRAPER
# =============================================================================

class FotMobScraper(BaseScraper):
    """Scraper for FotMob.com statistics via their API."""
    
    # Multiple API endpoints to try
    API_URLS = [
        "https://www.fotmob.com/api",
        "https://api.fotmob.com/api",
    ]
    
    def __init__(self):
        super().__init__()
        self.session.headers.update({
            'Accept': 'application/json',
            'Origin': 'https://www.fotmob.com',
            'Referer': 'https://www.fotmob.com/'
        })
        self.api_url = self.API_URLS[0]
    
    def get_matches(self, date: str = None) -> List[Dict]:
        """Get matches for a specific date."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        url = f"{self.API_URL}/matches?date={date}"
        response = self._get(url)
        
        if not response:
            return []
        
        try:
            data = response.json()
            matches = []
            
            leagues = data.get('leagues', [])
            for league in leagues:
                for match in league.get('matches', []):
                    matches.append({
                        'match_id': match.get('id'),
                        'home_team': match.get('home', {}).get('name', ''),
                        'away_team': match.get('away', {}).get('name', ''),
                        'league': league.get('name', ''),
                        'status': match.get('status', {}).get('utcTime', ''),
                        'source': 'fotmob'
                    })
            
            logger.info(f"Fetched {len(matches)} matches from FotMob")
            return matches
            
        except Exception as e:
            logger.error(f"Error parsing FotMob matches: {e}")
            return []
    
    def get_match_details(self, match_id: int) -> Dict:
        """Get detailed match statistics."""
        url = f"{self.API_URL}/matchDetails?matchId={match_id}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            
            stats = {
                'home_team': data.get('general', {}).get('homeTeam', {}).get('name', ''),
                'away_team': data.get('general', {}).get('awayTeam', {}).get('name', ''),
            }
            
            # Extract xG if available
            content = data.get('content', {})
            stats_data = content.get('stats', {}).get('Ede', [])
            
            for stat_group in stats_data:
                for stat in stat_group.get('stats', []):
                    stat_name = stat.get('title', '').lower().replace(' ', '_')
                    stats_values = stat.get('stats', [])
                    if len(stats_values) >= 2:
                        stats[f'home_{stat_name}'] = stats_values[0]
                        stats[f'away_{stat_name}'] = stats_values[1]
            
            # Get momentum if available
            momentum = content.get('momentum', {})
            if momentum:
                stats['momentum'] = momentum
            
            stats['source'] = 'fotmob'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing FotMob match details: {e}")
            return {}
    
    def get_team_stats(self, team_id: int) -> Dict:
        """Get team statistics from FotMob."""
        url = f"{self.API_URL}/teams?id={team_id}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            
            stats = {
                'name': data.get('details', {}).get('name', ''),
            }
            
            # Get form
            form = data.get('stats', {}).get('form', [])
            if form:
                stats['form'] = ''.join([r.get('result', '?') for r in form[-5:]])
            
            # Get statistics
            stat_data = data.get('stats', {}).get('topStats', [])
            for stat in stat_data:
                stat_name = stat.get('title', '').lower().replace(' ', '_')
                stats[stat_name] = stat.get('value', 0)
            
            stats['source'] = 'fotmob'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing FotMob team: {e}")
            return {}
    
    def get_league_stats(self, league_id: int) -> Dict:
        """Get league overview and team rankings."""
        url = f"{self.API_URL}/leagues?id={league_id}"
        response = self._get(url)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            
            stats = {
                'name': data.get('details', {}).get('name', ''),
                'teams': {}
            }
            
            # Parse standings
            table = data.get('table', [])
            for section in table:
                for row in section.get('table', {}).get('all', []):
                    team_name = row.get('name', '')
                    stats['teams'][team_name] = {
                        'position': row.get('idx', 0),
                        'played': row.get('played', 0),
                        'won': row.get('wins', 0),
                        'drawn': row.get('draws', 0),
                        'lost': row.get('losses', 0),
                        'goals_for': row.get('scoresFor', 0),
                        'goals_against': row.get('scoresAgainst', 0),
                        'points': row.get('pts', 0)
                    }
            
            stats['source'] = 'fotmob'
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing FotMob league: {e}")
            return {}


# =============================================================================
# UNIFIED MULTI-SOURCE SCRAPER
# =============================================================================

class MultiSourceScraper:
    """Unified scraper that aggregates data from all sources."""
    
    def __init__(self):
        self.predictz = PredictZScraper()
        self.sofascore = SofaScoreScraper()
        self.footystats = FootyStatsScraper()
        self.soccerstats = SoccerStatsScraper()
        self.whoscored = WhoScoredScraper()
        self.fotmob = FotMobScraper()
        
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "multi_source"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_all_predictions(self, league: str = "premier-league") -> List[Dict]:
        """Get predictions from all sources."""
        all_predictions = []
        
        # Predictz
        try:
            preds = self.predictz.get_predictions(league)
            all_predictions.extend(preds)
        except Exception as e:
            logger.warning(f"Predictz scraping failed: {e}")
        
        return all_predictions
    
    def scrape_team_stats(self, team_name: str) -> Dict:
        """Get team stats from all available sources and merge."""
        merged_stats = {
            'team_name': team_name,
            'sources': []
        }
        
        # SofaScore form (if we had team ID)
        # For now, we'll aggregate what we can
        
        # FootyStats
        try:
            team_slug = team_name.lower().replace(' ', '-')
            footy_stats = self.footystats.get_team_stats(team_slug)
            if footy_stats:
                merged_stats.update({f"fs_{k}": v for k, v in footy_stats.items()})
                merged_stats['sources'].append('footystats')
        except Exception as e:
            logger.warning(f"FootyStats scraping failed for {team_name}: {e}")
        
        # SoccerStats
        try:
            soccer_stats = self.soccerstats.get_team_stats(team_name)
            if soccer_stats:
                merged_stats.update({f"ss_{k}": v for k, v in soccer_stats.items()})
                merged_stats['sources'].append('soccerstats')
        except Exception as e:
            logger.warning(f"SoccerStats scraping failed for {team_name}: {e}")
        
        return merged_stats
    
    def scrape_league_data(self, league: str = "premier-league") -> Dict:
        """Get league data from all sources."""
        league_data = {
            'league': league,
            'predictions': [],
            'team_stats': {},
            'league_stats': {}
        }
        
        # Predictz predictions
        try:
            league_data['predictions'] = self.predictz.get_predictions(league)
        except Exception as e:
            logger.warning(f"Predictz predictions failed: {e}")
        
        # FootyStats league stats
        try:
            footy_league = self.footystats.get_league_stats(f"england/{league}")
            league_data['league_stats']['footystats'] = footy_league
        except Exception as e:
            logger.warning(f"FootyStats league failed: {e}")
        
        # SoccerStats
        try:
            soccer_league = self.soccerstats.get_league_stats(f"eng-{league}")
            league_data['league_stats']['soccerstats'] = soccer_league
        except Exception as e:
            logger.warning(f"SoccerStats league failed: {e}")
        
        return league_data
    
    def scrape_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Get upcoming matches from all sources."""
        all_matches = []
        seen = set()
        
        # SofaScore
        try:
            sofa_matches = self.sofascore.get_upcoming_matches(days)
            for match in sofa_matches:
                key = f"{match['home_team']}_{match['away_team']}"
                if key not in seen:
                    all_matches.append(match)
                    seen.add(key)
        except Exception as e:
            logger.warning(f"SofaScore matches failed: {e}")
        
        # FotMob
        try:
            for day_offset in range(min(days, 3)):  # FotMob date-based
                date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y%m%d")
                fm_matches = self.fotmob.get_matches(date)
                for match in fm_matches:
                    key = f"{match['home_team']}_{match['away_team']}"
                    if key not in seen:
                        all_matches.append(match)
                        seen.add(key)
        except Exception as e:
            logger.warning(f"FotMob matches failed: {e}")
        
        logger.info(f"Total unique matches from all sources: {len(all_matches)}")
        return all_matches
    
    def save_data(self, data: Any, filename: str):
        """Save scraped data to file."""
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved data to {filepath}")
        return filepath


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for multi-source scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Source Statistics Scraper')
    parser.add_argument('--source', type=str, choices=['all', 'predictz', 'sofascore', 'footystats', 'soccerstats', 'whoscored', 'fotmob'],
                       default='all', help='Source to scrape')
    parser.add_argument('--league', type=str, default='premier-league', help='League to scrape')
    parser.add_argument('--days', type=int, default=7, help='Days ahead to scrape')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    scraper = MultiSourceScraper()
    
    print("="*60)
    print(f"Multi-Source Statistics Scraper")
    print("="*60)
    
    if args.source == 'all':
        # Scrape from all sources
        print("\n📊 Scraping upcoming matches...")
        matches = scraper.scrape_upcoming_matches(args.days)
        print(f"   Found {len(matches)} matches")
        
        print("\n📈 Scraping league data...")
        league_data = scraper.scrape_league_data(args.league)
        print(f"   Predictions: {len(league_data.get('predictions', []))}")
        
        if args.save:
            scraper.save_data(matches, f"matches_{datetime.now().strftime('%Y%m%d')}.json")
            scraper.save_data(league_data, f"league_{args.league}_{datetime.now().strftime('%Y%m%d')}.json")
    
    elif args.source == 'predictz':
        predictions = scraper.predictz.get_predictions(args.league)
        print(f"Predictz predictions: {len(predictions)}")
        for pred in predictions[:5]:
            print(f"  {pred['home_team']} vs {pred['away_team']}: {pred.get('prediction', 'N/A')}")
    
    elif args.source == 'fotmob':
        matches = scraper.fotmob.get_matches()
        print(f"FotMob matches today: {len(matches)}")
        for m in matches[:5]:
            print(f"  {m['home_team']} vs {m['away_team']} ({m['league']})")
    
    elif args.source == 'sofascore':
        matches = scraper.sofascore.get_upcoming_matches(args.days)
        print(f"SofaScore matches: {len(matches)}")
        for m in matches[:5]:
            print(f"  {m['home_team']} vs {m['away_team']} ({m['tournament']})")
    
    print("\n✅ Scraping complete!")


if __name__ == "__main__":
    main()
