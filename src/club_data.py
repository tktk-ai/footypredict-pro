"""
Club Football Data Provider

Fetches data from top European leagues for training.
Sources: football-data.org, OpenLigaDB, and other free providers.
"""

import logging
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CLUB_DATA_DIR = DATA_DIR / "club_football"
CLUB_DATA_DIR.mkdir(parents=True, exist_ok=True)


class ClubFootballProvider:
    """Fetch club football data from multiple free sources"""
    
    # Football-data.org (free tier: 10 req/min)
    FD_API = "https://api.football-data.org/v4"
    
    # Available competitions (free tier)
    COMPETITIONS = {
        'PL': 'Premier League',
        'ELC': 'Championship',
        'BL1': 'Bundesliga',
        'SA': 'Serie A',
        'PD': 'La Liga',
        'FL1': 'Ligue 1',
        'DED': 'Eredivisie',
        'PPL': 'Primeira Liga',
        'CL': 'Champions League',
        'EC': 'European Championship'
    }
    
    def __init__(self):
        self.api_key = os.environ.get('FOOTBALL_DATA_API_KEY', '')
        self.cache: Dict[str, Dict] = {}
        self.last_request = datetime.min
        self.rate_limit = 6  # seconds between requests
    
    async def _fetch(self, endpoint: str) -> Dict:
        """Fetch from football-data.org with rate limiting"""
        # Rate limiting
        since_last = (datetime.now() - self.last_request).seconds
        if since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - since_last)
        
        self.last_request = datetime.now()
        
        headers = {}
        if self.api_key:
            headers['X-Auth-Token'] = self.api_key
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.FD_API}/{endpoint}"
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        logger.warning("Rate limited, waiting...")
                        await asyncio.sleep(60)
                    else:
                        logger.warning(f"API returned {resp.status}")
        except Exception as e:
            logger.error(f"Fetch error: {e}")
        
        return {}
    
    async def get_matches(self, competition: str = 'PL', season: str = None) -> List[Dict]:
        """Get matches for a competition"""
        season = season or str(datetime.now().year)
        
        # Check cache
        cache_key = f"{competition}_{season}"
        cache_file = CLUB_DATA_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # If recent data, use cache
                cache_time = datetime.fromisoformat(data.get('fetched', '2000-01-01'))
                if (datetime.now() - cache_time).days < 1:
                    return data.get('matches', [])
        
        # Fetch fresh data
        data = await self._fetch(f"competitions/{competition}/matches?season={season}")
        matches = data.get('matches', [])
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                'matches': matches,
                'fetched': datetime.now().isoformat()
            }, f)
        
        return matches
    
    async def get_team_matches(self, team_id: int, limit: int = 20) -> List[Dict]:
        """Get recent matches for a team"""
        data = await self._fetch(f"teams/{team_id}/matches?limit={limit}")
        return data.get('matches', [])
    
    async def get_standings(self, competition: str = 'PL') -> Dict:
        """Get current standings"""
        data = await self._fetch(f"competitions/{competition}/standings")
        return data
    
    def format_for_training(self, matches: List[Dict]) -> List[Dict]:
        """Format matches for ML training"""
        formatted = []
        
        for m in matches:
            if m.get('status') != 'FINISHED':
                continue
            
            score = m.get('score', {}).get('fullTime', {})
            if score.get('home') is None:
                continue
            
            formatted.append({
                'date': m.get('utcDate', '')[:10],
                'home_team': m.get('homeTeam', {}).get('name', ''),
                'away_team': m.get('awayTeam', {}).get('name', ''),
                'home_score': score.get('home', 0),
                'away_score': score.get('away', 0),
                'competition': m.get('competition', {}).get('name', ''),
                'matchday': m.get('matchday'),
                'venue': m.get('venue', ''),
            })
        
        return formatted
    
    async def download_all_training_data(self) -> int:
        """Download training data from all available competitions"""
        all_matches = []
        
        for code, name in self.COMPETITIONS.items():
            try:
                logger.info(f"Fetching {name}...")
                matches = await self.get_matches(code)
                formatted = self.format_for_training(matches)
                all_matches.extend(formatted)
                logger.info(f"  Got {len(formatted)} matches")
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
        
        # Save combined data
        if all_matches:
            output_file = CLUB_DATA_DIR / "all_club_matches.json"
            with open(output_file, 'w') as f:
                json.dump(all_matches, f, indent=2)
            logger.info(f"Saved {len(all_matches)} total club matches")
        
        return len(all_matches)


class LiveDataPipeline:
    """Real-time data updates and live scores"""
    
    # Free live score sources
    LIVESCORE_API = "https://api.football-data.org/v4/matches"
    
    def __init__(self):
        self.api_key = os.environ.get('FOOTBALL_DATA_API_KEY', '')
        self.live_matches: Dict[str, Dict] = {}
        self.update_callbacks: List = []
    
    async def get_live_matches(self) -> List[Dict]:
        """Get currently live matches"""
        headers = {'X-Auth-Token': self.api_key} if self.api_key else {}
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {'status': 'LIVE'}
                async with session.get(self.LIVESCORE_API, headers=headers, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('matches', [])
        except Exception as e:
            logger.error(f"Live data error: {e}")
        
        return []
    
    async def get_todays_matches(self) -> List[Dict]:
        """Get all matches scheduled for today"""
        today = datetime.now().strftime('%Y-%m-%d')
        headers = {'X-Auth-Token': self.api_key} if self.api_key else {}
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {'dateFrom': today, 'dateTo': today}
                async with session.get(self.LIVESCORE_API, headers=headers, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('matches', [])
        except Exception as e:
            logger.error(f"Today's matches error: {e}")
        
        return []
    
    def format_live_match(self, match: Dict) -> Dict:
        """Format live match for display"""
        score = match.get('score', {})
        return {
            'id': match.get('id'),
            'home_team': match.get('homeTeam', {}).get('name', '?'),
            'away_team': match.get('awayTeam', {}).get('name', '?'),
            'home_score': score.get('fullTime', {}).get('home', 0),
            'away_score': score.get('fullTime', {}).get('away', 0),
            'minute': match.get('minute'),
            'status': match.get('status'),
            'competition': match.get('competition', {}).get('name', ''),
            'in_play': match.get('status') == 'IN_PLAY'
        }
    
    async def start_live_updates(self, interval: int = 60):
        """Start polling for live updates"""
        while True:
            try:
                matches = await self.get_live_matches()
                for match in matches:
                    formatted = self.format_live_match(match)
                    match_id = str(formatted['id'])
                    
                    # Check for changes
                    old = self.live_matches.get(match_id)
                    if old and (old['home_score'] != formatted['home_score'] or 
                               old['away_score'] != formatted['away_score']):
                        # Score changed! Notify callbacks
                        for callback in self.update_callbacks:
                            try:
                                callback(formatted, 'goal')
                            except:
                                pass
                    
                    self.live_matches[match_id] = formatted
                
            except Exception as e:
                logger.error(f"Live update error: {e}")
            
            await asyncio.sleep(interval)
    
    def on_update(self, callback):
        """Register callback for live updates"""
        self.update_callbacks.append(callback)


# Global instances
_club_provider: Optional[ClubFootballProvider] = None
_live_pipeline: Optional[LiveDataPipeline] = None

def get_club_provider() -> ClubFootballProvider:
    global _club_provider
    if _club_provider is None:
        _club_provider = ClubFootballProvider()
    return _club_provider

def get_live_pipeline() -> LiveDataPipeline:
    global _live_pipeline
    if _live_pipeline is None:
        _live_pipeline = LiveDataPipeline()
    return _live_pipeline

def download_club_data() -> int:
    """Download all club football data"""
    loop = asyncio.new_event_loop()
    count = loop.run_until_complete(get_club_provider().download_all_training_data())
    loop.close()
    return count

def get_live_matches() -> List[Dict]:
    """Get live matches (sync)"""
    loop = asyncio.new_event_loop()
    matches = loop.run_until_complete(get_live_pipeline().get_live_matches())
    loop.close()
    return [get_live_pipeline().format_live_match(m) for m in matches]

def get_todays_fixtures() -> List[Dict]:
    """Get today's fixtures (sync)"""
    loop = asyncio.new_event_loop()
    matches = loop.run_until_complete(get_live_pipeline().get_todays_matches())
    loop.close()
    return matches
