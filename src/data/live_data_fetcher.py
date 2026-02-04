"""
Live Data Fetcher
==================

Automatically fetches:
- Daily fixtures from OpenLigaDB
- Match results after games complete
- Live score updates during matches
"""

import requests
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import schedule
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "predictions.db"


class LiveDataFetcher:
    """
    Fetches live data from multiple sources.
    """
    
    # API endpoints
    OPENLIGA_BASE = "https://api.openligadb.de"
    SPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"
    
    # Supported leagues for OpenLigaDB
    OPENLIGA_LEAGUES = {
        'bl1': {'name': 'Bundesliga', 'season': 2025},
        'bl2': {'name': 'Bundesliga 2', 'season': 2025},
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.db_conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database connection and tables."""
        self.db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        
        # Create live data tables
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_fixtures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT UNIQUE,
                source TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                match_date DATETIME,
                match_time TEXT,
                status TEXT DEFAULT 'scheduled',
                home_score INTEGER,
                away_score INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER,
                market TEXT,
                prediction TEXT,
                probability REAL,
                actual_outcome TEXT,
                is_correct INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fixture_id) REFERENCES live_fixtures(id)
            )
        ''')
        
        self.db_conn.commit()
    
    def fetch_today_fixtures(self) -> List[Dict]:
        """Fetch today's fixtures from all sources."""
        today = datetime.now().strftime('%Y-%m-%d')
        all_fixtures = []
        
        # OpenLigaDB
        for league_id, config in self.OPENLIGA_LEAGUES.items():
            try:
                url = f"{self.OPENLIGA_BASE}/getmatchdata/{league_id}/{config['season']}"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    matches = response.json()
                    
                    for match in matches:
                        match_date = match.get('matchDateTime', '')[:10]
                        
                        if match_date == today:
                            fixture = {
                                'external_id': f"oldb_{match.get('matchID')}",
                                'source': 'openligadb',
                                'league': config['name'],
                                'home_team': match.get('team1', {}).get('teamName', ''),
                                'away_team': match.get('team2', {}).get('teamName', ''),
                                'match_date': match_date,
                                'match_time': match.get('matchDateTime', '')[11:16],
                                'status': 'finished' if match.get('matchIsFinished') else 'scheduled',
                                'home_score': None,
                                'away_score': None,
                            }
                            
                            if match.get('matchIsFinished'):
                                results = match.get('matchResults', [])
                                if results:
                                    final = results[-1]
                                    fixture['home_score'] = final.get('pointsTeam1')
                                    fixture['away_score'] = final.get('pointsTeam2')
                            
                            all_fixtures.append(fixture)
                            
            except Exception as e:
                logger.error(f"Error fetching OpenLigaDB {league_id}: {e}")
        
        # TheSportsDB (for more leagues)
        try:
            url = f"{self.SPORTSDB_BASE}/eventsday.php?d={today}&s=Soccer"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', []) or []
                
                for event in events:
                    fixture = {
                        'external_id': f"tsdb_{event.get('idEvent')}",
                        'source': 'thesportsdb',
                        'league': event.get('strLeague', ''),
                        'home_team': event.get('strHomeTeam', ''),
                        'away_team': event.get('strAwayTeam', ''),
                        'match_date': event.get('dateEvent', ''),
                        'match_time': event.get('strTime', '')[:5] if event.get('strTime') else '',
                        'status': 'finished' if event.get('intHomeScore') is not None else 'scheduled',
                        'home_score': int(event.get('intHomeScore')) if event.get('intHomeScore') else None,
                        'away_score': int(event.get('intAwayScore')) if event.get('intAwayScore') else None,
                    }
                    all_fixtures.append(fixture)
                    
        except Exception as e:
            logger.error(f"Error fetching TheSportsDB: {e}")
        
        logger.info(f"Fetched {len(all_fixtures)} fixtures for {today}")
        return all_fixtures
    
    def update_results(self) -> int:
        """Update results for pending fixtures."""
        cursor = self.db_conn.cursor()
        
        # Get pending fixtures
        cursor.execute('''
            SELECT id, external_id, source, league FROM live_fixtures 
            WHERE status = 'scheduled' AND match_date <= date('now')
        ''')
        pending = cursor.fetchall()
        
        updated_count = 0
        
        for fixture_id, external_id, source, league in pending:
            try:
                if source == 'openligadb':
                    # Fetch from OpenLigaDB
                    match_id = external_id.replace('oldb_', '')
                    url = f"{self.OPENLIGA_BASE}/getmatchdata/{match_id}"
                    response = self.session.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        match = response.json()
                        
                        if match.get('matchIsFinished'):
                            results = match.get('matchResults', [])
                            if results:
                                final = results[-1]
                                home_score = final.get('pointsTeam1')
                                away_score = final.get('pointsTeam2')
                                
                                cursor.execute('''
                                    UPDATE live_fixtures 
                                    SET status = 'finished', home_score = ?, away_score = ?, 
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                ''', (home_score, away_score, fixture_id))
                                
                                updated_count += 1
                                
            except Exception as e:
                logger.error(f"Error updating {external_id}: {e}")
        
        self.db_conn.commit()
        logger.info(f"Updated {updated_count} fixtures with results")
        
        return updated_count
    
    def save_fixtures(self, fixtures: List[Dict]):
        """Save fixtures to database."""
        cursor = self.db_conn.cursor()
        
        for fixture in fixtures:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO live_fixtures 
                    (external_id, source, league, home_team, away_team, match_date, 
                     match_time, status, home_score, away_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    fixture['external_id'],
                    fixture['source'],
                    fixture['league'],
                    fixture['home_team'],
                    fixture['away_team'],
                    fixture['match_date'],
                    fixture['match_time'],
                    fixture['status'],
                    fixture['home_score'],
                    fixture['away_score'],
                ))
            except Exception as e:
                logger.error(f"Error saving fixture: {e}")
        
        self.db_conn.commit()
        logger.info(f"Saved {len(fixtures)} fixtures to database")
    
    def get_upcoming_fixtures(self, days: int = 7) -> List[Dict]:
        """Get upcoming fixtures for next N days."""
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            SELECT id, league, home_team, away_team, match_date, match_time, status
            FROM live_fixtures 
            WHERE match_date >= date('now') 
              AND match_date <= date('now', '+' || ? || ' days')
              AND status = 'scheduled'
            ORDER BY match_date, match_time
        ''', (days,))
        
        fixtures = []
        for row in cursor.fetchall():
            fixtures.append({
                'id': row[0],
                'league': row[1],
                'home_team': row[2],
                'away_team': row[3],
                'match_date': row[4],
                'match_time': row[5],
                'status': row[6],
            })
        
        return fixtures
    
    def run_daily_update(self):
        """Run daily fixture and results update."""
        logger.info("Running daily data update...")
        
        # Fetch today's fixtures
        fixtures = self.fetch_today_fixtures()
        self.save_fixtures(fixtures)
        
        # Update any pending results
        self.update_results()
        
        logger.info("Daily update complete")
    
    def start_scheduler(self, run_now: bool = True):
        """Start background scheduler for automatic updates."""
        if run_now:
            self.run_daily_update()
        
        # Schedule daily updates at 6 AM
        schedule.every().day.at("06:00").do(self.run_daily_update)
        
        # Schedule result updates every 2 hours
        schedule.every(2).hours.do(self.update_results)
        
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        thread = threading.Thread(target=run_schedule, daemon=True)
        thread.start()
        
        logger.info("Scheduler started - updates at 6 AM daily, results every 2 hours")


def fetch_and_save_fixtures():
    """Quick function to fetch and save today's fixtures."""
    fetcher = LiveDataFetcher()
    fixtures = fetcher.fetch_today_fixtures()
    fetcher.save_fixtures(fixtures)
    return fixtures


def get_live_fetcher():
    """Get singleton instance of LiveDataFetcher."""
    return LiveDataFetcher()


if __name__ == "__main__":
    fetcher = LiveDataFetcher()
    fetcher.run_daily_update()
    
    # Show upcoming fixtures
    upcoming = fetcher.get_upcoming_fixtures(days=3)
    print(f"\n📅 Upcoming fixtures ({len(upcoming)}):")
    for f in upcoming[:10]:
        print(f"  {f['match_date']} {f['match_time']}: {f['home_team']} vs {f['away_team']} ({f['league']})")
