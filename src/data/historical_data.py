"""
Historical Data Storage (SQLite)

Stores historical match data for:
- ML model training
- H2H analysis
- ELO rating calculation
- Form tracking
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'football_history.db')


@dataclass
class HistoricalMatch:
    """Historical match record"""
    id: str
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    league: str
    season: str


class HistoricalDatabase:
    """
    SQLite database for storing historical match data.
    Used for ML training, H2H lookup, and ELO calculation.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Matches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_score INTEGER,
                    away_score INTEGER,
                    league TEXT,
                    season TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Team ELO ratings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_elo (
                    team TEXT PRIMARY KEY,
                    elo REAL DEFAULT 1500,
                    matches_played INTEGER DEFAULT 0,
                    last_updated TEXT
                )
            ''')
            
            # Team statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_stats (
                    team TEXT PRIMARY KEY,
                    wins INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    goals_for INTEGER DEFAULT 0,
                    goals_against INTEGER DEFAULT 0,
                    last_5_results TEXT DEFAULT '[]',
                    last_updated TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league)')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_match(self, match: Dict) -> bool:
        """Store a match result"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO matches 
                    (id, date, home_team, away_team, home_score, away_score, league, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match.get('id', f"{match['home_team']}_{match['away_team']}_{match['date']}"),
                    match.get('date'),
                    match.get('home_team'),
                    match.get('away_team'),
                    match.get('home_score'),
                    match.get('away_score'),
                    match.get('league', 'unknown'),
                    match.get('season', '2024-25')
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error storing match: {e}")
            return False
    
    def store_matches_batch(self, matches: List[Dict]) -> int:
        """Store multiple matches at once"""
        stored = 0
        for match in matches:
            if self.store_match(match):
                stored += 1
        return stored
    
    def get_h2h(self, team1: str, team2: str, limit: int = 10) -> List[Dict]:
        """Get head-to-head matches between two teams"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM matches 
                WHERE (home_team LIKE ? AND away_team LIKE ?)
                   OR (home_team LIKE ? AND away_team LIKE ?)
                ORDER BY date DESC
                LIMIT ?
            ''', (f'%{team1}%', f'%{team2}%', f'%{team2}%', f'%{team1}%', limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_h2h_stats(self, team1: str, team2: str) -> Dict:
        """Get H2H statistics between two teams"""
        matches = self.get_h2h(team1, team2, limit=50)
        
        if not matches:
            return {'found': False, 'total_matches': 0}
        
        team1_wins = 0
        draws = 0
        team2_wins = 0
        team1_goals = 0
        team2_goals = 0
        last_5 = []
        
        for match in matches:
            home = match['home_team']
            h_score = match['home_score'] or 0
            a_score = match['away_score'] or 0
            
            is_team1_home = team1.lower() in home.lower()
            
            if is_team1_home:
                team1_goals += h_score
                team2_goals += a_score
                if h_score > a_score:
                    team1_wins += 1
                elif h_score == a_score:
                    draws += 1
                else:
                    team2_wins += 1
            else:
                team1_goals += a_score
                team2_goals += h_score
                if a_score > h_score:
                    team1_wins += 1
                elif a_score == h_score:
                    draws += 1
                else:
                    team2_wins += 1
            
            if len(last_5) < 5:
                last_5.append({
                    'date': match['date'],
                    'home_score': h_score,
                    'away_score': a_score,
                    'result': 'H' if h_score > a_score else ('D' if h_score == a_score else 'A')
                })
        
        total = len(matches)
        return {
            'found': True,
            'total_matches': total,
            'team1': team1,
            'team2': team2,
            'team1_wins': team1_wins,
            'draws': draws,
            'team2_wins': team2_wins,
            'team1_goals': team1_goals,
            'team2_goals': team2_goals,
            'avg_goals': round((team1_goals + team2_goals) / total, 2) if total > 0 else 0,
            'last_5': last_5
        }
    
    def get_team_form(self, team: str, limit: int = 5) -> List[str]:
        """Get team's recent form (W/D/L)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM matches 
                WHERE home_team LIKE ? OR away_team LIKE ?
                ORDER BY date DESC
                LIMIT ?
            ''', (f'%{team}%', f'%{team}%', limit))
            
            results = []
            for row in cursor.fetchall():
                match = dict(row)
                h_score = match['home_score'] or 0
                a_score = match['away_score'] or 0
                is_home = team.lower() in match['home_team'].lower()
                
                if is_home:
                    if h_score > a_score:
                        results.append('W')
                    elif h_score == a_score:
                        results.append('D')
                    else:
                        results.append('L')
                else:
                    if a_score > h_score:
                        results.append('W')
                    elif a_score == h_score:
                        results.append('D')
                    else:
                        results.append('L')
            
            return results
    
    def update_team_elo(self, team: str, new_elo: float):
        """Update team's ELO rating"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO team_elo (team, elo, matches_played, last_updated)
                VALUES (?, ?, COALESCE((SELECT matches_played FROM team_elo WHERE team = ?) + 1, 1), ?)
            ''', (team, new_elo, team, datetime.now().isoformat()))
            conn.commit()
    
    def get_team_elo(self, team: str) -> float:
        """Get team's ELO rating"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT elo FROM team_elo WHERE team LIKE ?', (f'%{team}%',))
            row = cursor.fetchone()
            return row['elo'] if row else 1500.0
    
    def get_match_count(self) -> int:
        """Get total number of stored matches"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM matches')
            return cursor.fetchone()['count']
    
    def get_all_teams(self) -> List[str]:
        """Get all unique team names"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT home_team FROM matches
                UNION
                SELECT DISTINCT away_team FROM matches
            ''')
            return [row['home_team'] for row in cursor.fetchall()]


# Global instance
history_db = HistoricalDatabase()


def sync_from_api():
    """Sync historical data from Football-Data.org API"""
    from src.data.api_clients import FootballDataOrgClient
    
    client = FootballDataOrgClient()
    leagues = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
    
    total_stored = 0
    for league in leagues:
        matches = client.get_finished_matches(league, limit=100)
        for match in matches:
            db_match = {
                'id': str(match.get('id')),
                'date': match.get('utcDate', '')[:10],
                'home_team': match.get('homeTeam', {}).get('name', ''),
                'away_team': match.get('awayTeam', {}).get('name', ''),
                'home_score': match.get('score', {}).get('fullTime', {}).get('home'),
                'away_score': match.get('score', {}).get('fullTime', {}).get('away'),
                'league': league,
                'season': match.get('season', {}).get('id', '2024-25')
            }
            if history_db.store_match(db_match):
                total_stored += 1
    
    return total_stored
