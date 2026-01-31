"""
Result Tracker - Automatic accuracy tracking for predictions
============================================================

This module:
1. Stores predictions with match IDs in a SQLite database
2. Fetches actual match results from football-data.org API
3. Compares predictions vs results and calculates accuracy metrics
4. Provides running accuracy statistics
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultTracker:
    """Track prediction results and calculate accuracy metrics."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent.parent / "data" / "results.db")
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT UNIQUE,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT,
                match_date TEXT NOT NULL,
                market TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL,
                confidence TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT UNIQUE,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                match_date TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                status TEXT DEFAULT 'FINISHED',
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES predictions (match_id)
            )
        ''')
        
        # Accuracy tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accuracy_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                market TEXT NOT NULL,
                predictions_count INTEGER,
                correct_count INTEGER,
                accuracy REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized results database at {self.db_path}")
    
    def store_prediction(self, prediction: Dict) -> bool:
        """Store a prediction in the database."""
        try:
            home = prediction.get('home_team', prediction.get('match', '').split(' vs ')[0])
            away = prediction.get('away_team', prediction.get('match', '').split(' vs ')[1] if ' vs ' in prediction.get('match', '') else '')
            match_date = prediction.get('match_date', datetime.now().strftime('%Y-%m-%d'))
            
            # Generate unique match ID
            match_id = f"{home}_{away}_{match_date}".lower().replace(' ', '_')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (match_id, home_team, away_team, league, match_date, market, prediction, probability, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id,
                home,
                away,
                prediction.get('league', ''),
                match_date,
                prediction.get('market', ''),
                prediction.get('prediction', ''),
                prediction.get('probability', 0),
                prediction.get('confidence', 'MEDIUM')
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return False
    
    def store_predictions_batch(self, predictions: List[Dict]) -> int:
        """Store multiple predictions at once."""
        stored = 0
        for pred in predictions:
            if self.store_prediction(pred):
                stored += 1
        return stored
    
    def fetch_results(self, date: str = None) -> List[Dict]:
        """Fetch match results from football-data.org API."""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        results = []
        
        # Football-Data.org API (free tier)
        api_key = "YOUR_API_KEY"  # Replace with actual API key
        base_url = "https://api.football-data.org/v4"
        
        try:
            # Get finished matches for the date
            response = requests.get(
                f"{base_url}/matches",
                params={
                    'dateFrom': date,
                    'dateTo': date,
                    'status': 'FINISHED'
                },
                headers={'X-Auth-Token': api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for match in data.get('matches', []):
                    home = match['homeTeam']['name']
                    away = match['awayTeam']['name']
                    match_id = f"{home}_{away}_{date}".lower().replace(' ', '_')
                    
                    result = {
                        'match_id': match_id,
                        'home_team': home,
                        'away_team': away,
                        'match_date': date,
                        'home_score': match['score']['fullTime']['home'],
                        'away_score': match['score']['fullTime']['away'],
                        'status': 'FINISHED'
                    }
                    results.append(result)
                    
                    # Store result in database
                    self._store_result(result)
                    
        except Exception as e:
            logger.error(f"Error fetching results: {e}")
        
        return results
    
    def _store_result(self, result: Dict):
        """Store a single result in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO results
                (match_id, home_team, away_team, match_date, home_score, away_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['match_id'],
                result['home_team'],
                result['away_team'],
                result['match_date'],
                result['home_score'],
                result['away_score'],
                result['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing result: {e}")
    
    def evaluate_predictions(self, date: str = None) -> Dict:
        """Evaluate predictions against actual results."""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predictions with matching results
        cursor.execute('''
            SELECT p.match_id, p.market, p.prediction, p.probability,
                   r.home_score, r.away_score
            FROM predictions p
            JOIN results r ON p.match_id = r.match_id
            WHERE p.match_date = ?
        ''', (date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        evaluations = {
            'date': date,
            'total': len(rows),
            'by_market': {},
            'overall_accuracy': 0
        }
        
        correct_total = 0
        
        for row in rows:
            match_id, market, prediction, probability, home_score, away_score = row
            
            # Evaluate based on market type
            is_correct = self._evaluate_single(market, prediction, home_score, away_score)
            
            if market not in evaluations['by_market']:
                evaluations['by_market'][market] = {'total': 0, 'correct': 0, 'accuracy': 0}
            
            evaluations['by_market'][market]['total'] += 1
            if is_correct:
                evaluations['by_market'][market]['correct'] += 1
                correct_total += 1
        
        # Calculate accuracies
        if evaluations['total'] > 0:
            evaluations['overall_accuracy'] = round(correct_total / evaluations['total'] * 100, 1)
        
        for market in evaluations['by_market']:
            market_data = evaluations['by_market'][market]
            if market_data['total'] > 0:
                market_data['accuracy'] = round(market_data['correct'] / market_data['total'] * 100, 1)
        
        # Log to accuracy table
        self._log_accuracy(evaluations)
        
        return evaluations
    
    def _evaluate_single(self, market: str, prediction: str, home_score: int, away_score: int) -> bool:
        """Evaluate a single prediction against the result."""
        total_goals = home_score + away_score
        
        market_lower = market.lower()
        prediction_lower = prediction.lower()
        
        # Over/Under goals
        if 'over 0.5' in market_lower:
            return total_goals >= 1
        elif 'over 1.5' in market_lower:
            return total_goals >= 2
        elif 'over 2.5' in market_lower:
            return total_goals >= 3
        
        # Double Chance
        elif 'double chance' in market_lower or 'dc 1x' in market_lower:
            if '1x' in market_lower or 'or draw' in prediction_lower:
                return home_score >= away_score  # Home win or draw
            elif 'x2' in market_lower:
                return away_score >= home_score  # Away win or draw
        
        # Result prediction
        elif 'win' in market_lower or 'result' in market_lower:
            if 'home' in prediction_lower:
                return home_score > away_score
            elif 'away' in prediction_lower:
                return away_score > home_score
            elif 'draw' in prediction_lower:
                return home_score == away_score
        
        # BTTS
        elif 'btts' in market_lower or 'both teams' in market_lower:
            return (home_score > 0 and away_score > 0) == ('yes' in prediction_lower)
        
        # First half (if we have HT data - for now assume based on FT)
        elif 'ht' in market_lower or 'first half' in market_lower:
            # Approximate - assume if total goals high, HT likely had goals
            return total_goals >= 2 if 'over' in market_lower else total_goals < 2
        
        return False
    
    def _log_accuracy(self, evaluations: Dict):
        """Log accuracy results to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for market, data in evaluations['by_market'].items():
                cursor.execute('''
                    INSERT INTO accuracy_log (date, market, predictions_count, correct_count, accuracy)
                    VALUES (?, ?, ?, ?, ?)
                ''', (evaluations['date'], market, data['total'], data['correct'], data['accuracy']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging accuracy: {e}")
    
    def get_accuracy_summary(self, days: int = 7) -> Dict:
        """Get accuracy summary for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT market, 
                   SUM(predictions_count) as total,
                   SUM(correct_count) as correct,
                   ROUND(SUM(correct_count) * 100.0 / SUM(predictions_count), 1) as accuracy
            FROM accuracy_log
            WHERE date >= ?
            GROUP BY market
            ORDER BY accuracy DESC
        ''', (since_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {
            'period': f'Last {days} days',
            'since': since_date,
            'by_market': {
                row[0]: {'total': row[1], 'correct': row[2], 'accuracy': row[3]}
                for row in rows
            }
        }


# Singleton
_tracker = None

def get_result_tracker() -> ResultTracker:
    global _tracker
    if _tracker is None:
        _tracker = ResultTracker()
    return _tracker


if __name__ == "__main__":
    tracker = get_result_tracker()
    
    # Example: Store some test predictions
    test_predictions = [
        {
            'match': 'Liverpool vs Manchester United',
            'match_date': '2026-01-30',
            'market': 'Double Chance 1X',
            'prediction': 'Liverpool or Draw',
            'probability': 90,
            'confidence': 'HIGH'
        },
        {
            'match': 'Bayern Munich vs Dortmund',
            'match_date': '2026-01-30',
            'market': 'Over 1.5 Goals',
            'prediction': 'Yes',
            'probability': 88,
            'confidence': 'HIGH'
        }
    ]
    
    stored = tracker.store_predictions_batch(test_predictions)
    print(f"Stored {stored} predictions")
    
    # Get accuracy summary
    summary = tracker.get_accuracy_summary()
    print(f"\nAccuracy Summary: {json.dumps(summary, indent=2)}")
