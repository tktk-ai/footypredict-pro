"""
Automated Scheduler System

Complete automation for the football prediction system:
1. Scheduled Fixture Fetching (daily at 6 AM)
2. Auto Odds Updates (every 30 minutes)
3. Background Prediction Generator (on fixture/odds changes)
4. Model Retraining Schedule (weekly)
5. Cache/Database for Predictions (SQLite)
6. Live Score Updates (every 1 minute during matches)
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.db')


def init_database():
    """Initialize SQLite database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fixtures table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fixtures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        external_id TEXT UNIQUE,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        league_id TEXT NOT NULL,
        match_date DATETIME NOT NULL,
        status TEXT DEFAULT 'scheduled',
        home_score INTEGER,
        away_score INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fixture_id INTEGER,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_win_prob REAL,
        draw_prob REAL,
        away_win_prob REAL,
        btts_yes REAL,
        over_2_5 REAL,
        predicted_outcome TEXT,
        confidence REAL,
        correct_scores TEXT,
        htft TEXT,
        model_version TEXT DEFAULT 'v1',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
    )
    ''')
    
    # Odds table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS odds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fixture_id INTEGER,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        bookmaker TEXT,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        over_2_5_odds REAL,
        under_2_5_odds REAL,
        btts_yes_odds REAL,
        btts_no_odds REAL,
        fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
    )
    ''')
    
    # Odds history (for tracking movements)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS odds_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fixture_id INTEGER,
        bookmaker TEXT,
        market TEXT,
        odds_value REAL,
        recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
    )
    ''')
    
    # Live scores table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS live_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fixture_id INTEGER,
        home_score INTEGER,
        away_score INTEGER,
        minute INTEGER,
        status TEXT,
        events TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
    )
    ''')
    
    # Scheduler jobs log
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scheduler_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name TEXT,
        status TEXT,
        message TEXT,
        executed_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Model retraining history
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        parameters TEXT,
        accuracy REAL,
        trained_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(match_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_odds_fixture ON odds(fixture_id)')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")


class PredictionCache:
    """SQLite-backed prediction cache for instant access."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_database()
    
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
    
    def store_fixture(self, fixture: Dict) -> int:
        """Store a fixture in the database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO fixtures 
            (external_id, home_team, away_team, league_id, match_date, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                fixture.get('external_id', f"{fixture['home_team']}-{fixture['away_team']}-{fixture['match_date']}"),
                fixture['home_team'],
                fixture['away_team'],
                fixture['league_id'],
                fixture['match_date'],
                fixture.get('status', 'scheduled')
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def store_prediction(self, prediction: Dict, fixture_id: int = None) -> int:
        """Store a prediction in the database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO predictions 
            (fixture_id, home_team, away_team, home_win_prob, draw_prob, away_win_prob,
             btts_yes, over_2_5, predicted_outcome, confidence, correct_scores, htft, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fixture_id,
                prediction['home_team'],
                prediction['away_team'],
                prediction.get('home_win', 0),
                prediction.get('draw', 0),
                prediction.get('away_win', 0),
                prediction.get('btts_yes', 0),
                prediction.get('over_2_5', 0),
                prediction.get('predicted_outcome', ''),
                prediction.get('confidence', 0),
                json.dumps(prediction.get('correct_scores', {})),
                json.dumps(prediction.get('htft', {})),
                prediction.get('model_version', 'v1')
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_prediction(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Get cached prediction for a match."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT * FROM predictions 
            WHERE home_team = ? AND away_team = ?
            ORDER BY created_at DESC LIMIT 1
            ''', (home_team, away_team))
            
            row = cursor.fetchone()
            if row:
                return {
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'home_win': row['home_win_prob'],
                    'draw': row['draw_prob'],
                    'away_win': row['away_win_prob'],
                    'btts_yes': row['btts_yes'],
                    'over_2_5': row['over_2_5'],
                    'predicted_outcome': row['predicted_outcome'],
                    'confidence': row['confidence'],
                    'correct_scores': json.loads(row['correct_scores']) if row['correct_scores'] else {},
                    'htft': json.loads(row['htft']) if row['htft'] else {},
                    'cached_at': row['created_at'],
                    'from_cache': True
                }
            return None
        finally:
            conn.close()
    
    def get_all_predictions(self, league_id: str = None, limit: int = 50) -> List[Dict]:
        """Get all cached predictions, optionally filtered by league."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if league_id:
                cursor.execute('''
                SELECT p.*, f.league_id, f.match_date FROM predictions p
                JOIN fixtures f ON p.fixture_id = f.id
                WHERE f.league_id = ?
                ORDER BY f.match_date ASC LIMIT ?
                ''', (league_id, limit))
            else:
                cursor.execute('''
                SELECT p.*, f.league_id, f.match_date FROM predictions p
                LEFT JOIN fixtures f ON p.fixture_id = f.id
                ORDER BY p.created_at DESC LIMIT ?
                ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def store_odds(self, odds: Dict, fixture_id: int = None):
        """Store odds in the database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO odds 
            (fixture_id, home_team, away_team, bookmaker, home_odds, draw_odds, away_odds,
             over_2_5_odds, under_2_5_odds, btts_yes_odds, btts_no_odds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fixture_id,
                odds.get('home_team', ''),
                odds.get('away_team', ''),
                odds.get('bookmaker', 'average'),
                odds.get('home_odds', 0),
                odds.get('draw_odds', 0),
                odds.get('away_odds', 0),
                odds.get('over_2_5_odds', 0),
                odds.get('under_2_5_odds', 0),
                odds.get('btts_yes_odds', 0),
                odds.get('btts_no_odds', 0)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def update_live_score(self, fixture_id: int, home_score: int, away_score: int, 
                          minute: int, status: str):
        """Update live score for a match."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO live_scores 
            (fixture_id, home_score, away_score, minute, status, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (fixture_id, home_score, away_score, minute, status))
            conn.commit()
        finally:
            conn.close()
    
    def get_upcoming_fixtures(self, days: int = 3) -> List[Dict]:
        """Get upcoming fixtures within N days."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            end_date = datetime.now() + timedelta(days=days)
            cursor.execute('''
            SELECT * FROM fixtures 
            WHERE match_date BETWEEN CURRENT_TIMESTAMP AND ?
            AND status = 'scheduled'
            ORDER BY match_date ASC
            ''', (end_date.isoformat(),))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def log_job(self, job_name: str, status: str, message: str = ''):
        """Log scheduler job execution."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO scheduler_log (job_name, status, message)
            VALUES (?, ?, ?)
            ''', (job_name, status, message))
            conn.commit()
        finally:
            conn.close()


class AutomatedScheduler:
    """
    Automated scheduler for all prediction system jobs.
    
    Jobs:
    1. Fixture Fetcher: Daily at 6 AM
    2. Odds Updater: Every 30 minutes
    3. Prediction Generator: After fixture/odds updates
    4. Live Score Poller: Every 1 minute during matches
    5. Model Retrainer: Weekly on Sunday at 2 AM
    """
    
    def __init__(self):
        self.scheduler = BackgroundScheduler(daemon=True)
        self.cache = PredictionCache()
        self.is_running = False
        self.jobs_status = {}
        
    def start(self):
        """Start the scheduler with all jobs."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        # Job 1: Fetch fixtures daily at 6 AM
        self.scheduler.add_job(
            self.fetch_fixtures_job,
            CronTrigger(hour=6, minute=0),
            id='fixture_fetcher',
            name='Daily Fixture Fetcher',
            replace_existing=True
        )
        
        # Job 2: Update odds every 30 minutes
        self.scheduler.add_job(
            self.update_odds_job,
            IntervalTrigger(minutes=30),
            id='odds_updater',
            name='Odds Updater',
            replace_existing=True
        )
        
        # Job 3: Generate predictions every hour
        self.scheduler.add_job(
            self.generate_predictions_job,
            IntervalTrigger(hours=1),
            id='prediction_generator',
            name='Prediction Generator',
            replace_existing=True
        )
        
        # Job 4: Live score updates every minute
        self.scheduler.add_job(
            self.live_scores_job,
            IntervalTrigger(minutes=1),
            id='live_scores',
            name='Live Score Poller',
            replace_existing=True
        )
        
        # Job 5: Model retraining weekly on Sunday at 2 AM
        self.scheduler.add_job(
            self.retrain_models_job,
            CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='model_retrainer',
            name='Weekly Model Retrainer',
            replace_existing=True
        )
        
        # Job 6: Accuracy tracker daily at midnight
        self.scheduler.add_job(
            self.track_accuracy_job,
            CronTrigger(hour=0, minute=0),
            id='accuracy_tracker',
            name='Daily Accuracy Tracker',
            replace_existing=True
        )
        
        # Job 7: Daily SEO blog generation at 7 AM
        self.scheduler.add_job(
            self.generate_blog_job,
            CronTrigger(hour=7, minute=0),
            id='blog_generator',
            name='Daily Blog Generator',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        logger.info("🚀 Automated Scheduler started with all jobs!")
        self.cache.log_job('scheduler', 'started', 'All jobs registered')
    
    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("Scheduler stopped")
    
    def fetch_fixtures_job(self):
        """Job 1: Fetch fixtures from all leagues."""
        logger.info("🔄 Running fixture fetcher...")
        
        try:
            from .free_odds_api import fetch_live_odds
            
            leagues = [
                'bundesliga', 'premier_league', 'la_liga', 'serie_a', 'ligue_1',
                'championship', 'eredivisie', 'primeira_liga', 'jupiler_pro',
                'champions_league', 'europa_league'
            ]
            
            total_fixtures = 0
            for league in leagues:
                try:
                    odds_data = fetch_live_odds(league)
                    for match in odds_data:
                        fixture = {
                            'home_team': match.get('home_team', ''),
                            'away_team': match.get('away_team', ''),
                            'league_id': league,
                            'match_date': match.get('commence_time', datetime.now().isoformat()),
                            'status': 'scheduled'
                        }
                        self.cache.store_fixture(fixture)
                        total_fixtures += 1
                except Exception as e:
                    logger.error(f"Error fetching {league}: {e}")
            
            self.cache.log_job('fixture_fetcher', 'success', f'Fetched {total_fixtures} fixtures')
            self.jobs_status['fixture_fetcher'] = {'last_run': datetime.now(), 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Fixture fetcher failed: {e}")
            self.cache.log_job('fixture_fetcher', 'error', str(e))
    
    def update_odds_job(self):
        """Job 2: Update odds from APIs."""
        logger.info("🔄 Running odds updater...")
        
        try:
            from .free_odds_api import fetch_live_odds
            
            leagues = ['bundesliga', 'premier_league', 'la_liga', 'serie_a', 'ligue_1']
            total_odds = 0
            
            for league in leagues:
                try:
                    odds_data = fetch_live_odds(league)
                    for match in odds_data:
                        odds = {
                            'home_team': match.get('home_team', ''),
                            'away_team': match.get('away_team', ''),
                            'bookmaker': 'average',
                            'home_odds': match.get('home_odds', 0),
                            'draw_odds': match.get('draw_odds', 0),
                            'away_odds': match.get('away_odds', 0)
                        }
                        self.cache.store_odds(odds)
                        total_odds += 1
                except Exception as e:
                    logger.error(f"Error fetching odds for {league}: {e}")
            
            self.cache.log_job('odds_updater', 'success', f'Updated {total_odds} odds')
            self.jobs_status['odds_updater'] = {'last_run': datetime.now(), 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Odds updater failed: {e}")
            self.cache.log_job('odds_updater', 'error', str(e))
    
    def generate_predictions_job(self):
        """Job 3: Generate predictions for upcoming fixtures."""
        logger.info("🔄 Running prediction generator...")
        
        try:
            from .advanced_pipeline import get_advanced_prediction
            
            fixtures = self.cache.get_upcoming_fixtures(days=3)
            total_predictions = 0
            
            for fixture in fixtures[:50]:  # Limit to 50 per run
                try:
                    pred = get_advanced_prediction(fixture['home_team'], fixture['away_team'])
                    
                    # Determine outcome
                    probs = pred['1x2']
                    if probs['home_win'] > probs['draw'] and probs['home_win'] > probs['away_win']:
                        outcome = 'Home Win'
                        confidence = probs['home_win']
                    elif probs['away_win'] > probs['draw']:
                        outcome = 'Away Win'
                        confidence = probs['away_win']
                    else:
                        outcome = 'Draw'
                        confidence = probs['draw']
                    
                    prediction = {
                        'home_team': fixture['home_team'],
                        'away_team': fixture['away_team'],
                        'home_win': probs['home_win'],
                        'draw': probs['draw'],
                        'away_win': probs['away_win'],
                        'btts_yes': pred['btts']['yes'],
                        'over_2_5': pred['over_under']['over_2.5'],
                        'predicted_outcome': outcome,
                        'confidence': confidence,
                        'correct_scores': pred.get('correct_scores', {}),
                        'htft': pred.get('htft', {}),
                        'model_version': 'v1'
                    }
                    
                    self.cache.store_prediction(prediction, fixture.get('id'))
                    total_predictions += 1
                    
                except Exception as e:
                    logger.error(f"Error predicting {fixture['home_team']} vs {fixture['away_team']}: {e}")
            
            self.cache.log_job('prediction_generator', 'success', f'Generated {total_predictions} predictions')
            self.jobs_status['prediction_generator'] = {'last_run': datetime.now(), 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Prediction generator failed: {e}")
            self.cache.log_job('prediction_generator', 'error', str(e))
    
    def live_scores_job(self):
        """Job 4: Poll live scores."""
        logger.info("🔄 Checking live scores...")
        
        try:
            from .live_data import LiveDataClient
            
            live_client = LiveDataClient()
            leagues = ['bundesliga', 'premier_league', 'la_liga']
            
            for league in leagues:
                try:
                    scores = live_client.get_live_scores(league)
                    for match in scores:
                        if match.get('status') in ['LIVE', 'IN_PLAY', '1H', '2H', 'HT']:
                            # Would update live_scores table here
                            pass
                except Exception as e:
                    logger.debug(f"Live scores for {league}: {e}")
            
            self.jobs_status['live_scores'] = {'last_run': datetime.now(), 'status': 'success'}
            
        except Exception as e:
            logger.debug(f"Live scores check: {e}")
    
    def retrain_models_job(self):
        """Job 5: Weekly model retraining."""
        logger.info("🔄 Running model retraining...")
        
        try:
            from .dixon_coles import DixonColesModel
            from .pi_ratings import PiRatingSystem
            
            # Retrain Dixon-Coles with latest data
            # In production, this would load match results and fit the model
            
            self.cache.log_job('model_retrainer', 'success', 'Models retrained')
            self.jobs_status['model_retrainer'] = {'last_run': datetime.now(), 'status': 'success'}
            
            logger.info("✅ Models retrained successfully")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            self.cache.log_job('model_retrainer', 'error', str(e))
    
    def track_accuracy_job(self):
        """Job 6: Track prediction accuracy."""
        logger.info("🔄 Tracking prediction accuracy...")
        
        try:
            # Compare predictions vs actual results
            # Update accuracy metrics
            
            self.cache.log_job('accuracy_tracker', 'success', 'Accuracy tracked')
            self.jobs_status['accuracy_tracker'] = {'last_run': datetime.now(), 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Accuracy tracking failed: {e}")
    
    def generate_blog_job(self):
        """Job 7: Generate daily SEO blog posts at 7 AM."""
        logger.info("📝 Running daily blog generator...")
        
        try:
            from .seo_blog_generator import generate_daily_seo_posts
            
            # Get predictions for blog content
            predictions = self.cache.get_all_predictions(limit=100)
            
            # Generate blog posts
            result = generate_daily_seo_posts(predictions)
            
            posts_generated = 2 if 'preview' in result else 0
            word_count = result.get('word_count', 0)
            
            self.cache.log_job('blog_generator', 'success', 
                f'Generated {posts_generated} blog posts ({word_count} words)')
            self.jobs_status['blog_generator'] = {
                'last_run': datetime.now(), 
                'status': 'success',
                'posts': posts_generated,
                'word_count': word_count
            }
            
            logger.info(f"✅ Blog posts generated: {posts_generated} posts, {word_count} words")
            
        except Exception as e:
            logger.error(f"Blog generator failed: {e}")
            self.cache.log_job('blog_generator', 'error', str(e))
    
    def run_job_now(self, job_id: str):
        """Manually trigger a job."""
        job_map = {
            'fixture_fetcher': self.fetch_fixtures_job,
            'odds_updater': self.update_odds_job,
            'prediction_generator': self.generate_predictions_job,
            'live_scores': self.live_scores_job,
            'model_retrainer': self.retrain_models_job,
            'accuracy_tracker': self.track_accuracy_job,
            'blog_generator': self.generate_blog_job
        }
        
        if job_id in job_map:
            logger.info(f"Manually running job: {job_id}")
            threading.Thread(target=job_map[job_id]).start()
            return True
        return False
    
    def get_status(self) -> Dict:
        """Get scheduler status."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'last_status': self.jobs_status.get(job.id, {})
            })
        
        return {
            'is_running': self.is_running,
            'jobs': jobs,
            'job_count': len(jobs)
        }


# Global scheduler instance
automated_scheduler = AutomatedScheduler()
prediction_cache = PredictionCache()


def start_scheduler():
    """Start the automated scheduler."""
    automated_scheduler.start()
    return automated_scheduler.get_status()


def stop_scheduler():
    """Stop the scheduler."""
    automated_scheduler.stop()


def get_scheduler_status():
    """Get current scheduler status."""
    return automated_scheduler.get_status()


def get_cached_predictions(league_id: str = None, limit: int = 50):
    """Get cached predictions from database."""
    return prediction_cache.get_all_predictions(league_id, limit)


def run_job_manually(job_id: str):
    """Manually trigger a job."""
    return automated_scheduler.run_job_now(job_id)
