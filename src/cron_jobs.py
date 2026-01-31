"""
Cron Jobs Scheduler

Automated daily tasks:
- Send morning predictions (9 AM)
- Send evening results (10 PM)
- Weekly accuracy reports (Sunday)
- Auto-retrain (Sunday night)
"""

import logging
import threading
from datetime import datetime, time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Try APScheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False


class CronJobManager:
    """Manage scheduled tasks"""
    
    def __init__(self):
        self.scheduler: Optional[BackgroundScheduler] = None
        self.jobs: List[dict] = []
        self.is_running = False
        
        if HAS_SCHEDULER:
            self.scheduler = BackgroundScheduler()
    
    def add_job(self, job_id: str, func: Callable, trigger: str, **kwargs):
        """Add a scheduled job"""
        if not HAS_SCHEDULER:
            logger.warning("APScheduler not installed")
            return False
        
        try:
            self.scheduler.add_job(
                func, 
                CronTrigger.from_crontab(trigger),
                id=job_id,
                replace_existing=True,
                **kwargs
            )
            self.jobs.append({
                'id': job_id,
                'trigger': trigger,
                'added_at': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {e}")
            return False
    
    def start(self):
        """Start the scheduler"""
        if self.scheduler and not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("Cron scheduler started")
            return True
        return False
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            return True
        return False
    
    def get_status(self):
        """Get scheduler status"""
        jobs_info = []
        if self.scheduler:
            for job in self.scheduler.get_jobs():
                jobs_info.append({
                    'id': job.id,
                    'next_run': str(job.next_run_time) if job.next_run_time else None
                })
        
        return {
            'is_running': self.is_running,
            'jobs': jobs_info,
            'scheduler_available': HAS_SCHEDULER
        }


def send_morning_predictions():
    """Send daily prediction digest at 9 AM"""
    from src.telegram_bot import send_daily_digest
    from src.enhanced_predictor_v2 import enhanced_predict
    
    logger.info("Sending morning predictions...")
    
    # Get top predictions for today
    predictions = []
    # In production, fetch actual fixtures for today
    sample_matches = [
        ('Manchester United', 'Liverpool'),
        ('Arsenal', 'Chelsea'),
        ('Barcelona', 'Real Madrid')
    ]
    
    for home, away in sample_matches:
        try:
            pred = enhanced_predict(home, away)
            predictions.append({
                'match': {'home_team': {'name': home}, 'away_team': {'name': away}},
                'prediction': pred.get('final_prediction', {}),
                'goals': pred.get('goals', {})
            })
        except:
            pass
    
    if predictions:
        send_daily_digest(predictions)
        logger.info(f"Sent {len(predictions)} predictions")


def send_evening_results():
    """Send results summary at 10 PM"""
    from src.telegram_bot import send_accuracy_update
    from src.accuracy_dashboard import get_accuracy_stats
    
    logger.info("Sending evening results...")
    
    stats = get_accuracy_stats('today')
    if stats.get('total', 0) > 0:
        send_accuracy_update(stats)
        logger.info("Sent accuracy update")


def send_weekly_report():
    """Send weekly accuracy report on Sunday"""
    from src.telegram_bot import send_accuracy_update
    from src.accuracy_dashboard import get_accuracy_stats
    
    logger.info("Sending weekly report...")
    
    stats = get_accuracy_stats('week')
    send_accuracy_update(stats)


def weekly_retrain():
    """Weekly model retraining on Sunday night"""
    from src.models.local_trainer import retrain_models
    from src.models.auto_tuner import get_hyperparams
    
    logger.info("Starting weekly retrain...")
    
    config = get_hyperparams()
    params = config.get('hyperparameters', {})
    result = retrain_models(params, async_mode=True)
    logger.info(f"Retrain initiated: {result}")


def generate_daily_blog():
    """Generate daily blog posts from predictions - runs at 7 AM"""
    from src.blog_generator import generate_daily_blog_posts
    
    logger.info("Generating daily blog posts...")
    
    try:
        result = generate_daily_blog_posts()
        logger.info(f"Blog posts generated: {result}")
        return result
    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        return {'error': str(e)}


# Global manager
_manager: Optional[CronJobManager] = None

def get_cron_manager() -> CronJobManager:
    global _manager
    if _manager is None:
        _manager = CronJobManager()
    return _manager


def setup_default_cron_jobs():
    """Setup default scheduled jobs"""
    manager = get_cron_manager()
    
    # Blog generation at 7 AM (after predictions are ready)
    manager.add_job('daily_blog', generate_daily_blog, '0 7 * * *')
    
    # Morning predictions at 9 AM
    manager.add_job('morning_predictions', send_morning_predictions, '0 9 * * *')
    
    # Evening results at 10 PM
    manager.add_job('evening_results', send_evening_results, '0 22 * * *')
    
    # Weekly report Sunday 8 PM
    manager.add_job('weekly_report', send_weekly_report, '0 20 * * 0')
    
    # Weekly retrain Sunday 2 AM
    manager.add_job('weekly_retrain', weekly_retrain, '0 2 * * 0')
    
    manager.start()
    logger.info("Default cron jobs configured (including blog generation)")
    return manager.get_status()


def start_cron():
    """Start cron with default jobs"""
    return setup_default_cron_jobs()


def stop_cron():
    """Stop cron"""
    return get_cron_manager().stop()


def get_cron_status():
    """Get cron status"""
    return get_cron_manager().get_status()
