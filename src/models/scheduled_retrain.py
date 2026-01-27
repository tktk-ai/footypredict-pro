"""
Scheduled Auto-Retraining System

Runs weekly retraining automatically using APScheduler.
Also supports cron-style scheduling.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Try to use APScheduler if available
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False
    logger.warning("APScheduler not installed. Run: pip install apscheduler")


class ScheduledRetrainer:
    """Automatically retrain models on a schedule"""
    
    def __init__(self):
        self.scheduler: Optional[BackgroundScheduler] = None
        self.is_running = False
        self.last_run = None
        self.next_run = None
        
        if HAS_SCHEDULER:
            self.scheduler = BackgroundScheduler()
    
    def _retrain_job(self):
        """Job that runs on schedule - includes daily calibration"""
        logger.info("Scheduled retraining started...")
        self.last_run = datetime.now()
        
        try:
            # Step 1: Run daily pipeline (collect results, calibrate, retrain)
            try:
                from src.cron.daily_pipeline import run_daily_pipeline
                pipeline_result = run_daily_pipeline()
                logger.info(f"Daily pipeline complete: {pipeline_result.get('steps', {}).get('metrics', {})}")
            except ImportError:
                logger.warning("Daily pipeline not available, using legacy retrain")
                pipeline_result = None
            
            # Step 2: Run full model retraining if needed
            from src.models.local_trainer import retrain_models
            from src.models.auto_tuner import get_hyperparams
            
            config = get_hyperparams()
            params = config.get('hyperparameters', {})
            result = retrain_models(params, async_mode=False)
            logger.info(f"Scheduled retraining complete: {result}")
            
        except Exception as e:
            logger.error(f"Scheduled retraining failed: {e}")
    
    def start_weekly(self, day: str = 'sun', hour: int = 3):
        """Start weekly retraining (default: Sunday 3 AM)"""
        if not HAS_SCHEDULER:
            return {'error': 'APScheduler not installed'}
        
        if self.is_running:
            return {'status': 'already_running'}
        
        self.scheduler.add_job(
            self._retrain_job,
            CronTrigger(day_of_week=day, hour=hour),
            id='weekly_retrain',
            replace_existing=True
        )
        self.scheduler.start()
        self.is_running = True
        
        job = self.scheduler.get_job('weekly_retrain')
        self.next_run = job.next_run_time if job else None
        
        logger.info(f"Scheduled weekly retraining: {day} at {hour}:00")
        return {
            'status': 'scheduled',
            'schedule': f'{day} at {hour}:00',
            'next_run': str(self.next_run) if self.next_run else None
        }
    
    def start_daily(self, hour: int = 4):
        """Start daily retraining"""
        if not HAS_SCHEDULER:
            return {'error': 'APScheduler not installed'}
        
        if self.is_running:
            self.stop()
        
        self.scheduler.add_job(
            self._retrain_job,
            CronTrigger(hour=hour),
            id='daily_retrain',
            replace_existing=True
        )
        self.scheduler.start()
        self.is_running = True
        
        job = self.scheduler.get_job('daily_retrain')
        self.next_run = job.next_run_time if job else None
        
        return {
            'status': 'scheduled',
            'schedule': f'daily at {hour}:00',
            'next_run': str(self.next_run)
        }
    
    def stop(self):
        """Stop scheduled retraining"""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown(wait=False)
            self.scheduler = BackgroundScheduler() if HAS_SCHEDULER else None
            self.is_running = False
            return {'status': 'stopped'}
        return {'status': 'not_running'}
    
    def get_status(self):
        """Get scheduler status"""
        return {
            'is_running': self.is_running,
            'last_run': str(self.last_run) if self.last_run else None,
            'next_run': str(self.next_run) if self.next_run else None,
            'scheduler_available': HAS_SCHEDULER
        }


# Global instance
_scheduler: Optional[ScheduledRetrainer] = None

def get_scheduler() -> ScheduledRetrainer:
    global _scheduler
    if _scheduler is None:
        _scheduler = ScheduledRetrainer()
    return _scheduler

def start_weekly_retrain(day: str = 'sun', hour: int = 3):
    return get_scheduler().start_weekly(day, hour)

def start_daily_retrain(hour: int = 4):
    return get_scheduler().start_daily(hour)

def stop_scheduled_retrain():
    return get_scheduler().stop()

def get_schedule_status():
    return get_scheduler().get_status()
