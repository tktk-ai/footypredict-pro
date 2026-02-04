"""
Weekly Scheduler
================

Provides alternative scheduling via Python (for systems without cron).
Uses APScheduler for background task scheduling.

Usage:
    python -m src.automation.weekly_scheduler           # Start scheduler
    python -m src.automation.weekly_scheduler --once    # Run once and exit
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'scheduler.log')
    ]
)
logger = logging.getLogger(__name__)


def run_weekly_training():
    """Run the weekly training pipeline."""
    logger.info("🤖 Starting weekly training...")
    from src.training.weekly_training_pipeline import WeeklyTrainingPipeline
    
    pipeline = WeeklyTrainingPipeline()
    results = pipeline.run_weekly_cycle()
    
    logger.info(f"Weekly training completed - Success: {results.get('steps', {}).get('model_training', {})}")
    return results


def run_daily_fixtures():
    """Run daily fixture collection."""
    logger.info("📥 Collecting daily fixtures...")
    from src.data.sportybet_scraper import SportyBetScraper
    
    scraper = SportyBetScraper()
    fixtures = scraper.get_all_fixtures(days=7)
    filepath = scraper.save_fixtures_to_csv(fixtures, f"fixtures_{datetime.now().strftime('%Y%m%d')}.csv")
    
    logger.info(f"Collected {len(fixtures)} fixtures, saved to {filepath}")
    return len(fixtures)


def run_hourly_odds():
    """Run hourly odds update."""
    logger.info("💰 Updating today's odds...")
    from src.data.sportybet_scraper import SportyBetScraper
    
    scraper = SportyBetScraper()
    fixtures = scraper.get_todays_fixtures()
    scraper.save_fixtures_to_csv(fixtures, "today_odds.csv")
    
    logger.info(f"Updated odds for {len(fixtures)} fixtures")
    return len(fixtures)


def start_scheduler():
    """Start the background scheduler."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Install with: pip install apscheduler")
        logger.info("Alternatively, use cron: scripts/cron_setup.sh")
        sys.exit(1)
    
    scheduler = BlockingScheduler()
    
    # Weekly training - Sundays at 2 AM
    scheduler.add_job(
        run_weekly_training,
        CronTrigger(day_of_week='sun', hour=2, minute=0),
        id='weekly_training',
        name='Weekly Model Training'
    )
    
    # Daily fixtures - Every day at 6 AM
    scheduler.add_job(
        run_daily_fixtures,
        CronTrigger(hour=6, minute=0),
        id='daily_fixtures',
        name='Daily Fixture Collection'
    )
    
    # Hourly odds - Every hour at :30
    scheduler.add_job(
        run_hourly_odds,
        CronTrigger(minute=30),
        id='hourly_odds',
        name='Hourly Odds Update'
    )
    
    logger.info("="*50)
    logger.info("⏰ FootyPredict Scheduler Started")
    logger.info("="*50)
    logger.info("Scheduled tasks:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")
    logger.info("\nPress Ctrl+C to stop")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n🛑 Scheduler stopped")


def main():
    parser = argparse.ArgumentParser(description='FootyPredict Scheduler')
    parser.add_argument('--once', action='store_true', help='Run all tasks once and exit')
    parser.add_argument('--training', action='store_true', help='Run training only')
    parser.add_argument('--fixtures', action='store_true', help='Run fixture collection only')
    parser.add_argument('--odds', action='store_true', help='Run odds update only')
    
    args = parser.parse_args()
    
    # Create logs directory
    (project_root / 'logs').mkdir(exist_ok=True)
    
    if args.training:
        run_weekly_training()
    elif args.fixtures:
        run_daily_fixtures()
    elif args.odds:
        run_hourly_odds()
    elif args.once:
        logger.info("Running all tasks once...")
        run_daily_fixtures()
        run_hourly_odds()
        logger.info("✅ All tasks completed")
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
