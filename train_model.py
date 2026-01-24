#!/usr/bin/env python3
"""
ML Model Auto-Training Script

Run this script periodically (e.g., weekly via cron) to:
1. Fetch new historical match data from APIs
2. Retrain the ML model on updated data
3. Save the new model weights

Usage:
    python train_model.py              # One-time training
    python train_model.py --schedule   # Start scheduler (runs weekly)

Cron example (run every Sunday at 3 AM):
    0 3 * * 0 cd /home/netboss/Desktop/pers_bus/soccer && /home/netboss/Desktop/pers_bus/soccer/venv/bin/python train_model.py
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def train_model():
    """Run the full training pipeline"""
    print(f"\n{'='*60}")
    print(f"🤖 ML Model Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        from src.ml.data_pipeline import MLTrainer
        
        trainer = MLTrainer()
        
        print("📊 Fetching historical data from APIs...")
        result = trainer.train(fetch_new=True)
        
        if result['success']:
            print(f"✅ Training complete!")
            print(f"   - Samples fetched: {result['samples_fetched']}")
            print(f"   - Total samples: {result['total_samples']}")
            print(f"   - Model saved: {result['model_path']}")
        else:
            print(f"❌ Training failed: {result.get('message', 'Unknown error')}")
            return False
        
        # Sync to historical database
        print("\n💾 Syncing to SQLite database...")
        try:
            from src.data.historical_data import sync_from_api
            stored = sync_from_api()
            print(f"   - Matches stored: {stored}")
        except Exception as e:
            print(f"   - Sync warning: {e}")
        
        print(f"\n{'='*60}")
        print("✅ Auto-training complete!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_scheduler(interval_hours: int = 168):  # Default: weekly (168 hours)
    """Run training on a schedule"""
    print(f"⏰ Starting scheduler (interval: {interval_hours} hours)")
    print("   Press Ctrl+C to stop\n")
    
    while True:
        train_model()
        
        next_run = datetime.now().timestamp() + (interval_hours * 3600)
        next_run_str = datetime.fromtimestamp(next_run).strftime('%Y-%m-%d %H:%M:%S')
        print(f"💤 Next training: {next_run_str}")
        
        try:
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped")
            break


def main():
    parser = argparse.ArgumentParser(description='ML Model Auto-Training')
    parser.add_argument('--schedule', action='store_true', 
                        help='Run on a schedule (default: weekly)')
    parser.add_argument('--interval', type=int, default=168,
                        help='Interval in hours between training (default: 168 = weekly)')
    
    args = parser.parse_args()
    
    if args.schedule:
        run_scheduler(args.interval)
    else:
        success = train_model()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
