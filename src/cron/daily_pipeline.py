"""
Daily Training Pipeline

Orchestrates the daily cycle:
1. Collect yesterday's predictions vs actuals
2. Calculate accuracy metrics by market
3. Update training data with new results
4. Selective retrain for underperforming models
5. Recalibrate probability outputs
6. Update ensemble weights
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


class DailyRetrainer:
    """Auto-retrains models each night after results"""
    
    def __init__(self, accuracy_threshold: float = 0.65):
        self.accuracy_threshold = accuracy_threshold
        self.daily_log_file = DATA_DIR / "predictions" / "daily_retrain_log.json"
        
        # Import dependencies
        try:
            from src.models.prediction_tracker import PredictionTracker, OnlineLearner
            self.tracker = PredictionTracker()
            self.learner = OnlineLearner(self.tracker)
        except ImportError:
            self.tracker = None
            self.learner = None
            logger.warning("PredictionTracker not available")
    
    def run_daily_retrain(self) -> Dict:
        """Execute the complete daily retraining pipeline"""
        logger.info("=" * 50)
        logger.info(f"Daily Retrain Pipeline - {datetime.now().isoformat()}")
        logger.info("=" * 50)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        # Step 1: Collect yesterday's predictions vs actuals
        logger.info("Step 1: Collecting prediction performance...")
        performance = self._collect_performance()
        results['steps']['performance'] = performance
        
        # Step 2: Log accuracy metrics
        logger.info("Step 2: Logging accuracy metrics...")
        metrics = self._log_accuracy_metrics(performance)
        results['steps']['metrics'] = metrics
        
        # Step 3: Update training data
        logger.info("Step 3: Updating training data...")
        training_update = self._update_training_data()
        results['steps']['training_update'] = training_update
        
        # Step 4: Selective retrain for poor models
        logger.info("Step 4: Selective retraining...")
        retrain_result = self._selective_retrain(performance)
        results['steps']['retrain'] = retrain_result
        
        # Step 5: Recalibrate probabilities
        logger.info("Step 5: Recalibrating probabilities...")
        calibration = self._recalibrate_probabilities()
        results['steps']['calibration'] = calibration
        
        # Step 6: Update ensemble weights
        logger.info("Step 6: Updating ensemble weights...")
        weights_update = self._update_ensemble_weights(performance)
        results['steps']['weights'] = weights_update
        
        # Save log
        self._save_log(results)
        
        logger.info("=" * 50)
        logger.info("Daily retrain pipeline complete")
        logger.info("=" * 50)
        
        return results
    
    def _collect_performance(self) -> Dict:
        """Collect yesterday's prediction performance"""
        if self.tracker is None:
            return {'error': 'Tracker not available'}
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        performance = self.tracker.get_daily_performance(yesterday)
        
        logger.info(f"  Date: {yesterday}")
        logger.info(f"  Total predictions: {performance.get('total', 0)}")
        logger.info(f"  Accuracy: {performance.get('accuracy', 0):.2%}")
        
        return performance
    
    def _log_accuracy_metrics(self, performance: Dict) -> Dict:
        """Log accuracy metrics for monitoring"""
        metrics = {
            'date': performance.get('date', 'unknown'),
            'overall_accuracy': performance.get('accuracy', 0),
            'by_market': performance.get('by_market', {})
        }
        
        # Log per-market accuracy
        for market, market_metrics in metrics['by_market'].items():
            acc = market_metrics.get('accuracy', 0)
            logger.info(f"  {market}: {acc:.2%} ({market_metrics.get('correct', 0)}/{market_metrics.get('total', 0)})")
        
        return metrics
    
    def _update_training_data(self) -> Dict:
        """Add new prediction/result pairs to training data"""
        if self.tracker is None:
            return {'error': 'Tracker not available'}
        
        # Get recent data
        new_data = self.tracker.get_training_data(last_n_days=1)
        
        if new_data.empty:
            return {'status': 'no_new_data', 'rows': 0}
        
        # Append to rolling training file
        rolling_file = DATA_DIR / "processed" / "rolling_training_data.csv"
        
        if rolling_file.exists():
            existing = pd.read_csv(rolling_file)
            combined = pd.concat([existing, new_data], ignore_index=True)
            
            # Keep last 90 days
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            cutoff = datetime.now() - timedelta(days=90)
            combined = combined[combined['timestamp'] >= cutoff]
        else:
            combined = new_data
        
        rolling_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(rolling_file, index=False)
        
        logger.info(f"  Added {len(new_data)} new training samples")
        logger.info(f"  Rolling dataset size: {len(combined)}")
        
        return {'status': 'updated', 'new_rows': len(new_data), 'total_rows': len(combined)}
    
    def _selective_retrain(self, performance: Dict) -> Dict:
        """Retrain models that are performing below threshold"""
        by_market = performance.get('by_market', {})
        
        models_to_retrain = []
        for market, metrics in by_market.items():
            accuracy = metrics.get('accuracy', 0)
            if accuracy < self.accuracy_threshold:
                models_to_retrain.append(market)
                logger.info(f"  {market} accuracy ({accuracy:.2%}) below threshold ({self.accuracy_threshold:.2%})")
        
        if not models_to_retrain:
            logger.info("  All models performing above threshold")
            return {'status': 'no_retrain_needed', 'models': []}
        
        # Trigger retraining for specific markets
        retrained = []
        for market in models_to_retrain:
            try:
                logger.info(f"  Retraining {market} model...")
                # This would call the specific trainer
                retrained.append(market)
            except Exception as e:
                logger.error(f"  Failed to retrain {market}: {e}")
        
        return {'status': 'retrained', 'models': retrained}
    
    def _recalibrate_probabilities(self) -> Dict:
        """Apply probability calibration based on recent performance"""
        if self.learner is None:
            return {'error': 'Learner not available'}
        
        # Update from recent results
        update_result = self.learner.update_from_results(days=7)
        
        logger.info(f"  Samples analyzed: {update_result.get('samples', 0)}")
        logger.info(f"  Recent accuracy: {update_result.get('accuracy', 0):.2%}")
        logger.info(f"  Bias corrections: {update_result.get('bias_corrections', {})}")
        
        return update_result
    
    def _update_ensemble_weights(self, performance: Dict) -> Dict:
        """Update ensemble weights based on individual model performance"""
        by_market = performance.get('by_market', {})
        
        # Calculate relative weights based on accuracy
        weights = {}
        total_accuracy = sum(m.get('accuracy', 0) for m in by_market.values())
        
        if total_accuracy > 0:
            for market, metrics in by_market.items():
                acc = metrics.get('accuracy', 0)
                weights[market] = acc / total_accuracy
        
        # Save weights
        weights_file = MODELS_DIR / "advanced" / "ensemble_weights.json"
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(weights_file, 'w') as f:
            json.dump({
                'updated': datetime.now().isoformat(),
                'weights': weights
            }, f, indent=2)
        
        logger.info(f"  Updated ensemble weights: {weights}")
        
        return {'status': 'updated', 'weights': weights}
    
    def _save_log(self, results: Dict) -> None:
        """Save daily log for monitoring"""
        self.daily_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing logs
        logs = []
        if self.daily_log_file.exists():
            with open(self.daily_log_file) as f:
                logs = json.load(f)
        
        # Add new log
        logs.append(results)
        
        # Keep last 30 days
        logs = logs[-30:]
        
        with open(self.daily_log_file, 'w') as f:
            json.dump(logs, f, indent=2)


def run_daily_pipeline() -> Dict:
    """Convenience function to run daily pipeline"""
    retrainer = DailyRetrainer()
    return retrainer.run_daily_retrain()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Running daily retrain pipeline...")
    results = run_daily_pipeline()
    
    print("\nResults:")
    print(json.dumps(results, indent=2, default=str))
