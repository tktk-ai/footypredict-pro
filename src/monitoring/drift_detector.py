"""
Model Drift Detector
=====================

Monitors model performance and detects:
- Accuracy degradation over time
- Distribution shift in predictions
- Confidence calibration issues
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "predictions.db"


class DriftDetector:
    """
    Detects model drift and performance degradation.
    """
    
    def __init__(self, window_size: int = 100, accuracy_threshold: float = 0.60):
        """
        Initialize drift detector.
        
        Args:
            window_size: Number of recent predictions to consider
            accuracy_threshold: Minimum acceptable accuracy
        """
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        
        self.predictions_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        self.alerts = []
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables for drift monitoring."""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                window_size INTEGER,
                alert_triggered INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                resolved INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_prediction(self, prediction: str, actual: str, confidence: float, 
                          market: str = 'result'):
        """
        Record a prediction result for drift monitoring.
        
        Args:
            prediction: Predicted outcome
            actual: Actual outcome
            confidence: Prediction confidence
            market: Market type
        """
        is_correct = 1 if prediction == actual else 0
        
        self.predictions_window.append({
            'market': market,
            'prediction': prediction,
            'actual': actual,
            'correct': is_correct,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        self.confidence_window.append((confidence, is_correct))
        
        # Check for drift after each prediction
        self._check_drift()
    
    def _check_drift(self):
        """Check for various types of drift."""
        if len(self.predictions_window) < 20:
            return  # Not enough data
        
        # 1. Accuracy drift
        recent_accuracy = self._calculate_rolling_accuracy()
        
        if recent_accuracy < self.accuracy_threshold:
            self._trigger_alert(
                'accuracy_drift',
                f'Rolling accuracy dropped to {recent_accuracy:.1%} (threshold: {self.accuracy_threshold:.1%})',
                'high'
            )
        
        # 2. Confidence calibration
        calibration_error = self._calculate_calibration_error()
        
        if calibration_error > 0.15:  # 15% calibration error
            self._trigger_alert(
                'calibration_drift',
                f'Confidence calibration error: {calibration_error:.1%}',
                'medium'
            )
        
        # 3. Prediction distribution shift
        distribution_shift = self._check_distribution_shift()
        
        if distribution_shift:
            self._trigger_alert(
                'distribution_drift',
                'Prediction distribution has shifted significantly',
                'low'
            )
    
    def _calculate_rolling_accuracy(self) -> float:
        """Calculate rolling accuracy over the window."""
        if not self.predictions_window:
            return 0.0
        
        correct = sum(1 for p in self.predictions_window if p['correct'])
        return correct / len(self.predictions_window)
    
    def _calculate_calibration_error(self) -> float:
        """
        Calculate expected calibration error (ECE).
        
        Measures if predicted probabilities match actual outcomes.
        """
        if len(self.confidence_window) < 20:
            return 0.0
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_correct = [[] for _ in range(10)]
        bin_confidence = [[] for _ in range(10)]
        
        for conf, correct in self.confidence_window:
            bin_idx = min(int(conf * 10), 9)
            bin_correct[bin_idx].append(correct)
            bin_confidence[bin_idx].append(conf)
        
        # Calculate ECE
        ece = 0.0
        total = 0
        
        for i in range(10):
            if bin_correct[i]:
                avg_correct = np.mean(bin_correct[i])
                avg_confidence = np.mean(bin_confidence[i])
                ece += len(bin_correct[i]) * abs(avg_correct - avg_confidence)
                total += len(bin_correct[i])
        
        return ece / total if total > 0 else 0.0
    
    def _check_distribution_shift(self) -> bool:
        """Check if prediction distribution has shifted."""
        if len(self.predictions_window) < 50:
            return False
        
        # Compare first half to second half
        predictions = list(self.predictions_window)
        mid = len(predictions) // 2
        
        first_half = [p['prediction'] for p in predictions[:mid]]
        second_half = [p['prediction'] for p in predictions[mid:]]
        
        # Count outcomes
        first_counts = {}
        second_counts = {}
        
        for p in first_half:
            first_counts[p] = first_counts.get(p, 0) + 1
        for p in second_half:
            second_counts[p] = second_counts.get(p, 0) + 1
        
        # Check for significant differences
        all_outcomes = set(first_counts.keys()) | set(second_counts.keys())
        
        for outcome in all_outcomes:
            first_rate = first_counts.get(outcome, 0) / len(first_half)
            second_rate = second_counts.get(outcome, 0) / len(second_half)
            
            if abs(first_rate - second_rate) > 0.20:  # 20% shift
                return True
        
        return False
    
    def _trigger_alert(self, alert_type: str, message: str, severity: str):
        """Trigger a drift alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        
        # Log alert
        if severity == 'high':
            logger.warning(f"🚨 DRIFT ALERT: {message}")
        else:
            logger.info(f"⚠️ Drift notice: {message}")
        
        # Save to database
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_alerts (alert_type, message, severity)
            VALUES (?, ?, ?)
        ''', (alert_type, message, severity))
        
        conn.commit()
        conn.close()
    
    def get_current_status(self) -> Dict:
        """Get current drift status."""
        return {
            'window_size': self.window_size,
            'predictions_count': len(self.predictions_window),
            'rolling_accuracy': self._calculate_rolling_accuracy(),
            'calibration_error': self._calculate_calibration_error(),
            'active_alerts': len([a for a in self.alerts if a.get('resolved', False) == False]),
            'needs_retrain': self._calculate_rolling_accuracy() < self.accuracy_threshold,
        }
    
    def get_accuracy_trend(self, periods: int = 5) -> List[float]:
        """Get accuracy trend over time periods."""
        if len(self.predictions_window) < periods * 10:
            return []
        
        predictions = list(self.predictions_window)
        period_size = len(predictions) // periods
        
        trend = []
        for i in range(periods):
            start = i * period_size
            end = start + period_size
            period_preds = predictions[start:end]
            accuracy = sum(1 for p in period_preds if p['correct']) / len(period_preds)
            trend.append(accuracy)
        
        return trend
    
    def save_metrics(self):
        """Save current metrics to database."""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        metrics = {
            'rolling_accuracy': self._calculate_rolling_accuracy(),
            'calibration_error': self._calculate_calibration_error(),
            'predictions_count': len(self.predictions_window),
        }
        
        for name, value in metrics.items():
            cursor.execute('''
                INSERT INTO drift_metrics (metric_name, metric_value, window_size)
                VALUES (?, ?, ?)
            ''', (name, value, self.window_size))
        
        conn.commit()
        conn.close()


# Global drift detector instance
_drift_detector = None


def get_drift_detector() -> DriftDetector:
    """Get singleton drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


def record_result(prediction: str, actual: str, confidence: float, market: str = 'result'):
    """Convenience function to record a prediction result."""
    detector = get_drift_detector()
    detector.record_prediction(prediction, actual, confidence, market)


if __name__ == "__main__":
    # Test drift detector
    detector = DriftDetector()
    
    # Simulate some predictions
    import random
    
    for _ in range(50):
        pred = random.choice(['H', 'D', 'A'])
        actual = random.choice(['H', 'D', 'A'])
        conf = random.uniform(0.4, 0.9)
        
        detector.record_prediction(pred, actual, conf)
    
    status = detector.get_current_status()
    print(f"\nDrift Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
