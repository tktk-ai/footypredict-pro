"""
Enhanced Calibration System
Multiple calibration methods with online adaptation and monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    log_loss: float
    reliability_diagram: Dict[str, List[float]]
    timestamp: datetime = field(default_factory=datetime.now)


class BaseCalibrator(ABC):
    """Abstract base class for calibrators."""
    
    @abstractmethod
    def fit(self, probas: np.ndarray, y_true: np.ndarray):
        """Fit calibrator."""
        pass
    
    @abstractmethod
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        pass


class TemperatureScaling(BaseCalibrator):
    """
    Temperature scaling calibration.
    Learns a single temperature parameter T to scale logits.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, probas: np.ndarray, y_true: np.ndarray):
        """Find optimal temperature."""
        # Convert probas to logits
        epsilon = 1e-10
        probas = np.clip(probas, epsilon, 1 - epsilon)
        
        if probas.ndim == 1:
            logits = np.log(probas / (1 - probas))
        else:
            logits = np.log(probas + epsilon)
        
        def objective(T):
            scaled_logits = logits / T[0]
            if probas.ndim == 1:
                scaled_probas = 1 / (1 + np.exp(-scaled_logits))
                return -np.mean(y_true * np.log(scaled_probas + epsilon) + 
                               (1 - y_true) * np.log(1 - scaled_probas + epsilon))
            else:
                scaled_probas = softmax(scaled_logits, axis=1)
                # Cross-entropy loss
                return -np.mean(np.log(scaled_probas[np.arange(len(y_true)), y_true] + epsilon))
        
        result = minimize(objective, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        logger.info(f"Temperature scaling: T = {self.temperature:.4f}")
    
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        epsilon = 1e-10
        probas = np.clip(probas, epsilon, 1 - epsilon)
        
        if probas.ndim == 1:
            logits = np.log(probas / (1 - probas))
            scaled_logits = logits / self.temperature
            return 1 / (1 + np.exp(-scaled_logits))
        else:
            logits = np.log(probas + epsilon)
            scaled_logits = logits / self.temperature
            return softmax(scaled_logits, axis=1)


class BetaCalibration(BaseCalibrator):
    """
    Beta calibration using beta distribution fitting.
    """
    
    def __init__(self):
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
    
    def fit(self, probas: np.ndarray, y_true: np.ndarray):
        """Fit beta calibration parameters."""
        if probas.ndim > 1:
            # Use max probability for multi-class
            probas = probas.max(axis=1)
        
        epsilon = 1e-10
        probas = np.clip(probas, epsilon, 1 - epsilon)
        
        # Transform to logit space
        logits = np.log(probas / (1 - probas))
        
        def objective(params):
            a, b, c = params
            mapped = 1 / (1 + np.exp(-(a * logits + b * (1 - probas) + c)))
            mapped = np.clip(mapped, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(mapped) + (1 - y_true) * np.log(1 - mapped))
        
        result = minimize(objective, [1.0, 0.0, 0.0], method='L-BFGS-B')
        self.a, self.b, self.c = result.x
        logger.info(f"Beta calibration: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
    
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        original_shape = probas.shape
        
        if probas.ndim > 1:
            max_probas = probas.max(axis=1)
        else:
            max_probas = probas
        
        epsilon = 1e-10
        max_probas = np.clip(max_probas, epsilon, 1 - epsilon)
        
        logits = np.log(max_probas / (1 - max_probas))
        calibrated = 1 / (1 + np.exp(-(self.a * logits + self.b * (1 - max_probas) + self.c)))
        
        if probas.ndim > 1:
            # Scale all probabilities proportionally
            scale_factor = calibrated / (max_probas + epsilon)
            calibrated_multi = probas * scale_factor[:, np.newaxis]
            # Renormalize
            calibrated_multi = calibrated_multi / calibrated_multi.sum(axis=1, keepdims=True)
            return calibrated_multi
        
        return calibrated


class IsotonicCalibration(BaseCalibrator):
    """
    Isotonic regression calibration.
    Non-parametric calibration using isotonic regression.
    """
    
    def __init__(self):
        self.calibrators = {}
    
    def fit(self, probas: np.ndarray, y_true: np.ndarray):
        """Fit isotonic regression for each class."""
        if probas.ndim == 1:
            self.calibrators[0] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators[0].fit(probas, y_true)
        else:
            n_classes = probas.shape[1]
            for c in range(n_classes):
                y_binary = (y_true == c).astype(int)
                self.calibrators[c] = IsotonicRegression(out_of_bounds='clip')
                self.calibrators[c].fit(probas[:, c], y_binary)
        
        logger.info(f"Isotonic calibration fitted for {len(self.calibrators)} classes")
    
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if probas.ndim == 1:
            return self.calibrators[0].predict(probas)
        
        calibrated = np.zeros_like(probas)
        for c in range(probas.shape[1]):
            if c in self.calibrators:
                calibrated[:, c] = self.calibrators[c].predict(probas[:, c])
        
        # Renormalize
        calibrated = calibrated / (calibrated.sum(axis=1, keepdims=True) + 1e-10)
        return calibrated


class AdaptiveCalibrator:
    """
    Adaptive calibrator that selects best method automatically.
    """
    
    def __init__(self):
        self.calibrators = {
            'temperature': TemperatureScaling(),
            'beta': BetaCalibration(),
            'isotonic': IsotonicCalibration()
        }
        self.best_method = None
        self.metrics = {}
    
    def fit(self, probas: np.ndarray, y_true: np.ndarray, val_probas: np.ndarray = None, val_y: np.ndarray = None):
        """Fit all calibrators and select best."""
        # Use validation set if provided, else use same data
        val_probas = val_probas if val_probas is not None else probas
        val_y = val_y if val_y is not None else y_true
        
        best_score = float('inf')
        
        for name, calibrator in self.calibrators.items():
            try:
                calibrator.fit(probas, y_true)
                calibrated = calibrator.calibrate(val_probas)
                
                # Calculate ECE
                ece = self._expected_calibration_error(calibrated, val_y)
                self.metrics[name] = ece
                
                if ece < best_score:
                    best_score = ece
                    self.best_method = name
                
                logger.info(f"  {name}: ECE = {ece:.4f}")
                
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        
        logger.info(f"Best calibration method: {self.best_method}")
    
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Apply best calibration method."""
        if self.best_method is None:
            return probas
        
        return self.calibrators[self.best_method].calibrate(probas)
    
    def _expected_calibration_error(self, probas: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        if probas.ndim > 1:
            max_probas = probas.max(axis=1)
            predictions = probas.argmax(axis=1)
            accuracies = (predictions == y_true).astype(float)
        else:
            max_probas = probas
            predictions = (probas > 0.5).astype(int)
            accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (max_probas > bin_boundaries[i]) & (max_probas <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                avg_confidence = max_probas[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += in_bin.sum() * np.abs(avg_confidence - avg_accuracy)
        
        return ece / len(y_true)


class OnlineCalibrator:
    """
    Online calibration that adapts over time.
    Uses sliding window for continuous adaptation.
    """
    
    def __init__(self, window_size: int = 1000, recalibrate_threshold: float = 0.05):
        self.window_size = window_size
        self.recalibrate_threshold = recalibrate_threshold
        
        self.calibrator = AdaptiveCalibrator()
        self.history_probas = []
        self.history_true = []
        self.last_ece = None
        self.recalibration_count = 0
    
    def update(self, probas: np.ndarray, y_true: np.ndarray):
        """Update with new observations."""
        # Add to history
        for p, y in zip(probas, y_true):
            self.history_probas.append(p)
            self.history_true.append(y)
        
        # Keep only recent history
        if len(self.history_probas) > self.window_size:
            self.history_probas = self.history_probas[-self.window_size:]
            self.history_true = self.history_true[-self.window_size:]
        
        # Check if recalibration needed
        if len(self.history_probas) >= 100:
            current_ece = self._calculate_current_ece()
            
            if self.last_ece is None or (current_ece - self.last_ece) > self.recalibrate_threshold:
                logger.info(f"Triggering recalibration (ECE: {current_ece:.4f})")
                self._recalibrate()
                self.recalibration_count += 1
    
    def _calculate_current_ece(self) -> float:
        """Calculate current ECE on recent data."""
        probas = np.array(self.history_probas[-100:])
        y_true = np.array(self.history_true[-100:])
        
        calibrated = self.calibrator.calibrate(probas)
        return self.calibrator._expected_calibration_error(calibrated, y_true)
    
    def _recalibrate(self):
        """Recalibrate on recent history."""
        probas = np.array(self.history_probas)
        y_true = np.array(self.history_true)
        
        # Split for train/val
        split = int(len(probas) * 0.8)
        
        self.calibrator.fit(
            probas[:split], y_true[:split],
            probas[split:], y_true[split:]
        )
        
        self.last_ece = self._calculate_current_ece()
    
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Apply current calibration."""
        return self.calibrator.calibrate(probas)
    
    def get_status(self) -> Dict[str, Any]:
        """Get calibration status."""
        return {
            'history_size': len(self.history_probas),
            'last_ece': self.last_ece,
            'recalibration_count': self.recalibration_count,
            'best_method': self.calibrator.best_method,
            'needs_more_data': len(self.history_probas) < 100
        }


class CalibrationMonitor:
    """
    Monitors calibration quality over time and triggers alerts.
    """
    
    def __init__(self, alert_threshold: float = 0.1, check_interval: int = 100):
        self.alert_threshold = alert_threshold
        self.check_interval = check_interval
        
        self.metrics_history: List[CalibrationMetrics] = []
        self.alerts: List[Dict] = []
        self.prediction_count = 0
    
    def record_prediction(self, probas: np.ndarray, y_true: np.ndarray = None):
        """Record prediction for monitoring."""
        self.prediction_count += len(probas) if isinstance(probas, np.ndarray) else 1
        
        if y_true is not None and self.prediction_count % self.check_interval == 0:
            metrics = self._calculate_metrics(probas, y_true)
            self.metrics_history.append(metrics)
            self._check_alerts(metrics)
    
    def _calculate_metrics(self, probas: np.ndarray, y_true: np.ndarray) -> CalibrationMetrics:
        """Calculate calibration metrics."""
        if probas.ndim > 1:
            max_probas = probas.max(axis=1)
            predictions = probas.argmax(axis=1)
        else:
            max_probas = probas
            predictions = (probas > 0.5).astype(int)
        
        # Reliability diagram bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_confs = []
        bin_accs = []
        
        accuracies = (predictions == y_true).astype(float)
        
        for i in range(n_bins):
            in_bin = (max_probas > bin_boundaries[i]) & (max_probas <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_confs.append(float(max_probas[in_bin].mean()))
                bin_accs.append(float(accuracies[in_bin].mean()))
        
        # ECE
        ece = 0.0
        mce = 0.0
        for conf, acc in zip(bin_confs, bin_accs):
            gap = abs(conf - acc)
            ece += gap
            mce = max(mce, gap)
        ece /= len(bin_confs) if bin_confs else 1
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier_score_loss(y_true, max_probas) if probas.ndim == 1 else 0.0,
            log_loss=log_loss(y_true, probas) if len(np.unique(y_true)) > 1 else 0.0,
            reliability_diagram={'confidence': bin_confs, 'accuracy': bin_accs}
        )
    
    def _check_alerts(self, metrics: CalibrationMetrics):
        """Check if calibration alert should be raised."""
        if metrics.expected_calibration_error > self.alert_threshold:
            self.alerts.append({
                'type': 'HIGH_ECE',
                'message': f"ECE ({metrics.expected_calibration_error:.4f}) exceeds threshold ({self.alert_threshold})",
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            logger.warning(f"Calibration alert: {self.alerts[-1]['message']}")
        
        # Check for drift (if we have history)
        if len(self.metrics_history) > 5:
            recent_ece = np.mean([m.expected_calibration_error for m in self.metrics_history[-5:]])
            older_ece = np.mean([m.expected_calibration_error for m in self.metrics_history[-10:-5]])
            
            if recent_ece > older_ece * 1.5:  # 50% degradation
                self.alerts.append({
                    'type': 'CALIBRATION_DRIFT',
                    'message': f"Calibration degrading: {older_ece:.4f} -> {recent_ece:.4f}",
                    'timestamp': datetime.now().isoformat()
                })
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        return {
            'prediction_count': self.prediction_count,
            'metrics_history_size': len(self.metrics_history),
            'active_alerts': len(self.alerts),
            'latest_ece': self.metrics_history[-1].expected_calibration_error if self.metrics_history else None,
            'alerts': self.alerts[-5:]  # Last 5 alerts
        }


class EnhancedCalibrationSystem:
    """
    Complete calibration system with all V4.0 features.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.adaptive_calibrator = AdaptiveCalibrator()
        self.online_calibrator = OnlineCalibrator(
            window_size=self.config.get('window_size', 1000),
            recalibrate_threshold=self.config.get('recalibrate_threshold', 0.05)
        )
        self.monitor = CalibrationMonitor(
            alert_threshold=self.config.get('alert_threshold', 0.1)
        )
        
        self.is_fitted = False
    
    def fit(self, probas: np.ndarray, y_true: np.ndarray, 
            val_probas: np.ndarray = None, val_y: np.ndarray = None):
        """Fit calibration system."""
        logger.info("Fitting calibration system...")
        
        self.adaptive_calibrator.fit(probas, y_true, val_probas, val_y)
        self.is_fitted = True
        
        # Initialize online calibrator with calibrated probabilities
        calibrated = self.adaptive_calibrator.calibrate(probas)
        self.online_calibrator.history_probas = list(calibrated)
        self.online_calibrator.history_true = list(y_true)
        
        return self.get_metrics(probas, y_true)
    
    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning original probabilities")
            return probas
        
        return self.adaptive_calibrator.calibrate(probas)
    
    def update(self, probas: np.ndarray, y_true: np.ndarray):
        """Update with new observations for online adaptation."""
        calibrated = self.calibrate(probas)
        self.online_calibrator.update(calibrated, y_true)
        self.monitor.record_prediction(calibrated, y_true)
    
    def get_metrics(self, probas: np.ndarray, y_true: np.ndarray) -> CalibrationMetrics:
        """Get calibration metrics."""
        calibrated = self.calibrate(probas)
        return self.monitor._calculate_metrics(calibrated, y_true)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'is_fitted': self.is_fitted,
            'best_method': self.adaptive_calibrator.best_method,
            'online_status': self.online_calibrator.get_status(),
            'monitor_status': self.monitor.get_status()
        }
    
    def should_recalibrate(self) -> bool:
        """Check if recalibration is needed."""
        monitor_status = self.monitor.get_status()
        return (
            len(self.monitor.alerts) > 0 or
            (monitor_status.get('latest_ece') or 0) > 0.1
        )


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED CALIBRATION SYSTEM - TEST")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_classes = 3
    
    # Simulated model predictions (uncalibrated)
    raw_logits = np.random.randn(n_samples, n_classes)
    probas = softmax(raw_logits * 2, axis=1)  # Overconfident
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Split
    train_idx = slice(0, 400)
    val_idx = slice(400, 500)
    
    print("\n1. Testing Temperature Scaling...")
    temp_cal = TemperatureScaling()
    temp_cal.fit(probas[train_idx], y_true[train_idx])
    temp_calibrated = temp_cal.calibrate(probas[val_idx])
    print(f"   Temperature: {temp_cal.temperature:.4f}")
    
    print("\n2. Testing Adaptive Calibrator...")
    adaptive = AdaptiveCalibrator()
    adaptive.fit(probas[train_idx], y_true[train_idx])
    print(f"   Best method: {adaptive.best_method}")
    
    print("\n3. Testing Enhanced Calibration System...")
    system = EnhancedCalibrationSystem()
    metrics = system.fit(
        probas[train_idx], y_true[train_idx],
        probas[val_idx], y_true[val_idx]
    )
    print(f"   ECE: {metrics.expected_calibration_error:.4f}")
    print(f"   Status: {system.get_status()}")
    
    print("\n✅ Calibration system test complete!")
