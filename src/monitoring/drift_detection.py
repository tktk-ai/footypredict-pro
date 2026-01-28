"""
Drift Detection Module - V4.0
Comprehensive monitoring for data drift, concept drift, and model performance degradation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    reference_window: int = 1000  # Number of samples in reference window
    detection_window: int = 100   # Number of samples in detection window
    ks_threshold: float = 0.05   # KS test p-value threshold
    psi_threshold: float = 0.2   # PSI threshold (0.1=slight, 0.25=significant)
    adwin_delta: float = 0.002   # ADWIN sensitivity
    check_frequency: int = 50    # Check every N predictions
    alert_cooldown_hours: int = 24  # Min hours between same alert


class KSTest:
    """
    Kolmogorov-Smirnov Test for distribution shift detection.
    Compares two distributions to detect if they come from the same source.
    """
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """
        Perform KS test between reference and current distributions.
        
        Returns:
            Dictionary with statistic, p-value, and drift detection result
        """
        if len(reference) < 10 or len(current) < 10:
            return {'drift_detected': False, 'statistic': 0, 'p_value': 1.0, 'reason': 'Insufficient samples'}
        
        statistic, p_value = stats.ks_2samp(reference, current)
        drift_detected = p_value < self.threshold
        
        return {
            'drift_detected': drift_detected,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'threshold': self.threshold,
            'severity': self._calculate_severity(statistic)
        }
    
    def _calculate_severity(self, statistic: float) -> str:
        """Calculate drift severity based on KS statistic."""
        if statistic < 0.1:
            return 'low'
        elif statistic < 0.2:
            return 'medium'
        elif statistic < 0.3:
            return 'high'
        else:
            return 'critical'


class PSICalculator:
    """
    Population Stability Index (PSI) for feature distribution monitoring.
    Measures how much a variable's distribution has shifted.
    """
    
    def __init__(self, n_bins: int = 10, threshold: float = 0.2):
        self.n_bins = n_bins
        self.threshold = threshold  # 0.1 = slight shift, 0.25 = significant shift
    
    def calculate(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """
        Calculate PSI between reference and current distributions.
        
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.25: Moderate shift (monitor)
        PSI >= 0.25: Significant shift (action required)
        """
        if len(reference) < 10 or len(current) < 10:
            return {'psi': 0, 'drift_detected': False, 'reason': 'Insufficient samples'}
        
        # Create bins from reference distribution
        try:
            _, bin_edges = np.histogram(reference, bins=self.n_bins)
            
            # Add small epsilon to handle edge cases
            eps = 1e-10
            
            # Calculate percentages per bin
            ref_counts = np.histogram(reference, bins=bin_edges)[0]
            cur_counts = np.histogram(current, bins=bin_edges)[0]
            
            ref_pct = (ref_counts + eps) / (len(reference) + eps * self.n_bins)
            cur_pct = (cur_counts + eps) / (len(current) + eps * self.n_bins)
            
            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            return {
                'psi': float(psi),
                'drift_detected': psi >= self.threshold,
                'threshold': self.threshold,
                'interpretation': self._interpret_psi(psi),
                'bin_details': {
                    'reference_percentages': ref_pct.tolist(),
                    'current_percentages': cur_pct.tolist()
                }
            }
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return {'psi': 0, 'drift_detected': False, 'error': str(e)}
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return 'No significant shift'
        elif psi < 0.25:
            return 'Moderate shift - monitoring recommended'
        else:
            return 'Significant shift - action required'


class ADWIN:
    """
    Adaptive Windowing (ADWIN) for concept drift detection.
    Automatically adjusts window size to detect changes in data distribution.
    """
    
    def __init__(self, delta: float = 0.002):
        """
        Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter (smaller = more sensitive, larger = more robust)
        """
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.drift_detected = False
        self.last_drift_index = -1
        self.n_observations = 0
    
    def add_element(self, value: float) -> bool:
        """
        Add new observation and check for drift.
        
        Returns:
            True if drift is detected
        """
        self.window.append(value)
        self.total += value
        self.width += 1
        self.n_observations += 1
        
        # Update running variance (Welford's algorithm)
        if self.width > 1:
            delta = value - (self.total / self.width)
            self.variance += delta * (value - self.total / self.width)
        
        self.drift_detected = self._check_drift()
        
        if self.drift_detected:
            self.last_drift_index = self.n_observations
        
        return self.drift_detected
    
    def _check_drift(self) -> bool:
        """Check if distribution has drifted using ADWIN algorithm."""
        if self.width < 10:
            return False
        
        # Try different split points
        for i in range(1, self.width):
            if self._test_split(i):
                # Remove old data up to split point
                for _ in range(i):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1
                return True
        
        return False
    
    def _test_split(self, split_point: int) -> bool:
        """Test if split point indicates drift."""
        if split_point < 5 or self.width - split_point < 5:
            return False
        
        # Calculate means for both windows
        window_list = list(self.window)
        left = window_list[:split_point]
        right = window_list[split_point:]
        
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        
        n1, n2 = len(left), len(right)
        n = n1 + n2
        
        # Calculate epsilon cut using Hoeffding bound
        m = 1.0 / ((1.0 / n1) + (1.0 / n2))
        epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))
        
        return abs(left_mean - right_mean) >= epsilon
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current ADWIN statistics."""
        return {
            'window_size': self.width,
            'mean': self.total / max(self.width, 1),
            'drift_detected': self.drift_detected,
            'last_drift_index': self.last_drift_index,
            'total_observations': self.n_observations
        }
    
    def reset(self):
        """Reset the detector."""
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.drift_detected = False


class PerformanceDriftDetector:
    """
    Detects degradation in model performance over time.
    Tracks accuracy, calibration, and other metrics.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.baseline_accuracy = None
        self.baseline_brier = None
    
    def add_prediction(self, predicted_prob: float, actual: int, timestamp: datetime = None):
        """Add a prediction-actual pair."""
        self.predictions.append(predicted_prob)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())
        
        # Set baseline after initial window
        if len(self.predictions) == self.window_size and self.baseline_accuracy is None:
            self._set_baseline()
    
    def _set_baseline(self):
        """Set baseline performance metrics."""
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        predicted_classes = (preds > 0.5).astype(int)
        self.baseline_accuracy = np.mean(predicted_classes == acts)
        self.baseline_brier = np.mean((preds - acts) ** 2)
        
        logger.info(f"Baseline set - Accuracy: {self.baseline_accuracy:.3f}, Brier: {self.baseline_brier:.4f}")
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for performance drift."""
        if len(self.predictions) < self.window_size:
            return {'drift_detected': False, 'reason': 'Insufficient samples'}
        
        if self.baseline_accuracy is None:
            self._set_baseline()
            return {'drift_detected': False, 'reason': 'Baseline just set'}
        
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        predicted_classes = (preds > 0.5).astype(int)
        current_accuracy = np.mean(predicted_classes == acts)
        current_brier = np.mean((preds - acts) ** 2)
        
        accuracy_drop = self.baseline_accuracy - current_accuracy
        brier_increase = current_brier - self.baseline_brier
        
        drift_detected = (accuracy_drop > self.threshold) or (brier_increase > self.threshold)
        
        return {
            'drift_detected': drift_detected,
            'baseline_accuracy': float(self.baseline_accuracy),
            'current_accuracy': float(current_accuracy),
            'accuracy_change': float(-accuracy_drop),
            'baseline_brier': float(self.baseline_brier),
            'current_brier': float(current_brier),
            'brier_change': float(brier_increase),
            'sample_size': len(self.predictions),
            'severity': 'high' if accuracy_drop > 0.1 else 'medium' if drift_detected else 'low'
        }


class DriftAlert:
    """Represents a drift alert."""
    
    def __init__(self, alert_type: str, severity: str, details: Dict[str, Any],
                 feature_name: str = None, recommendations: List[str] = None):
        self.id = f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.alert_type = alert_type
        self.severity = severity  # low, medium, high, critical
        self.details = details
        self.feature_name = feature_name
        self.recommendations = recommendations or []
        self.timestamp = datetime.now()
        self.acknowledged = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.alert_type,
            'severity': self.severity,
            'feature': self.feature_name,
            'details': self.details,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


class DriftDetector:
    """
    Comprehensive drift detection system combining multiple methods.
    Monitors features, predictions, and model performance.
    """
    
    def __init__(self, config: DriftConfig = None, storage_path: str = None):
        self.config = config or DriftConfig()
        self.storage_path = Path(storage_path or 'data/drift_monitoring')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize detectors
        self.ks_test = KSTest(threshold=self.config.ks_threshold)
        self.psi_calculator = PSICalculator(threshold=self.config.psi_threshold)
        
        # ADWIN detectors per feature
        self.adwin_detectors: Dict[str, ADWIN] = {}
        
        # Performance drift detector
        self.performance_detector = PerformanceDriftDetector()
        
        # Reference distributions
        self.reference_data: Dict[str, np.ndarray] = {}
        
        # Current window data
        self.current_data: Dict[str, List[float]] = {}
        
        # Alerts
        self.alerts: List[DriftAlert] = []
        self.alert_history: List[Dict] = []
        
        # Monitoring state
        self.n_samples_processed = 0
        self.last_check = datetime.now()
        self.monitoring_active = True
        
        logger.info("DriftDetector initialized")
    
    def set_reference(self, feature_name: str, data: np.ndarray):
        """Set reference distribution for a feature."""
        self.reference_data[feature_name] = np.array(data)
        self.current_data[feature_name] = []
        self.adwin_detectors[feature_name] = ADWIN(delta=self.config.adwin_delta)
        logger.info(f"Reference set for {feature_name}: {len(data)} samples")
    
    def set_reference_batch(self, data: pd.DataFrame):
        """Set reference distributions for all numeric features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            valid_data = data[col].dropna().values
            if len(valid_data) >= self.config.reference_window:
                self.set_reference(col, valid_data[:self.config.reference_window])
    
    def add_sample(self, features: Dict[str, float], prediction: float = None, actual: int = None):
        """Add a new sample for monitoring."""
        self.n_samples_processed += 1
        
        # Update feature windows
        for feature_name, value in features.items():
            if feature_name not in self.current_data:
                self.current_data[feature_name] = []
            
            self.current_data[feature_name].append(value)
            
            # Update ADWIN
            if feature_name in self.adwin_detectors:
                adwin_drift = self.adwin_detectors[feature_name].add_element(value)
                if adwin_drift:
                    self._raise_alert('adwin_drift', 'medium',
                                     self.adwin_detectors[feature_name].get_stats(),
                                     feature_name=feature_name)
            
            # Trim current window
            if len(self.current_data[feature_name]) > self.config.detection_window:
                self.current_data[feature_name] = self.current_data[feature_name][-self.config.detection_window:]
        
        # Update performance tracker
        if prediction is not None and actual is not None:
            self.performance_detector.add_prediction(prediction, actual)
        
        # Periodic full check
        if self.n_samples_processed % self.config.check_frequency == 0:
            self.run_drift_checks()
    
    def run_drift_checks(self) -> Dict[str, Any]:
        """Run all drift detection checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'samples_processed': self.n_samples_processed,
            'feature_drift': {},
            'performance_drift': {},
            'alerts': []
        }
        
        # Check each feature
        for feature_name in self.reference_data.keys():
            if feature_name not in self.current_data:
                continue
            
            current = np.array(self.current_data[feature_name])
            if len(current) < 10:
                continue
            
            reference = self.reference_data[feature_name]
            
            # KS Test
            ks_result = self.ks_test.test(reference, current)
            
            # PSI
            psi_result = self.psi_calculator.calculate(reference, current)
            
            results['feature_drift'][feature_name] = {
                'ks': ks_result,
                'psi': psi_result
            }
            
            # Raise alerts if needed
            if ks_result.get('drift_detected'):
                self._raise_alert('ks_drift', ks_result.get('severity', 'medium'),
                                 ks_result, feature_name=feature_name)
            
            if psi_result.get('drift_detected'):
                severity = 'high' if psi_result['psi'] > 0.3 else 'medium'
                self._raise_alert('psi_drift', severity, psi_result, feature_name=feature_name)
        
        # Check performance drift
        perf_result = self.performance_detector.check_drift()
        results['performance_drift'] = perf_result
        
        if perf_result.get('drift_detected'):
            self._raise_alert('performance_drift', perf_result.get('severity', 'high'),
                             perf_result, recommendations=[
                                 'Consider retraining the model',
                                 'Check for changes in data quality',
                                 'Review recent feature distributions'
                             ])
        
        # Add active alerts
        results['alerts'] = [a.to_dict() for a in self.alerts if not a.acknowledged]
        
        self.last_check = datetime.now()
        return results
    
    def _raise_alert(self, alert_type: str, severity: str, details: Dict[str, Any],
                    feature_name: str = None, recommendations: List[str] = None):
        """Raise a drift alert."""
        # Check cooldown
        recent_same_alerts = [
            a for a in self.alerts
            if a.alert_type == alert_type and a.feature_name == feature_name
            and (datetime.now() - a.timestamp).total_seconds() < self.config.alert_cooldown_hours * 3600
        ]
        
        if recent_same_alerts:
            return  # Skip due to cooldown
        
        # Generate recommendations if not provided
        if recommendations is None:
            recommendations = self._generate_recommendations(alert_type, severity, details)
        
        alert = DriftAlert(
            alert_type=alert_type,
            severity=severity,
            details=details,
            feature_name=feature_name,
            recommendations=recommendations
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert.to_dict())
        
        logger.warning(f"DRIFT ALERT: {alert_type} (severity: {severity}) - {feature_name or 'model'}")
    
    def _generate_recommendations(self, alert_type: str, severity: str, 
                                  details: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on alert type."""
        recommendations = []
        
        if alert_type in ['ks_drift', 'psi_drift']:
            recommendations.extend([
                'Investigate recent changes in data source',
                'Check for data quality issues',
                'Consider updating reference distribution'
            ])
            if severity in ['high', 'critical']:
                recommendations.append('Prioritize model retraining')
        
        elif alert_type == 'adwin_drift':
            recommendations.extend([
                'Concept drift detected - data characteristics changing',
                'Monitor closely for performance degradation',
                'Consider collecting more recent training data'
            ])
        
        elif alert_type == 'performance_drift':
            recommendations.extend([
                'Model accuracy has degraded significantly',
                'Trigger retraining pipeline',
                'Review prediction errors for patterns'
            ])
        
        return recommendations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        active_alerts = [a for a in self.alerts if not a.acknowledged]
        
        return {
            'status': 'ALERT' if active_alerts else 'OK',
            'samples_processed': self.n_samples_processed,
            'last_check': self.last_check.isoformat(),
            'features_monitored': len(self.reference_data),
            'active_alerts': len(active_alerts),
            'alert_breakdown': {
                'critical': len([a for a in active_alerts if a.severity == 'critical']),
                'high': len([a for a in active_alerts if a.severity == 'high']),
                'medium': len([a for a in active_alerts if a.severity == 'medium']),
                'low': len([a for a in active_alerts if a.severity == 'low'])
            }
        }
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def save_state(self):
        """Save monitoring state to disk."""
        state = {
            'n_samples_processed': self.n_samples_processed,
            'last_check': self.last_check.isoformat(),
            'alert_history': self.alert_history[-100:],  # Keep last 100
            'reference_stats': {
                name: {'mean': float(np.mean(data)), 'std': float(np.std(data)), 'n': len(data)}
                for name, data in self.reference_data.items()
            }
        }
        
        state_file = self.storage_path / 'drift_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Drift state saved to {state_file}")
    
    def load_state(self):
        """Load monitoring state from disk."""
        state_file = self.storage_path / 'drift_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.n_samples_processed = state.get('n_samples_processed', 0)
            self.alert_history = state.get('alert_history', [])
            logger.info(f"Drift state loaded from {state_file}")


# Factory function
def create_drift_detector(reference_data: pd.DataFrame = None, 
                         config: DriftConfig = None) -> DriftDetector:
    """Create and initialize a drift detector."""
    detector = DriftDetector(config=config)
    
    if reference_data is not None:
        detector.set_reference_batch(reference_data)
    
    return detector


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("DRIFT DETECTION MODULE - TEST")
    print("=" * 60)
    
    # Create sample reference data
    np.random.seed(42)
    reference = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.uniform(0, 10, 1000)
    })
    
    # Create detector
    detector = create_drift_detector(reference)
    
    # Simulate predictions with gradual drift
    print("\nSimulating predictions with drift...")
    
    for i in range(200):
        # Add increasing drift
        drift = i * 0.01
        features = {
            'feature1': np.random.normal(drift, 1),
            'feature2': np.random.normal(5 + drift, 2),
            'feature3': np.random.uniform(0, 10)
        }
        
        prediction = np.random.random()
        actual = 1 if np.random.random() > 0.5 else 0
        
        detector.add_sample(features, prediction, actual)
    
    # Get summary
    summary = detector.get_summary()
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    
    # Run drift checks
    results = detector.run_drift_checks()
    print(f"\nAlerts: {len(results['alerts'])}")
    
    print("\n✅ Drift detection module working correctly!")
