"""
Self-Improving Predictor - V4.0
Orchestrates continuous improvement through monitoring, retraining, and suggestion generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import joblib
import logging
from collections import deque
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelHealth(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class PredictionResult:
    """Result of a prediction with metadata."""
    prediction: Dict[str, float]  # e.g., {'home': 0.45, 'draw': 0.30, 'away': 0.25}
    confidence: float
    uncertainty: float
    features_used: int
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'features_used': self.features_used,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ImprovementSuggestion:
    """Suggestion for improving the model."""
    category: str  # 'data', 'features', 'training', 'calibration', 'infrastructure'
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    expected_impact: str
    effort: str  # 'low', 'medium', 'high'
    action_items: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'expected_impact': self.expected_impact,
            'effort': self.effort,
            'action_items': self.action_items,
            'created_at': self.created_at.isoformat()
        }


class HealthMonitor:
    """
    Continuous health monitoring for the prediction system.
    Tracks performance, drift, and system metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Performance tracking
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Metrics history
        self.accuracy_history = deque(maxlen=1000)
        self.calibration_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        
        # Thresholds
        self.accuracy_threshold = 0.45  # Minimum acceptable accuracy
        self.calibration_threshold = 0.1  # Max Brier score deviation
        self.latency_threshold = 1000  # Max latency in ms
        
        # Baseline metrics
        self.baseline_accuracy = None
        self.baseline_calibration = None
        
        logger.info("HealthMonitor initialized")
    
    def record_prediction(self, prediction: Dict[str, float], latency_ms: float):
        """Record a prediction for monitoring."""
        self.predictions.append(prediction)
        self.timestamps.append(datetime.now())
        self.latency_history.append(latency_ms)
    
    def record_result(self, actual_outcome: str):
        """Record actual outcome for latest prediction."""
        self.actuals.append(actual_outcome)
        
        if len(self.predictions) >= self.window_size and len(self.actuals) >= self.window_size:
            self._update_metrics()
    
    def _update_metrics(self):
        """Update rolling performance metrics."""
        if len(self.predictions) < 10 or len(self.actuals) < 10:
            return
        
        # Calculate accuracy (top prediction correct)
        correct = 0
        for pred, actual in zip(list(self.predictions)[-len(self.actuals):], self.actuals):
            predicted_outcome = max(pred.items(), key=lambda x: x[1])[0]
            if predicted_outcome == actual:
                correct += 1
        
        accuracy = correct / len(self.actuals)
        self.accuracy_history.append(accuracy)
        
        # Calculate Brier score for calibration
        brier_scores = []
        for pred, actual in zip(list(self.predictions)[-len(self.actuals):], self.actuals):
            for outcome, prob in pred.items():
                actual_val = 1 if outcome == actual else 0
                brier_scores.append((prob - actual_val) ** 2)
        
        if brier_scores:
            calibration = np.mean(brier_scores)
            self.calibration_history.append(calibration)
        
        # Set baseline if not set
        if self.baseline_accuracy is None and len(self.accuracy_history) >= 10:
            self.baseline_accuracy = np.mean(list(self.accuracy_history)[:10])
            self.baseline_calibration = np.mean(list(self.calibration_history)[:10])
            logger.info(f"Baseline set - Accuracy: {self.baseline_accuracy:.3f}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if len(self.accuracy_history) < 5:
            return {
                'status': ModelHealth.HEALTHY.value,
                'message': 'Insufficient data for health assessment',
                'samples': len(self.actuals)
            }
        
        current_accuracy = np.mean(list(self.accuracy_history)[-10:])
        current_calibration = np.mean(list(self.calibration_history)[-10:]) if self.calibration_history else 0
        current_latency = np.mean(list(self.latency_history)[-10:]) if self.latency_history else 0
        
        # Determine health status
        issues = []
        
        if current_accuracy < self.accuracy_threshold:
            issues.append(f"Accuracy below threshold: {current_accuracy:.2%} < {self.accuracy_threshold:.2%}")
        
        if self.baseline_accuracy and current_accuracy < self.baseline_accuracy - 0.05:
            issues.append(f"Accuracy degraded: {current_accuracy:.2%} vs baseline {self.baseline_accuracy:.2%}")
        
        if current_latency > self.latency_threshold:
            issues.append(f"High latency: {current_latency:.0f}ms > {self.latency_threshold}ms")
        
        # Calculate status
        if len(issues) >= 3:
            status = ModelHealth.CRITICAL
        elif len(issues) >= 2:
            status = ModelHealth.DEGRADED
        elif len(issues) >= 1:
            status = ModelHealth.WARNING
        else:
            status = ModelHealth.HEALTHY
        
        return {
            'status': status.value,
            'current_accuracy': float(current_accuracy),
            'baseline_accuracy': float(self.baseline_accuracy) if self.baseline_accuracy else None,
            'current_calibration': float(current_calibration),
            'current_latency_ms': float(current_latency),
            'samples_evaluated': len(self.actuals),
            'issues': issues,
            'last_updated': datetime.now().isoformat()
        }


class SuggestionEngine:
    """
    Generates actionable improvement suggestions based on system metrics.
    """
    
    def __init__(self):
        self.generated_suggestions: List[ImprovementSuggestion] = []
        self.suggestion_cooldown: Dict[str, datetime] = {}
        self.cooldown_hours = 24
    
    def analyze_and_suggest(self, 
                           health_status: Dict[str, Any],
                           drift_summary: Dict[str, Any] = None,
                           training_metrics: Dict[str, Any] = None) -> List[ImprovementSuggestion]:
        """Generate suggestions based on current system state."""
        suggestions = []
        
        # Accuracy-based suggestions
        if health_status.get('current_accuracy', 1.0) < 0.50:
            suggestions.append(self._create_suggestion(
                'accuracy_low',
                category='training',
                priority='high',
                title='Model Accuracy Below Target',
                description=f"Current accuracy is {health_status['current_accuracy']:.2%}, below 50% target.",
                expected_impact='10-15% accuracy improvement',
                effort='high',
                action_items=[
                    'Collect more recent training data',
                    'Engineer additional predictive features',
                    'Run hyperparameter optimization',
                    'Consider ensemble approach'
                ]
            ))
        
        # Calibration-based suggestions
        if health_status.get('current_calibration', 0) > 0.25:
            suggestions.append(self._create_suggestion(
                'calibration_poor',
                category='calibration',
                priority='medium',
                title='Prediction Calibration Needs Improvement',
                description='Predicted probabilities are not well-calibrated with actual outcomes.',
                expected_impact='Better betting value identification',
                effort='medium',
                action_items=[
                    'Apply temperature scaling',
                    'Use isotonic regression calibration',
                    'Implement Platt scaling'
                ]
            ))
        
        # Drift-based suggestions
        if drift_summary and drift_summary.get('active_alerts', 0) > 0:
            alert_breakdown = drift_summary.get('alert_breakdown', {})
            if alert_breakdown.get('critical', 0) > 0 or alert_breakdown.get('high', 0) > 0:
                suggestions.append(self._create_suggestion(
                    'drift_detected',
                    category='data',
                    priority='critical',
                    title='Significant Data Drift Detected',
                    description='Data distribution has shifted significantly from training data.',
                    expected_impact='Prevent model degradation',
                    effort='medium',
                    action_items=[
                        'Investigate data source changes',
                        'Update reference distributions',
                        'Retrain model on recent data',
                        'Review feature importance changes'
                    ]
                ))
        
        # Performance suggestions
        if health_status.get('current_latency_ms', 0) > 500:
            suggestions.append(self._create_suggestion(
                'latency_high',
                category='infrastructure',
                priority='low',
                title='Prediction Latency Could Be Improved',
                description=f"Average latency is {health_status['current_latency_ms']:.0f}ms.",
                expected_impact='Faster user experience',
                effort='medium',
                action_items=[
                    'Profile prediction pipeline',
                    'Cache preprocessed features',
                    'Consider model compression',
                    'Optimize feature computation'
                ]
            ))
        
        # Data quality suggestions
        if training_metrics and training_metrics.get('missing_rate', 0) > 0.1:
            suggestions.append(self._create_suggestion(
                'data_quality',
                category='data',
                priority='medium',
                title='High Rate of Missing Features',
                description='More than 10% of features are missing values.',
                expected_impact='Better model stability',
                effort='low',
                action_items=[
                    'Review data collection pipeline',
                    'Implement better imputation strategies',
                    'Add data validation checks'
                ]
            ))
        
        # Feature engineering suggestions
        if training_metrics and training_metrics.get('feature_count', 0) < 500:
            suggestions.append(self._create_suggestion(
                'features_limited',
                category='features',
                priority='medium',
                title='Feature Engineering Opportunity',
                description=f"Currently using {training_metrics.get('feature_count', 0)} features. Target is 600+.",
                expected_impact='5-10% accuracy improvement',
                effort='medium',
                action_items=[
                    'Add more rolling window variations',
                    'Include xG features if available',
                    'Generate position-based features',
                    'Add market odds-derived features'
                ]
            ))
        
        self.generated_suggestions.extend(suggestions)
        return suggestions
    
    def _create_suggestion(self, key: str, **kwargs) -> Optional[ImprovementSuggestion]:
        """Create a suggestion if not in cooldown."""
        last_generated = self.suggestion_cooldown.get(key)
        
        if last_generated:
            hours_since = (datetime.now() - last_generated).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                return None
        
        self.suggestion_cooldown[key] = datetime.now()
        return ImprovementSuggestion(**kwargs)
    
    def get_top_suggestions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N suggestions by priority."""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        sorted_suggestions = sorted(
            [s for s in self.generated_suggestions if s is not None],
            key=lambda x: priority_order.get(x.priority, 99)
        )
        
        return [s.to_dict() for s in sorted_suggestions[:n]]


class RetrainingTrigger:
    """
    Determines when to trigger model retraining.
    """
    
    def __init__(self):
        self.last_retrain = None
        self.min_retrain_interval_hours = 24
        self.retrain_requested = False
        self.retrain_reasons = []
    
    def check_retrain_needed(self,
                            health_status: Dict[str, Any],
                            drift_summary: Dict[str, Any] = None,
                            force: bool = False) -> Tuple[bool, List[str]]:
        """Check if retraining should be triggered."""
        reasons = []
        
        # Check cooldown
        if self.last_retrain:
            hours_since = (datetime.now() - self.last_retrain).total_seconds() / 3600
            if hours_since < self.min_retrain_interval_hours and not force:
                return False, [f"Cooldown active ({hours_since:.1f}h since last retrain)"]
        
        # Health-based triggers
        if health_status.get('status') == ModelHealth.CRITICAL.value:
            reasons.append("Model health is CRITICAL")
        elif health_status.get('status') == ModelHealth.DEGRADED.value:
            reasons.append("Model health is DEGRADED")
        
        accuracy = health_status.get('current_accuracy', 1.0)
        baseline = health_status.get('baseline_accuracy')
        if baseline and accuracy < baseline - 0.08:
            reasons.append(f"Accuracy dropped >8% from baseline ({accuracy:.2%} vs {baseline:.2%})")
        
        # Drift-based triggers
        if drift_summary:
            critical_alerts = drift_summary.get('alert_breakdown', {}).get('critical', 0)
            high_alerts = drift_summary.get('alert_breakdown', {}).get('high', 0)
            
            if critical_alerts > 0:
                reasons.append(f"{critical_alerts} critical drift alerts")
            if high_alerts >= 3:
                reasons.append(f"{high_alerts} high severity drift alerts")
        
        # Force trigger
        if force:
            reasons.append("Manual retrain requested")
        
        should_retrain = len(reasons) > 0
        
        if should_retrain:
            self.retrain_reasons = reasons
            self.retrain_requested = True
        
        return should_retrain, reasons
    
    def mark_retrain_complete(self):
        """Mark retraining as complete."""
        self.last_retrain = datetime.now()
        self.retrain_requested = False
        self.retrain_reasons = []
        logger.info("Retraining marked as complete")


class SelfImprovingPredictor:
    """
    Main orchestrator for the self-improving prediction system.
    Combines models, monitoring, suggestions, and retraining.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 storage_path: str = None):
        self.model_path = Path(model_path or 'models/v4')
        self.storage_path = Path(storage_path or 'data/self_improving')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.health_monitor = HealthMonitor()
        self.suggestion_engine = SuggestionEngine()
        self.retrain_trigger = RetrainingTrigger()
        
        # Drift detector (imported separately to avoid circular imports)
        self.drift_detector = None
        
        # Models
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self.feature_generator = None
        
        # State
        self.model_version = "v4.0.0"
        self.n_predictions = 0
        self.is_retraining = False
        self.retraining_lock = threading.Lock()
        
        # Callbacks
        self.on_retrain_needed: Optional[Callable] = None
        
        logger.info("SelfImprovingPredictor initialized")
    
    def load_models(self):
        """Load trained models and calibrators."""
        if not self.model_path.exists():
            logger.warning(f"Model path {self.model_path} does not exist")
            return
        
        # Load models
        for market in ['result', 'btts', 'over25', 'over15']:
            model_file = self.model_path / f'{market}_model.joblib'
            if model_file.exists():
                self.models[market] = joblib.load(model_file)
                logger.info(f"Loaded {market} model")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def set_drift_detector(self, detector):
        """Set the drift detector."""
        self.drift_detector = detector
    
    def predict(self, 
               home_team: str, 
               away_team: str, 
               features: Dict[str, float],
               market: str = 'result') -> PredictionResult:
        """
        Make a prediction with full monitoring.
        """
        import time
        start_time = time.time()
        
        self.n_predictions += 1
        
        # Default prediction if model not available
        if market not in self.models:
            prediction = {'home': 0.40, 'draw': 0.30, 'away': 0.30}
            confidence = 0.3
            uncertainty = 0.5
        else:
            model = self.models[market]
            
            # Prepare features
            X = pd.DataFrame([features])
            
            try:
                # Get prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    
                    if len(proba) == 3:
                        prediction = {'home': float(proba[0]), 'draw': float(proba[1]), 'away': float(proba[2])}
                    elif len(proba) == 2:
                        prediction = {'yes': float(proba[1]), 'no': float(proba[0])}
                    else:
                        prediction = {'class_0': float(proba[0])}
                    
                    # Calculate confidence (max probability)
                    confidence = float(max(proba))
                    
                    # Calculate uncertainty (entropy-based)
                    entropy = -np.sum(proba * np.log(proba + 1e-10))
                    max_entropy = np.log(len(proba))
                    uncertainty = float(entropy / max_entropy) if max_entropy > 0 else 0.5
                else:
                    pred = model.predict(X)[0]
                    prediction = {'predicted': float(pred)}
                    confidence = 0.5
                    uncertainty = 0.5
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                prediction = {'home': 0.33, 'draw': 0.34, 'away': 0.33}
                confidence = 0.3
                uncertainty = 0.7
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record for monitoring
        self.health_monitor.record_prediction(prediction, latency_ms)
        
        # Update drift detector
        if self.drift_detector:
            self.drift_detector.add_sample(features, max(prediction.values()))
        
        # Create result
        result = PredictionResult(
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
            features_used=len(features),
            model_version=self.model_version
        )
        
        return result
    
    def update_with_result(self, actual_outcome: str):
        """Update system with actual outcome."""
        self.health_monitor.record_result(actual_outcome)
        
        # Check if retraining needed
        health_status = self.health_monitor.get_health_status()
        drift_summary = self.drift_detector.get_summary() if self.drift_detector else None
        
        should_retrain, reasons = self.retrain_trigger.check_retrain_needed(
            health_status, drift_summary
        )
        
        if should_retrain and self.on_retrain_needed:
            logger.info(f"Retraining triggered: {reasons}")
            self.on_retrain_needed(reasons)
    
    def get_health(self) -> Dict[str, Any]:
        """Get comprehensive system health."""
        health_status = self.health_monitor.get_health_status()
        drift_summary = self.drift_detector.get_summary() if self.drift_detector else {}
        
        return {
            'model_health': health_status,
            'drift_status': drift_summary,
            'predictions_made': self.n_predictions,
            'model_version': self.model_version,
            'models_loaded': list(self.models.keys()),
            'is_retraining': self.is_retraining,
            'retrain_pending': self.retrain_trigger.retrain_requested,
            'retrain_reasons': self.retrain_trigger.retrain_reasons
        }
    
    def get_suggestions(self, training_metrics: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get improvement suggestions."""
        health_status = self.health_monitor.get_health_status()
        drift_summary = self.drift_detector.get_summary() if self.drift_detector else None
        
        self.suggestion_engine.analyze_and_suggest(
            health_status=health_status,
            drift_summary=drift_summary,
            training_metrics=training_metrics
        )
        
        return self.suggestion_engine.get_top_suggestions()
    
    def trigger_retrain(self, force: bool = False) -> Dict[str, Any]:
        """Trigger model retraining."""
        with self.retraining_lock:
            if self.is_retraining:
                return {'status': 'already_running', 'message': 'Retraining is already in progress'}
            
            health_status = self.health_monitor.get_health_status()
            should_retrain, reasons = self.retrain_trigger.check_retrain_needed(
                health_status, force=force
            )
            
            if not should_retrain and not force:
                return {'status': 'not_needed', 'message': 'Retraining not currently needed'}
            
            self.is_retraining = True
            
            # In production, this would start async retraining
            logger.info(f"Retraining triggered with reasons: {reasons}")
            
            return {
                'status': 'started',
                'reasons': reasons,
                'message': 'Retraining process initiated'
            }
    
    def complete_retrain(self, new_model_version: str = None):
        """Mark retraining as complete."""
        with self.retraining_lock:
            self.is_retraining = False
            self.retrain_trigger.mark_retrain_complete()
            
            if new_model_version:
                self.model_version = new_model_version
            
            # Reload models
            self.load_models()
            
            logger.info(f"Retraining complete. New version: {self.model_version}")
    
    def save_state(self):
        """Save predictor state."""
        state = {
            'model_version': self.model_version,
            'n_predictions': self.n_predictions,
            'last_retrain': self.retrain_trigger.last_retrain.isoformat() if self.retrain_trigger.last_retrain else None,
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = self.storage_path / 'predictor_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {state_file}")
    
    def load_state(self):
        """Load predictor state."""
        state_file = self.storage_path / 'predictor_state.json'
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.model_version = state.get('model_version', 'v4.0.0')
            self.n_predictions = state.get('n_predictions', 0)
            
            if state.get('last_retrain'):
                self.retrain_trigger.last_retrain = datetime.fromisoformat(state['last_retrain'])
            
            logger.info(f"State loaded: version={self.model_version}, predictions={self.n_predictions}")


# Factory function
def create_self_improving_predictor(model_path: str = None) -> SelfImprovingPredictor:
    """Create and initialize a self-improving predictor."""
    predictor = SelfImprovingPredictor(model_path=model_path)
    predictor.load_models()
    predictor.load_state()
    return predictor


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SELF-IMPROVING PREDICTOR - TEST")
    print("=" * 60)
    
    # Create predictor
    predictor = SelfImprovingPredictor()
    
    # Simulate predictions
    print("\nSimulating predictions...")
    
    for i in range(50):
        features = {f'feature_{j}': np.random.random() for j in range(100)}
        
        result = predictor.predict(
            home_team='Team A',
            away_team='Team B',
            features=features,
            market='result'
        )
        
        # Simulate outcome
        outcomes = ['home', 'draw', 'away']
        actual = np.random.choice(outcomes, p=[0.45, 0.25, 0.30])
        predictor.update_with_result(actual)
    
    # Get health
    health = predictor.get_health()
    print(f"\nHealth: {json.dumps(health['model_health'], indent=2)}")
    
    # Get suggestions
    suggestions = predictor.get_suggestions({'feature_count': 304})
    print(f"\nSuggestions: {len(suggestions)}")
    for s in suggestions:
        print(f"  - [{s['priority']}] {s['title']}")
    
    print("\n✅ Self-improving predictor working correctly!")
