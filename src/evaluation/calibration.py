"""
Calibration Module
Probability calibration for model predictions.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ProbabilityCalibrator:
    """
    Calibrates prediction probabilities.
    
    Methods:
    - Platt scaling
    - Isotonic regression
    - Temperature scaling
    """
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrators = {}
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation data.
        """
        if len(y_proba.shape) == 1:
            y_proba = y_proba.reshape(-1, 1)
        
        n_classes = y_proba.shape[1]
        
        for c in range(n_classes):
            class_true = (y_true == c).astype(int)
            class_proba = y_proba[:, c]
            
            if self.method == 'isotonic' and SKLEARN_AVAILABLE:
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(class_proba, class_true)
                self.calibrators[c] = calibrator
            
            elif self.method == 'temperature':
                # Optimize temperature
                self.temperature = self._optimize_temperature(y_true, y_proba)
        
        self.is_fitted = True
        logger.info(f"Calibrator fitted with method: {self.method}")
        
        return self
    
    def _optimize_temperature(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> float:
        """Optimize temperature for scaling."""
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in np.linspace(0.1, 5.0, 50):
            scaled = self._apply_temperature(y_proba, temp)
            
            # Calculate cross-entropy loss
            loss = 0
            for i, label in enumerate(y_true):
                if isinstance(label, (int, np.integer)) and label < scaled.shape[1]:
                    loss -= np.log(max(scaled[i, label], 1e-10))
            loss /= len(y_true)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        return best_temp
    
    def _apply_temperature(
        self,
        proba: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling."""
        # Convert to logits
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        logits = np.log(proba)
        
        # Scale and convert back
        scaled_logits = logits / temperature
        scaled_proba = np.exp(scaled_logits)
        scaled_proba = scaled_proba / scaled_proba.sum(axis=1, keepdims=True)
        
        return scaled_proba
    
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if not self.is_fitted:
            return y_proba
        
        if len(y_proba.shape) == 1:
            y_proba = y_proba.reshape(-1, 1)
        
        if self.method == 'isotonic':
            calibrated = np.zeros_like(y_proba)
            for c, calibrator in self.calibrators.items():
                if c < y_proba.shape[1]:
                    calibrated[:, c] = calibrator.predict(y_proba[:, c])
            
            # Normalize
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
            return calibrated
        
        elif self.method == 'temperature':
            return self._apply_temperature(y_proba, self.temperature)
        
        return y_proba
    
    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """Evaluate calibration quality."""
        calibrated = self.calibrate(y_proba)
        
        # Expected calibration error
        ece = 0
        total_samples = len(y_true)
        
        for c in range(calibrated.shape[1]):
            class_true = (y_true == c).astype(float)
            class_proba = calibrated[:, c]
            
            bins = np.linspace(0, 1, n_bins + 1)
            
            for i in range(n_bins):
                mask = (class_proba >= bins[i]) & (class_proba < bins[i+1])
                if mask.sum() > 0:
                    bin_accuracy = class_true[mask].mean()
                    bin_confidence = class_proba[mask].mean()
                    ece += (mask.sum() / total_samples) * abs(bin_accuracy - bin_confidence)
        
        ece /= calibrated.shape[1]
        
        return {
            'expected_calibration_error': round(ece, 4),
            'temperature': self.temperature if self.method == 'temperature' else None,
            'method': self.method
        }


_calibrator: Optional[ProbabilityCalibrator] = None

def get_calibrator() -> ProbabilityCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = ProbabilityCalibrator()
    return _calibrator
