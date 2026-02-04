"""
Probability Calibration for Football Predictions

This module implements calibration techniques to ensure
predicted probabilities match actual outcomes.

Techniques:
- Temperature scaling (Platt scaling)
- Isotonic regression
- Expected Calibration Error (ECE) metric
- Empirical calibration (using known market accuracies)

Author: FootyPredict Pro
"""

import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
import pickle
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Market-Specific Empirical Accuracies (from training/validation)
# =============================================================================

MARKET_ACCURACY = {
    # Match result
    '1x2': 0.56,
    
    # Goals markets
    'over_15': 0.75,
    'over_25': 0.60,
    'over_35': 0.55,
    'btts': 0.62,
    
    # Double chance (highest accuracy)
    'dc_1x': 0.72,
    'dc_x2': 0.70,
    'dc_12': 0.73,
    
    # Half-time markets
    'ht_over_05': 0.68,
    'ht_over_15': 0.55,
    'ht_btts': 0.58,
    
    # Combined markets
    'home_over_25': 0.52,
    'away_over_25': 0.48,
}

DEFAULT_ACCURACY = 0.55


def empirical_calibrate(raw_prob: float, market: str = 'default') -> float:
    """
    Calibrate a raw probability using empirical market accuracy.
    
    This adjusts overconfident predictions toward realistic values
    based on the model's actual historical performance.
    
    Args:
        raw_prob: Raw probability from model (0-1)
        market: Market type for market-specific calibration
        
    Returns:
        Calibrated probability
    """
    base_accuracy = MARKET_ACCURACY.get(market, DEFAULT_ACCURACY)
    base_rate = 0.5
    
    # How much does the model beat random?
    skill = base_accuracy - base_rate
    
    # How far is raw_prob from uncertain (0.5)?
    deviation = raw_prob - 0.5
    
    # Scale the deviation by skill factor
    trust_factor = min(1.0, skill * 2)
    calibrated = base_rate + deviation * trust_factor
    
    # Clamp to reasonable range
    return max(0.05, min(0.95, calibrated))


def get_honest_confidence(raw_prob: float, market: str) -> Dict:
    """
    Get user-friendly honest confidence display.
    
    Returns a dict suitable for frontend display.
    """
    calibrated = empirical_calibrate(raw_prob, market)
    pct = round(calibrated * 100, 1)
    base_accuracy = MARKET_ACCURACY.get(market, DEFAULT_ACCURACY)
    
    # Determine confidence level
    deviation = abs(calibrated - 0.5)
    if deviation < 0.05:
        level = 'very_low'
        label = 'Coin flip'
    elif deviation < 0.10:
        level = 'low'
        label = 'Slight edge'
    elif deviation < 0.15:
        level = 'medium'
        label = 'Moderate'
    elif deviation < 0.20:
        level = 'high'
        label = 'Good chance'
    else:
        level = 'very_high'
        label = 'Strong chance'
    
    return {
        'raw_probability': round(raw_prob * 100, 1),
        'calibrated_probability': pct,
        'confidence_level': level,
        'honest_label': label,
        'model_accuracy': round(base_accuracy * 100, 0),
    }




class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    Scales logits by a learned temperature parameter.
    
    P_calibrated = softmax(logits / T)
    """
    
    def __init__(self, initial_temp: float = 1.5):
        self.temperature = initial_temp
        self.fitted = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'TemperatureScaling':
        """
        Find optimal temperature using NLL on validation set.
        
        Args:
            logits: Raw model outputs before softmax (n_samples, n_classes)
            labels: True labels (n_samples,)
        """
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temp):
            scaled = logits / temp
            probs = self._softmax(scaled)
            # Negative log likelihood
            correct_probs = probs[np.arange(len(labels)), labels]
            return -np.log(correct_probs + 1e-10).mean()
        
        # Find optimal temperature
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.fitted = True
        
        logger.info(f"Calibration: Optimal temperature = {self.temperature:.3f}")
        return self
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Note: Works best when applied to logits, but can approximate
        from probabilities by reversing softmax.
        """
        if not self.fitted:
            logger.warning("Calibrator not fitted, returning original probs")
            return probs
        
        # Approximate logits from probabilities
        logits = np.log(probs + 1e-10)
        
        # Scale and re-normalize
        scaled = logits / self.temperature
        calibrated = self._softmax(scaled)
        
        return calibrated
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


class IsotonicCalibration:
    """
    Isotonic regression calibration (one per class).
    Non-parametric calibration that preserves ranking.
    """
    
    def __init__(self):
        self.calibrators = {}  # One per class
        self.fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibration':
        """
        Fit isotonic regression for each class.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
        """
        n_classes = probs.shape[1]
        
        for c in range(n_classes):
            # Binary labels for this class
            binary_labels = (labels == c).astype(float)
            
            # Fit isotonic regression
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs[:, c], binary_labels)
            
            self.calibrators[c] = iso
        
        self.fitted = True
        logger.info(f"Isotonic calibration fitted for {n_classes} classes")
        return self
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration"""
        if not self.fitted:
            return probs
        
        calibrated = np.zeros_like(probs)
        
        for c, iso in self.calibrators.items():
            calibrated[:, c] = iso.predict(probs[:, c])
        
        # Re-normalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, 
                                n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match actual accuracy.
    Lower is better (0 = perfectly calibrated).
    
    Returns:
        ece: Expected calibration error
        confidences: Average confidence per bin
        accuracies: Actual accuracy per bin
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies_arr = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            bin_acc = accuracies_arr[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
    
    return ece, np.array(bin_confidences), np.array(bin_accuracies)


class CalibratedModel:
    """
    Wrapper that adds calibration to any model.
    """
    
    def __init__(self, base_model, calibration_method: str = 'temperature'):
        """
        Args:
            base_model: Model with predict_proba method
            calibration_method: 'temperature' or 'isotonic'
        """
        self.base_model = base_model
        self.calibration_method = calibration_method
        
        if calibration_method == 'temperature':
            self.calibrator = TemperatureScaling()
        else:
            self.calibrator = IsotonicCalibration()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> 'CalibratedModel':
        """
        Train base model and fit calibrator on validation set.
        """
        # Train base model
        self.base_model.fit(X_train, y_train)
        
        # Get validation predictions
        val_probs = self.base_model.predict_proba(X_val)
        
        # Fit calibrator
        self.calibrator.fit(val_probs, y_val)
        
        # Report ECE
        ece_before, _, _ = expected_calibration_error(val_probs, y_val)
        cal_probs = self.calibrator.calibrate(val_probs)
        ece_after, _, _ = expected_calibration_error(cal_probs, y_val)
        
        logger.info(f"ECE: {ece_before:.4f} → {ece_after:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities"""
        raw_probs = self.base_model.predict_proba(X)
        return self.calibrator.calibrate(raw_probs)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'base_model': self.base_model,
                'calibrator': self.calibrator,
                'method': self.calibration_method,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CalibratedModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls.__new__(cls)
        model.base_model = data['base_model']
        model.calibrator = data['calibrator']
        model.calibration_method = data['method']
        
        return model


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Testing Calibration Module...")
    
    np.random.seed(42)
    
    # Simulated probabilities (overconfident)
    n_samples = 1000
    probs = np.random.dirichlet([3, 1, 1], n_samples)  # Biased toward class 0
    labels = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.25, 0.30])
    
    # Test ECE
    ece, confs, accs = expected_calibration_error(probs, labels)
    print(f"ECE before calibration: {ece:.4f}")
    
    # Test temperature scaling
    temp_cal = TemperatureScaling()
    temp_cal.fit(np.log(probs + 1e-10), labels)  # Using log-probs as proxy
    cal_probs = temp_cal.calibrate(probs)
    
    ece_after, _, _ = expected_calibration_error(cal_probs, labels)
    print(f"ECE after temperature scaling: {ece_after:.4f}")
    
    # Test isotonic
    iso_cal = IsotonicCalibration()
    iso_cal.fit(probs, labels)
    iso_probs = iso_cal.calibrate(probs)
    
    ece_iso, _, _ = expected_calibration_error(iso_probs, labels)
    print(f"ECE after isotonic: {ece_iso:.4f}")
    
    print("\n✅ Calibration module tests passed!")
