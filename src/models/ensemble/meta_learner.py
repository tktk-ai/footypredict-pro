"""
Meta Learner
Learns to combine base model predictions optimally.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MetaLearner:
    """
    Meta-learner that learns optimal combination of base models.
    
    Features:
    - Learns from base model predictions
    - Calibrated probability outputs
    - Automatic weight learning
    """
    
    def __init__(
        self,
        meta_model: str = 'logistic',
        calibrate: bool = True
    ):
        self.meta_model_type = meta_model
        self.calibrate = calibrate
        self.meta_model = None
        self.base_model_names: List[str] = []
        self.is_fitted = False
    
    def fit(
        self,
        base_predictions: Dict[str, np.ndarray],
        targets: np.ndarray
    ) -> 'MetaLearner':
        """
        Fit meta-learner on base model predictions.
        
        Args:
            base_predictions: Dict of model_name -> predictions array
            targets: True labels
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using simple averaging")
            return self
        
        self.base_model_names = list(base_predictions.keys())
        
        # Stack predictions as features
        X = np.column_stack([base_predictions[name] for name in self.base_model_names])
        
        # Create meta-model
        if self.meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(max_iter=1000)
        elif self.meta_model_type == 'gbm':
            self.meta_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3
            )
        else:
            self.meta_model = LogisticRegression(max_iter=1000)
        
        self.meta_model.fit(X, targets)
        self.is_fitted = True
        
        logger.info(f"Meta-learner fitted with {len(self.base_model_names)} base models")
        
        return self
    
    def predict(
        self,
        base_predictions: Dict[str, Dict]
    ) -> Dict:
        """
        Make prediction using meta-learner.
        
        Args:
            base_predictions: Dict of model_name -> prediction_dict
        """
        if not self.is_fitted or self.meta_model is None:
            # Fall back to averaging
            return self._average_predictions(base_predictions)
        
        # Extract probabilities from each model
        features = []
        for name in self.base_model_names:
            if name in base_predictions and '1x2' in base_predictions[name]:
                probs = base_predictions[name]['1x2']
                features.extend([
                    probs.get('home', 0.33),
                    probs.get('draw', 0.33),
                    probs.get('away', 0.34)
                ])
            else:
                features.extend([0.33, 0.33, 0.34])
        
        X = np.array(features).reshape(1, -1)
        
        probs = self.meta_model.predict_proba(X)[0]
        
        return {
            '1x2': {
                'home': round(float(probs[0]), 4),
                'draw': round(float(probs[1]), 4) if len(probs) > 1 else 0.25,
                'away': round(float(probs[2]), 4) if len(probs) > 2 else 0.35
            },
            'method': 'meta_learner',
            'base_models': self.base_model_names
        }
    
    def _average_predictions(
        self,
        base_predictions: Dict[str, Dict]
    ) -> Dict:
        """Simple average fallback."""
        home = draw = away = 0
        count = 0
        
        for name, pred in base_predictions.items():
            if '1x2' in pred:
                home += pred['1x2'].get('home', 0)
                draw += pred['1x2'].get('draw', 0)
                away += pred['1x2'].get('away', 0)
                count += 1
        
        if count == 0:
            return {'1x2': {'home': 0.4, 'draw': 0.25, 'away': 0.35}}
        
        return {
            '1x2': {
                'home': round(home / count, 4),
                'draw': round(draw / count, 4),
                'away': round(away / count, 4)
            },
            'method': 'average_fallback'
        }
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get learned weights for base models."""
        if not self.is_fitted or self.meta_model is None:
            return {name: 1.0 for name in self.base_model_names}
        
        if hasattr(self.meta_model, 'coef_'):
            coefs = np.abs(self.meta_model.coef_).mean(axis=0)
            
            # Group by model (3 features per model)
            weights = {}
            for i, name in enumerate(self.base_model_names):
                start_idx = i * 3
                weights[name] = float(coefs[start_idx:start_idx + 3].mean())
            
            # Normalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            
            return weights
        
        return {name: 1.0 / len(self.base_model_names) for name in self.base_model_names}


_meta_learner: Optional[MetaLearner] = None

def get_meta_learner() -> MetaLearner:
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
    return _meta_learner
