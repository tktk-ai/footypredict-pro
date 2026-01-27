"""
LIME Explainer Module
Local interpretable model explanations.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class LIMEExplainer:
    """
    LIME-based model explanation.
    
    Features:
    - Local explanations
    - Model-agnostic
    - Feature contributions
    """
    
    def __init__(self):
        self.explainer = None
        self.feature_names = []
        self.class_names = ['H', 'D', 'A']
    
    def fit(
        self,
        X_train: np.ndarray,
        feature_names: List[str] = None,
        class_names: List[str] = None,
        mode: str = 'classification'
    ) -> 'LIMEExplainer':
        """
        Fit LIME explainer.
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or self.class_names
        self.mode = mode
        
        if not LIME_AVAILABLE:
            logger.warning("LIME not available")
            return self
        
        try:
            self.explainer = LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=mode
            )
            logger.info("LIME explainer fitted successfully")
        except Exception as e:
            logger.warning(f"Could not create LIME explainer: {e}")
        
        return self
    
    def explain_prediction(
        self,
        X: np.ndarray,
        predict_fn: callable,
        num_features: int = 10
    ) -> Dict:
        """
        Explain a single prediction.
        """
        if self.explainer is None or not LIME_AVAILABLE:
            return self._fallback_explanation()
        
        try:
            if len(X.shape) > 1:
                X = X[0]
            
            explanation = self.explainer.explain_instance(
                X,
                predict_fn,
                num_features=num_features
            )
            
            # Extract features
            feature_weights = explanation.as_list()
            
            return {
                'top_features': [
                    {'feature': f, 'contribution': round(w, 4)}
                    for f, w in feature_weights
                ],
                'positive_contributors': [
                    {'feature': f, 'contribution': round(w, 4)}
                    for f, w in feature_weights if w > 0
                ],
                'negative_contributors': [
                    {'feature': f, 'contribution': round(w, 4)}
                    for f, w in feature_weights if w < 0
                ],
                'local_prediction': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None
            }
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return self._fallback_explanation()
    
    def _fallback_explanation(self) -> Dict:
        """Fallback when LIME unavailable."""
        return {
            'method': 'unavailable',
            'top_features': [],
            'message': 'LIME not available or explanation failed'
        }
    
    def generate_html_explanation(
        self,
        X: np.ndarray,
        predict_fn: callable
    ) -> str:
        """Generate HTML explanation."""
        if self.explainer is None or not LIME_AVAILABLE:
            return "<p>LIME not available</p>"
        
        try:
            explanation = self.explainer.explain_instance(X[0] if len(X.shape) > 1 else X, predict_fn)
            return explanation.as_html()
        except Exception:
            return "<p>Explanation failed</p>"


_explainer: Optional[LIMEExplainer] = None

def get_explainer() -> LIMEExplainer:
    global _explainer
    if _explainer is None:
        _explainer = LIMEExplainer()
    return _explainer
