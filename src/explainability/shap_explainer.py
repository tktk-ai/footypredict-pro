"""
SHAP and LIME Explainability Module
Provides interpretable AI explanations for predictions.

Based on the blueprint for model explainability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for explainability libraries
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    logger.warning("LIME not installed. Install with: pip install lime")


class SHAPExplainer:
    """
    SHAP-based model explainability.
    Provides feature importance and prediction explanations.
    """
    
    def __init__(self, model: Any, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.background_data = None
        
    def fit(self, background_data: np.ndarray, sample_size: int = 100):
        """Initialize explainer with background data."""
        if not HAS_SHAP:
            logger.warning("SHAP not available")
            return
        
        # Sample background data
        if len(background_data) > sample_size:
            indices = np.random.choice(len(background_data), sample_size, replace=False)
            self.background_data = background_data[indices]
        else:
            self.background_data = background_data
        
        # Create explainer
        try:
            # Try TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
        except Exception:
            try:
                # Fall back to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    self.background_data
                )
                logger.info("Using KernelExplainer")
            except Exception as e:
                logger.error(f"Failed to create SHAP explainer: {e}")
    
    def explain_prediction(
        self,
        features: np.ndarray,
        class_index: int = None
    ) -> Dict:
        """
        Explain a single prediction.
        
        Returns:
            Dictionary with feature importances and base value
        """
        if self.explainer is None:
            return self._fallback_explanation(features)
        
        try:
            shap_values = self.explainer.shap_values(features.reshape(1, -1))
            
            # Handle multi-class
            if isinstance(shap_values, list):
                if class_index is not None:
                    values = shap_values[class_index][0]
                else:
                    values = shap_values[1][0]  # Default to positive class
            else:
                values = shap_values[0]
            
            # Get feature importance ranking
            importance = np.abs(values)
            sorted_idx = np.argsort(importance)[::-1]
            
            top_features = []
            for idx in sorted_idx[:10]:  # Top 10 features
                if self.feature_names and idx < len(self.feature_names):
                    name = self.feature_names[idx]
                else:
                    name = f"Feature_{idx}"
                
                top_features.append({
                    'feature': name,
                    'importance': float(importance[idx]),
                    'contribution': float(values[idx]),
                    'direction': 'positive' if values[idx] > 0 else 'negative'
                })
            
            return {
                'top_features': top_features,
                'base_value': float(self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value),
                'total_contribution': float(np.sum(values))
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(features)
    
    def get_global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Get global feature importance across dataset."""
        if self.explainer is None:
            return pd.DataFrame()
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                values = shap_values[1]  # Positive class
            else:
                values = shap_values
            
            importance = np.abs(values).mean(axis=0)
            
            df = pd.DataFrame({
                'feature': self.feature_names if self.feature_names else [f'Feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
            
            return df.sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Global importance failed: {e}")
            return pd.DataFrame()
    
    def _fallback_explanation(self, features: np.ndarray) -> Dict:
        """Simple fallback when SHAP unavailable."""
        # Use feature magnitudes as proxy
        importance = np.abs(features)
        sorted_idx = np.argsort(importance)[::-1]
        
        top_features = []
        for idx in sorted_idx[:10]:
            if self.feature_names and idx < len(self.feature_names):
                name = self.feature_names[idx]
            else:
                name = f"Feature_{idx}"
            
            top_features.append({
                'feature': name,
                'importance': float(importance[idx]),
                'contribution': float(features[idx]),
                'direction': 'positive' if features[idx] > 0 else 'negative'
            })
        
        return {
            'top_features': top_features,
            'base_value': 0.0,
            'total_contribution': 0.0,
            'note': 'Fallback explanation (SHAP unavailable)'
        }


class LIMEExplainer:
    """
    LIME-based model explainability.
    Provides local interpretable model-agnostic explanations.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str] = None,
        class_names: List[str] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Away', 'Draw', 'Home']
        self.explainer = None
        
    def fit(self, training_data: np.ndarray):
        """Initialize LIME explainer with training data."""
        if not HAS_LIME:
            logger.warning("LIME not available")
            return
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        logger.info("LIME explainer initialized")
    
    def explain_prediction(
        self,
        features: np.ndarray,
        num_features: int = 10
    ) -> Dict:
        """
        Explain a single prediction using LIME.
        """
        if self.explainer is None:
            return self._fallback_explanation(features)
        
        try:
            # Get prediction function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = lambda x: self.model.predict(x)
            
            explanation = self.explainer.explain_instance(
                features,
                predict_fn,
                num_features=num_features
            )
            
            # Extract feature contributions
            feature_weights = explanation.as_list()
            
            top_features = []
            for feature_desc, weight in feature_weights:
                top_features.append({
                    'feature': feature_desc,
                    'importance': abs(weight),
                    'contribution': weight,
                    'direction': 'positive' if weight > 0 else 'negative'
                })
            
            return {
                'top_features': top_features,
                'local_prediction': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None,
                'score': explanation.score
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return self._fallback_explanation(features)
    
    def _fallback_explanation(self, features: np.ndarray) -> Dict:
        """Simple fallback when LIME unavailable."""
        importance = np.abs(features)
        sorted_idx = np.argsort(importance)[::-1]
        
        top_features = []
        for idx in sorted_idx[:10]:
            if self.feature_names and idx < len(self.feature_names):
                name = self.feature_names[idx]
            else:
                name = f"Feature_{idx}"
            
            top_features.append({
                'feature': name,
                'importance': float(importance[idx]),
                'contribution': float(features[idx]),
                'direction': 'positive' if features[idx] > 0 else 'negative'
            })
        
        return {
            'top_features': top_features,
            'note': 'Fallback explanation (LIME unavailable)'
        }


class PredictionExplainer:
    """
    Combined explainability system using SHAP and LIME.
    Provides comprehensive prediction explanations.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str] = None,
        use_shap: bool = True,
        use_lime: bool = True
    ):
        self.model = model
        self.feature_names = feature_names
        
        self.shap_explainer = None
        self.lime_explainer = None
        
        if use_shap and HAS_SHAP:
            self.shap_explainer = SHAPExplainer(model, feature_names)
        
        if use_lime and HAS_LIME:
            self.lime_explainer = LIMEExplainer(model, feature_names)
    
    def fit(self, training_data: np.ndarray):
        """Initialize all explainers."""
        if self.shap_explainer:
            self.shap_explainer.fit(training_data)
        
        if self.lime_explainer:
            self.lime_explainer.fit(training_data)
    
    def explain(
        self,
        features: np.ndarray,
        prediction: Dict = None
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction.
        """
        result = {
            'prediction': prediction,
            'feature_values': {}
        }
        
        # Add feature values
        if self.feature_names:
            for i, name in enumerate(self.feature_names[:20]):  # Top 20
                if i < len(features):
                    result['feature_values'][name] = float(features[i])
        
        # SHAP explanation
        if self.shap_explainer:
            result['shap'] = self.shap_explainer.explain_prediction(features)
        
        # LIME explanation
        if self.lime_explainer:
            result['lime'] = self.lime_explainer.explain_prediction(features)
        
        # Generate human-readable summary
        result['summary'] = self._generate_summary(result)
        
        return result
    
    def _generate_summary(self, explanation: Dict) -> str:
        """Generate human-readable summary."""
        summary_parts = []
        
        if 'shap' in explanation and explanation['shap'].get('top_features'):
            top_3 = explanation['shap']['top_features'][:3]
            
            positive_factors = [f['feature'] for f in top_3 if f['direction'] == 'positive']
            negative_factors = [f['feature'] for f in top_3 if f['direction'] == 'negative']
            
            if positive_factors:
                summary_parts.append(f"Key positive factors: {', '.join(positive_factors)}")
            if negative_factors:
                summary_parts.append(f"Key negative factors: {', '.join(negative_factors)}")
        
        return '. '.join(summary_parts) if summary_parts else "No explanation available"


# Global instance
_explainer: Optional[PredictionExplainer] = None


def get_explainer(model: Any = None, feature_names: List[str] = None) -> PredictionExplainer:
    """Get or create prediction explainer."""
    global _explainer
    
    if _explainer is None and model is not None:
        _explainer = PredictionExplainer(model, feature_names)
    
    return _explainer


def explain_prediction(features: np.ndarray, prediction: Dict = None, model: Any = None) -> Dict:
    """Quick function to explain a prediction."""
    explainer = get_explainer(model)
    
    if explainer:
        return explainer.explain(features, prediction)
    
    return {'error': 'No explainer available'}
