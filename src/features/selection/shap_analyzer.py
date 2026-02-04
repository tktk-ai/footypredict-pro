"""
SHAP Feature Analyzer
Uses SHAP values for feature analysis and selection.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPAnalyzer:
    """
    SHAP-based feature analysis and selection.
    
    Provides:
    - Global feature importance
    - Feature interaction analysis
    - Selection based on SHAP values
    """
    
    def __init__(self, model=None):
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.feature_names: List[str] = []
        self.importance_df: Optional[pd.DataFrame] = None
    
    def set_model(self, model):
        """Set the model to analyze."""
        self.model = model
        self._create_explainer()
    
    def _create_explainer(self):
        """Create SHAP explainer based on model type."""
        if not SHAP_AVAILABLE or self.model is None:
            return
        
        try:
            # Try TreeExplainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
        except Exception:
            try:
                # Fall back to KernelExplainer
                self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(np.zeros((1, 100))))
                logger.info("Using KernelExplainer")
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
    
    def analyze(
        self,
        X: pd.DataFrame,
        sample_size: int = 500
    ) -> Dict:
        """
        Analyze feature importance using SHAP.
        
        Args:
            X: Feature DataFrame
            sample_size: Number of samples to analyze
        """
        self.feature_names = X.columns.tolist()
        
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_analysis(X)
        
        # Sample if needed
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        try:
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class (use first class or average)
            if isinstance(self.shap_values, list):
                self.shap_values = np.mean(np.abs(self.shap_values), axis=0)
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(self.shap_values).mean(axis=0)
            
            self.importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            
            return {
                'top_features': self.get_top_features(20),
                'feature_importance': self.importance_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return self._fallback_analysis(X)
    
    def _fallback_analysis(self, X: pd.DataFrame) -> Dict:
        """Fallback when SHAP is unavailable."""
        # Use variance as a simple importance proxy
        variances = X.var()
        
        self.importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': variances.values
        }).sort_values('importance', ascending=False)
        
        return {
            'top_features': self.get_top_features(20),
            'method': 'variance_fallback'
        }
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features."""
        if self.importance_df is None:
            return []
        return self.importance_df.head(n)['feature'].tolist()
    
    def select_features(
        self,
        threshold: float = None,
        top_n: int = None
    ) -> List[str]:
        """
        Select features based on SHAP importance.
        
        Args:
            threshold: Minimum importance threshold
            top_n: Number of top features to select
        """
        if self.importance_df is None:
            return []
        
        if top_n:
            return self.importance_df.head(top_n)['feature'].tolist()
        
        if threshold:
            selected = self.importance_df[
                self.importance_df['importance'] >= threshold
            ]
            return selected['feature'].tolist()
        
        # Default: features with above-average importance
        mean_imp = self.importance_df['importance'].mean()
        selected = self.importance_df[
            self.importance_df['importance'] >= mean_imp
        ]
        return selected['feature'].tolist()
    
    def get_feature_interactions(
        self,
        X: pd.DataFrame,
        top_n: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Get top feature interactions."""
        if not SHAP_AVAILABLE or self.shap_values is None:
            return []
        
        try:
            # Calculate SHAP interaction values
            interaction_values = self.explainer.shap_interaction_values(X.head(100))
            
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0]
            
            # Average across samples
            mean_interactions = np.abs(interaction_values).mean(axis=0)
            
            # Extract top interactions (excluding diagonal)
            interactions = []
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    interactions.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        mean_interactions[i, j]
                    ))
            
            interactions.sort(key=lambda x: x[2], reverse=True)
            return interactions[:top_n]
            
        except Exception as e:
            logger.warning(f"Interaction analysis failed: {e}")
            return []
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame
    ) -> Dict:
        """Explain a single prediction."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {}
        
        try:
            shap_values = self.explainer.shap_values(X_single)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            explanation = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_values[0]):
                    explanation[feature] = float(shap_values[0][i])
            
            # Sort by absolute value
            sorted_exp = sorted(
                explanation.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            return {
                'contributions': dict(sorted_exp[:10]),
                'positive_factors': [(k, v) for k, v in sorted_exp if v > 0][:5],
                'negative_factors': [(k, v) for k, v in sorted_exp if v < 0][:5]
            }
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {e}")
            return {}


# Global instance
_analyzer: Optional[SHAPAnalyzer] = None


def get_analyzer(model=None) -> SHAPAnalyzer:
    """Get or create SHAP analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SHAPAnalyzer()
    if model:
        _analyzer.set_model(model)
    return _analyzer


def analyze_features(X: pd.DataFrame, model) -> Dict:
    """Quick function to analyze features."""
    analyzer = SHAPAnalyzer(model)
    return analyzer.analyze(X)
