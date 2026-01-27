"""
Boruta Feature Selector
Implements Boruta algorithm for feature selection.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BorutaSelector:
    """
    Boruta feature selection algorithm.
    
    Identifies all-relevant features by comparing original features
    against shadow (permuted) features.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 50,
        alpha: float = 0.05,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        
        self.selected_features: List[str] = []
        self.rejected_features: List[str] = []
        self.tentative_features: List[str] = []
        self.importance_history: List[dict] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BorutaSelector':
        """
        Fit the Boruta selector.
        
        Args:
            X: Feature DataFrame
            y: Target variable
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using simplified selection")
            return self._simplified_fit(X, y)
        
        feature_names = X.columns.tolist()
        n_features = len(feature_names)
        
        # Track hits (feature beats all shadows)
        hits = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # Create shadow features (permuted copies)
            X_shadow = X.apply(np.random.permutation)
            X_shadow.columns = [f'shadow_{c}' for c in X_shadow.columns]
            
            # Combine original and shadow
            X_combined = pd.concat([X, X_shadow], axis=1)
            
            # Train random forest
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state + iteration,
                n_jobs=-1
            )
            
            try:
                rf.fit(X_combined, y)
            except Exception as e:
                logger.warning(f"RF fit failed at iteration {iteration}: {e}")
                continue
            
            # Get importances
            importances = rf.feature_importances_
            original_imp = importances[:n_features]
            shadow_imp = importances[n_features:]
            shadow_max = shadow_imp.max()
            
            # Count hits
            hits += (original_imp > shadow_max).astype(int)
            
            self.importance_history.append({
                'iteration': iteration,
                'importances': dict(zip(feature_names, original_imp)),
                'shadow_max': shadow_max
            })
        
        # Determine selected features using binomial test
        from scipy import stats
        
        for i, feature in enumerate(feature_names):
            p_value = stats.binom_test(
                int(hits[i]),
                self.max_iter,
                0.5,
                alternative='greater'
            ) if hasattr(stats, 'binom_test') else 0.5  # Fallback
            
            if p_value < self.alpha:
                self.selected_features.append(feature)
            elif p_value < 0.5:
                self.tentative_features.append(feature)
            else:
                self.rejected_features.append(feature)
        
        logger.info(f"Selected {len(self.selected_features)} features, "
                   f"rejected {len(self.rejected_features)}")
        
        return self
    
    def _simplified_fit(self, X: pd.DataFrame, y: pd.Series) -> 'BorutaSelector':
        """Simplified selection without full Boruta."""
        # Use correlation-based selection
        correlations = X.corrwith(y).abs()
        threshold = correlations.median()
        
        self.selected_features = correlations[correlations >= threshold].index.tolist()
        self.rejected_features = correlations[correlations < threshold].index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only the important features."""
        available_features = [f for f in self.selected_features if f in X.columns]
        return X[available_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self) -> dict:
        """Get feature selection support information."""
        return {
            'selected': self.selected_features,
            'rejected': self.rejected_features,
            'tentative': self.tentative_features,
            'n_selected': len(self.selected_features),
            'n_rejected': len(self.rejected_features)
        }
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """Get ranking of features by importance."""
        if not self.importance_history:
            return pd.DataFrame()
        
        # Average importance across iterations
        avg_importance = {}
        for hist in self.importance_history:
            for feature, imp in hist['importances'].items():
                if feature not in avg_importance:
                    avg_importance[feature] = []
                avg_importance[feature].append(imp)
        
        ranking = pd.DataFrame([
            {'feature': f, 'avg_importance': np.mean(imps), 'std_importance': np.std(imps)}
            for f, imps in avg_importance.items()
        ])
        
        return ranking.sort_values('avg_importance', ascending=False)


# Global instance
_selector: Optional[BorutaSelector] = None


def get_selector() -> BorutaSelector:
    """Get or create Boruta selector."""
    global _selector
    if _selector is None:
        _selector = BorutaSelector()
    return _selector


def select_features(X: pd.DataFrame, y: pd.Series) -> List[str]:
    """Quick function to select features."""
    selector = BorutaSelector()
    selector.fit(X, y)
    return selector.selected_features
