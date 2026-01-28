"""
Stacking Ensemble Model
=======================
Implements stacking ensemble that combines multiple base models using a meta-learner.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble."""
    n_folds: int = 5
    use_features_in_meta: bool = True
    meta_model_type: str = 'logistic'  # 'logistic', 'ridge', 'rf', 'gbm'
    random_state: int = 42


class StackingEnsemble:
    """
    Stacking Ensemble that combines multiple base models.
    
    Uses out-of-fold predictions from base models as features
    for a meta-learner to make final predictions.
    """
    
    def __init__(self, config: StackingConfig = None):
        self.config = config or StackingConfig()
        self.base_models = {}
        self.meta_model = None
        self.is_fitted = False
        self.feature_names = []
        
    def add_base_model(self, name: str, model: Any) -> None:
        """Add a base model to the ensemble."""
        self.base_models[name] = {
            'model': model,
            'fitted': False
        }
        logger.info(f"Added base model: {name}")
    
    def _create_meta_model(self):
        """Create the meta-learner model."""
        if self.config.meta_model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state
            )
        elif self.config.meta_model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=self.config.random_state)
        elif self.config.meta_model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state
            )
        elif self.config.meta_model_type == 'gbm':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.config.random_state
            )
        else:
            return LogisticRegression(max_iter=1000)
    
    def _get_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Get out-of-fold predictions from all base models.
        
        Returns:
            Array of shape (n_samples, n_base_models * n_classes)
        """
        n_samples = X.shape[0]
        oof_predictions = []
        
        kfold = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            
            # Get OOF predictions
            if hasattr(model, 'predict_proba'):
                oof_pred = cross_val_predict(
                    model, X, y,
                    cv=kfold,
                    method='predict_proba'
                )
            else:
                oof_pred = cross_val_predict(
                    model, X, y,
                    cv=kfold
                ).reshape(-1, 1)
            
            oof_predictions.append(oof_pred)
            logger.info(f"Generated OOF predictions for {name}: shape {oof_pred.shape}")
        
        # Stack all OOF predictions
        stacked = np.hstack(oof_predictions)
        
        # Optionally include original features
        if self.config.use_features_in_meta:
            stacked = np.hstack([stacked, X])
        
        return stacked
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Training labels
            feature_names: Names of features
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models added. Use add_base_model() first.")
        
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        
        logger.info(f"Fitting stacking ensemble with {len(self.base_models)} base models")
        
        # Step 1: Get OOF predictions for meta-features
        meta_features = self._get_oof_predictions(X, y)
        logger.info(f"Meta features shape: {meta_features.shape}")
        
        # Step 2: Fit all base models on full training data
        for name, model_info in self.base_models.items():
            model_info['model'].fit(X, y)
            model_info['fitted'] = True
            logger.info(f"Fitted base model: {name}")
        
        # Step 3: Fit meta-model on OOF predictions
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        logger.info("Stacking ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        meta_features = self._get_meta_features(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # For regression-based meta models
            pred = self.meta_model.predict(meta_features)
            return np.column_stack([1 - pred, pred])
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base model predictions."""
        base_predictions = []
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            
            base_predictions.append(pred)
        
        stacked = np.hstack(base_predictions)
        
        if self.config.use_features_in_meta:
            stacked = np.hstack([stacked, X])
        
        return stacked
    
    def predict_match(
        self,
        home_features: Dict[str, float],
        away_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict match outcome.
        
        Args:
            home_features: Features for home team
            away_features: Features for away team
            
        Returns:
            Probabilities for home/draw/away
        """
        # Combine features
        features = []
        for fname in self.feature_names:
            if fname.startswith('home_'):
                key = fname.replace('home_', '')
                features.append(home_features.get(key, 0))
            elif fname.startswith('away_'):
                key = fname.replace('away_', '')
                features.append(away_features.get(key, 0))
            else:
                features.append(home_features.get(fname, away_features.get(fname, 0)))
        
        X = np.array(features).reshape(1, -1)
        probs = self.predict_proba(X)[0]
        
        if len(probs) == 3:
            return {
                'home_win': float(probs[0]),
                'draw': float(probs[1]),
                'away_win': float(probs[2])
            }
        else:
            return {
                'home_win': float(probs[1]),
                'not_home_win': float(probs[0])
            }
    
    def save(self, path: str) -> None:
        """Save the ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'config': self.config,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Saved stacking ensemble to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StackingEnsemble':
        """Load ensemble from disk."""
        save_data = joblib.load(path)
        
        ensemble = cls(config=save_data['config'])
        ensemble.base_models = save_data['base_models']
        ensemble.meta_model = save_data['meta_model']
        ensemble.feature_names = save_data['feature_names']
        ensemble.is_fitted = save_data['is_fitted']
        
        logger.info(f"Loaded stacking ensemble from {path}")
        return ensemble
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-model."""
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        if hasattr(self.meta_model, 'coef_'):
            coefs = np.abs(self.meta_model.coef_).mean(axis=0) if self.meta_model.coef_.ndim > 1 else np.abs(self.meta_model.coef_)
            
            # Base model contributions
            idx = 0
            for name in self.base_models.keys():
                importance[f"base_{name}"] = float(coefs[idx:idx+3].sum()) if idx+3 <= len(coefs) else float(coefs[idx])
                idx += 3
            
            # Original features if included
            if self.config.use_features_in_meta:
                for i, fname in enumerate(self.feature_names):
                    if idx + i < len(coefs):
                        importance[fname] = float(coefs[idx + i])
        
        elif hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
            
            idx = 0
            for name in self.base_models.keys():
                importance[f"base_{name}"] = float(importances[idx:idx+3].sum()) if idx+3 <= len(importances) else float(importances[idx])
                idx += 3
        
        return importance


# Global instance
_ensemble: Optional[StackingEnsemble] = None


def get_ensemble() -> StackingEnsemble:
    """Get or create stacking ensemble."""
    global _ensemble
    if _ensemble is None:
        _ensemble = StackingEnsemble()
    return _ensemble


def create_default_ensemble() -> StackingEnsemble:
    """Create ensemble with default base models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    ensemble = StackingEnsemble(StackingConfig(
        n_folds=5,
        use_features_in_meta=True,
        meta_model_type='logistic'
    ))
    
    # Add base models
    ensemble.add_base_model('logistic', LogisticRegression(max_iter=1000))
    ensemble.add_base_model('rf', RandomForestClassifier(n_estimators=100, max_depth=10))
    ensemble.add_base_model('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=5))
    ensemble.add_base_model('nb', GaussianNB())
    
    return ensemble
