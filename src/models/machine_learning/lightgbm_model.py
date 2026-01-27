"""
LightGBM Model Wrapper
Standardized wrapper for LightGBM football prediction.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class LightGBMModel:
    """
    LightGBM model wrapper for football predictions.
    
    Supports:
    - Multi-output predictions
    - Fast training
    - Categorical features
    """
    
    DEFAULT_PARAMS = {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'n_estimators': 200,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    def __init__(
        self,
        params: Dict = None,
        model_dir: str = "models/saved_models/lightgbm"
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.result_model = None
        self.goals_model = None
        self.btts_model = None
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y_result: pd.Series,
        y_home_goals: pd.Series = None,
        y_away_goals: pd.Series = None,
        y_btts: pd.Series = None,
        categorical_features: List[str] = None
    ) -> 'LightGBMModel':
        """Fit all prediction models."""
        if not LGB_AVAILABLE:
            logger.error("LightGBM not installed")
            return self
        
        self.feature_names = X.columns.tolist()
        
        fit_params = {}
        if categorical_features:
            fit_params['categorical_feature'] = categorical_features
        
        # Result model
        logger.info("Training LightGBM result model...")
        self.result_model = lgb.LGBMClassifier(**self.params)
        self.result_model.fit(X, y_result, **fit_params)
        
        # Goals models
        if y_home_goals is not None:
            logger.info("Training LightGBM goals model...")
            goals_params = {**self.params}
            goals_params['objective'] = 'regression'
            del goals_params['num_class']
            
            self.goals_model = {
                'home': lgb.LGBMRegressor(**goals_params),
                'away': lgb.LGBMRegressor(**goals_params)
            }
            self.goals_model['home'].fit(X, y_home_goals)
            
            if y_away_goals is not None:
                self.goals_model['away'].fit(X, y_away_goals)
        
        # BTTS model
        if y_btts is not None:
            logger.info("Training LightGBM BTTS model...")
            btts_params = {**self.params}
            btts_params['objective'] = 'binary'
            del btts_params['num_class']
            
            self.btts_model = lgb.LGBMClassifier(**btts_params)
            self.btts_model.fit(X, y_btts)
        
        self.is_fitted = True
        logger.info("LightGBM training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Make predictions for all markets."""
        if not self.is_fitted:
            return {}
        
        predictions = {}
        
        # Result
        if self.result_model:
            probs = self.result_model.predict_proba(X)
            predictions['1x2'] = {
                'home': float(probs[0, 0]),
                'draw': float(probs[0, 1]),
                'away': float(probs[0, 2])
            }
            predictions['result'] = ['H', 'D', 'A'][np.argmax(probs[0])]
        
        # Goals
        if self.goals_model:
            predictions['home_goals'] = float(self.goals_model['home'].predict(X)[0])
            predictions['away_goals'] = float(self.goals_model['away'].predict(X)[0])
            
            total = predictions['home_goals'] + predictions['away_goals']
            predictions['over_2.5'] = 1 / (1 + np.exp(-(total - 2.5)))
        
        # BTTS
        if self.btts_model:
            btts_prob = self.btts_model.predict_proba(X)
            predictions['btts'] = float(btts_prob[0, 1]) if btts_prob.shape[1] > 1 else 0.5
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if not self.result_model:
            return pd.DataFrame()
        
        importance = self.result_model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
    
    def save(self, name: str = "lightgbm_model"):
        """Save models."""
        if self.result_model:
            joblib.dump(self.result_model, self.model_dir / f"{name}_result.joblib")
        if self.goals_model:
            joblib.dump(self.goals_model, self.model_dir / f"{name}_goals.joblib")
        if self.btts_model:
            joblib.dump(self.btts_model, self.model_dir / f"{name}_btts.joblib")
    
    def load(self, name: str = "lightgbm_model") -> bool:
        """Load models."""
        try:
            result_path = self.model_dir / f"{name}_result.joblib"
            if result_path.exists():
                self.result_model = joblib.load(result_path)
            
            goals_path = self.model_dir / f"{name}_goals.joblib"
            if goals_path.exists():
                self.goals_model = joblib.load(goals_path)
            
            btts_path = self.model_dir / f"{name}_btts.joblib"
            if btts_path.exists():
                self.btts_model = joblib.load(btts_path)
            
            self.is_fitted = self.result_model is not None
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False


_model: Optional[LightGBMModel] = None

def get_model() -> LightGBMModel:
    global _model
    if _model is None:
        _model = LightGBMModel()
    return _model
