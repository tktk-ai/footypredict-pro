"""
XGBoost Model Wrapper
Standardized wrapper for XGBoost football prediction.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBoostModel:
    """
    XGBoost model wrapper for football predictions.
    
    Supports:
    - Multi-output (result, goals, BTTS, O/U)
    - Probability calibration
    - Feature importance
    """
    
    DEFAULT_PARAMS = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 200,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    def __init__(
        self,
        params: Dict = None,
        model_dir: str = "models/saved_models/xgboost"
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
        eval_set: tuple = None
    ) -> 'XGBoostModel':
        """
        Fit all prediction models.
        
        Args:
            X: Features
            y_result: Result labels (0=H, 1=D, 2=A)
            y_home_goals: Home goals
            y_away_goals: Away goals  
            y_btts: BTTS labels
        """
        if not XGB_AVAILABLE:
            logger.error("XGBoost not installed")
            return self
        
        self.feature_names = X.columns.tolist()
        
        # Result model (1X2)
        logger.info("Training result model...")
        self.result_model = xgb.XGBClassifier(**self.params)
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False
        
        self.result_model.fit(X, y_result, **fit_params)
        
        # Goals models
        if y_home_goals is not None:
            logger.info("Training home goals model...")
            goals_params = {**self.params}
            goals_params['objective'] = 'reg:squarederror'
            del goals_params['num_class']
            
            self.goals_model = {
                'home': xgb.XGBRegressor(**goals_params),
                'away': xgb.XGBRegressor(**goals_params)
            }
            self.goals_model['home'].fit(X, y_home_goals)
            
            if y_away_goals is not None:
                self.goals_model['away'].fit(X, y_away_goals)
        
        # BTTS model
        if y_btts is not None:
            logger.info("Training BTTS model...")
            btts_params = {**self.params}
            btts_params['objective'] = 'binary:logistic'
            del btts_params['num_class']
            
            self.btts_model = xgb.XGBClassifier(**btts_params)
            self.btts_model.fit(X, y_btts)
        
        self.is_fitted = True
        logger.info("XGBoost training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Make predictions for all markets."""
        if not self.is_fitted:
            logger.warning("Model not fitted")
            return {}
        
        predictions = {}
        
        # Result probabilities
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
        """Get feature importance from result model."""
        if not self.result_model:
            return pd.DataFrame()
        
        importance = self.result_model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    def save(self, name: str = "xgboost_model"):
        """Save models to disk."""
        if self.result_model:
            joblib.dump(self.result_model, self.model_dir / f"{name}_result.joblib")
        if self.goals_model:
            joblib.dump(self.goals_model, self.model_dir / f"{name}_goals.joblib")
        if self.btts_model:
            joblib.dump(self.btts_model, self.model_dir / f"{name}_btts.joblib")
        
        logger.info(f"Saved XGBoost models to {self.model_dir}")
    
    def load(self, name: str = "xgboost_model") -> bool:
        """Load models from disk."""
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
            logger.info(f"Loaded XGBoost models from {self.model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False


# Global instance
_model: Optional[XGBoostModel] = None


def get_model() -> XGBoostModel:
    """Get or create XGBoost model."""
    global _model
    if _model is None:
        _model = XGBoostModel()
    return _model
