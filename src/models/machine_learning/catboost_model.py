"""
CatBoost Model Wrapper
Standardized wrapper for CatBoost football prediction.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False


class CatBoostModel:
    """
    CatBoost model wrapper for football predictions.
    
    Supports:
    - Native categorical feature handling
    - GPU training
    - Ordered boosting
    """
    
    DEFAULT_PARAMS = {
        'learning_rate': 0.05,
        'depth': 6,
        'iterations': 200,
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'MultiClass'
    }
    
    def __init__(
        self,
        params: Dict = None,
        model_dir: str = "models/saved_models/catboost"
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
        cat_features: List[str] = None
    ) -> 'CatBoostModel':
        """Fit all prediction models."""
        if not CAT_AVAILABLE:
            logger.error("CatBoost not installed")
            return self
        
        self.feature_names = X.columns.tolist()
        cat_idx = [X.columns.get_loc(c) for c in (cat_features or []) if c in X.columns]
        
        # Result model
        logger.info("Training CatBoost result model...")
        self.result_model = CatBoostClassifier(**self.params)
        self.result_model.fit(X, y_result, cat_features=cat_idx if cat_idx else None)
        
        # Goals models
        if y_home_goals is not None:
            logger.info("Training CatBoost goals model...")
            goals_params = {**self.params}
            goals_params['loss_function'] = 'RMSE'
            
            self.goals_model = {
                'home': CatBoostRegressor(**goals_params),
                'away': CatBoostRegressor(**goals_params)
            }
            self.goals_model['home'].fit(X, y_home_goals, cat_features=cat_idx if cat_idx else None)
            
            if y_away_goals is not None:
                self.goals_model['away'].fit(X, y_away_goals, cat_features=cat_idx if cat_idx else None)
        
        # BTTS model
        if y_btts is not None:
            logger.info("Training CatBoost BTTS model...")
            btts_params = {**self.params}
            btts_params['loss_function'] = 'Logloss'
            
            self.btts_model = CatBoostClassifier(**btts_params)
            self.btts_model.fit(X, y_btts, cat_features=cat_idx if cat_idx else None)
        
        self.is_fitted = True
        logger.info("CatBoost training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Make predictions."""
        if not self.is_fitted:
            return {}
        
        predictions = {}
        
        if self.result_model:
            probs = self.result_model.predict_proba(X)
            predictions['1x2'] = {
                'home': float(probs[0, 0]),
                'draw': float(probs[0, 1]),
                'away': float(probs[0, 2])
            }
            predictions['result'] = ['H', 'D', 'A'][np.argmax(probs[0])]
        
        if self.goals_model:
            predictions['home_goals'] = float(self.goals_model['home'].predict(X)[0])
            predictions['away_goals'] = float(self.goals_model['away'].predict(X)[0])
            
            total = predictions['home_goals'] + predictions['away_goals']
            predictions['over_2.5'] = 1 / (1 + np.exp(-(total - 2.5)))
        
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
    
    def save(self, name: str = "catboost_model"):
        """Save models."""
        if self.result_model:
            self.result_model.save_model(str(self.model_dir / f"{name}_result.cbm"))
        if self.goals_model:
            self.goals_model['home'].save_model(str(self.model_dir / f"{name}_home_goals.cbm"))
            self.goals_model['away'].save_model(str(self.model_dir / f"{name}_away_goals.cbm"))
        if self.btts_model:
            self.btts_model.save_model(str(self.model_dir / f"{name}_btts.cbm"))
    
    def load(self, name: str = "catboost_model") -> bool:
        """Load models."""
        try:
            result_path = self.model_dir / f"{name}_result.cbm"
            if result_path.exists():
                self.result_model = CatBoostClassifier()
                self.result_model.load_model(str(result_path))
            
            home_path = self.model_dir / f"{name}_home_goals.cbm"
            away_path = self.model_dir / f"{name}_away_goals.cbm"
            if home_path.exists():
                self.goals_model = {
                    'home': CatBoostRegressor(),
                    'away': CatBoostRegressor()
                }
                self.goals_model['home'].load_model(str(home_path))
                if away_path.exists():
                    self.goals_model['away'].load_model(str(away_path))
            
            btts_path = self.model_dir / f"{name}_btts.cbm"
            if btts_path.exists():
                self.btts_model = CatBoostClassifier()
                self.btts_model.load_model(str(btts_path))
            
            self.is_fitted = self.result_model is not None
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False


_model: Optional[CatBoostModel] = None

def get_model() -> CatBoostModel:
    global _model
    if _model is None:
        _model = CatBoostModel()
    return _model
