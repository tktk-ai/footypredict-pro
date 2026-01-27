"""Machine Learning Models Package."""

from .xgboost_model import XGBoostModel, get_model as get_xgboost
from .lightgbm_model import LightGBMModel, get_model as get_lightgbm
from .catboost_model import CatBoostModel, get_model as get_catboost

# Import existing stacking if available
try:
    from src.models.stacking_ensemble import StackingEnsemble
except ImportError:
    StackingEnsemble = None

__all__ = [
    'XGBoostModel', 'get_xgboost',
    'LightGBMModel', 'get_lightgbm',
    'CatBoostModel', 'get_catboost',
    'StackingEnsemble'
]
