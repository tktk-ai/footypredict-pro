"""
ML Models Package

Provides pre-trained and custom ML models for match prediction.

Usage:
    from src.models import predict, get_registry, list_models
    
    # Quick prediction
    pred = predict('Bayern', 'Dortmund')
    print(pred.home_win_prob, pred.confidence)
    
    # Full registry access
    registry = get_registry()
    health = registry.health_check()
"""

from .model_registry import (
    ModelRegistry,
    get_registry,
    predict
)

from .ensemble import (
    ModelEnsemble,
    EnsemblePrediction
)

from .pretrained_loader import (
    PretrainedModelLoader,
    get_loader,
    download_all,
    get_model,
    list_models
)

from .mock_models import (
    MockPodosPredictor,
    MockFootballerPredictor,
    MockXGBoostPredictor,
    MockPrediction,
    create_mock_predictor
)

# SportyBet specialized models
from .sportybet_predictor import (
    SportyBetPredictor,
    SportyBetPrediction,
    SportyBetMultiPrediction,
    get_sportybet_predictor,
    sportybet_predict,
    get_available_sportybet_markets
)

# Advanced Models Integration (XGBoost + LightGBM)
from .advanced_integration import (
    AdvancedModelsPredictor,
    AdvancedPrediction,
    get_advanced_predictor,
    advanced_predict
)

__all__ = [
    # Registry
    'ModelRegistry',
    'get_registry',
    'predict',
    
    # Ensemble
    'ModelEnsemble',
    'EnsemblePrediction',
    
    # Loader
    'PretrainedModelLoader',
    'get_loader',
    'download_all',
    'get_model',
    'list_models',
    
    # Mock
    'MockPodosPredictor',
    'MockFootballerPredictor',
    'MockXGBoostPredictor',
    'MockPrediction',
    'create_mock_predictor',
    
    # SportyBet
    'SportyBetPredictor',
    'SportyBetPrediction',
    'SportyBetMultiPrediction',
    'get_sportybet_predictor',
    'sportybet_predict',
    'get_available_sportybet_markets',
    
    # Advanced Models
    'AdvancedModelsPredictor',
    'AdvancedPrediction',
    'get_advanced_predictor',
    'advanced_predict'
]

