"""
Model Registry - Central management for all prediction models

Handles:
- Loading/unloading models
- Model health checks
- Prediction routing
- Fallback strategies
"""

import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict

from .pretrained_loader import PretrainedModelLoader, get_loader
from .mock_models import create_mock_predictor
from .ensemble import ModelEnsemble, EnsemblePrediction

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Status of a registered model"""
    name: str
    loaded: bool
    type: str  # 'pretrained', 'trained', 'mock'
    healthy: bool
    last_prediction_time: Optional[float] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """
    Central registry for all prediction models.
    
    Manages model lifecycle and provides unified prediction interface.
    
    Usage:
        registry = ModelRegistry()
        registry.initialize()
        prediction = registry.predict('Bayern', 'Dortmund')
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.ensemble = ModelEnsemble()
        self.loader = get_loader()
        self._initialized = False
    
    def initialize(self, use_pretrained: bool = True, 
                   use_mock_fallback: bool = True) -> bool:
        """
        Initialize all models.
        
        Args:
            use_pretrained: Try to load HuggingFace models
            use_mock_fallback: Use mock models if pretrained unavailable
            
        Returns:
            True if at least one model loaded successfully
        """
        logger.info("Initializing model registry...")
        
        # Try to load pretrained models
        if use_pretrained:
            self._load_pretrained_models()
        
        # Fill in with mock models if needed
        if use_mock_fallback:
            self._load_mock_fallbacks()
        
        # Set up ensemble with loaded models
        for name, model in self.models.items():
            self.ensemble.register_model(name, model)
        
        self._initialized = len(self.models) > 0
        
        if self._initialized:
            logger.info(f"Registry initialized with {len(self.models)} models")
        else:
            logger.error("No models loaded!")
        
        return self._initialized
    
    def _load_pretrained_models(self):
        """Load pretrained models from HuggingFace"""
        model_names = ['podos', 'footballer']
        
        for name in model_names:
            try:
                model = self.loader.get_model(name)
                if model:
                    self.models[name] = model
                    self.model_status[name] = ModelStatus(
                        name=name,
                        loaded=True,
                        type='pretrained',
                        healthy=True
                    )
                    logger.info(f"Loaded pretrained model: {name}")
            except Exception as e:
                logger.warning(f"Could not load pretrained {name}: {e}")
    
    def _load_mock_fallbacks(self):
        """Load mock models for missing predictors"""
        required_models = ['podos', 'footballer', 'xgboost']
        
        for name in required_models:
            if name not in self.models:
                try:
                    mock = create_mock_predictor(name)
                    self.models[name] = mock
                    self.model_status[name] = ModelStatus(
                        name=name,
                        loaded=True,
                        type='mock',
                        healthy=True
                    )
                    logger.info(f"Loaded mock model: {name}")
                except Exception as e:
                    logger.error(f"Could not create mock {name}: {e}")
    
    def load_trained_model(self, name: str, path: Path) -> bool:
        """
        Load a custom trained model from file.
        
        Args:
            name: Model identifier
            path: Path to model file
            
        Returns:
            True if successful
        """
        if not path.exists():
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            import pickle
            
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            self.models[name] = model
            self.model_status[name] = ModelStatus(
                name=name,
                loaded=True,
                type='trained',
                healthy=True
            )
            
            # Update ensemble
            self.ensemble.register_model(name, model)
            
            logger.info(f"Loaded trained model: {name} from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {name}: {e}")
            return False
    
    def predict(self, home_team: str, away_team: str, 
                use_ensemble: bool = True, **features) -> EnsemblePrediction:
        """
        Get prediction using registered models.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            use_ensemble: Use ensemble (True) or single best model (False)
            **features: Additional prediction features
            
        Returns:
            EnsemblePrediction with probabilities and metadata
        """
        if not self._initialized:
            self.initialize()
        
        if not self.models:
            raise RuntimeError("No models available for prediction")
        
        try:
            if use_ensemble:
                return self.ensemble.predict(home_team, away_team, **features)
            else:
                # Use single best model (podos preferred)
                best_model = self.models.get('podos') or next(iter(self.models.values()))
                pred = best_model.predict(home_team, away_team, **features)
                
                # Wrap in EnsemblePrediction format
                return EnsemblePrediction(
                    home_win_prob=pred.home_win_prob,
                    draw_prob=pred.draw_prob,
                    away_win_prob=pred.away_win_prob,
                    predicted_outcome=self._get_outcome(pred),
                    confidence=pred.confidence,
                    model_agreement=1.0,
                    individual_predictions={pred.model_name: pred.to_dict()},
                    calibrated=False
                )
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback prediction
            return self._fallback_prediction(home_team, away_team)
    
    def _get_outcome(self, pred) -> str:
        """Get predicted outcome from prediction object"""
        probs = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
        outcomes = ['Home Win', 'Draw', 'Away Win']
        return outcomes[probs.index(max(probs))]
    
    def _fallback_prediction(self, home_team: str, away_team: str) -> EnsemblePrediction:
        """Return a safe fallback prediction when all models fail"""
        return EnsemblePrediction(
            home_win_prob=0.40,
            draw_prob=0.28,
            away_win_prob=0.32,
            predicted_outcome='Home Win',
            confidence=0.45,
            model_agreement=0.0,
            individual_predictions={'fallback': {'error': 'All models failed'}},
            calibrated=False
        )
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all registered models"""
        return {name: status.to_dict() for name, status in self.model_status.items()}
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self.models.keys())
    
    def unload_model(self, name: str):
        """Unload a model from memory"""
        if name in self.models:
            del self.models[name]
            if name in self.model_status:
                self.model_status[name].loaded = False
            logger.info(f"Unloaded model: {name}")
    
    def reload_model(self, name: str) -> bool:
        """Reload a model"""
        self.unload_model(name)
        
        # Try pretrained first
        model = self.loader.get_model(name)
        if model:
            self.models[name] = model
            self.model_status[name].loaded = True
            return True
        
        # Fall back to mock
        mock = create_mock_predictor(name)
        self.models[name] = mock
        self.model_status[name] = ModelStatus(
            name=name, loaded=True, type='mock', healthy=True
        )
        return True
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all models"""
        health = {}
        
        for name, model in self.models.items():
            try:
                # Try a simple prediction
                pred = model.predict('TestHome', 'TestAway')
                is_healthy = (
                    hasattr(pred, 'home_win_prob') and
                    0 <= pred.home_win_prob <= 1
                )
                health[name] = is_healthy
                self.model_status[name].healthy = is_healthy
            except Exception as e:
                health[name] = False
                self.model_status[name].healthy = False
                self.model_status[name].error_count += 1
        
        return health


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.initialize()
    return _registry


def predict(home_team: str, away_team: str, **features) -> EnsemblePrediction:
    """Convenience function for quick predictions"""
    return get_registry().predict(home_team, away_team, **features)
