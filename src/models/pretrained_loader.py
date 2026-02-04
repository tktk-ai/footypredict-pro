"""
Pre-trained Model Loader

Downloads and manages pre-trained models from HuggingFace.
No training required - just load and predict!

Available models:
- Podos Transformer: 276K params, trained on 100K games
- FootballerModel: Classification model for match outcomes
- XGBoost (optional): Loaded from local trained model
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model storage directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TRAINED_DIR = MODELS_DIR / "trained"
CONFIG_DIR = MODELS_DIR / "config"

# Ensure directories exist
for dir_path in [PRETRAINED_DIR, TRAINED_DIR, CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    version: str
    source: str  # 'huggingface', 'kaggle', 'local'
    params: int
    loaded: bool
    path: Optional[str] = None
    error: Optional[str] = None


class PretrainedModelLoader:
    """
    Downloads and loads pre-trained models from HuggingFace.
    
    Usage:
        loader = PretrainedModelLoader()
        loader.download_all()
        podos = loader.get_model('podos')
    """
    
    # Model registry with HuggingFace repo info
    MODELS = {
        'podos': {
            'repo_id': 'podos/soccer-match-predictor',
            'description': 'Transformer model trained on 100K games',
            'params': 276000,
            'type': 'transformer',
            'fallback_repo': None  # Will use mock if not available
        },
        'footballer': {
            'repo_id': 'AmjadKha/FootballerModel',
            'description': 'Match outcome classifier',
            'params': 50000,
            'type': 'classifier',
            'fallback_repo': None
        }
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or PRETRAINED_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._hf_available = self._check_huggingface()
    
    def _check_huggingface(self) -> bool:
        """Check if HuggingFace libraries are available"""
        try:
            import torch
            from huggingface_hub import hf_hub_download
            return True
        except ImportError:
            logger.warning("HuggingFace/PyTorch not installed. Using fallback models.")
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        Download a specific model from HuggingFace.
        
        Args:
            model_name: Name of model ('podos', 'footballer')
            force: Force re-download even if cached
            
        Returns:
            True if successful
        """
        if model_name not in self.MODELS:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_config = self.MODELS[model_name]
        model_path = self.cache_dir / f"{model_name}_model.pt"
        
        # Check cache
        if model_path.exists() and not force:
            logger.info(f"Model {model_name} already cached at {model_path}")
            return True
        
        if not self._hf_available:
            logger.warning(f"Creating mock model for {model_name} (HuggingFace not available)")
            self._create_mock_model(model_name, model_path)
            return True
        
        try:
            from huggingface_hub import hf_hub_download, HfHubHTTPError
            
            logger.info(f"Downloading {model_name} from HuggingFace...")
            
            try:
                # Try to download from HuggingFace
                downloaded_path = hf_hub_download(
                    repo_id=model_config['repo_id'],
                    filename="pytorch_model.bin",
                    cache_dir=self.cache_dir / "hf_cache",
                    local_dir=self.cache_dir
                )
                
                # Copy/move to standard location
                import shutil
                shutil.copy(downloaded_path, model_path)
                logger.info(f"Model {model_name} downloaded to {model_path}")
                return True
                
            except Exception as e:
                logger.warning(f"Could not download from HuggingFace: {e}")
                logger.info(f"Creating local mock model for {model_name}")
                self._create_mock_model(model_name, model_path)
                return True
                
        except ImportError:
            logger.warning("huggingface_hub not installed")
            self._create_mock_model(model_name, model_path)
            return True
    
    def _create_mock_model(self, model_name: str, model_path: Path):
        """Create a mock model when HuggingFace is unavailable"""
        from .mock_models import create_mock_predictor
        
        mock_model = create_mock_predictor(model_name)
        
        # Save mock model info
        mock_info = {
            'name': model_name,
            'type': 'mock',
            'version': '0.1.0',
            'description': f'Mock {model_name} model (HuggingFace unavailable)'
        }
        
        with open(model_path.with_suffix('.json'), 'w') as f:
            json.dump(mock_info, f)
        
        # Save the mock model
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        logger.info(f"Created mock model at {model_path}")
    
    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """Download all available models"""
        results = {}
        for model_name in self.MODELS:
            results[model_name] = self.download_model(model_name, force)
        return results
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded model object or None
        """
        if model_name in self._models:
            return self._models[model_name]
        
        model_path = self.cache_dir / f"{model_name}_model.pt"
        
        if not model_path.exists():
            logger.info(f"Model {model_name} not found, downloading...")
            if not self.download_model(model_name):
                return None
        
        try:
            # Try PyTorch load first
            if self._hf_available:
                import torch
                try:
                    model = torch.load(model_path, map_location='cpu')
                    self._models[model_name] = model
                    self._model_info[model_name] = ModelInfo(
                        name=model_name,
                        version='1.0.0',
                        source='huggingface',
                        params=self.MODELS[model_name]['params'],
                        loaded=True,
                        path=str(model_path)
                    )
                    return model
                except Exception:
                    pass
            
            # Fall back to pickle (mock models)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self._models[model_name] = model
            self._model_info[model_name] = ModelInfo(
                name=model_name,
                version='0.1.0',
                source='mock',
                params=self.MODELS.get(model_name, {}).get('params', 0),
                loaded=True,
                path=str(model_path)
            )
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self._model_info[model_name] = ModelInfo(
                name=model_name,
                version='0.0.0',
                source='error',
                params=0,
                loaded=False,
                error=str(e)
            )
            return None
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model, loading it if necessary"""
        if model_name not in self._models:
            self.load_model(model_name)
        return self._models.get(model_name)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get info about a model"""
        if model_name not in self._model_info:
            self.load_model(model_name)
        return self._model_info.get(model_name)
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with their status"""
        models = []
        for name, config in self.MODELS.items():
            model_path = self.cache_dir / f"{name}_model.pt"
            info = self._model_info.get(name)
            
            models.append({
                'name': name,
                'description': config['description'],
                'params': config['params'],
                'type': config['type'],
                'downloaded': model_path.exists(),
                'loaded': name in self._models,
                'info': info.__dict__ if info else None
            })
        
        return models
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    def clear_cache(self):
        """Clear all cached models"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models.clear()
        self._model_info.clear()
        logger.info("Model cache cleared")


# Global loader instance
_loader: Optional[PretrainedModelLoader] = None


def get_loader() -> PretrainedModelLoader:
    """Get the global model loader instance"""
    global _loader
    if _loader is None:
        _loader = PretrainedModelLoader()
    return _loader


def download_all() -> Dict[str, bool]:
    """Download all pre-trained models"""
    return get_loader().download_all()


def get_model(model_name: str) -> Optional[Any]:
    """Get a pre-trained model by name"""
    return get_loader().get_model(model_name)


def list_models() -> List[Dict]:
    """List all available models"""
    return get_loader().list_available_models()
