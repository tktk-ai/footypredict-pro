"""
HuggingFace Model Loader
========================

This module provides functions to download and load models from HuggingFace Hub.
It handles caching, fallback to local models, and lazy loading.

Usage:
    from src.models.hf_loader import load_model_from_hf, get_model_path

    # Get path to a model (downloads if needed)
    model_path = get_model_path("sportybet_ensemble/btts_ensemble.pkl")

    # Load a model directly
    model = load_model_from_hf("sportybet_ensemble/btts_ensemble.pkl")
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any, Union
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
HF_REPO = os.getenv("HUGGINGFACE_REPO", "nananie143/footypredict-models")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LOCAL_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
CACHE_DIR = Path.home() / ".cache" / "footypredict" / "models"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_hf_hub():
    """Lazy import of huggingface_hub to avoid import errors if not installed."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
        return hf_hub_download, HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed. Using local models only.")
        return None, None


@lru_cache(maxsize=100)
def get_model_path(
    model_name: str,
    prefer_local: bool = True,
    force_download: bool = False
) -> Optional[Path]:
    """
    Get the path to a model file.
    
    First checks local models directory, then tries HuggingFace Hub.
    Results are cached for fast subsequent access.
    
    Args:
        model_name: Relative path to model (e.g., "sportybet_ensemble/btts_ensemble.pkl")
        prefer_local: If True, prefer local models over HuggingFace
        force_download: If True, always download from HuggingFace
        
    Returns:
        Path to the model file, or None if not found
    """
    # Check local first (if preferred and not forcing download)
    if prefer_local and not force_download:
        local_path = LOCAL_MODELS_DIR / model_name
        if local_path.exists():
            logger.debug(f"Using local model: {local_path}")
            return local_path
    
    # Try HuggingFace Hub
    hf_hub_download, _ = _get_hf_hub()
    if hf_hub_download:
        try:
            logger.info(f"Downloading from HuggingFace: {HF_REPO}/{model_name}")
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=model_name,
                token=HF_TOKEN,
                cache_dir=str(CACHE_DIR),
            )
            return Path(downloaded_path)
        except Exception as e:
            logger.warning(f"Failed to download from HuggingFace: {e}")
    
    # Fall back to local if not preferring local initially
    if not prefer_local:
        local_path = LOCAL_MODELS_DIR / model_name
        if local_path.exists():
            logger.debug(f"Falling back to local model: {local_path}")
            return local_path
    
    logger.error(f"Model not found: {model_name}")
    return None


def load_model_from_hf(
    model_name: str,
    prefer_local: bool = True,
    force_download: bool = False
) -> Optional[Any]:
    """
    Load a model from HuggingFace Hub or local storage.
    
    Supports various model formats:
    - .pkl, .joblib: Loaded with joblib
    - .json: Loaded as XGBoost model or JSON
    - .pt: Loaded as PyTorch model
    
    Args:
        model_name: Relative path to model
        prefer_local: If True, prefer local models over HuggingFace
        force_download: If True, always download from HuggingFace
        
    Returns:
        Loaded model object, or None if failed
    """
    model_path = get_model_path(model_name, prefer_local, force_download)
    
    if model_path is None:
        return None
    
    suffix = model_path.suffix.lower()
    
    try:
        if suffix in ['.pkl', '.joblib']:
            import joblib
            return joblib.load(model_path)
        
        elif suffix == '.json':
            # Try XGBoost first
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                return model
            except:
                # Fall back to regular JSON
                import json
                with open(model_path, 'r') as f:
                    return json.load(f)
        
        elif suffix == '.pt':
            import torch
            return torch.load(model_path, map_location='cpu')
        
        else:
            logger.warning(f"Unknown model format: {suffix}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None


def list_available_models() -> dict:
    """
    List all available models from HuggingFace Hub.
    
    Returns:
        Dictionary with model categories and their files
    """
    _, HfApi = _get_hf_hub()
    
    if HfApi is None:
        logger.warning("Cannot list HuggingFace models - library not installed")
        return {}
    
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=HF_REPO, token=HF_TOKEN)
        
        # Organize by directory
        models = {}
        for f in files:
            if '/' in f:
                category = f.split('/')[0]
                if category not in models:
                    models[category] = []
                models[category].append(f)
            else:
                if 'root' not in models:
                    models['root'] = []
                models['root'].append(f)
        
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {}


# Convenience functions for specific model types
def load_sportybet_ensemble(market: str) -> Optional[Any]:
    """Load a SportyBet ensemble model for a specific market."""
    return load_model_from_hf(f"sportybet_ensemble/{market}_ensemble.pkl")


def load_sportybet_xgb(market: str) -> tuple:
    """Load a SportyBet XGBoost model and its scaler."""
    model = load_model_from_hf(f"trained/sportybet/{market}_model.json")
    scaler = load_model_from_hf(f"trained/sportybet/{market}_scaler.pkl")
    return model, scaler


def load_main_xgb() -> tuple:
    """Load the main XGBoost model and scaler."""
    model = load_model_from_hf("trained/xgb_football_76features.json")
    scaler = load_model_from_hf("trained/scaler.pkl")
    return model, scaler


# Pre-warm cache on import (optional)
def preload_essential_models():
    """Pre-download essential models for faster inference."""
    essential = [
        "trained/scaler.pkl",
        "trained/xgb_football_76features.json",
        "sportybet_ensemble/btts_ensemble.pkl",
        "sportybet_ensemble/result_1x2_ensemble.pkl",
    ]
    
    logger.info("Pre-loading essential models...")
    for model in essential:
        get_model_path(model)
    logger.info("Essential models loaded.")


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    
    print("Testing HuggingFace Model Loader")
    print("=" * 50)
    
    # List available models
    models = list_available_models()
    print(f"\n📦 Available models in {HF_REPO}:")
    for category, files in models.items():
        print(f"\n  {category}/")
        for f in files[:5]:  # Show first 5
            print(f"    - {f}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
    
    # Test loading a model
    print("\n\n🔄 Testing model loading...")
    model, scaler = load_main_xgb()
    if model:
        print(f"✅ Successfully loaded main XGBoost model")
    else:
        print(f"❌ Failed to load model")
