"""
Feature Store

Centralized storage and retrieval of features for model training and inference.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Centralized feature storage and retrieval.
    
    Features:
    - Store computed features for reuse
    - Versioning of feature sets
    - Feature metadata tracking
    - Lazy loading of features
    """
    
    def __init__(
        self,
        store_path: Path = None,
        cache_in_memory: bool = True
    ):
        self.store_path = store_path or Path('data/features')
        self.cache_in_memory = cache_in_memory
        self._memory_cache = {}
        self._metadata = {}
        
        # Create store directory
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self._load_metadata()
    
    def store_features(
        self,
        features: pd.DataFrame,
        name: str,
        version: str = None,
        metadata: Dict = None
    ) -> str:
        """
        Store features to the feature store.
        
        Args:
            features: DataFrame with features
            name: Feature set name
            version: Version string (auto-generated if not provided)
            metadata: Additional metadata
            
        Returns:
            Feature store key
        """
        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"{name}_v{version}"
        
        # Save features
        feature_path = self.store_path / f"{key}.parquet"
        features.to_parquet(feature_path, index=False)
        
        # Store metadata
        self._metadata[key] = {
            'name': name,
            'version': version,
            'columns': list(features.columns),
            'shape': features.shape,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Cache in memory
        if self.cache_in_memory:
            self._memory_cache[key] = features.copy()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Stored features: {key} ({features.shape})")
        return key
    
    def get_features(
        self,
        key: str = None,
        name: str = None,
        version: str = 'latest',
        columns: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve features from the store.
        
        Args:
            key: Direct feature set key
            name: Feature set name (used with version)
            version: Version to retrieve ('latest' for most recent)
            columns: Specific columns to retrieve
            
        Returns:
            Features DataFrame
        """
        # Resolve key
        if key is None:
            if name is None:
                raise ValueError("Either key or name must be provided")
            
            if version == 'latest':
                key = self._get_latest_version(name)
            else:
                key = f"{name}_v{version}"
        
        if key is None:
            logger.warning(f"No features found for name: {name}")
            return None
        
        # Try memory cache first
        if key in self._memory_cache:
            features = self._memory_cache[key]
        else:
            # Load from disk
            feature_path = self.store_path / f"{key}.parquet"
            
            if not feature_path.exists():
                logger.warning(f"Feature file not found: {feature_path}")
                return None
            
            features = pd.read_parquet(feature_path)
            
            # Cache if enabled
            if self.cache_in_memory:
                self._memory_cache[key] = features.copy()
        
        # Select columns if specified
        if columns:
            available = [c for c in columns if c in features.columns]
            missing = [c for c in columns if c not in features.columns]
            
            if missing:
                logger.warning(f"Missing columns: {missing}")
            
            features = features[available]
        
        return features
    
    def list_feature_sets(self, name: str = None) -> List[Dict]:
        """List available feature sets."""
        results = []
        
        for key, meta in self._metadata.items():
            if name is None or meta['name'] == name:
                results.append({
                    'key': key,
                    **meta
                })
        
        return sorted(results, key=lambda x: x['created_at'], reverse=True)
    
    def get_feature_info(self, key: str) -> Optional[Dict]:
        """Get metadata for a feature set."""
        return self._metadata.get(key)
    
    def delete_features(self, key: str) -> bool:
        """Delete a feature set."""
        feature_path = self.store_path / f"{key}.parquet"
        
        if feature_path.exists():
            feature_path.unlink()
        
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        if key in self._metadata:
            del self._metadata[key]
            self._save_metadata()
        
        logger.info(f"Deleted features: {key}")
        return True
    
    def compute_feature_hash(self, features: pd.DataFrame) -> str:
        """Compute a hash of the feature DataFrame for versioning."""
        content = str(features.columns.tolist()) + str(features.shape)
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version key for a feature set name."""
        matching = [
            (k, v) for k, v in self._metadata.items()
            if v['name'] == name
        ]
        
        if not matching:
            return None
        
        # Sort by created_at and return latest
        matching.sort(key=lambda x: x[1]['created_at'], reverse=True)
        return matching[0][0]
    
    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = self.store_path / 'metadata.json'
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self._metadata = {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = self.store_path / 'metadata.json'
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._memory_cache.clear()
        logger.info("Cleared feature store cache")


# Global store instance
_store: Optional[FeatureStore] = None


def get_store() -> FeatureStore:
    """Get the global feature store instance."""
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


def store_features(features: pd.DataFrame, name: str, **kwargs) -> str:
    """Convenience function to store features."""
    return get_store().store_features(features, name, **kwargs)


def get_features(name: str, **kwargs) -> Optional[pd.DataFrame]:
    """Convenience function to retrieve features."""
    return get_store().get_features(name=name, **kwargs)
