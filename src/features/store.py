"""
Feature Store Module
Centralized storage and retrieval of computed features.

Part of the complete blueprint implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Centralized feature storage and retrieval.
    
    Features:
    - Feature versioning
    - Caching
    - Feature metadata
    - Batch retrieval
    """
    
    def __init__(self, store_dir: str = "data/features"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.store_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self.cache: Dict[str, pd.DataFrame] = {}
        
    def _load_metadata(self) -> Dict:
        """Load feature metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {'features': {}, 'versions': {}}
    
    def _save_metadata(self):
        """Save feature metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for versioning."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:12]
    
    def save_features(
        self,
        features: pd.DataFrame,
        name: str,
        description: str = "",
        version: str = None
    ) -> str:
        """
        Save features to the store.
        
        Args:
            features: Feature DataFrame
            name: Feature set name
            description: Description of features
            version: Optional version string
        
        Returns:
            Version identifier
        """
        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save DataFrame
        filename = f"{name}_v{version}.parquet"
        filepath = self.store_dir / filename
        features.to_parquet(filepath, index=False)
        
        # Update metadata
        if name not in self.metadata['features']:
            self.metadata['features'][name] = {
                'description': description,
                'columns': features.columns.tolist(),
                'versions': []
            }
        
        version_info = {
            'version': version,
            'filename': filename,
            'rows': len(features),
            'columns': len(features.columns),
            'created_at': datetime.now().isoformat(),
            'hash': self._compute_hash(features)
        }
        
        self.metadata['features'][name]['versions'].append(version_info)
        self.metadata['features'][name]['latest_version'] = version
        self._save_metadata()
        
        # Update cache
        self.cache[f"{name}_latest"] = features
        
        logger.info(f"Saved feature set '{name}' v{version} ({len(features)} rows)")
        return version
    
    def load_features(
        self,
        name: str,
        version: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Load features from the store.
        
        Args:
            name: Feature set name
            version: Specific version or None for latest
        """
        # Check cache
        cache_key = f"{name}_{version or 'latest'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if name not in self.metadata['features']:
            logger.warning(f"Feature set '{name}' not found")
            return None
        
        feature_meta = self.metadata['features'][name]
        
        if version:
            # Find specific version
            version_info = next(
                (v for v in feature_meta['versions'] if v['version'] == version),
                None
            )
        else:
            # Get latest
            version_info = feature_meta['versions'][-1] if feature_meta['versions'] else None
        
        if not version_info:
            logger.warning(f"Version not found for '{name}'")
            return None
        
        filepath = self.store_dir / version_info['filename']
        
        if not filepath.exists():
            logger.warning(f"Feature file not found: {filepath}")
            return None
        
        df = pd.read_parquet(filepath)
        self.cache[cache_key] = df
        
        return df
    
    def get_features_for_match(
        self,
        home_team: str,
        away_team: str,
        feature_sets: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get all features for a specific match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            feature_sets: Specific feature sets to load
        """
        feature_sets = feature_sets or list(self.metadata['features'].keys())
        
        result = {
            'match': f"{home_team} vs {away_team}",
            'features': {}
        }
        
        for name in feature_sets:
            df = self.load_features(name)
            if df is None:
                continue
            
            # Try to find matching rows
            if 'home_team' in df.columns and 'away_team' in df.columns:
                match_data = df[
                    (df['home_team'] == home_team) & 
                    (df['away_team'] == away_team)
                ]
                if not match_data.empty:
                    result['features'][name] = match_data.iloc[0].to_dict()
        
        return result
    
    def list_feature_sets(self) -> List[Dict]:
        """List all available feature sets."""
        return [
            {
                'name': name,
                'description': meta.get('description', ''),
                'columns': len(meta.get('columns', [])),
                'versions': len(meta.get('versions', [])),
                'latest': meta.get('latest_version')
            }
            for name, meta in self.metadata['features'].items()
        ]
    
    def get_feature_columns(self, name: str) -> List[str]:
        """Get columns for a feature set."""
        if name in self.metadata['features']:
            return self.metadata['features'][name].get('columns', [])
        return []
    
    def delete_feature_set(self, name: str, version: str = None):
        """Delete a feature set or specific version."""
        if name not in self.metadata['features']:
            return
        
        if version:
            # Delete specific version
            versions = self.metadata['features'][name]['versions']
            version_info = next(
                (v for v in versions if v['version'] == version),
                None
            )
            if version_info:
                filepath = self.store_dir / version_info['filename']
                if filepath.exists():
                    filepath.unlink()
                versions.remove(version_info)
        else:
            # Delete all versions
            for v in self.metadata['features'][name]['versions']:
                filepath = self.store_dir / v['filename']
                if filepath.exists():
                    filepath.unlink()
            del self.metadata['features'][name]
        
        self._save_metadata()
        logger.info(f"Deleted feature set '{name}' {version or '(all versions)'}")
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()


# Global instance
_store: Optional[FeatureStore] = None


def get_store() -> FeatureStore:
    """Get or create feature store."""
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


def save_features(features: pd.DataFrame, name: str) -> str:
    """Quick function to save features."""
    return get_store().save_features(features, name)


def load_features(name: str) -> Optional[pd.DataFrame]:
    """Quick function to load features."""
    return get_store().load_features(name)
