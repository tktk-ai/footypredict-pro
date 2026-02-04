#!/usr/bin/env python3
"""
FootyPredict Pro - Local Training Pipeline Test
Tests the V4.0 ensemble training locally before deploying to Kaggle.
"""

import os
import sys
import time
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb
import lightgbm as lgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models' / 'v4_local'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    'n_splits': 3,
    'test_size': 100,
    'random_state': 42,
    'markets': ['result', 'over25', 'btts'],
    'use_gpu': False  # Set to True if you have CUDA
}


def load_training_data():
    """Load all available training data."""
    logger.info("Loading training data...")
    
    # Priority order for data sources
    data_sources = [
        DATA_DIR / 'comprehensive_training_data.csv',
        DATA_DIR / 'collected' / 'merged_training_data.parquet',
        DATA_DIR / 'training_data.csv',
    ]
    
    data = None
    for source in data_sources:
        if source.exists():
            logger.info(f"Found: {source}")
            if source.suffix == '.parquet':
                data = pd.read_parquet(source)
            else:
                data = pd.read_csv(source)
            logger.info(f"Loaded {len(data)} matches from {source.name}")
            break
    
    if data is None:
        raise FileNotFoundError("No training data found!")
    
    return data


def generate_features(data):
    """Generate features from raw match data."""
    logger.info("Generating features...")
    
    features = pd.DataFrame(index=data.index)
    
    # Basic columns that might exist
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Copy numeric columns as features
    for col in numeric_cols:
        if col not in ['Unnamed: 0', 'index']:
            features[col] = data[col]
    
    # Derived features
    if 'FTHG' in data.columns and 'FTAG' in data.columns:
        features['total_goals'] = data['FTHG'] + data['FTAG']
        features['goal_diff'] = data['FTHG'] - data['FTAG']
        features['home_win'] = (data['FTHG'] > data['FTAG']).astype(int)
        features['away_win'] = (data['FTHG'] < data['FTAG']).astype(int)
        features['draw'] = (data['FTHG'] == data['FTAG']).astype(int)
    
    # Odds features
    odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']
    for col in odds_cols:
        if col in data.columns:
            features[f'{col}_prob'] = 1 / data[col].replace(0, np.nan)
    
    # Rolling features (need to group by team ideally, but simplified here)
    for col in ['FTHG', 'FTAG']:
        if col in data.columns:
            for window in [3, 5, 10]:
                features[f'{col}_rolling_{window}'] = data[col].rolling(window, min_periods=1).mean()
    
    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    # Sanitize column names for LightGBM (no special JSON characters)
    import re
    clean_cols = {}
    seen = {}
    for col in features.columns:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        # Handle duplicates by adding suffix
        if clean_name in seen:
            seen[clean_name] += 1
            clean_name = f"{clean_name}_{seen[clean_name]}"
        else:
            seen[clean_name] = 0
        clean_cols[col] = clean_name
    features = features.rename(columns=clean_cols)
    
    logger.info(f"Generated {len(features.columns)} features")
    return features


def prepare_targets(data):
    """Prepare target variables for each market."""
    targets = {}
    
    if 'FTR' in data.columns:
        targets['result'] = data['FTR']
    
    if 'FTHG' in data.columns and 'FTAG' in data.columns:
        total_goals = data['FTHG'] + data['FTAG']
        targets['over25'] = (total_goals > 2.5).map({True: 'Over', False: 'Under'})
        targets['btts'] = ((data['FTHG'] > 0) & (data['FTAG'] > 0)).map({True: 'Yes', False: 'No'})
    
    return targets


class LocalEnsembleTrainer:
    """Train ensemble models locally."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
    
    def train(self, X, y, target_name='result'):
        """Train ensemble for a single market."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {target_name.upper()} model")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        # LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # Ensemble predictions (average)
        xgb_proba = xgb_model.predict_proba(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)
        ensemble_proba = (xgb_proba + lgb_proba) / 2
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        try:
            logloss = log_loss(y_test, ensemble_proba)
        except:
            logloss = 0.0
        
        train_time = time.time() - start_time
        
        logger.info(f"\nResults for {target_name}:")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Log Loss: {logloss:.4f}")
        logger.info(f"  Time: {train_time:.1f}s")
        
        # Store model bundle
        self.models[target_name] = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'label_encoder': le,
            'feature_names': list(X.columns),
            'classes': list(le.classes_)
        }
        
        self.results[target_name] = {
            'accuracy': float(accuracy),
            'log_loss': float(logloss),
            'train_time': float(train_time),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(X.columns)
        }
        
        return accuracy
    
    def save_models(self, output_dir):
        """Save all trained models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for target_name, model_dict in self.models.items():
            model_path = output_dir / f'{target_name}_ensemble.joblib'
            joblib.dump(model_dict, model_path)
            logger.info(f"Saved: {model_path}")
        
        # Save results
        results_path = output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': CONFIG,
                'results': self.results
            }, f, indent=2)
        logger.info(f"Saved: {results_path}")


def main():
    """Run local training pipeline."""
    logger.info("="*60)
    logger.info("FOOTYPREDICT V4.0 - LOCAL TRAINING TEST")
    logger.info("="*60)
    
    start_time = time.time()
    
    # 1. Load data
    data = load_training_data()
    
    # 2. Generate features
    features = generate_features(data)
    
    # 3. Prepare targets
    targets = prepare_targets(data)
    
    # 4. Train models
    trainer = LocalEnsembleTrainer(CONFIG)
    
    for market in CONFIG['markets']:
        if market in targets:
            y = targets[market]
            valid_idx = ~y.isna()
            X = features[valid_idx]
            y = y[valid_idx]
            
            if len(X) > 100:
                trainer.train(X, y, target_name=market)
            else:
                logger.warning(f"Skipping {market} - insufficient data ({len(X)} samples)")
        else:
            logger.warning(f"Skipping {market} - target not available")
    
    # 5. Save models
    trainer.save_models(MODELS_DIR)
    
    total_time = time.time() - start_time
    
    # 6. Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Models saved to: {MODELS_DIR}")
    
    logger.info("\nResults Summary:")
    for market, results in trainer.results.items():
        logger.info(f"  {market.upper()}: {results['accuracy']:.2%} accuracy")
    
    return trainer.results


if __name__ == '__main__':
    results = main()
