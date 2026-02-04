#!/usr/bin/env python3
"""
V4.0 Training Script
Trains all market models using the enhanced training pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("ULTIMATE FOOTBALL PREDICTION SYSTEM V4.0 - TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Load training data
    data_path = Path("data/comprehensive_training_data.csv")
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        # Try alternative paths
        alt_paths = [
            Path("data/training_data.csv"),
            Path("data/matches.csv"),
        ]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break
        else:
            logger.error("No training data found!")
            return
    
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} matches")
    
    # Prepare data
    if 'Date' in data.columns:
        data['match_date'] = pd.to_datetime(data['Date'], dayfirst=True, format='mixed')
    elif 'date' in data.columns:
        data['match_date'] = pd.to_datetime(data['date'], dayfirst=True, format='mixed')
    
    if 'HomeTeam' in data.columns:
        data['home_team'] = data['HomeTeam']
        data['away_team'] = data['AwayTeam']
    
    if 'FTHG' in data.columns:
        data['home_goals'] = data['FTHG']
        data['away_goals'] = data['FTAG']
    
    if 'HTHG' in data.columns:
        data['home_goals_ht'] = data['HTHG']
        data['away_goals_ht'] = data['HTAG']
    
    # Sort by date
    data = data.sort_values('match_date').reset_index(drop=True)
    
    # Use last 3 years
    cutoff = data['match_date'].max() - pd.Timedelta(days=365*3)
    recent_data = data[data['match_date'] >= cutoff].copy()
    logger.info(f"Using {len(recent_data)} recent matches (last 3 years)")
    
    # Generate features
    print("\n" + "=" * 40)
    print("PHASE 1: FEATURE ENGINEERING")
    print("=" * 40)
    
    try:
        from src.features.enhanced_engineering import EnhancedFeatureGenerator, FeatureConfig
        
        config = FeatureConfig(
            rolling_windows=[3, 5, 10],  # Faster with fewer windows
            include_clusters=True
        )
        
        generator = EnhancedFeatureGenerator(recent_data, config)
        features = generator.generate_all_features()
        
        logger.info(f"Generated {len(features.columns)} features")
        
        # Remove NaN-heavy features
        nan_threshold = 0.5
        features = features.dropna(axis=1, thresh=int(len(features) * (1 - nan_threshold)))
        features = features.fillna(0)
        
        logger.info(f"After cleanup: {len(features.columns)} features")
        
    except ImportError as e:
        logger.warning(f"Enhanced features not available: {e}")
        logger.info("Using basic features...")
        
        from src.features.real_data_features import RealDataFeatureGenerator
        generator = RealDataFeatureGenerator()
        
        feature_list = []
        for idx, row in recent_data.iterrows():
            feats = generator.generate(
                home_team=row['home_team'],
                away_team=row['away_team'],
                match_date=row['match_date']
            )
            feature_list.append(feats)
        
        features = pd.DataFrame(feature_list)
        features = features.fillna(0)
        logger.info(f"Generated {len(features.columns)} features")
    
    # Training
    print("\n" + "=" * 40)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 40)
    
    from src.training.enhanced_training_pipeline import (
        TrainingConfig, EnhancedTrainingPipeline
    )
    
    config = TrainingConfig(
        n_splits=3,
        test_size=100,
        n_optuna_trials=20,  # Faster with fewer trials
        model_dir="models/v4"
    )
    
    pipeline = EnhancedTrainingPipeline(config)
    
    # Train all markets
    markets = ['result', 'btts', 'over25', 'over15']
    results = pipeline.train_all_markets(recent_data, features, markets)
    
    # Print results
    print("\n" + "=" * 40)
    print("TRAINING RESULTS")
    print("=" * 40)
    
    for market, result in results.items():
        print(f"\n{market.upper()}:")
        print(f"  CV Accuracy: {result.val_accuracy:.4f}")
        print(f"  Test Accuracy: {result.test_accuracy:.4f}")
        print(f"  Training Time: {result.training_time_seconds:.1f}s")
        if result.suggestions:
            print(f"  Suggestions: {result.suggestions[0]}")
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE at {datetime.now()}")
    print("Models saved to models/v4/")
    print("=" * 60)


if __name__ == "__main__":
    main()
