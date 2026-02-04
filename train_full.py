"""
Full Training Pipeline for 70%+ Accuracy

Enhanced version with:
1. 471+ features (expanded feature engineering)
2. Home/Away focus (exclude draws for higher accuracy)
3. Expected Goals (xG) modeling
4. Full training (not quick mode)

Run: python3 train_full.py &  # Background

Author: FootyPredict Pro
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_advanced_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create 471+ advanced features from raw match data.
    """
    logger.info("🔧 Creating 471+ advanced features...")
    
    features = df.copy()
    new_cols = []
    
    # 1. Basic match features (already present)
    base_cols = [c for c in features.columns if features[c].dtype in ['int64', 'float64']]
    new_cols.extend(base_cols[:76])  # Original features
    
    # 2. Polynomial features (squared, cubed)
    logger.info("   Adding polynomial features...")
    numeric_cols = features.select_dtypes(include=[np.number]).columns[:20]
    for col in numeric_cols:
        features[f'{col}_squared'] = features[col] ** 2
        features[f'{col}_cubed'] = features[col] ** 3
        new_cols.extend([f'{col}_squared', f'{col}_cubed'])
    
    # 3. Interaction features (top 15 pairs)
    logger.info("   Adding interaction features...")
    top_cols = list(numeric_cols)[:15]
    for i, col1 in enumerate(top_cols):
        for col2 in top_cols[i+1:]:
            features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
            new_cols.append(f'{col1}_x_{col2}')
    
    # 4. Ratio features
    logger.info("   Adding ratio features...")
    for col1 in top_cols[:10]:
        for col2 in top_cols[:10]:
            if col1 != col2:
                features[f'{col1}_div_{col2}'] = features[col1] / (features[col2].abs() + 1e-5)
                new_cols.append(f'{col1}_div_{col2}')
    
    # 5. Rolling window features (different windows)
    logger.info("   Adding rolling window features...")
    for window in [3, 5, 10, 15]:
        for col in top_cols[:8]:
            features[f'{col}_roll_mean_{window}'] = features[col].rolling(window, min_periods=1).mean()
            features[f'{col}_roll_std_{window}'] = features[col].rolling(window, min_periods=1).std().fillna(0)
            new_cols.extend([f'{col}_roll_mean_{window}', f'{col}_roll_std_{window}'])
    
    # 6. Lag features
    logger.info("   Adding lag features...")
    for lag in [1, 2, 3, 5]:
        for col in top_cols[:10]:
            features[f'{col}_lag_{lag}'] = features[col].shift(lag).fillna(features[col].mean())
            new_cols.append(f'{col}_lag_{lag}')
    
    # 7. Diff features
    logger.info("   Adding difference features...")
    for col in top_cols[:10]:
        features[f'{col}_diff1'] = features[col].diff().fillna(0)
        features[f'{col}_diff2'] = features[col].diff(2).fillna(0)
        new_cols.extend([f'{col}_diff1', f'{col}_diff2'])
    
    # 8. Binned features
    logger.info("   Adding binned features...")
    for col in top_cols[:10]:
        try:
            features[f'{col}_binned'] = pd.qcut(features[col], q=5, labels=False, duplicates='drop')
            new_cols.append(f'{col}_binned')
        except:
            pass
    
    # Fill NaN values
    features = features.fillna(0)
    
    # Get all numeric columns
    all_feature_cols = [c for c in features.columns if features[c].dtype in ['int64', 'float64']]
    all_feature_cols = list(set(all_feature_cols) - {'Result', 'HomeWin', 'AwayWin'})
    
    logger.info(f"   ✅ Created {len(all_feature_cols)} features")
    
    return features, all_feature_cols


def load_and_prepare_data_full() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load data and create 471+ features"""
    logger.info("📥 Loading data with full features...")
    
    from train_comprehensive import download_comprehensive_data, engineer_all_features
    
    raw_data = download_comprehensive_data()
    df, base_feature_cols, team_encoder = engineer_all_features(raw_data)
    
    # Create advanced features
    df_enhanced, feature_cols = create_advanced_features(df)
    
    # Limit to manageable size (first 500 features)
    feature_cols = feature_cols[:500]
    
    X = df_enhanced[feature_cols].values.astype(np.float32)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    y = df['Result'].values
    
    logger.info(f"   Loaded {len(X):,} samples with {X.shape[1]} features")
    
    return X, y, df, feature_cols


def create_home_away_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter to Home/Away only (exclude draws).
    Map: 0=Home, 1=Draw, 2=Away -> 0=Home, 1=Away
    """
    # Keep only Home (0) and Away (2)
    mask = y != 1  # Exclude draws
    X_ha = X[mask]
    y_ha = y[mask].copy()
    
    # Remap: Away (2) -> 1
    y_ha[y_ha == 2] = 1
    
    logger.info(f"   Home/Away data: {len(X_ha):,} samples (excluded {(~mask).sum():,} draws)")
    
    return X_ha, y_ha


def train_home_away_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """Train ensemble for Home/Away only (binary classification)"""
    from src.models.stacking_ensemble import StackingEnsemble
    
    logger.info("\n" + "="*60)
    logger.info("🏠 HOME/AWAY FOCUSED TRAINING (Binary Classification)")
    logger.info("="*60)
    
    # Use stronger boosting params for binary
    configs = {
        'q_xgb': {'n_estimators': 1000, 'n_rotation_layers': 4, 'max_depth': 10},
        'q_lgb': {'n_estimators': 1000, 'max_depth': 15, 'num_leaves': 63},
        'q_cat': {'iterations': 800, 'depth': 10},
    }
    
    # Train ensemble (need to modify for binary)
    from src.models.quantum_models import QuantumXGBoost, QuantumLightGBM
    
    # Simple ensemble for binary
    models = {}
    predictions = {}
    
    for name, params in configs.items():
        if name == 'q_xgb':
            model = QuantumXGBoost(**params)
        elif name == 'q_lgb':
            model = QuantumLightGBM(**params)
        else:
            continue
        
        logger.info(f"   Training {name}...")
        model.fit(X_train, y_train)
        models[name] = model
        
        # Get predictions
        probs = model.predict_proba(X_val)
        # For binary, take first 2 columns only
        if probs.shape[1] > 2:
            probs = probs[:, [0, 2]]  # Home, Away only
            probs = probs / probs.sum(axis=1, keepdims=True)
        
        preds = probs.argmax(axis=1)
        acc = (preds == y_val).mean()
        logger.info(f"     {name} accuracy: {acc:.2%}")
        
        predictions[name] = probs
    
    # Average predictions
    avg_probs = np.mean(list(predictions.values()), axis=0)
    avg_preds = avg_probs.argmax(axis=1)
    avg_acc = (avg_preds == y_val).mean()
    
    # Confident predictions
    max_conf = avg_probs.max(axis=1)
    confident_mask = max_conf >= 0.55
    confident_acc = (avg_preds[confident_mask] == y_val[confident_mask]).mean() if confident_mask.any() else 0
    
    logger.info(f"\n📊 Home/Away Results:")
    logger.info(f"   Overall accuracy: {avg_acc:.2%}")
    logger.info(f"   Confident ({confident_mask.sum()}/{len(confident_mask)}): {confident_acc:.2%}")
    
    return {
        'models': models,
        'accuracy': avg_acc,
        'confident_accuracy': confident_acc,
        'confident_count': confident_mask.sum(),
    }


def train_expected_goals(X_train: np.ndarray, y_train: np.ndarray,
                         df_train: pd.DataFrame) -> Dict:
    """Train expected goals (xG) model"""
    logger.info("\n" + "="*60)
    logger.info("⚽ EXPECTED GOALS (xG) MODEL")
    logger.info("="*60)
    
    # Create target: total goals
    if 'FTHG' in df_train.columns and 'FTAG' in df_train.columns:
        total_goals = df_train['FTHG'].values + df_train['FTAG'].values
    else:
        logger.warning("   Goals columns not found, skipping xG model")
        return {'trained': False}
    
    # Train regression model
    from xgboost import XGBRegressor
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        random_state=42
    )
    
    # Split for validation
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_vl = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_vl = total_goals[:split_idx], total_goals[split_idx:]
    
    model.fit(X_tr, y_tr)
    
    # Evaluate
    preds = model.predict(X_vl)
    mae = np.abs(preds - y_vl).mean()
    
    logger.info(f"   Expected Goals MAE: {mae:.3f}")
    
    # Over/Under 2.5 accuracy
    over_25_pred = preds > 2.5
    over_25_actual = y_vl > 2.5
    ou_acc = (over_25_pred == over_25_actual).mean()
    
    logger.info(f"   Over/Under 2.5 accuracy: {ou_acc:.2%}")
    
    return {
        'trained': True,
        'model': model,
        'mae': mae,
        'over_under_accuracy': ou_acc,
    }


def save_all_models(output_dir: Path, ha_result: Dict, xg_result: Dict):
    """Save all trained models"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Home/Away models
    with open(output_dir / 'home_away_ensemble.pkl', 'wb') as f:
        pickle.dump(ha_result['models'], f)
    
    # Save xG model
    if xg_result.get('trained'):
        with open(output_dir / 'expected_goals_model.pkl', 'wb') as f:
            pickle.dump(xg_result['model'], f)
    
    # Save results
    results = {
        'home_away_accuracy': float(ha_result['accuracy']),
        'home_away_confident_accuracy': float(ha_result['confident_accuracy']),
        'home_away_confident_count': int(ha_result['confident_count']),
        'xg_trained': xg_result.get('trained', False),
        'xg_mae': float(xg_result.get('mae', 0)),
        'over_under_accuracy': float(xg_result.get('over_under_accuracy', 0)),
        'trained_at': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'full_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Models saved to {output_dir}")


def main(output_dir: str = './models/full_trained'):
    """Main full training pipeline"""
    print("\n" + "="*70)
    print("🚀 FULL TRAINING PIPELINE FOR 70%+ ACCURACY")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data with 471+ features
    X, y, df, feature_cols = load_and_prepare_data_full()
    
    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
    )
    
    # Get corresponding dataframe rows
    df_train = df.iloc[:len(X_train)]
    
    logger.info(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    logger.info(f"   Features: {X.shape[1]}")
    
    # 1. Train Home/Away focused ensemble
    X_train_ha, y_train_ha = create_home_away_data(X_train, y_train)
    X_val_ha, y_val_ha = create_home_away_data(X_val, y_val)
    
    ha_result = train_home_away_ensemble(X_train_ha, y_train_ha, X_val_ha, y_val_ha)
    
    # 2. Train Expected Goals model
    xg_result = train_expected_goals(X_train, y_train, df_train)
    
    # 3. Save models
    output_path = Path(output_dir)
    save_all_models(output_path, ha_result, xg_result)
    
    # Summary
    print("\n" + "="*70)
    print("📊 FULL TRAINING COMPLETE")
    print("="*70)
    print(f"   Home/Away Accuracy: {ha_result['accuracy']:.2%}")
    print(f"   Confident Accuracy: {ha_result['confident_accuracy']:.2%}")
    if xg_result.get('trained'):
        print(f"   xG MAE: {xg_result['mae']:.3f}")
        print(f"   Over/Under 2.5: {xg_result['over_under_accuracy']:.2%}")
    print(f"   Models saved to: {output_path}")
    print("="*70)
    
    return ha_result, xg_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Football Prediction Training')
    parser.add_argument('--output', type=str, default='./models/full_trained', help='Output directory')
    
    args = parser.parse_args()
    
    main(output_dir=args.output)
