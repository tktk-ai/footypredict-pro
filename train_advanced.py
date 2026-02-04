"""
Advanced Training Pipeline for 65-70% Accuracy Target

This script combines all advanced techniques:
1. Stacking Ensemble with OOF predictions
2. Advanced feature engineering  
3. Focal Loss for class imbalance
4. Monte Carlo Dropout uncertainty
5. Probability calibration
6. Confidence filtering

Run: python3 train_advanced.py [--quick] [--cv]

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
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load data and engineer features"""
    logger.info("📥 Loading data...")
    
    from train_comprehensive import download_comprehensive_data, engineer_all_features
    
    raw_data = download_comprehensive_data()
    df, feature_cols, team_encoder = engineer_all_features(raw_data)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['Result'].values
    
    logger.info(f"   Loaded {len(X):,} samples with {X.shape[1]} features")
    logger.info(f"   Class distribution: H={sum(y==0)}, D={sum(y==1)}, A={sum(y==2)}")
    
    return X, y, df


def train_stacking_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            quick_mode: bool = False) -> Dict:
    """Train stacking ensemble with all models"""
    from src.models.stacking_ensemble import StackingEnsemble
    
    logger.info("\n" + "="*60)
    logger.info("🏗️ PHASE 1: STACKING ENSEMBLE")
    logger.info("="*60)
    
    # Model configurations
    if quick_mode:
        configs = {
            'q_xgb': {'n_estimators': 200, 'n_rotation_layers': 2},
            'q_lgb': {'n_estimators': 200, 'max_depth': 8},
        }
        n_folds = 3
    else:
        configs = {
            'q_xgb': {'n_estimators': 800, 'n_rotation_layers': 3},
            'q_lgb': {'n_estimators': 800, 'max_depth': 12},
            'q_cat': {'iterations': 800, 'depth': 10},
            'deep_nn': {'hidden_layers': 4, 'hidden_units': 256, 'use_attention': True},
        }
        n_folds = 5
    
    # Combine train/val for stacking (uses internal CV)
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])
    
    ensemble = StackingEnsemble(n_folds=n_folds, confidence_threshold=0.45)
    ensemble.fit(X_full, y_full, model_configs=configs)
    
    # Evaluate on validation set
    val_preds = ensemble.predict(X_val)
    val_acc = (val_preds == y_val).mean()
    
    # Confidence filtering
    preds, confident_mask, confidences = ensemble.predict_with_confidence(X_val)
    confident_acc = (preds[confident_mask] == y_val[confident_mask]).mean() if confident_mask.any() else 0
    
    logger.info(f"\n📊 Stacking Results:")
    logger.info(f"   Overall accuracy: {val_acc:.2%}")
    logger.info(f"   Confident predictions: {confident_mask.sum()}/{len(confident_mask)} ({confident_acc:.2%})")
    
    return {
        'ensemble': ensemble,
        'accuracy': val_acc,
        'confident_accuracy': confident_acc,
        'confident_count': confident_mask.sum(),
    }


def train_calibrated_ensemble(ensemble, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """Add probability calibration to ensemble"""
    from src.models.calibration import IsotonicCalibration, TemperatureScaling, expected_calibration_error
    
    logger.info("\n" + "="*60)
    logger.info("🎯 PHASE 2: PROBABILITY CALIBRATION")
    logger.info("="*60)
    
    # Get validation probabilities
    val_probs = ensemble.predict_proba(X_val)
    
    # ECE before calibration
    ece_before, _, _ = expected_calibration_error(val_probs, y_val)
    
    # Fit temperature scaling
    temp_cal = TemperatureScaling()
    temp_cal.fit(np.log(val_probs + 1e-10), y_val)
    
    # Fit isotonic (split val for fitting)
    val_split = int(len(X_val) * 0.5)
    iso_cal = IsotonicCalibration()
    iso_cal.fit(val_probs[:val_split], y_val[:val_split])
    
    # Calibrate probabilities
    temp_probs = temp_cal.calibrate(val_probs)
    iso_probs = iso_cal.calibrate(val_probs)
    
    ece_temp, _, _ = expected_calibration_error(temp_probs, y_val)
    ece_iso, _, _ = expected_calibration_error(iso_probs, y_val)
    
    # Use best calibrator
    if ece_iso < ece_temp:
        calibrator = iso_cal
        cal_probs = iso_probs
        ece_after = ece_iso
        method = "isotonic"
    else:
        calibrator = temp_cal
        cal_probs = temp_probs
        ece_after = ece_temp
        method = "temperature"
    
    # Accuracy after calibration
    cal_acc = (cal_probs.argmax(axis=1) == y_val).mean()
    
    logger.info(f"📊 Calibration Results:")
    logger.info(f"   ECE: {ece_before:.4f} → {ece_after:.4f} ({method})")
    logger.info(f"   Accuracy: {cal_acc:.2%}")
    
    return {
        'calibrator': calibrator,
        'method': method,
        'ece_before': ece_before,
        'ece_after': ece_after,
    }


def evaluate_final(ensemble, calibrator, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Final evaluation on held-out test set"""
    from src.models.calibration import expected_calibration_error
    
    logger.info("\n" + "="*60)
    logger.info("✅ FINAL EVALUATION")
    logger.info("="*60)
    
    # Get predictions
    raw_probs = ensemble.predict_proba(X_test)
    cal_probs = calibrator.calibrate(raw_probs)
    
    raw_preds = raw_probs.argmax(axis=1)
    cal_preds = cal_probs.argmax(axis=1)
    
    raw_acc = (raw_preds == y_test).mean()
    cal_acc = (cal_preds == y_test).mean()
    
    # Confidence filtering
    preds, confident, confs = ensemble.predict_with_confidence(X_test)
    confident_acc = (preds[confident] == y_test[confident]).mean() if confident.any() else 0
    
    # Per-class accuracy
    for c, name in [(0, 'Home'), (1, 'Draw'), (2, 'Away')]:
        mask = y_test == c
        class_acc = (raw_preds[mask] == y_test[mask]).mean()
        logger.info(f"   {name}: {class_acc:.2%} ({mask.sum()} samples)")
    
    ece, _, _ = expected_calibration_error(cal_probs, y_test)
    
    results = {
        'raw_accuracy': raw_acc,
        'calibrated_accuracy': cal_acc,
        'confident_accuracy': confident_acc,
        'confident_count': confident.sum(),
        'total_samples': len(y_test),
        'ece': ece,
    }
    
    logger.info(f"\n🎯 Final Results:")
    logger.info(f"   Raw accuracy: {raw_acc:.2%}")
    logger.info(f"   Calibrated accuracy: {cal_acc:.2%}")
    logger.info(f"   Confident ({confident.sum()}/{len(y_test)}): {confident_acc:.2%}")
    logger.info(f"   ECE: {ece:.4f}")
    
    return results


def save_models(output_dir: Path, ensemble, calibrator, results: Dict):
    """Save all trained models"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble
    ensemble.save(str(output_dir / 'stacking_ensemble.pkl'))
    
    # Save calibrator
    with open(output_dir / 'calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)
    
    # Save results
    with open(output_dir / 'training_results.json', 'w') as f:
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, float)):
                serializable[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                serializable[k] = int(v)
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)
    
    logger.info(f"\n💾 Models saved to {output_dir}")


def main(quick_mode: bool = False, output_dir: str = './models/advanced'):
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🚀 ADVANCED TRAINING PIPELINE FOR 65-70% ACCURACY")
    print(f"   Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data
    X, y, df = load_and_prepare_data()
    
    # Split: Train (70%) / Val (15%) / Test (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
    )
    
    logger.info(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Phase 1: Stacking Ensemble
    stacking_result = train_stacking_ensemble(X_train, y_train, X_val, y_val, quick_mode)
    
    # Phase 2: Calibration
    cal_result = train_calibrated_ensemble(
        stacking_result['ensemble'], X_train, y_train, X_val, y_val
    )
    
    # Final Evaluation
    final_result = evaluate_final(
        stacking_result['ensemble'], cal_result['calibrator'], X_test, y_test
    )
    
    # Save models
    output_path = Path(output_dir)
    save_models(
        output_path,
        stacking_result['ensemble'],
        cal_result['calibrator'],
        final_result
    )
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING COMPLETE")
    print("="*70)
    print(f"   Final Test Accuracy: {final_result['raw_accuracy']:.2%}")
    print(f"   Confident Accuracy: {final_result['confident_accuracy']:.2%}")
    print(f"   Models saved to: {output_path}")
    print("="*70)
    
    return final_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Football Prediction Training')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer epochs)')
    parser.add_argument('--output', type=str, default='./models/advanced', help='Output directory')
    
    args = parser.parse_args()
    
    main(quick_mode=args.quick, output_dir=args.output)
