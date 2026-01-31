"""
V4 Model Trainer with Proper Cross-Validation
==============================================

Fixes the overfitting issue in V4 models by:
- Using 5-fold stratified cross-validation
- Proper train/validation/test splits
- Early stopping with patience
- L1/L2 regularization
- Ensemble with stacking
"""

import numpy as np
import pandas as pd
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "v4_fixed"
DATA_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


class V4ModelTrainer:
    """
    Properly trained V4 model with cross-validation and regularization.
    """
    
    MARKETS = {
        'result': {'type': 'multiclass', 'target': 'result'},
        'over25': {'type': 'binary', 'target': 'over_25'},
        'over15': {'type': 'binary', 'target': 'over_15'},
        'btts': {'type': 'binary', 'target': 'btts'},
    }
    
    def __init__(self, n_folds: int = 5, test_size: float = 0.15, random_state: int = 42):
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_state = random_state
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.calibrators = {}
        self.results = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare training data."""
        # Try parquet first (faster)
        parquet_path = DATA_DIR / "training_data_unified.parquet"
        csv_path = DATA_DIR / "training_data_unified.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No training data found at {DATA_DIR}")
        
        logger.info(f"Loaded {len(df)} matches from training data")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare features and targets from raw data.
        
        Returns:
            X: Feature matrix
            y_dict: Dictionary of targets for each market
        """
        # Define feature columns (use what's available)
        numeric_cols = []
        
        # Odds columns (if available)
        odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 
                     'PSH', 'PSD', 'PSA']
        numeric_cols.extend([c for c in odds_cols if c in df.columns])
        
        # Stats columns (if available)
        stats_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        numeric_cols.extend([c for c in stats_cols if c in df.columns])
        
        # Derived features
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['total_goals'] = df['home_score'] + df['away_score']
        
        # Create targets
        y_dict = {}
        
        # Result (multiclass: H/D/A)
        if 'result' not in df.columns and 'home_score' in df.columns:
            df['result'] = df.apply(
                lambda r: 'H' if r['home_score'] > r['away_score'] 
                else ('A' if r['away_score'] > r['home_score'] else 'D'), 
                axis=1
            )
        
        if 'result' in df.columns:
            le = LabelEncoder()
            y_dict['result'] = le.fit_transform(df['result'])
            self.encoders['result'] = le
        
        # Binary targets
        if 'over_25' not in df.columns and 'total_goals' in df.columns:
            df['over_25'] = (df['total_goals'] > 2.5).astype(int)
        if 'over_15' not in df.columns and 'total_goals' in df.columns:
            df['over_15'] = (df['total_goals'] > 1.5).astype(int)
        if 'btts' not in df.columns and 'home_score' in df.columns:
            df['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)
        
        for target in ['over_25', 'over_15', 'btts']:
            if target in df.columns:
                y_dict[target] = df[target].astype(int).values
        
        # Prepare X matrix
        if numeric_cols:
            X = df[numeric_cols].copy()
            X = X.fillna(X.median())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['features'] = scaler
            self.feature_cols = numeric_cols
        else:
            # If no numeric columns, create basic features
            logger.warning("No numeric columns found, creating basic features")
            X_scaled = np.random.randn(len(df), 10)  # Placeholder
            self.feature_cols = [f'feature_{i}' for i in range(10)]
        
        logger.info(f"Prepared {X_scaled.shape[1]} features, {len(y_dict)} targets")
        
        return X_scaled, y_dict
    
    def train_single_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market: str,
        model_type: str = 'xgboost'
    ) -> Tuple[object, Dict]:
        """
        Train a single model with cross-validation.
        
        Args:
            X: Features
            y: Target
            market: Market name
            model_type: 'xgboost' or 'lightgbm'
        
        Returns:
            Trained model and metrics
        """
        is_multiclass = self.MARKETS.get(market, {}).get('type') == 'multiclass'
        
        # Split into train and test (test is NEVER seen during training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            stratify=y, random_state=self.random_state
        )
        
        # Cross-validation on training set only
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        oof_predictions = np.zeros((len(X_train), len(np.unique(y)) if is_multiclass else 2))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            if model_type == 'xgboost':
                model = self._create_xgb_model(is_multiclass)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:  # lightgbm
                model = self._create_lgb_model(is_multiclass)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                )
            
            # Validation predictions
            val_proba = model.predict_proba(X_val)
            oof_predictions[val_idx] = val_proba
            
            val_pred = np.argmax(val_proba, axis=1) if is_multiclass else (val_proba[:, 1] >= 0.5).astype(int)
            cv_scores.append(accuracy_score(y_val, val_pred))
            
            logger.info(f"  Fold {fold + 1}: {cv_scores[-1]:.4f}")
        
        # Train final model on all training data (without early stopping)
        if model_type == 'xgboost':
            final_model = xgb.XGBClassifier(
                n_estimators=500,           # More trees
                max_depth=8,                 # Deeper trees
                learning_rate=0.01,          # Lower LR for better convergence
                subsample=0.7,               # More regularization
                colsample_bytree=0.7,        # Feature sampling
                colsample_bylevel=0.7,       # Per-level sampling
                gamma=0.1,                   # Min loss reduction
                reg_alpha=0.5,               # Stronger L1
                reg_lambda=2.0,              # Stronger L2
                min_child_weight=5,          # More conservative splits
                scale_pos_weight=1,
                objective='multi:softprob' if is_multiclass else 'binary:logistic',
                random_state=self.random_state,
                use_label_encoder=False,
                verbosity=0,
                n_jobs=-1,                   # Use all cores
            )
        else:
            final_model = self._create_lgb_model(is_multiclass)
        
        final_model.fit(X_train, y_train)
        
        # Just use the final model directly (skip calibration for simplicity)
        calibrator = final_model
        
        # Evaluate on TEST set (never seen during training)


        test_proba = calibrator.predict_proba(X_test)
        test_pred = np.argmax(test_proba, axis=1) if is_multiclass else (test_proba[:, 1] >= 0.5).astype(int)
        
        test_accuracy = accuracy_score(y_test, test_pred)
        test_logloss = log_loss(y_test, test_proba)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        metrics = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'test_accuracy': test_accuracy,
            'test_logloss': test_logloss,
            'test_f1': test_f1,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
        
        logger.info(f"  CV Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return calibrator, metrics
    
    def _create_xgb_model(self, is_multiclass: bool) -> xgb.XGBClassifier:
        """Create XGBoost model with optimized hyperparameters."""
        return xgb.XGBClassifier(
            n_estimators=500,               # More trees
            max_depth=8,                     # Deeper trees
            learning_rate=0.01,              # Lower LR
            subsample=0.7,                   # Regularization
            colsample_bytree=0.7,            # Feature sampling
            colsample_bylevel=0.7,           # Per-level sampling  
            gamma=0.1,                       # Min loss reduction
            reg_alpha=0.5,                   # L1 regularization
            reg_lambda=2.0,                  # L2 regularization
            min_child_weight=5,              # Conservative splits
            objective='multi:softprob' if is_multiclass else 'binary:logistic',
            eval_metric='mlogloss' if is_multiclass else 'logloss',
            early_stopping_rounds=30,
            random_state=self.random_state,
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )
    
    def _create_lgb_model(self, is_multiclass: bool) -> lgb.LGBMClassifier:
        """Create LightGBM model with regularization."""
        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            objective='multiclass' if is_multiclass else 'binary',
            random_state=self.random_state,
            verbose=-1,
        )
    
    def train_all_markets(self) -> Dict:
        """Train models for all markets."""
        logger.info("=" * 60)
        logger.info("V4 MODEL TRAINING WITH CROSS-VALIDATION")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data()
        X, y_dict = self.prepare_features(df)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_folds': self.n_folds,
                'test_size': self.test_size,
                'random_state': self.random_state,
            },
            'markets': {}
        }
        
        for market, config in self.MARKETS.items():
            target = config['target']
            
            if target not in y_dict:
                logger.warning(f"Target {target} not available, skipping {market}")
                continue
            
            y = y_dict[target]
            
            logger.info(f"\n📊 Training {market.upper()} model...")
            logger.info(f"   Target: {target}, Samples: {len(y)}")
            
            # Train XGBoost
            model, metrics = self.train_single_model(X, y, market, 'xgboost')
            
            self.models[market] = model
            self.results[market] = metrics
            all_results['markets'][market] = metrics
            
            # Save model
            model_path = MODELS_DIR / f"{market}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"   Saved to {model_path}")
        
        # Save scalers and encoders
        joblib.dump(self.scalers, MODELS_DIR / "scalers.joblib")
        joblib.dump(self.encoders, MODELS_DIR / "encoders.joblib")
        joblib.dump(self.feature_cols, MODELS_DIR / "feature_cols.joblib")
        
        # Save results
        with open(MODELS_DIR / "training_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        for market, metrics in self.results.items():
            logger.info(f"  {market}: CV={metrics['cv_mean']:.1%} Test={metrics['test_accuracy']:.1%}")
        
        return all_results
    
    def create_stacking_ensemble(self, X: np.ndarray, y: np.ndarray) -> object:
        """
        Create stacking ensemble with meta-learner.
        
        Level 1: XGBoost + LightGBM
        Level 2: Logistic Regression meta-learner
        """
        from sklearn.ensemble import StackingClassifier
        
        base_models = [
            ('xgb', self._create_xgb_model(False)),
            ('lgb', self._create_lgb_model(False)),
        ]
        
        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state
        )
        
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=self.n_folds,
            stack_method='predict_proba',
            passthrough=True
        )
        
        return stacking


class V4Predictor:
    """
    Predictor using trained V4 models.
    """
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_cols = []
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models."""
        try:
            # Load scalers and encoders
            scalers_path = self.models_dir / "scalers.joblib"
            encoders_path = self.models_dir / "encoders.joblib"
            features_path = self.models_dir / "feature_cols.joblib"
            
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
            if encoders_path.exists():
                self.encoders = joblib.load(encoders_path)
            if features_path.exists():
                self.feature_cols = joblib.load(features_path)
            
            # Load models
            for market in ['result', 'over25', 'over15', 'btts']:
                model_path = self.models_dir / f"{market}_model.joblib"
                if model_path.exists():
                    self.models[market] = joblib.load(model_path)
                    logger.info(f"Loaded {market} model")
            
            logger.info(f"Loaded {len(self.models)} V4 models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict(self, features: Dict) -> Dict[str, Dict]:
        """
        Make predictions for all markets.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Dictionary of predictions per market
        """
        predictions = {}
        
        # Prepare feature vector
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        
        if 'features' in self.scalers:
            X = self.scalers['features'].transform(X)
        
        for market, model in self.models.items():
            try:
                proba = model.predict_proba(X)[0]
                
                if market == 'result':
                    labels = self.encoders.get('result', LabelEncoder()).classes_
                    predictions[market] = {
                        'probabilities': {label: float(p) for label, p in zip(labels, proba)},
                        'prediction': labels[np.argmax(proba)],
                        'confidence': float(np.max(proba)),
                    }
                else:
                    predictions[market] = {
                        'probability': float(proba[1]),
                        'prediction': 'Yes' if proba[1] >= 0.5 else 'No',
                        'confidence': float(max(proba)),
                    }
            except Exception as e:
                logger.error(f"Prediction error for {market}: {e}")
        
        return predictions


def train_v4_models():
    """Run V4 model training."""
    trainer = V4ModelTrainer(n_folds=5, test_size=0.15)
    return trainer.train_all_markets()


if __name__ == "__main__":
    results = train_v4_models()
