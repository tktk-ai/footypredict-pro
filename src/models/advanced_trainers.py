"""
Advanced Trainers Module

Market-specific trainers with:
- Optuna hyperparameter tuning
- Stacking ensemble (XGBoost + LightGBM + CatBoost)
- Probability calibration
- Cross-validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import joblib
import logging
import json

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Base paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "advanced"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class BaseAdvancedTrainer:
    """Base class for advanced trainers with Optuna tuning and calibration"""
    
    def __init__(self, model_name: str, n_trials: int = 100):
        self.model_name = model_name
        self.n_trials = n_trials
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.best_params: Dict[str, Dict] = {}
        self.feature_cols: List[str] = []
        self.label_encoder = LabelEncoder()
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                             model_type: str = 'xgboost') -> Dict:
        """Use Optuna to find optimal hyperparameters"""
        if not HAS_OPTUNA:
            logger.warning("Optuna not installed, using default parameters")
            return self._get_default_params(model_type)
        
        def objective(trial):
            if model_type == 'xgboost' and HAS_XGB:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0)
                
            elif model_type == 'lightgbm' and HAS_LGB:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
                
            elif model_type == 'catboost' and HAS_CB:
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                }
                model = cb.CatBoostClassifier(**params, random_state=42, verbose=0)
            else:
                return 0.5
            
            # Cross-validation score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Best {model_type} accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def _get_default_params(self, model_type: str) -> Dict:
        """Get default hyperparameters"""
        defaults = {
            'xgboost': {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8
            },
            'catboost': {
                'iterations': 200, 'depth': 6, 'learning_rate': 0.1
            }
        }
        return defaults.get(model_type, {})
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, 
                      n_folds: int = 5) -> Dict[str, float]:
        """Train with cross-validation and return metrics"""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        metrics = {'accuracy': [], 'log_loss': []}
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train models
            for model_type in ['xgboost', 'lightgbm', 'catboost']:
                model = self._create_model(model_type)
                if model is not None:
                    model.fit(X_train_scaled, y_train)
                    
                    y_pred = model.predict(X_val_scaled)
                    y_proba = model.predict_proba(X_val_scaled)
                    
                    metrics['accuracy'].append(accuracy_score(y_val, y_pred))
                    metrics['log_loss'].append(log_loss(y_val, y_proba))
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _create_model(self, model_type: str, params: Optional[Dict] = None):
        """Create a model instance"""
        if params is None:
            params = self.best_params.get(model_type, self._get_default_params(model_type))
        
        if model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0)
        elif model_type == 'lightgbm' and HAS_LGB:
            return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        elif model_type == 'catboost' and HAS_CB:
            return cb.CatBoostClassifier(**params, random_state=42, verbose=0)
        return None
    
    def calibrate_probabilities(self, X: np.ndarray, y: np.ndarray) -> None:
        """Apply probability calibration to all models"""
        for model_type, model in self.models.items():
            if model is not None:
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated.fit(X, y)
                self.models[f"{model_type}_calibrated"] = calibrated
                logger.info(f"Calibrated {model_type} model")
    
    def save(self, output_dir: Optional[Path] = None) -> None:
        """Save all models and metadata"""
        output_dir = output_dir or MODELS_DIR / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if model is not None:
                joblib.dump(model, output_dir / f"{name}_model.joblib")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, output_dir / f"{name}_scaler.joblib")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_cols': self.feature_cols,
            'best_params': self.best_params
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved models to {output_dir}")
    
    def load(self, input_dir: Optional[Path] = None) -> bool:
        """Load saved models"""
        input_dir = input_dir or MODELS_DIR / self.model_name
        
        if not input_dir.exists():
            return False
        
        # Load models
        for model_file in input_dir.glob("*_model.joblib"):
            name = model_file.stem.replace("_model", "")
            self.models[name] = joblib.load(model_file)
        
        # Load scalers
        for scaler_file in input_dir.glob("*_scaler.joblib"):
            name = scaler_file.stem.replace("_scaler", "")
            self.scalers[name] = joblib.load(scaler_file)
        
        # Load metadata
        metadata_file = input_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            self.feature_cols = metadata.get('feature_cols', [])
            self.best_params = metadata.get('best_params', {})
        
        logger.info(f"Loaded {len(self.models)} models from {input_dir}")
        return True


class StackingEnsemble(BaseAdvancedTrainer):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, model_name: str = "stacking_ensemble"):
        super().__init__(model_name)
        self.meta_learner = None
        self.base_models: List[Tuple[str, Any]] = []
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              tune: bool = True) -> Dict[str, float]:
        """Train stacking ensemble"""
        logger.info(f"Training stacking ensemble on {len(X)} samples...")
        
        # Split for stacking
        from sklearn.model_selection import train_test_split
        X_train, X_meta, y_train, y_meta = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_train_scaled = self.scalers['main'].fit_transform(X_train)
        X_meta_scaled = self.scalers['main'].transform(X_meta)
        
        # Train and tune base models
        meta_features_train = []
        meta_features_meta = []
        
        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            if tune and HAS_OPTUNA:
                logger.info(f"Tuning {model_type}...")
                self.best_params[model_type] = self.tune_hyperparameters(
                    X_train_scaled, y_train, model_type
                )
            
            model = self._create_model(model_type)
            if model is not None:
                model.fit(X_train_scaled, y_train)
                self.models[model_type] = model
                self.base_models.append((model_type, model))
                
                # Get predictions for meta-learner
                meta_features_train.append(model.predict_proba(X_train_scaled))
                meta_features_meta.append(model.predict_proba(X_meta_scaled))
                
                acc = accuracy_score(y_meta, model.predict(X_meta_scaled))
                logger.info(f"  {model_type} accuracy: {acc:.4f}")
        
        # Stack predictions
        if meta_features_train:
            X_meta_train = np.hstack(meta_features_train)
            X_meta_test = np.hstack(meta_features_meta)
            
            # Train meta-learner (logistic regression for calibration)
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            self.meta_learner.fit(X_meta_train, y_train)
            self.models['meta_learner'] = self.meta_learner
            
            # Final accuracy
            y_pred = self.meta_learner.predict(X_meta_test)
            final_acc = accuracy_score(y_meta, y_pred)
            logger.info(f"Stacking ensemble accuracy: {final_acc:.4f}")
            
            return {'accuracy': final_acc}
        
        return {'accuracy': 0.0}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacking ensemble"""
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X
        
        # Get base model predictions
        meta_features = []
        for model_type, model in self.base_models:
            if model is not None:
                meta_features.append(model.predict_proba(X_scaled))
        
        if meta_features and self.meta_learner is not None:
            X_meta = np.hstack(meta_features)
            return self.meta_learner.predict(X_meta)
        
        # Fallback to first model
        if self.base_models:
            return self.base_models[0][1].predict(X_scaled)
        
        return np.zeros(len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X
        
        # Get base model predictions
        meta_features = []
        for model_type, model in self.base_models:
            if model is not None:
                meta_features.append(model.predict_proba(X_scaled))
        
        if meta_features and self.meta_learner is not None:
            X_meta = np.hstack(meta_features)
            return self.meta_learner.predict_proba(X_meta)
        
        # Fallback to averaging
        if meta_features:
            return np.mean(meta_features, axis=0)
        
        return np.ones((len(X), 3)) / 3


class Result1X2Trainer(StackingEnsemble):
    """Specialized trainer for 1X2 match result prediction"""
    
    def __init__(self):
        super().__init__("result_1x2")
    
    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable (H/D/A -> 0/1/2)"""
        result_col = 'result' if 'result' in df.columns else 'FTR'
        
        if result_col in df.columns:
            mapping = {'H': 0, 'D': 1, 'A': 2}
            return df[result_col].map(mapping).fillna(1).values.astype(int)
        
        return np.ones(len(df), dtype=int)


class GoalsTrainer(StackingEnsemble):
    """Specialized trainer for Over/Under goals prediction"""
    
    def __init__(self, threshold: float = 2.5):
        super().__init__(f"goals_over_{threshold}".replace('.', '_'))
        self.threshold = threshold
    
    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable (Over/Under threshold)"""
        home_goals = df.get('home_goals', df.get('FTHG', 0))
        away_goals = df.get('away_goals', df.get('FTAG', 0))
        
        total_goals = home_goals + away_goals
        return (total_goals > self.threshold).astype(int).values


class BTTSTrainer(StackingEnsemble):
    """Specialized trainer for Both Teams To Score prediction"""
    
    def __init__(self):
        super().__init__("btts")
    
    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable (BTTS yes/no)"""
        home_goals = df.get('home_goals', df.get('FTHG', 0))
        away_goals = df.get('away_goals', df.get('FTAG', 0))
        
        btts = (home_goals > 0) & (away_goals > 0)
        return btts.astype(int).values


# Convenience functions
def train_all_advanced_models(X: np.ndarray, y_result: np.ndarray,
                              y_goals: np.ndarray, y_btts: np.ndarray,
                              tune: bool = True) -> Dict[str, Any]:
    """Train all advanced models"""
    results = {}
    
    # 1X2 Result
    logger.info("Training 1X2 model...")
    result_trainer = Result1X2Trainer()
    results['1x2'] = result_trainer.train(X, y_result, tune=tune)
    result_trainer.save()
    
    # Over 2.5 Goals
    logger.info("Training Over 2.5 model...")
    goals_trainer = GoalsTrainer(2.5)
    results['over_25'] = goals_trainer.train(X, y_goals, tune=tune)
    goals_trainer.save()
    
    # BTTS
    logger.info("Training BTTS model...")
    btts_trainer = BTTSTrainer()
    results['btts'] = btts_trainer.train(X, y_btts, tune=tune)
    btts_trainer.save()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with random data
    print("Testing advanced trainers...")
    
    X = np.random.randn(1000, 100)
    y = np.random.randint(0, 3, 1000)
    
    trainer = StackingEnsemble("test")
    results = trainer.train(X, y, tune=False)
    
    print(f"Training results: {results}")
