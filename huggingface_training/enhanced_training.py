"""
Enhanced Self-Improving Training Pipeline
Walk-forward validation, hyperparameter optimization, and multi-model ensemble
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import joblib
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, f1_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    n_splits: int = 5
    test_size: int = 100
    gap: int = 0
    n_optuna_trials: int = 50
    early_stopping_rounds: int = 50
    calibrate: bool = True
    use_ensemble: bool = True
    random_state: int = 42
    model_dir: str = "models/v4"
    experiment_dir: str = "experiments"


@dataclass
class ExperimentResult:
    """Result from a training experiment."""
    experiment_id: str
    model_name: str
    timestamp: datetime
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    log_loss_val: float
    brier_score: float
    best_params: Dict
    feature_importance: Dict[str, float]
    training_time_seconds: float
    n_features: int
    n_train_samples: int
    suggestions: List[str] = field(default_factory=list)


class ExperimentTracker:
    """Tracks and analyzes training experiments."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[ExperimentResult] = []
        self._load_experiments()
    
    def _load_experiments(self):
        """Load previous experiments."""
        exp_file = self.experiment_dir / "experiments.json"
        if exp_file.exists():
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                self.experiments = [ExperimentResult(**e) for e in data]
            except:
                pass
    
    def save_experiment(self, result: ExperimentResult):
        """Save experiment result."""
        self.experiments.append(result)
        exp_file = self.experiment_dir / "experiments.json"
        
        data = []
        for exp in self.experiments:
            exp_dict = {
                'experiment_id': exp.experiment_id,
                'model_name': exp.model_name,
                'timestamp': exp.timestamp.isoformat() if isinstance(exp.timestamp, datetime) else exp.timestamp,
                'train_accuracy': exp.train_accuracy,
                'val_accuracy': exp.val_accuracy,
                'test_accuracy': exp.test_accuracy,
                'log_loss_val': exp.log_loss_val,
                'brier_score': exp.brier_score,
                'best_params': exp.best_params,
                'feature_importance': exp.feature_importance,
                'training_time_seconds': exp.training_time_seconds,
                'n_features': exp.n_features,
                'n_train_samples': exp.n_train_samples,
                'suggestions': exp.suggestions
            }
            data.append(exp_dict)
        
        with open(exp_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_best_experiment(self, metric: str = 'val_accuracy') -> Optional[ExperimentResult]:
        """Get best experiment by metric."""
        if not self.experiments:
            return None
        
        return max(self.experiments, key=lambda x: getattr(x, metric, 0))
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across experiments."""
        if len(self.experiments) < 3:
            return {}
        
        # Analyze best performing configurations
        sorted_exps = sorted(self.experiments, key=lambda x: x.val_accuracy, reverse=True)
        top_5 = sorted_exps[:5]
        
        # Common patterns in top performers
        common_params = {}
        for exp in top_5:
            for k, v in exp.best_params.items():
                if k not in common_params:
                    common_params[k] = []
                common_params[k].append(v)
        
        return {
            'best_accuracy': top_5[0].val_accuracy if top_5 else 0,
            'avg_top5_accuracy': np.mean([e.val_accuracy for e in top_5]),
            'common_params': {k: np.median(v) if isinstance(v[0], (int, float)) else v[0] 
                            for k, v in common_params.items()},
            'n_experiments': len(self.experiments)
        }


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def optimize_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_classes: int = 3
    ) -> Dict:
        """Optimize XGBoost hyperparameters."""
        if not HAS_OPTUNA:
            return self._default_xgb_params(n_classes)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
                'num_class': n_classes if n_classes > 2 else None,
                'random_state': self.config.random_state,
                'n_jobs': -1
            }
            
            if params['num_class'] is None:
                del params['num_class']
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = model.predict(X_val)
            return accuracy_score(y_val, preds)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['objective'] = 'multi:softprob' if n_classes > 2 else 'binary:logistic'
        if n_classes > 2:
            best_params['num_class'] = n_classes
        best_params['random_state'] = self.config.random_state
        best_params['n_jobs'] = -1
        
        return best_params
    
    def optimize_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_classes: int = 3
    ) -> Dict:
        """Optimize LightGBM hyperparameters."""
        if not HAS_OPTUNA:
            return self._default_lgb_params(n_classes)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'objective': 'multiclass' if n_classes > 2 else 'binary',
                'num_class': n_classes if n_classes > 2 else None,
                'random_state': self.config.random_state,
                'n_jobs': -1,
                'verbose': -1
            }
            
            if params['num_class'] is None:
                del params['num_class']
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
            
            preds = model.predict(X_val)
            return accuracy_score(y_val, preds)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['objective'] = 'multiclass' if n_classes > 2 else 'binary'
        if n_classes > 2:
            best_params['num_class'] = n_classes
        best_params['random_state'] = self.config.random_state
        best_params['n_jobs'] = -1
        best_params['verbose'] = -1
        
        return best_params
    
    def _default_xgb_params(self, n_classes: int) -> Dict:
        """Default XGBoost parameters."""
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.config.random_state,
            'n_jobs': -1
        }
        if n_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
        return params
    
    def _default_lgb_params(self, n_classes: int) -> Dict:
        """Default LightGBM parameters."""
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 50,
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        if n_classes > 2:
            params['objective'] = 'multiclass'
            params['num_class'] = n_classes
        return params


class AdvancedEnsemble:
    """Multi-model ensemble with stacking."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """Fit ensemble with stacking."""
        logger.info("Training ensemble models...")
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        n_classes = len(self.label_encoder.classes_)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Optimize and train base models
        optimizer = HyperparameterOptimizer(self.config)
        
        # XGBoost
        logger.info("  Training XGBoost...")
        xgb_params = optimizer.optimize_xgboost(
            X_train_scaled, y_train_enc, X_val_scaled, y_val_enc, n_classes
        )
        self.base_models['xgb'] = xgb.XGBClassifier(**xgb_params)
        self.base_models['xgb'].fit(X_train_scaled, y_train_enc)
        
        # LightGBM
        logger.info("  Training LightGBM...")
        lgb_params = optimizer.optimize_lightgbm(
            X_train_scaled, y_train_enc, X_val_scaled, y_val_enc, n_classes
        )
        self.base_models['lgb'] = lgb.LGBMClassifier(**lgb_params)
        self.base_models['lgb'].fit(X_train_scaled, y_train_enc)
        
        # CatBoost if available
        if HAS_CATBOOST:
            logger.info("  Training CatBoost...")
            cb_params = {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.1,
                'loss_function': 'MultiClass' if n_classes > 2 else 'Logloss',
                'random_state': self.config.random_state,
                'verbose': False
            }
            self.base_models['catboost'] = CatBoostClassifier(**cb_params)
            self.base_models['catboost'].fit(X_train_scaled, y_train_enc)
        
        # Generate meta-features
        meta_train = self._get_meta_features(X_train_scaled)
        meta_val = self._get_meta_features(X_val_scaled)
        
        # Train meta-model (simple LightGBM)
        logger.info("  Training meta-learner...")
        self.meta_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=self.config.random_state,
            verbose=-1
        )
        self.meta_model.fit(meta_train, y_train_enc)
        
        # Evaluate
        val_preds = self.predict(X_val)
        val_proba = self.predict_proba(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_preds),
            'log_loss': log_loss(y_val_enc, val_proba),
        }
        
        logger.info(f"  Ensemble validation accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        meta_features = []
        
        for name, model in self.base_models.items():
            proba = model.predict_proba(X)
            meta_features.append(proba)
        
        return np.hstack(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        meta_features = self._get_meta_features(X_scaled)
        preds_enc = self.meta_model.predict(meta_features)
        return self.label_encoder.inverse_transform(preds_enc)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        X_scaled = self.scaler.transform(X)
        meta_features = self._get_meta_features(X_scaled)
        return self.meta_model.predict_proba(meta_features)
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Get aggregated feature importance."""
        importance = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    feat_name = feature_names[i] if feature_names else f"feature_{i}"
                    importance[feat_name] = importance.get(feat_name, 0) + imp
        
        # Normalize
        total = sum(importance.values()) or 1
        importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50])


class SelfImprovingTrainer:
    """Self-improving trainer with experiment tracking and suggestions."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.tracker = ExperimentTracker(self.config.experiment_dir)
        self.model_dir = Path(self.config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None,
        target_type: str = 'result'  # result, btts, over25, etc.
    ) -> Tuple[AdvancedEnsemble, ExperimentResult]:
        """Train with walk-forward validation and generate suggestions."""
        import time
        start_time = time.time()
        
        experiment_id = f"{target_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Convert to numpy
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        feature_names = feature_names or (list(X.columns) if isinstance(X, pd.DataFrame) else None)
        
        # Walk-forward split
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits, test_size=self.config.test_size)
        
        cv_scores = []
        best_ensemble = None
        best_score = 0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            logger.info(f"  Fold {fold + 1}/{self.config.n_splits}")
            
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
            
            # Train ensemble
            ensemble = AdvancedEnsemble(self.config)
            metrics = ensemble.fit(X_train, y_train, X_val, y_val, feature_names)
            
            cv_scores.append(metrics['accuracy'])
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_ensemble = ensemble
        
        # Calculate final metrics
        train_time = time.time() - start_time
        
        # Final test evaluation (last fold)
        test_preds = best_ensemble.predict(X_val)
        test_proba = best_ensemble.predict_proba(X_val)
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_name=f"ensemble_{target_type}",
            timestamp=datetime.now(),
            train_accuracy=np.mean(cv_scores[:-1]) if len(cv_scores) > 1 else cv_scores[0],
            val_accuracy=np.mean(cv_scores),
            test_accuracy=accuracy_score(y_val, test_preds),
            log_loss_val=log_loss(
                LabelEncoder().fit_transform(y_val),
                test_proba
            ),
            brier_score=0.0,  # Multi-class doesn't use brier
            best_params={},
            feature_importance=best_ensemble.get_feature_importance(feature_names),
            training_time_seconds=train_time,
            n_features=X_arr.shape[1],
            n_train_samples=len(train_idx),
            suggestions=self._generate_suggestions(cv_scores, X_arr.shape[1])
        )
        
        # Save experiment
        self.tracker.save_experiment(result)
        
        # Save model
        model_path = self.model_dir / f"{experiment_id}.joblib"
        joblib.dump(best_ensemble, model_path)
        logger.info(f"Model saved to {model_path}")
        
        logger.info(f"Training complete! CV Accuracy: {result.val_accuracy:.4f}")
        
        return best_ensemble, result
    
    def _generate_suggestions(self, cv_scores: List[float], n_features: int) -> List[str]:
        """Generate improvement suggestions based on experiment."""
        suggestions = []
        
        avg_score = np.mean(cv_scores)
        score_std = np.std(cv_scores)
        
        # High variance suggestions
        if score_std > 0.05:
            suggestions.append(
                "High variance in CV scores ({:.3f}). Consider: "
                "1) More regularization, 2) Larger training set, 3) Feature selection".format(score_std)
            )
        
        # Low accuracy suggestions
        if avg_score < 0.55:
            suggestions.append(
                "Low accuracy ({:.3f}). Consider: "
                "1) More features, 2) Different model architecture, 3) Check data quality".format(avg_score)
            )
        
        # Feature count suggestions
        if n_features > 500:
            suggestions.append(
                f"High feature count ({n_features}). Consider feature selection or PCA."
            )
        elif n_features < 50:
            suggestions.append(
                f"Low feature count ({n_features}). Consider adding more engineered features."
            )
        
        # Compare with previous experiments
        patterns = self.tracker.analyze_patterns()
        if patterns and patterns.get('best_accuracy', 0) > avg_score + 0.02:
            suggestions.append(
                f"Previous best: {patterns['best_accuracy']:.4f}. "
                "Review best performing configuration parameters."
            )
        
        if not suggestions:
            suggestions.append("Performance looks good! Consider ensemble diversity or calibration.")
        
        return suggestions
    
    def get_improvement_recommendations(self) -> Dict[str, Any]:
        """Get overall improvement recommendations."""
        patterns = self.tracker.analyze_patterns()
        
        recommendations = {
            'experiment_count': patterns.get('n_experiments', 0),
            'best_accuracy': patterns.get('best_accuracy', 0),
            'avg_top5_accuracy': patterns.get('avg_top5_accuracy', 0),
            'recommended_params': patterns.get('common_params', {}),
            'actions': []
        }
        
        if patterns.get('n_experiments', 0) < 5:
            recommendations['actions'].append("Run more experiments to establish baseline")
        
        if patterns.get('best_accuracy', 0) < 0.6:
            recommendations['actions'].append("Focus on feature engineering")
            recommendations['actions'].append("Consider adding external data sources")
        
        return recommendations


class EnhancedTrainingPipeline:
    """Complete training pipeline with all V4.0 features."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.trainer = SelfImprovingTrainer(self.config)
        self.models = {}
    
    def train_all_markets(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        markets: List[str] = None
    ) -> Dict[str, ExperimentResult]:
        """Train models for all markets."""
        markets = markets or ['result', 'btts', 'over25']
        results = {}
        
        for market in markets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {market.upper()} model")
            logger.info(f"{'='*50}")
            
            # Get target for market
            y = self._get_target(data, market)
            
            if y is None or len(y) == 0:
                logger.warning(f"No target for {market}, skipping")
                continue
            
            # Align features and target
            valid_idx = ~y.isna()
            X = features[valid_idx]
            y = y[valid_idx]
            
            # Train
            model, result = self.trainer.train(
                X, y,
                feature_names=list(features.columns),
                target_type=market
            )
            
            self.models[market] = model
            results[market] = result
        
        return results
    
    def _get_target(self, data: pd.DataFrame, market: str) -> pd.Series:
        """Get target variable for market."""
        if market == 'result':
            # Convert result to H/D/A
            if 'result' in data.columns:
                return data['result']
            elif 'home_goals' in data.columns:
                return pd.Series(np.where(
                    data['home_goals'] > data['away_goals'], 'H',
                    np.where(data['home_goals'] < data['away_goals'], 'A', 'D')
                ))
        
        elif market == 'btts':
            if 'home_goals' in data.columns:
                return ((data['home_goals'] > 0) & (data['away_goals'] > 0)).astype(int)
        
        elif market == 'over25':
            if 'home_goals' in data.columns:
                return ((data['home_goals'] + data['away_goals']) > 2.5).astype(int)
        
        elif market == 'over15':
            if 'home_goals' in data.columns:
                return ((data['home_goals'] + data['away_goals']) > 1.5).astype(int)
        
        return None
    
    def predict(self, features: pd.DataFrame, market: str = 'result') -> Dict[str, Any]:
        """Make predictions with uncertainty."""
        if market not in self.models:
            raise ValueError(f"No model for market: {market}")
        
        model = self.models[market]
        
        proba = model.predict_proba(features.values)
        pred = model.predict(features.values)
        
        # Uncertainty quantification
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        max_conf = np.max(proba, axis=1)
        
        return {
            'prediction': pred.tolist(),
            'probabilities': proba.tolist(),
            'confidence': max_conf.tolist(),
            'uncertainty': entropy.tolist()
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED TRAINING PIPELINE - TEST")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice(['H', 'D', 'A'], n_samples))
    
    # Test trainer
    config = TrainingConfig(n_optuna_trials=5, n_splits=3)
    trainer = SelfImprovingTrainer(config)
    
    print("\n1. Testing SelfImprovingTrainer...")
    model, result = trainer.train(X, y, target_type='test')
    
    print(f"\n✅ Training complete!")
    print(f"   Accuracy: {result.val_accuracy:.4f}")
    print(f"   Training time: {result.training_time_seconds:.1f}s")
    print(f"   Suggestions: {result.suggestions}")
