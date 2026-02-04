"""
Enhanced Ensemble Training Pipeline
====================================

Advanced training with multiple model architectures for SportyBet predictions.

Models Used:
- XGBoost (Gradient Boosting)
- LightGBM (Fast Gradient Boosting)
- CatBoost (Categorical Boosting)
- Random Forest (Bagging)
- Neural Network (MLP)

Ensemble Methods:
- Stacking
- Weighted Average
- Voting
"""

import sys
import os
import json
import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = project_root / "models" / "sportybet_ensemble"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class EnsembleModel:
    """Ensemble model combining multiple classifiers."""
    
    def __init__(self, market_name: str, use_gpu: bool = False):
        self.market_name = market_name
        self.use_gpu = use_gpu
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.models = {}
        self.ensemble = None
        self.metrics = {}
        self.best_model = None
        
    def _create_base_models(self) -> Dict:
        """Create base model instances."""
        models = {}
        
        # XGBoost
        try:
            from xgboost import XGBClassifier
            models['xgboost'] = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        except ImportError:
            logger.warning("XGBoost not available")
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            models['lightgbm'] = LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        except ImportError:
            logger.warning("LightGBM not available")
        
        # CatBoost
        try:
            from catboost import CatBoostClassifier
            models['catboost'] = CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=False
            )
        except ImportError:
            logger.warning("CatBoost not available")
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # MLP Neural Network
        models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20
        )
        
        return models
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              ensemble_method: str = 'stacking') -> Dict:
        """
        Train ensemble of models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            ensemble_method: 'stacking', 'voting', or 'weighted'
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Training {self.market_name} ensemble ({ensemble_method})...")
        
        # Prepare features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_cols
        
        # Handle missing values
        X_clean = X[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train base models
        base_models = self._create_base_models()
        model_scores = {}
        
        for name, model in base_models.items():
            try:
                logger.info(f"  Training {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                acc = accuracy_score(y_val, y_pred)
                model_scores[name] = acc
                self.models[name] = model
                logger.info(f"    {name} accuracy: {acc:.4f}")
            except Exception as e:
                logger.warning(f"    {name} failed: {e}")
        
        if not self.models:
            raise ValueError("No models trained successfully")
        
        # Find best individual model
        self.best_model = max(model_scores, key=model_scores.get)
        logger.info(f"  Best single model: {self.best_model} ({model_scores[self.best_model]:.4f})")
        
        # Create ensemble
        if ensemble_method == 'stacking' and len(self.models) >= 2:
            estimators = [(name, model) for name, model in self.models.items() 
                         if name != 'mlp']  # MLP can be slow for stacking
            
            self.ensemble = StackingClassifier(
                estimators=estimators[:4],  # Max 4 for speed
                final_estimator=LogisticRegression(max_iter=500),
                cv=3,
                n_jobs=-1
            )
            
            try:
                logger.info("  Training stacking ensemble...")
                self.ensemble.fit(X_train_scaled, y_train)
                y_pred_ensemble = self.ensemble.predict(X_val_scaled)
                ensemble_acc = accuracy_score(y_val, y_pred_ensemble)
                model_scores['ensemble'] = ensemble_acc
                logger.info(f"    Stacking ensemble accuracy: {ensemble_acc:.4f}")
            except Exception as e:
                logger.warning(f"  Stacking failed, using best model: {e}")
                self.ensemble = self.models[self.best_model]
        
        elif ensemble_method == 'voting' and len(self.models) >= 2:
            estimators = [(name, model) for name, model in self.models.items()]
            
            self.ensemble = VotingClassifier(
                estimators=estimators[:4],
                voting='soft',
                n_jobs=-1
            )
            
            try:
                logger.info("  Training voting ensemble...")
                self.ensemble.fit(X_train_scaled, y_train)
                y_pred_ensemble = self.ensemble.predict(X_val_scaled)
                ensemble_acc = accuracy_score(y_val, y_pred_ensemble)
                model_scores['ensemble'] = ensemble_acc
                logger.info(f"    Voting ensemble accuracy: {ensemble_acc:.4f}")
            except Exception as e:
                logger.warning(f"  Voting failed, using best model: {e}")
                self.ensemble = self.models[self.best_model]
        else:
            self.ensemble = self.models[self.best_model]
        
        # Final evaluation
        y_pred_final = self.ensemble.predict(X_val_scaled)
        y_pred_proba = None
        
        if hasattr(self.ensemble, 'predict_proba'):
            y_pred_proba = self.ensemble.predict_proba(X_val_scaled)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred_final),
            'precision': precision_score(y_val, y_pred_final, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred_final, average='weighted', zero_division=0),
            'f1': f1_score(y_val, y_pred_final, average='weighted', zero_division=0),
            'individual_models': model_scores,
            'best_single_model': self.best_model,
            'ensemble_method': ensemble_method,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_features': len(self.feature_columns),
            'trained_at': datetime.now().isoformat()
        }
        
        # ROC-AUC for binary classification
        if len(np.unique(y)) == 2 and y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba[:, 1])
        
        logger.info(f"  Final ensemble accuracy: {self.metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.ensemble is None:
            raise ValueError("Model not trained")
        
        X_clean = X[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_clean)
        return self.ensemble.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        if self.ensemble is None:
            raise ValueError("Model not trained")
        
        X_clean = X[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_clean)
        
        if hasattr(self.ensemble, 'predict_proba'):
            return self.ensemble.predict_proba(X_scaled)
        else:
            # Return one-hot if no proba
            preds = self.ensemble.predict(X_scaled)
            n_classes = len(np.unique(preds))
            proba = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                proba[i, int(p)] = 1.0
            return proba
    
    def save(self, path: Path = None):
        """Save ensemble model."""
        if path is None:
            path = MODELS_DIR / f"{self.market_name}_ensemble.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'models': self.models,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'metrics': self.metrics,
                'best_model': self.best_model,
                'market_name': self.market_name
            }, f)
        
        logger.info(f"Saved {self.market_name} ensemble to {path}")
    
    def load(self, path: Path = None):
        """Load ensemble model."""
        if path is None:
            path = MODELS_DIR / f"{self.market_name}_ensemble.pkl"
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.ensemble = data['ensemble']
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.metrics = data['metrics']
        self.best_model = data['best_model']
        
        logger.info(f"Loaded {self.market_name} ensemble from {path}")


class EnhancedTrainingPipeline:
    """Enhanced training pipeline with advanced features."""
    
    MARKETS = {
        'result_1x2': {'type': 'multiclass', 'target': 'result'},
        'over_under_25': {'type': 'binary', 'target': 'over_25'},
        'btts': {'type': 'binary', 'target': 'btts'},
        'double_chance_1x': {'type': 'binary', 'target': 'dc_1x'},
        'double_chance_x2': {'type': 'binary', 'target': 'dc_x2'},
        'over_under_15': {'type': 'binary', 'target': 'over_15'},
        'first_half_over_05': {'type': 'binary', 'target': 'ht_over_05'},
        'home_win': {'type': 'binary', 'target': 'home_win'},
        'away_win': {'type': 'binary', 'target': 'away_win'},
    }
    
    def __init__(self):
        self.data_dir = project_root / "data"
        self.models = {}
        self.results = {}
    
    def run_full_training(self, use_advanced_features: bool = True) -> Dict:
        """Run complete training pipeline."""
        logger.info("="*60)
        logger.info("Enhanced Ensemble Training Pipeline")
        logger.info("="*60)
        
        results = {'started_at': datetime.now().isoformat(), 'markets': {}}
        
        # Step 1: Load and prepare data
        logger.info("\n📊 Loading training data...")
        try:
            df = self._load_data()
            logger.info(f"   Loaded {len(df)} samples")
        except Exception as e:
            logger.error(f"   Failed to load data: {e}")
            return {'error': str(e)}
        
        # Step 2: Generate features
        logger.info("\n🔧 Generating advanced features...")
        try:
            if use_advanced_features:
                features_df = self._generate_advanced_features(df)
            else:
                features_df = df
            logger.info(f"   Generated {len(features_df.columns)} features")
        except Exception as e:
            logger.error(f"   Feature generation failed: {e}")
            features_df = df
        
        # Step 3: Train models for each market
        logger.info("\n🤖 Training ensemble models...")
        
        for market_name, config in self.MARKETS.items():
            try:
                logger.info(f"\n  Training: {market_name}")
                
                # Prepare target
                target_df = self._prepare_target(features_df, config['target'])
                if target_df is None:
                    logger.warning(f"    Skipping - target not available")
                    continue
                
                X = target_df.drop(columns=[config['target']])
                y = target_df[config['target']]
                
                # Train ensemble
                model = EnsembleModel(market_name)
                metrics = model.train(X, y, ensemble_method='stacking')
                model.save()
                
                self.models[market_name] = model
                results['markets'][market_name] = metrics
                
            except Exception as e:
                logger.error(f"    {market_name} failed: {e}")
                results['markets'][market_name] = {'error': str(e)}
        
        # Step 4: Save results
        results['completed_at'] = datetime.now().isoformat()
        results_path = MODELS_DIR / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\n" + "="*60)
        logger.info("✅ Training Complete!")
        logger.info("="*60)
        
        # Print summary
        print("\n📊 Training Summary:")
        for market, res in results['markets'].items():
            if 'error' not in res:
                print(f"  {market}: {res['accuracy']*100:.1f}% accuracy")
            else:
                print(f"  {market}: FAILED")
        
        return results
    
    def _load_data(self) -> pd.DataFrame:
        """Load training data from various sources."""
        data_files = [
            self.data_dir / "training_data_200k.csv",
            self.data_dir / "comprehensive_training_data.csv",
            self.data_dir / "training_data.csv",
        ]
        
        for path in data_files:
            if path.exists():
                return pd.read_csv(path)
        
        # Try SportyBet data
        sportybet_dir = self.data_dir / "sportybet"
        if sportybet_dir.exists():
            csv_files = list(sportybet_dir.glob("*.csv"))
            if csv_files:
                dfs = [pd.read_csv(f) for f in csv_files]
                return pd.concat(dfs, ignore_index=True)
        
        raise FileNotFoundError("No training data found")
    
    def _generate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced features for training data."""
        # If odds columns exist, generate additional features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add derived features if not present
        if 'odds_home' in df.columns and 'prob_home_fair' not in df.columns:
            home = df['odds_home'].replace(0, np.nan)
            draw = df['odds_draw'].replace(0, np.nan) if 'odds_draw' in df.columns else 3.3
            away = df['odds_away'].replace(0, np.nan) if 'odds_away' in df.columns else 3.5
            
            if 'odds_draw' in df.columns and 'odds_away' in df.columns:
                overround = 1/home + 1/draw + 1/away
                df['prob_home_fair'] = (1/home) / overround
                df['prob_draw_fair'] = (1/draw) / overround
                df['prob_away_fair'] = (1/away) / overround
                df['odds_margin'] = overround - 1
        
        return df
    
    def _prepare_target(self, df: pd.DataFrame, target: str) -> Optional[pd.DataFrame]:
        """Prepare target variable for training."""
        df = df.copy()
        
        # Normalize column names for goals
        if 'home_score' in df.columns and 'home_goals' not in df.columns:
            df['home_goals'] = pd.to_numeric(df['home_score'], errors='coerce')
        if 'away_score' in df.columns and 'away_goals' not in df.columns:
            df['away_goals'] = pd.to_numeric(df['away_score'], errors='coerce')
        
        # Also check for FTHG/FTAG (football-data.co.uk format)
        if 'FTHG' in df.columns and 'home_goals' not in df.columns:
            df['home_goals'] = pd.to_numeric(df['FTHG'], errors='coerce')
        if 'FTAG' in df.columns and 'away_goals' not in df.columns:
            df['away_goals'] = pd.to_numeric(df['FTAG'], errors='coerce')
        
        # Filter to rows with valid goals data
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            valid_mask = df['home_goals'].notna() & df['away_goals'].notna()
            df = df[valid_mask].copy()
            logger.info(f"    Valid samples with results: {len(df)}")
        else:
            logger.warning(f"    No goal columns found")
            return None
        
        if len(df) < 100:
            logger.warning(f"    Insufficient data: {len(df)} rows")
            return None
        
        hg = df['home_goals']
        ag = df['away_goals']
        
        # Create target
        if target == 'result':
            df[target] = np.where(hg > ag, 2, np.where(hg == ag, 1, 0))
        elif target == 'over_25':
            df[target] = ((hg + ag) > 2.5).astype(int)
        elif target == 'over_15':
            df[target] = ((hg + ag) > 1.5).astype(int)
        elif target == 'btts':
            df[target] = ((hg > 0) & (ag > 0)).astype(int)
        elif target == 'dc_1x':
            df[target] = (hg >= ag).astype(int)
        elif target == 'dc_x2':
            df[target] = (ag >= hg).astype(int)
        elif target == 'home_win':
            df[target] = (hg > ag).astype(int)
        elif target == 'away_win':
            df[target] = (ag > hg).astype(int)
        elif target == 'ht_over_05':
            if 'ht_home_score' in df.columns and 'ht_away_score' in df.columns:
                ht_h = pd.to_numeric(df['ht_home_score'], errors='coerce').fillna(0)
                ht_a = pd.to_numeric(df['ht_away_score'], errors='coerce').fillna(0)
                df[target] = ((ht_h + ht_a) > 0.5).astype(int)
            else:
                return None
        else:
            return None
        
        # Return only numeric columns plus target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target not in numeric_cols:
            numeric_cols.append(target)
        
        result = df[numeric_cols].dropna(subset=[target])
        logger.info(f"    Target '{target}' created with {len(result)} samples")
        return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Ensemble Training')
    parser.add_argument('--market', type=str, help='Train specific market only')
    parser.add_argument('--basic-features', action='store_true', help='Use basic features only')
    
    args = parser.parse_args()
    
    pipeline = EnhancedTrainingPipeline()
    results = pipeline.run_full_training(use_advanced_features=not args.basic_features)
    
    return results


if __name__ == "__main__":
    main()
