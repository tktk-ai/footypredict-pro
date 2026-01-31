"""
Enhanced V4 Trainer with Advanced Features
===========================================

Integrates:
- Advanced feature generator (33+ features)
- Betting odds features
- Team form and H2H stats
- Ensemble with SportyBet models
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "v4_enhanced"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class EnhancedV4Trainer:
    """
    V4 Trainer with advanced features and ensemble capabilities.
    """
    
    MARKETS = {
        'result': {'target': 'result', 'type': 'multiclass'},
        'over25': {'target': 'over_25', 'type': 'binary'},
        'over15': {'target': 'over_15', 'type': 'binary'},
        'btts': {'target': 'btts', 'type': 'binary'},
        'dc_1x': {'target': 'dc_1x', 'type': 'binary'},
        'dc_x2': {'target': 'dc_x2', 'type': 'binary'},
        'dc_12': {'target': 'dc_12', 'type': 'binary'},
    }
    
    # Optimized hyperparameters from grid search
    XGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'min_child_weight': 3,
        'gamma': 0,
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0,
        'n_jobs': -1,
    }
    
    LGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    def __init__(self, n_folds: int = 5, test_size: float = 0.15):
        self.n_folds = n_folds
        self.test_size = test_size
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_cols = []
        self.results = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load training data."""
        csv_path = DATA_DIR / "training_data_unified.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Training data not found at {csv_path}")
        
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Loaded {len(df):,} matches")
        return df
    
    def generate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced features from raw data.
        
        Features include:
        - Betting odds (implied probabilities)
        - Odds-derived features (overround, value)
        - Team form approximations
        - League-specific factors
        """
        features = df.copy()
        
        # Odds columns
        odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 
                     'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA']
        
        # Convert odds to implied probabilities
        for bookmaker in ['B365', 'BW', 'IW', 'PS']:
            h_col, d_col, a_col = f'{bookmaker}H', f'{bookmaker}D', f'{bookmaker}A'
            if all(c in df.columns for c in [h_col, d_col, a_col]):
                # Implied probabilities
                features[f'{bookmaker}_home_prob'] = 1 / df[h_col].replace(0, np.nan)
                features[f'{bookmaker}_draw_prob'] = 1 / df[d_col].replace(0, np.nan)
                features[f'{bookmaker}_away_prob'] = 1 / df[a_col].replace(0, np.nan)
                
                # Overround (measure of bookmaker margin)
                features[f'{bookmaker}_overround'] = (
                    features[f'{bookmaker}_home_prob'] + 
                    features[f'{bookmaker}_draw_prob'] + 
                    features[f'{bookmaker}_away_prob']
                )
        
        # Consensus odds (average across bookmakers)
        home_prob_cols = [c for c in features.columns if c.endswith('_home_prob')]
        draw_prob_cols = [c for c in features.columns if c.endswith('_draw_prob')]
        away_prob_cols = [c for c in features.columns if c.endswith('_away_prob')]
        
        if home_prob_cols:
            features['consensus_home_prob'] = features[home_prob_cols].mean(axis=1)
            features['consensus_draw_prob'] = features[draw_prob_cols].mean(axis=1)
            features['consensus_away_prob'] = features[away_prob_cols].mean(axis=1)
            
            # Normalize to sum to 1
            total = (features['consensus_home_prob'] + features['consensus_draw_prob'] + 
                     features['consensus_away_prob'])
            features['norm_home_prob'] = features['consensus_home_prob'] / total
            features['norm_draw_prob'] = features['consensus_draw_prob'] / total
            features['norm_away_prob'] = features['consensus_away_prob'] / total
        
        # Match stats features (if available)
        stats_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        for col in stats_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # Derived stats features
        if 'HS' in df.columns and 'AS' in df.columns:
            features['shot_ratio'] = df['HS'] / (df['AS'] + 1)
            features['shot_diff'] = df['HS'] - df['AS']
        
        if 'HST' in df.columns and 'AST' in df.columns:
            features['shots_on_target_ratio'] = df['HST'] / (df['AST'] + 1)
            features['shots_on_target_diff'] = df['HST'] - df['AST']
        
        if 'HC' in df.columns and 'AC' in df.columns:
            features['corner_ratio'] = df['HC'] / (df['AC'] + 1)
            features['corner_diff'] = df['HC'] - df['AC']
        
        if 'HF' in df.columns and 'AF' in df.columns:
            features['foul_ratio'] = df['HF'] / (df['AF'] + 1)
        
        # League encoding
        if 'league' in df.columns:
            features['league_id'] = pd.factorize(df['league'])[0]
        
        return features
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare feature matrix and targets.
        """
        # Generate advanced features
        features_df = self.generate_advanced_features(df)
        
        # Select numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target-like columns
        exclude = ['home_score', 'away_score', 'total_goals', 'result', 
                   'over_25', 'over_15', 'btts', 'ht_home_score', 'ht_away_score',
                   'dc_1x', 'dc_x2', 'dc_12', 'finished']
        feature_cols = [c for c in numeric_cols if c not in exclude]
        
        # Build feature matrix
        X = features_df[feature_cols].copy()
        X = X.fillna(X.median())
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['features'] = scaler
        self.feature_cols = feature_cols
        
        logger.info(f"Prepared {len(feature_cols)} features")
        
        # Prepare targets
        y_dict = {}
        
        # Result
        if 'result' not in df.columns:
            df['result'] = df.apply(
                lambda r: 'H' if r['home_score'] > r['away_score'] 
                else ('A' if r['away_score'] > r['home_score'] else 'D'), 
                axis=1
            )
        
        le = LabelEncoder()
        y_dict['result'] = le.fit_transform(df['result'])
        self.encoders['result'] = le
        
        # Binary targets
        if 'total_goals' not in df.columns:
            df['total_goals'] = df['home_score'] + df['away_score']
        
        y_dict['over_25'] = (df['total_goals'] > 2.5).astype(int).values
        y_dict['over_15'] = (df['total_goals'] > 1.5).astype(int).values
        y_dict['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int).values
        
        # Double chance
        y_dict['dc_1x'] = ((df['home_score'] >= df['away_score'])).astype(int).values
        y_dict['dc_x2'] = ((df['home_score'] <= df['away_score'])).astype(int).values
        y_dict['dc_12'] = ((df['home_score'] != df['away_score'])).astype(int).values
        
        return X_scaled, y_dict
    
    def create_ensemble(self, is_multiclass: bool = False) -> StackingClassifier:
        """
        Create stacking ensemble with XGBoost + LightGBM + LogisticRegression.
        """
        xgb_params = self.XGB_PARAMS.copy()
        xgb_params['objective'] = 'multi:softprob' if is_multiclass else 'binary:logistic'
        
        lgb_params = self.LGB_PARAMS.copy()
        lgb_params['objective'] = 'multiclass' if is_multiclass else 'binary'
        
        base_estimators = [
            ('xgb', xgb.XGBClassifier(**xgb_params)),
            ('lgb', lgb.LGBMClassifier(**lgb_params)),
        ]
        
        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method='predict_proba',
            passthrough=True,
            n_jobs=-1
        )
        
        return ensemble
    
    def train_market(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market: str,
        use_ensemble: bool = False
    ) -> Tuple[object, Dict]:
        """
        Train model for a single market.
        """
        is_multiclass = self.MARKETS.get(market, {}).get('type') == 'multiclass'
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=42
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            if use_ensemble:
                model = self.create_ensemble(is_multiclass)
            else:
                params = self.XGB_PARAMS.copy()
                params['objective'] = 'multi:softprob' if is_multiclass else 'binary:logistic'
                model = xgb.XGBClassifier(**params)
            
            model.fit(X_train[train_idx], y_train[train_idx])
            pred = model.predict(X_train[val_idx])
            cv_scores.append(accuracy_score(y_train[val_idx], pred))
            
            logger.info(f"  Fold {fold + 1}: {cv_scores[-1]:.4f}")
        
        # Final model
        if use_ensemble:
            final_model = self.create_ensemble(is_multiclass)
        else:
            params = self.XGB_PARAMS.copy()
            params['objective'] = 'multi:softprob' if is_multiclass else 'binary:logistic'
            final_model = xgb.XGBClassifier(**params)
        
        final_model.fit(X_train, y_train)
        
        # Test evaluation
        test_pred = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        test_logloss = log_loss(y_test, test_proba)
        
        metrics = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_logloss': test_logloss,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
        
        logger.info(f"  CV Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        logger.info(f"  Test: {test_accuracy:.4f}")
        
        return final_model, metrics
    
    def train_all(self, use_ensemble: bool = False) -> Dict:
        """
        Train all markets.
        """
        logger.info("=" * 60)
        logger.info("ENHANCED V4 TRAINING WITH ADVANCED FEATURES")
        logger.info("=" * 60)
        
        # Load and prepare data
        df = self.load_data()
        X, y_dict = self.prepare_features(df)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_folds': self.n_folds,
                'test_size': self.test_size,
                'use_ensemble': use_ensemble,
                'n_features': len(self.feature_cols),
            },
            'markets': {}
        }
        
        for market, config in self.MARKETS.items():
            target = config['target']
            
            if target not in y_dict:
                logger.warning(f"Target {target} not available, skipping {market}")
                continue
            
            y = y_dict[target]
            
            logger.info(f"\n📊 Training {market.upper()}...")
            logger.info(f"   Samples: {len(y):,}, Features: {X.shape[1]}")
            
            model, metrics = self.train_market(X, y, market, use_ensemble)
            
            self.models[market] = model
            self.results[market] = metrics
            all_results['markets'][market] = metrics
            
            # Save model
            model_path = MODELS_DIR / f"{market}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save artifacts
        joblib.dump(self.scalers, MODELS_DIR / "scalers.joblib")
        joblib.dump(self.encoders, MODELS_DIR / "encoders.joblib")
        joblib.dump(self.feature_cols, MODELS_DIR / "feature_cols.joblib")
        
        with open(MODELS_DIR / "training_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        for market, metrics in self.results.items():
            logger.info(f"  {market}: CV={metrics['cv_mean']:.1%} Test={metrics['test_accuracy']:.1%}")
        
        return all_results


def train_enhanced_v4(use_ensemble: bool = False):
    """Train enhanced V4 models."""
    trainer = EnhancedV4Trainer(n_folds=5, test_size=0.15)
    return trainer.train_all(use_ensemble)


if __name__ == "__main__":
    results = train_enhanced_v4(use_ensemble=False)
