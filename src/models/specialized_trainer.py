#!/usr/bin/env python3
"""
Specialized Model Trainer for Football Predictions

Trains separate models for different bet types:
- BTTS (Both Teams To Score)
- Over/Under 2.5 Goals
- Double Chance (1X, X2, 12)
- Correct Score (common scorelines)

Each specialized model targets higher accuracy on its specific task.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
DATA_DIR = PROJECT_ROOT / "data"
SPECIALIZED_DIR = MODELS_DIR / "specialized"
SPECIALIZED_DIR.mkdir(parents=True, exist_ok=True)


class SpecializedTrainer:
    """Train specialized models for different bet types."""
    
    def __init__(self):
        self.data = None
        self.feature_cols = None
        self.models = {}
        
    def load_data(self) -> bool:
        """Load preprocessed training data."""
        cache_path = DATA_DIR / "comprehensive_training_data.csv"
        
        if cache_path.exists():
            logger.info(f"📂 Loading cached data from {cache_path}")
            self.data = pd.read_csv(cache_path)
            logger.info(f"   Loaded {len(self.data):,} matches")
            return True
        else:
            logger.warning("⚠️ No cached data found. Run ultimate_trainer.py first.")
            return False
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features from raw data."""
        # Basic feature engineering
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        team_to_idx = {t: i for i, t in enumerate(teams)}
        
        df = df.copy()
        df['HomeTeamEnc'] = df['HomeTeam'].map(team_to_idx)
        df['AwayTeamEnc'] = df['AwayTeam'].map(team_to_idx)
        
        # Encode league
        leagues = df['League'].unique() if 'League' in df.columns else ['Unknown']
        league_to_idx = {l: i for i, l in enumerate(leagues)}
        df['LeagueEnc'] = df['League'].map(league_to_idx) if 'League' in df.columns else 0
        
        # Available feature columns
        feature_cols = ['HomeTeamEnc', 'AwayTeamEnc', 'LeagueEnc']
        
        # Add odds if available
        odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 
                     'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA']
        for col in odds_cols:
            if col in df.columns:
                feature_cols.append(col)
                df[col] = df[col].fillna(df[col].median())
        
        # Add match stats if available
        stat_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
        for col in stat_cols:
            if col in df.columns:
                feature_cols.append(col)
                df[col] = df[col].fillna(0)
        
        self.feature_cols = feature_cols
        
        # Filter to only rows with valid data
        df = df.dropna(subset=['HomeTeamEnc', 'AwayTeamEnc', 'FTHG', 'FTAG'])
        
        X = df[feature_cols].values
        
        return X, df, feature_cols
    
    def train_btts_model(self, use_optuna: bool = True, n_trials: int = 30) -> Dict:
        """Train BTTS (Both Teams To Score) model."""
        logger.info("\n" + "="*60)
        logger.info("⚽ Training BTTS Model (Both Teams To Score)")
        logger.info("="*60)
        
        if self.data is None:
            self.load_data()
        
        X, df, feature_cols = self.prepare_features(self.data)
        
        # Create BTTS target: 1 if both teams scored, 0 otherwise
        y = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int).values
        
        logger.info(f"   Samples: {len(y):,}")
        logger.info(f"   BTTS Yes: {y.sum():,} ({y.mean():.1%})")
        logger.info(f"   BTTS No: {(1-y).sum():,} ({(1-y).mean():.1%})")
        
        # Split data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Train with Optuna if available
        if use_optuna:
            try:
                import optuna
                from sklearn.model_selection import cross_val_score
                import xgboost as xgb
                
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                        'random_state': 42,
                        'verbosity': 0,
                        'n_jobs': -1
                    }
                    model = xgb.XGBClassifier(**params)
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    return scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                logger.info(f"   Best CV Accuracy: {study.best_value:.2%}")
                
                model = xgb.XGBClassifier(**study.best_params, random_state=42, verbosity=0)
                
            except ImportError:
                logger.warning("   Optuna not available, using defaults")
                import xgboost as xgb
                model = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        else:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\n   ✅ BTTS Model Accuracy: {accuracy:.2%}")
        
        # Save model
        model.save_model(str(SPECIALIZED_DIR / 'btts_model.json'))
        
        with open(SPECIALIZED_DIR / 'btts_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(SPECIALIZED_DIR / 'btts_features.json', 'w') as f:
            json.dump(feature_cols, f)
        
        self.models['btts'] = model
        
        return {
            'model': 'BTTS',
            'accuracy': accuracy,
            'samples': len(y),
            'btts_rate': float(y.mean())
        }
    
    def train_over25_model(self, use_optuna: bool = True, n_trials: int = 30) -> Dict:
        """Train Over 2.5 Goals model."""
        logger.info("\n" + "="*60)
        logger.info("📊 Training Over 2.5 Goals Model")
        logger.info("="*60)
        
        if self.data is None:
            self.load_data()
        
        X, df, feature_cols = self.prepare_features(self.data)
        
        # Create Over 2.5 target: 1 if total goals > 2.5, 0 otherwise
        total_goals = df['FTHG'] + df['FTAG']
        y = (total_goals > 2.5).astype(int).values
        
        logger.info(f"   Samples: {len(y):,}")
        logger.info(f"   Over 2.5: {y.sum():,} ({y.mean():.1%})")
        logger.info(f"   Under 2.5: {(1-y).sum():,} ({(1-y).mean():.1%})")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        if use_optuna:
            try:
                import optuna
                from sklearn.model_selection import cross_val_score
                import xgboost as xgb
                
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                        'random_state': 42,
                        'verbosity': 0,
                        'n_jobs': -1
                    }
                    model = xgb.XGBClassifier(**params)
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    return scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                logger.info(f"   Best CV Accuracy: {study.best_value:.2%}")
                
                model = xgb.XGBClassifier(**study.best_params, random_state=42, verbosity=0)
                
            except ImportError:
                import xgboost as xgb
                model = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        else:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        
        model.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\n   ✅ Over 2.5 Model Accuracy: {accuracy:.2%}")
        
        model.save_model(str(SPECIALIZED_DIR / 'over25_model.json'))
        
        with open(SPECIALIZED_DIR / 'over25_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        self.models['over25'] = model
        
        return {
            'model': 'Over 2.5',
            'accuracy': accuracy,
            'samples': len(y),
            'over_rate': float(y.mean())
        }
    
    def train_double_chance_models(self, use_optuna: bool = True, n_trials: int = 30) -> Dict:
        """Train Double Chance models (1X, X2, 12)."""
        logger.info("\n" + "="*60)
        logger.info("🎯 Training Double Chance Models")
        logger.info("="*60)
        
        if self.data is None:
            self.load_data()
        
        X, df, feature_cols = self.prepare_features(self.data)
        
        results = {}
        
        # Double Chance 1X (Home Win or Draw)
        logger.info("\n   📌 Training 1X (Home or Draw)...")
        y_1x = (df['FTR'].isin(['H', 'D'])).astype(int).values
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        scaler_1x = StandardScaler()
        X_scaled = scaler_1x.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_1x, test_size=0.15, random_state=42, stratify=y_1x
        )
        
        import xgboost as xgb
        
        if use_optuna:
            try:
                import optuna
                from sklearn.model_selection import cross_val_score
                
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'random_state': 42,
                        'verbosity': 0,
                        'n_jobs': -1
                    }
                    model = xgb.XGBClassifier(**params)
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    return scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                model_1x = xgb.XGBClassifier(**study.best_params, random_state=42, verbosity=0)
            except:
                model_1x = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        else:
            model_1x = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        
        model_1x.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score
        y_pred = model_1x.predict(X_test)
        acc_1x = accuracy_score(y_test, y_pred)
        logger.info(f"      ✅ 1X Accuracy: {acc_1x:.2%}")
        
        model_1x.save_model(str(SPECIALIZED_DIR / 'dc_1x_model.json'))
        results['1X'] = {'accuracy': acc_1x, 'rate': float(y_1x.mean())}
        
        # Double Chance X2 (Draw or Away)
        logger.info("\n   📌 Training X2 (Draw or Away)...")
        y_x2 = (df['FTR'].isin(['D', 'A'])).astype(int).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_x2, test_size=0.15, random_state=42, stratify=y_x2
        )
        
        model_x2 = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        model_x2.fit(X_train, y_train)
        
        y_pred = model_x2.predict(X_test)
        acc_x2 = accuracy_score(y_test, y_pred)
        logger.info(f"      ✅ X2 Accuracy: {acc_x2:.2%}")
        
        model_x2.save_model(str(SPECIALIZED_DIR / 'dc_x2_model.json'))
        results['X2'] = {'accuracy': acc_x2, 'rate': float(y_x2.mean())}
        
        # Double Chance 12 (Home or Away, no draw)
        logger.info("\n   📌 Training 12 (Home or Away)...")
        y_12 = (df['FTR'].isin(['H', 'A'])).astype(int).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_12, test_size=0.15, random_state=42, stratify=y_12
        )
        
        model_12 = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        model_12.fit(X_train, y_train)
        
        y_pred = model_12.predict(X_test)
        acc_12 = accuracy_score(y_test, y_pred)
        logger.info(f"      ✅ 12 Accuracy: {acc_12:.2%}")
        
        model_12.save_model(str(SPECIALIZED_DIR / 'dc_12_model.json'))
        results['12'] = {'accuracy': acc_12, 'rate': float(y_12.mean())}
        
        # Save scaler
        with open(SPECIALIZED_DIR / 'dc_scaler.pkl', 'wb') as f:
            pickle.dump(scaler_1x, f)
        
        self.models['dc_1x'] = model_1x
        self.models['dc_x2'] = model_x2
        self.models['dc_12'] = model_12
        
        return {
            'model': 'Double Chance',
            'results': results,
            'avg_accuracy': (acc_1x + acc_x2 + acc_12) / 3
        }
    
    def train_all(self, use_optuna: bool = True, n_trials: int = 30) -> Dict:
        """Train all specialized models."""
        logger.info("\n" + "="*70)
        logger.info("🏆 SPECIALIZED MODEL TRAINING")
        logger.info(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        results = {
            'started': datetime.now().isoformat(),
            'models': {}
        }
        
        # Load data once
        self.load_data()
        
        # Train BTTS
        btts_result = self.train_btts_model(use_optuna, n_trials)
        results['models']['BTTS'] = btts_result
        
        # Train Over 2.5
        over_result = self.train_over25_model(use_optuna, n_trials)
        results['models']['Over25'] = over_result
        
        # Train Double Chance
        dc_result = self.train_double_chance_models(use_optuna, n_trials)
        results['models']['DoubleChance'] = dc_result
        
        results['completed'] = datetime.now().isoformat()
        
        # Save results
        with open(SPECIALIZED_DIR / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 SPECIALIZED TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"   BTTS Accuracy: {btts_result['accuracy']:.2%}")
        logger.info(f"   Over 2.5 Accuracy: {over_result['accuracy']:.2%}")
        logger.info(f"   Double Chance Avg: {dc_result['avg_accuracy']:.2%}")
        logger.info("="*70)
        
        return results


# Prediction functions for specialized models
class SpecializedPredictor:
    """Make predictions using specialized models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_loaded = False
    
    def load_models(self):
        """Load all specialized models."""
        try:
            import xgboost as xgb
            
            # BTTS
            btts_path = SPECIALIZED_DIR / 'btts_model.json'
            if btts_path.exists():
                self.models['btts'] = xgb.XGBClassifier()
                self.models['btts'].load_model(str(btts_path))
                logger.info("✅ BTTS model loaded")
            
            # Over 2.5
            over_path = SPECIALIZED_DIR / 'over25_model.json'
            if over_path.exists():
                self.models['over25'] = xgb.XGBClassifier()
                self.models['over25'].load_model(str(over_path))
                logger.info("✅ Over 2.5 model loaded")
            
            # Double Chance 1X
            dc1x_path = SPECIALIZED_DIR / 'dc_1x_model.json'
            if dc1x_path.exists():
                self.models['dc_1x'] = xgb.XGBClassifier()
                self.models['dc_1x'].load_model(str(dc1x_path))
                logger.info("✅ Double Chance 1X model loaded")
            
            # Double Chance X2
            dcx2_path = SPECIALIZED_DIR / 'dc_x2_model.json'
            if dcx2_path.exists():
                self.models['dc_x2'] = xgb.XGBClassifier()
                self.models['dc_x2'].load_model(str(dcx2_path))
                logger.info("✅ Double Chance X2 model loaded")
            
            # Double Chance 12
            dc12_path = SPECIALIZED_DIR / 'dc_12_model.json'
            if dc12_path.exists():
                self.models['dc_12'] = xgb.XGBClassifier()
                self.models['dc_12'].load_model(str(dc12_path))
                logger.info("✅ Double Chance 12 model loaded")
            
            self.is_loaded = len(self.models) > 0
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"Error loading specialized models: {e}")
            return False
    
    def predict_all(self, features: np.ndarray) -> Dict:
        """Get all specialized predictions."""
        if not self.is_loaded:
            self.load_models()
        
        results = {}
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # BTTS
        if 'btts' in self.models:
            proba = self.models['btts'].predict_proba(features)[0]
            results['btts'] = {
                'prediction': 'Yes' if proba[1] > 0.5 else 'No',
                'confidence': float(max(proba)),
                'yes_prob': float(proba[1]),
                'no_prob': float(proba[0])
            }
        
        # Over 2.5
        if 'over25' in self.models:
            proba = self.models['over25'].predict_proba(features)[0]
            results['over25'] = {
                'prediction': 'Over' if proba[1] > 0.5 else 'Under',
                'confidence': float(max(proba)),
                'over_prob': float(proba[1]),
                'under_prob': float(proba[0])
            }
        
        # Double Chance 1X
        if 'dc_1x' in self.models:
            proba = self.models['dc_1x'].predict_proba(features)[0]
            results['double_chance_1x'] = {
                'prediction': '1X' if proba[1] > 0.5 else 'Away',
                'confidence': float(max(proba)),
                'hit_prob': float(proba[1])
            }
        
        # Double Chance X2
        if 'dc_x2' in self.models:
            proba = self.models['dc_x2'].predict_proba(features)[0]
            results['double_chance_x2'] = {
                'prediction': 'X2' if proba[1] > 0.5 else 'Home',
                'confidence': float(max(proba)),
                'hit_prob': float(proba[1])
            }
        
        # Double Chance 12
        if 'dc_12' in self.models:
            proba = self.models['dc_12'].predict_proba(features)[0]
            results['double_chance_12'] = {
                'prediction': '12' if proba[1] > 0.5 else 'Draw',
                'confidence': float(max(proba)),
                'hit_prob': float(proba[1])
            }
        
        return results


# Global instances
trainer = SpecializedTrainer()
predictor = SpecializedPredictor()


def train_specialized_models(use_optuna: bool = True, n_trials: int = 30) -> Dict:
    """Convenience function to train all specialized models."""
    return trainer.train_all(use_optuna, n_trials)


def get_specialized_predictions(features: np.ndarray) -> Dict:
    """Convenience function to get all specialized predictions."""
    return predictor.predict_all(features)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Specialized Models')
    parser.add_argument('--optuna-trials', type=int, default=30)
    parser.add_argument('--no-optuna', action='store_true')
    
    args = parser.parse_args()
    
    results = train_specialized_models(
        use_optuna=not args.no_optuna,
        n_trials=args.optuna_trials
    )
    
    print(json.dumps(results, indent=2, default=str))
