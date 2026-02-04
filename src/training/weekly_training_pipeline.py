"""
Weekly Training Pipeline
========================

Automated weekly training cycle for FootyPredict:
1. Collect new data from SportyBet
2. Generate features
3. Train market-specific models (1X2, O/U, BTTS, DC, etc.)
4. Evaluate and deploy models

Usage:
    python -m src.training.weekly_training_pipeline          # Run full pipeline
    python -m src.training.weekly_training_pipeline --test   # Dry run
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model directory
MODELS_DIR = project_root / "models" / "sportybet_markets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class MarketModel:
    """Base class for market-specific models."""
    
    def __init__(self, market_name: str):
        self.market_name = market_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the model. Override in subclasses."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / f"{self.market_name}_model.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'metrics': self.metrics,
                'market_name': self.market_name,
                'trained_at': datetime.now().isoformat()
            }, f)
        logger.info(f"Saved {self.market_name} model to {path}")
        
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / f"{self.market_name}_model.pkl"
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.metrics = data['metrics']
        logger.info(f"Loaded {self.market_name} model from {path}")


class XGBoostMarketModel(MarketModel):
    """XGBoost-based model for market predictions."""
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train XGBoost model."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("XGBoost not available, using sklearn GradientBoosting")
            XGBClassifier = None
        
        # Prepare features
        feature_cols = [c for c in X.columns if X[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        self.feature_columns = feature_cols
        
        X_train, X_val, y_train, y_val = train_test_split(X[feature_cols], y, test_size=0.2, random_state=42)
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        if XGBClassifier:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss' if len(y.unique()) > 2 else 'logloss'
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_features': len(feature_cols)
        }
        
        if len(y.unique()) == 2:
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            self.metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
        
        logger.info(f"{self.market_name} - Accuracy: {self.metrics['accuracy']:.4f}, F1: {self.metrics['f1']:.4f}")
        
        return self.metrics


class WeeklyTrainingPipeline:
    """Complete weekly training pipeline."""
    
    MARKETS = {
        'result_1x2': {
            'target': 'result',  # 0=Away, 1=Draw, 2=Home
            'type': 'multiclass',
            'description': 'Match Result (1X2)'
        },
        'over_under_25': {
            'target': 'over_25',  # 0=Under, 1=Over
            'type': 'binary',
            'description': 'Over/Under 2.5 Goals'
        },
        'btts': {
            'target': 'btts',  # 0=No, 1=Yes
            'type': 'binary',
            'description': 'Both Teams to Score'
        },
        'double_chance_1x': {
            'target': 'dc_1x',  # 0=No, 1=Yes
            'type': 'binary',
            'description': 'Double Chance 1X'
        },
        'double_chance_x2': {
            'target': 'dc_x2',  # 0=No, 1=Yes
            'type': 'binary',
            'description': 'Double Chance X2'
        },
        'first_half_over_05': {
            'target': 'ht_over_05',  # 0=Under, 1=Over
            'type': 'binary',
            'description': 'First Half Over 0.5 Goals'
        }
    }
    
    def __init__(self, data_dir: Path = None, models_dir: Path = None):
        self.data_dir = data_dir or (project_root / "data")
        self.models_dir = models_dir or MODELS_DIR
        self.models = {}
        self.training_results = {}
        
    def run_weekly_cycle(self, dry_run: bool = False) -> Dict:
        """Run the complete weekly training cycle."""
        logger.info("="*60)
        logger.info(f"Weekly Training Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("="*60)
        
        results = {
            'started_at': datetime.now().isoformat(),
            'steps': {}
        }
        
        # Step 1: Collect new data
        logger.info("\n📥 Step 1: Collecting data from SportyBet...")
        try:
            fixtures = self._collect_data()
            results['steps']['data_collection'] = {
                'success': True,
                'fixtures_collected': len(fixtures)
            }
            logger.info(f"   Collected {len(fixtures)} fixtures")
        except Exception as e:
            logger.error(f"   Data collection failed: {e}")
            results['steps']['data_collection'] = {'success': False, 'error': str(e)}
            return results
        
        # Step 2: Generate features
        logger.info("\n🔧 Step 2: Generating features...")
        try:
            features_df = self._generate_features(fixtures)
            results['steps']['feature_generation'] = {
                'success': True,
                'samples': len(features_df),
                'features': len(features_df.columns)
            }
            logger.info(f"   Generated {len(features_df.columns)} features for {len(features_df)} samples")
        except Exception as e:
            logger.error(f"   Feature generation failed: {e}")
            results['steps']['feature_generation'] = {'success': False, 'error': str(e)}
            return results
        
        if dry_run:
            logger.info("\n⚠️  Dry run - skipping model training")
            results['dry_run'] = True
            return results
        
        # Step 3: Load historical data for training
        logger.info("\n📊 Step 3: Loading historical training data...")
        try:
            training_data = self._load_training_data()
            results['steps']['data_loading'] = {
                'success': True,
                'samples': len(training_data)
            }
            logger.info(f"   Loaded {len(training_data)} historical samples")
        except Exception as e:
            logger.error(f"   Data loading failed: {e}")
            results['steps']['data_loading'] = {'success': False, 'error': str(e)}
            # Continue with just collected data
            training_data = features_df
        
        # Step 4: Train models for each market
        logger.info("\n🤖 Step 4: Training market-specific models...")
        results['steps']['model_training'] = {}
        
        for market_name, market_config in self.MARKETS.items():
            try:
                metrics = self._train_market_model(market_name, market_config, training_data)
                results['steps']['model_training'][market_name] = {
                    'success': True,
                    'metrics': metrics
                }
            except Exception as e:
                logger.error(f"   {market_name} training failed: {e}")
                results['steps']['model_training'][market_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Step 5: Save results
        logger.info("\n💾 Step 5: Saving training results...")
        results['completed_at'] = datetime.now().isoformat()
        self._save_results(results)
        
        logger.info("\n" + "="*60)
        logger.info("✅ Weekly training cycle complete!")
        logger.info("="*60)
        
        return results
    
    def _collect_data(self) -> List[Dict]:
        """Collect fixtures from SportyBet."""
        from src.data.sportybet_scraper import SportyBetScraper
        
        scraper = SportyBetScraper()
        fixtures = scraper.get_all_fixtures(days=7)
        
        # Save to CSV
        scraper.save_fixtures_to_csv(fixtures, f"weekly_fixtures_{datetime.now().strftime('%Y%m%d')}.csv")
        
        return fixtures
    
    def _generate_features(self, fixtures: List[Dict]) -> pd.DataFrame:
        """Generate features for fixtures."""
        from src.features.sportybet_feature_engineering import generate_features_for_fixtures
        
        return generate_features_for_fixtures(fixtures)
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load historical training data."""
        # Try multiple data sources
        data_files = [
            self.data_dir / "training_data_200k.csv",
            self.data_dir / "comprehensive_training_data.csv",
            self.data_dir / "training_data.csv",
            self.data_dir / "collected" / "merged_training_data.parquet"
        ]
        
        for path in data_files:
            if path.exists():
                logger.info(f"   Loading from {path.name}")
                if path.suffix == '.parquet':
                    return pd.read_parquet(path)
                return pd.read_csv(path)
        
        # Generate from SportyBet historical
        sportybet_dir = self.data_dir / "sportybet"
        if sportybet_dir.exists():
            csv_files = list(sportybet_dir.glob("*.csv"))
            if csv_files:
                dfs = [pd.read_csv(f) for f in csv_files]
                return pd.concat(dfs, ignore_index=True)
        
        raise FileNotFoundError("No training data found")
    
    def _train_market_model(self, market_name: str, config: Dict, data: pd.DataFrame) -> Dict:
        """Train a single market model."""
        logger.info(f"   Training {config['description']}...")
        
        # Create target variable
        target_col = config['target']
        
        # If target doesn't exist, create from odds/results
        if target_col not in data.columns:
            data = self._create_target(data, target_col)
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Remove rows with missing target
        data = data.dropna(subset=[target_col])
        
        if len(data) < 100:
            raise ValueError(f"Insufficient training data: {len(data)} samples")
        
        # Get features (numeric columns only)
        feature_cols = [c for c in data.columns if c.startswith('odds_') or c.startswith('prob_') or 
                       c.startswith('league_') or c.startswith('home_') or c.startswith('away_') or
                       c.startswith('expected_') or c.startswith('is_') or c.startswith('dc_') or
                       c.startswith('ou') or c.startswith('btts')]
        
        if not feature_cols:
            # Use all numeric columns except target
            feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != target_col]
        
        X = data[feature_cols].fillna(0)
        y = data[target_col]
        
        # Train model
        model = XGBoostMarketModel(market_name)
        metrics = model.train(X, y)
        
        # Save model
        model.save()
        self.models[market_name] = model
        
        return metrics
    
    def _create_target(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create target variable from available columns."""
        if target_col == 'result' and 'home_goals' in data.columns:
            data['result'] = data.apply(
                lambda r: 2 if r['home_goals'] > r['away_goals'] else 
                         (1 if r['home_goals'] == r['away_goals'] else 0), 
                axis=1
            )
        elif target_col == 'over_25':
            if 'home_goals' in data.columns and 'away_goals' in data.columns:
                data['over_25'] = (data['home_goals'] + data['away_goals'] > 2.5).astype(int)
        elif target_col == 'btts':
            if 'home_goals' in data.columns and 'away_goals' in data.columns:
                data['btts'] = ((data['home_goals'] > 0) & (data['away_goals'] > 0)).astype(int)
        elif target_col == 'dc_1x':
            if 'home_goals' in data.columns:
                data['dc_1x'] = (data['home_goals'] >= data['away_goals']).astype(int)
        elif target_col == 'dc_x2':
            if 'away_goals' in data.columns:
                data['dc_x2'] = (data['away_goals'] >= data['home_goals']).astype(int)
        
        return data
    
    def _save_results(self, results: Dict):
        """Save training results."""
        results_path = self.models_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"   Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Weekly Training Pipeline')
    parser.add_argument('--test', action='store_true', help='Dry run without training')
    parser.add_argument('--market', type=str, help='Train specific market only')
    
    args = parser.parse_args()
    
    pipeline = WeeklyTrainingPipeline()
    results = pipeline.run_weekly_cycle(dry_run=args.test)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    
    for market, result in results.get('steps', {}).get('model_training', {}).items():
        if result.get('success'):
            metrics = result.get('metrics', {})
            print(f"  ✅ {market}: Accuracy={metrics.get('accuracy', 0):.2%}")
        else:
            print(f"  ❌ {market}: {result.get('error', 'Failed')}")


if __name__ == "__main__":
    main()
