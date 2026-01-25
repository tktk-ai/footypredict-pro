"""
Local Model Trainer

Automatically retrains models when hyperparameters change.
Features:
- Background training (non-blocking)
- Hot-swaps models without restart
- Keeps backup of old models
- Tracks training history
"""

import json
import pickle
import logging
import threading
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
TRAINED_DIR = MODELS_DIR / "trained"
CONFIG_DIR = MODELS_DIR / "config"
BACKUP_DIR = MODELS_DIR / "backup"
DATA_DIR = BASE_DIR / "data"

TRAINED_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class LocalTrainer:
    """Train models locally with automatic hot-reload"""
    
    def __init__(self):
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.training_history = []
        self.on_complete_callback: Optional[Callable] = None
        self.last_training_result = None
        
        # Load training data cache
        self._data_cache = None
        self._elo_ratings = {}
        
    def get_training_data(self) -> pd.DataFrame:
        """Load or cache training data"""
        if self._data_cache is not None:
            return self._data_cache
        
        # Try local cache first
        cache_file = DATA_DIR / "training_data.csv"
        if cache_file.exists():
            self._data_cache = pd.read_csv(cache_file)
            logger.info(f"Loaded {len(self._data_cache)} matches from cache")
            return self._data_cache
        
        # Download from GitHub
        try:
            url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
            self._data_cache = pd.read_csv(url)
            self._data_cache.to_csv(cache_file, index=False)
            logger.info(f"Downloaded {len(self._data_cache)} matches")
            return self._data_cache
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            # Create minimal sample data
            np.random.seed(42)
            self._data_cache = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
                'home_team': np.random.choice(['Team A', 'Team B', 'Team C'], 1000),
                'away_team': np.random.choice(['Team A', 'Team B', 'Team C'], 1000),
                'home_score': np.random.randint(0, 5, 1000),
                'away_score': np.random.randint(0, 5, 1000)
            })
            return self._data_cache
    
    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training"""
        from sklearn.preprocessing import LabelEncoder
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Result
        df['result'] = np.where(df['home_score'] > df['away_score'], 'H',
                       np.where(df['home_score'] < df['away_score'], 'A', 'D'))
        
        # Elo ratings
        K = 32
        for _, row in df.iterrows():
            home, away = row['home_team'], row['away_team']
            h_elo = self._elo_ratings.get(home, 1500)
            a_elo = self._elo_ratings.get(away, 1500)
            
            exp_h = 1 / (1 + 10**((a_elo - h_elo) / 400))
            
            if row['result'] == 'H': s_h, s_a = 1, 0
            elif row['result'] == 'A': s_h, s_a = 0, 1
            else: s_h, s_a = 0.5, 0.5
            
            self._elo_ratings[home] = h_elo + K * (s_h - exp_h)
            self._elo_ratings[away] = a_elo + K * (s_a - (1 - exp_h))
        
        # Features
        all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        le_team = LabelEncoder().fit(all_teams)
        le_result = LabelEncoder().fit(['A', 'D', 'H'])
        
        df['home_enc'] = le_team.transform(df['home_team'])
        df['away_enc'] = le_team.transform(df['away_team'])
        df['home_elo'] = df['home_team'].map(lambda t: self._elo_ratings.get(t, 1500))
        df['away_elo'] = df['away_team'].map(lambda t: self._elo_ratings.get(t, 1500))
        df['elo_diff'] = df['home_elo'] - df['away_elo']
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dow'] = df['date'].dt.dayofweek
        
        features = ['home_enc', 'away_enc', 'home_elo', 'away_elo', 'elo_diff', 'year', 'month', 'dow']
        X = df[features].values
        y = le_result.transform(df['result'])
        
        return X, y, le_team, le_result
    
    def _backup_models(self):
        """Backup current models before retraining"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = BACKUP_DIR / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for model_file in TRAINED_DIR.glob('*'):
            if model_file.is_file():
                shutil.copy(model_file, backup_path / model_file.name)
        
        logger.info(f"Backed up models to {backup_path}")
    
    def _train_models(self, params: Dict) -> Dict:
        """Train all models with given hyperparameters"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Starting model training...")
        results = {}
        
        # Get data
        df = self.get_training_data()
        X, y, le_team, le_result = self._prepare_features(df)
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train XGBoost
        try:
            from xgboost import XGBClassifier
            xgb_params = params.get('xgb', {})
            xgb = XGBClassifier(**xgb_params, random_state=42, eval_metric='mlogloss', use_label_encoder=False)
            xgb.fit(X_train, y_train)
            xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
            xgb.save_model(str(TRAINED_DIR / 'xgb_football.json'))
            results['xgb'] = {'accuracy': xgb_acc, 'status': 'success'}
            logger.info(f"XGBoost accuracy: {xgb_acc:.4f}")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results['xgb'] = {'status': 'failed', 'error': str(e)}
        
        # Train LightGBM
        try:
            from lightgbm import LGBMClassifier
            lgb_params = params.get('lgb', {})
            lgb = LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
            lgb.fit(X_train, y_train)
            lgb_acc = accuracy_score(y_test, lgb.predict(X_test))
            lgb.booster_.save_model(str(TRAINED_DIR / 'lgb_football.txt'))
            results['lgb'] = {'accuracy': lgb_acc, 'status': 'success'}
            logger.info(f"LightGBM accuracy: {lgb_acc:.4f}")
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            results['lgb'] = {'status': 'failed', 'error': str(e)}
        
        # Train CatBoost
        try:
            from catboost import CatBoostClassifier
            cat_params = params.get('cat', {})
            cat = CatBoostClassifier(**cat_params, random_state=42, verbose=0)
            cat.fit(X_train, y_train)
            cat_acc = accuracy_score(y_test, cat.predict(X_test))
            cat.save_model(str(TRAINED_DIR / 'cat_football.cbm'))
            results['cat'] = {'accuracy': cat_acc, 'status': 'success'}
            logger.info(f"CatBoost accuracy: {cat_acc:.4f}")
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            results['cat'] = {'status': 'failed', 'error': str(e)}
        
        # Train Neural Net
        try:
            import torch
            import torch.nn as nn
            
            class FootballNet(nn.Module):
                def __init__(self, input_dim=8, hidden=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden),
                        nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(hidden, 64),
                        nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 3)
                    )
                def forward(self, x):
                    return self.net(x)
            
            net = FootballNet()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            X_t = torch.FloatTensor(X_train_s)
            y_t = torch.LongTensor(y_train)
            
            for epoch in range(100):
                net.train()
                optimizer.zero_grad()
                loss = criterion(net(X_t), y_t)
                loss.backward()
                optimizer.step()
            
            net.eval()
            with torch.no_grad():
                nn_preds = net(torch.FloatTensor(X_test_s)).argmax(1).numpy()
            nn_acc = accuracy_score(y_test, nn_preds)
            torch.save(net.state_dict(), TRAINED_DIR / 'nn_football.pt')
            results['nn'] = {'accuracy': nn_acc, 'status': 'success'}
            logger.info(f"Neural Net accuracy: {nn_acc:.4f}")
        except Exception as e:
            logger.error(f"Neural Net training failed: {e}")
            results['nn'] = {'status': 'failed', 'error': str(e)}
        
        # Save encoders and config
        with open(CONFIG_DIR / 'encoders.pkl', 'wb') as f:
            pickle.dump({'team_enc': le_team, 'result_enc': le_result, 'scaler': scaler}, f)
        
        with open(CONFIG_DIR / 'elo_ratings.json', 'w') as f:
            json.dump(self._elo_ratings, f)
        
        # Update metadata
        accuracies = [r.get('accuracy', 0) for r in results.values() if r.get('status') == 'success']
        avg_acc = np.mean(accuracies) if accuracies else 0
        
        with open(CONFIG_DIR / 'model_meta.json', 'w') as f:
            json.dump({
                'features': ['home_enc', 'away_enc', 'home_elo', 'away_elo', 'elo_diff', 'year', 'month', 'dow'],
                'classes': ['A', 'D', 'H'],
                'ensemble_weights': params.get('ensemble_weights', {'xgb': 0.3, 'lgb': 0.3, 'cat': 0.25, 'nn': 0.15}),
                'accuracy': float(avg_acc),
                'num_teams': len(self._elo_ratings),
                'trained_at': datetime.now().isoformat(),
                'hyperparameters': params
            }, f, indent=2)
        
        return results
    
    def _reload_models(self):
        """Hot-reload models into the registry"""
        try:
            from src.models.trained_loader import get_trained_loader
            loader = get_trained_loader()
            loader._loaded = False
            loader.models = {}
            loader.load_all()
            logger.info("Hot-reloaded models into registry")
        except Exception as e:
            logger.error(f"Failed to hot-reload: {e}")
    
    def train_async(self, params: Dict, callback: Optional[Callable] = None):
        """Start training in background thread"""
        if self.is_training:
            return {'status': 'already_training'}
        
        self.on_complete_callback = callback
        self.is_training = True
        
        def _train():
            try:
                self._backup_models()
                results = self._train_models(params)
                self._reload_models()
                
                self.last_training_result = {
                    'status': 'complete',
                    'results': results,
                    'completed_at': datetime.now().isoformat()
                }
                
                self.training_history.append(self.last_training_result)
                
                if self.on_complete_callback:
                    self.on_complete_callback(results)
                    
            except Exception as e:
                logger.error(f"Training failed: {e}")
                self.last_training_result = {
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
            finally:
                self.is_training = False
        
        self.training_thread = threading.Thread(target=_train, daemon=True)
        self.training_thread.start()
        
        return {'status': 'training_started', 'message': 'Training in background...'}
    
    def train_sync(self, params: Dict) -> Dict:
        """Train synchronously (blocking)"""
        if self.is_training:
            return {'status': 'already_training'}
        
        self.is_training = True
        try:
            self._backup_models()
            results = self._train_models(params)
            self._reload_models()
            return {'status': 'complete', 'results': results}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
        finally:
            self.is_training = False
    
    def get_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'last_result': self.last_training_result,
            'history_count': len(self.training_history)
        }


# Global instance
_trainer: Optional[LocalTrainer] = None

def get_trainer() -> LocalTrainer:
    global _trainer
    if _trainer is None:
        _trainer = LocalTrainer()
    return _trainer

def retrain_models(params: Dict, async_mode: bool = True) -> Dict:
    """Retrain models with new hyperparameters"""
    trainer = get_trainer()
    if async_mode:
        return trainer.train_async(params)
    else:
        return trainer.train_sync(params)

def get_training_status() -> Dict:
    """Get current training status"""
    return get_trainer().get_status()
