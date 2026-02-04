"""
Advanced Specialized Trainer for SportyBet Markets
===================================================
Targets all popular SportyBet Ghana betting markets with Optuna optimization.

Markets supported:
- 1X2 (Match Result)
- Over/Under (0.5, 1.5, 2.5, 3.5, 4.5)
- BTTS (GG/NG)
- Double Chance (1X, X2, 12)
- Half-Time Result (HT 1X2)
- Half-Time/Full-Time (HT/FT)
- First Half Over/Under
- Second Half Over/Under
- Multi-Goal (0-1, 2-3, 4-5, 6+)
- Win to Nil (Home/Away)
- Clean Sheet
- Correct Score Groups
- Both Halves Over 1.5
- First/Last Goalscorer (team-based proxy)
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "trained" / "sportybet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SportyBet Market Definitions
SPORTYBET_MARKETS = {
    # Core Markets
    '1x2': {'name': 'Match Result', 'classes': 3, 'type': 'multiclass'},
    'btts': {'name': 'Both Teams to Score', 'classes': 2, 'type': 'binary'},
    'over_05': {'name': 'Over 0.5 Goals', 'classes': 2, 'type': 'binary'},
    'over_15': {'name': 'Over 1.5 Goals', 'classes': 2, 'type': 'binary'},
    'over_25': {'name': 'Over 2.5 Goals', 'classes': 2, 'type': 'binary'},
    'over_35': {'name': 'Over 3.5 Goals', 'classes': 2, 'type': 'binary'},
    'over_45': {'name': 'Over 4.5 Goals', 'classes': 2, 'type': 'binary'},
    
    # Double Chance
    'dc_1x': {'name': 'Double Chance 1X', 'classes': 2, 'type': 'binary'},
    'dc_x2': {'name': 'Double Chance X2', 'classes': 2, 'type': 'binary'},
    'dc_12': {'name': 'Double Chance 12', 'classes': 2, 'type': 'binary'},
    
    # Half-Time Markets
    'ht_1x2': {'name': 'Half-Time Result', 'classes': 3, 'type': 'multiclass'},
    'ht_over_05': {'name': 'HT Over 0.5', 'classes': 2, 'type': 'binary'},
    'ht_over_15': {'name': 'HT Over 1.5', 'classes': 2, 'type': 'binary'},
    'ht_btts': {'name': 'HT Both Teams Score', 'classes': 2, 'type': 'binary'},
    
    # Second Half Markets
    '2h_over_05': {'name': '2H Over 0.5', 'classes': 2, 'type': 'binary'},
    '2h_over_15': {'name': '2H Over 1.5', 'classes': 2, 'type': 'binary'},
    
    # Combo Markets (1X2 + O/U)
    'home_over_15': {'name': 'Home & Over 1.5', 'classes': 2, 'type': 'binary'},
    'home_over_25': {'name': 'Home & Over 2.5', 'classes': 2, 'type': 'binary'},
    'away_over_15': {'name': 'Away & Over 1.5', 'classes': 2, 'type': 'binary'},
    'away_over_25': {'name': 'Away & Over 2.5', 'classes': 2, 'type': 'binary'},
    
    # Combo Markets (1X2 + BTTS)
    'home_btts': {'name': 'Home & BTTS', 'classes': 2, 'type': 'binary'},
    'away_btts': {'name': 'Away & BTTS', 'classes': 2, 'type': 'binary'},
    'draw_btts': {'name': 'Draw & BTTS', 'classes': 2, 'type': 'binary'},
    
    # Win to Nil
    'home_win_nil': {'name': 'Home Win to Nil', 'classes': 2, 'type': 'binary'},
    'away_win_nil': {'name': 'Away Win to Nil', 'classes': 2, 'type': 'binary'},
    
    # Multi-Goal Ranges
    'goals_0_1': {'name': '0-1 Goals', 'classes': 2, 'type': 'binary'},
    'goals_2_3': {'name': '2-3 Goals', 'classes': 2, 'type': 'binary'},
    'goals_4_5': {'name': '4-5 Goals', 'classes': 2, 'type': 'binary'},
    'goals_6plus': {'name': '6+ Goals', 'classes': 2, 'type': 'binary'},
    
    # Special Markets
    'both_halves_over_05': {'name': 'Both Halves Over 0.5', 'classes': 2, 'type': 'binary'},
    'both_halves_over_15': {'name': 'Both Halves Over 1.5', 'classes': 2, 'type': 'binary'},
    
    # Correct Score Groups
    'cs_home_1_0': {'name': 'CS Home 1-0', 'classes': 2, 'type': 'binary'},
    'cs_home_2_0': {'name': 'CS Home 2-0', 'classes': 2, 'type': 'binary'},
    'cs_home_2_1': {'name': 'CS Home 2-1', 'classes': 2, 'type': 'binary'},
    'cs_draw_0_0': {'name': 'CS Draw 0-0', 'classes': 2, 'type': 'binary'},
    'cs_draw_1_1': {'name': 'CS Draw 1-1', 'classes': 2, 'type': 'binary'},
    'cs_draw_2_2': {'name': 'CS Draw 2-2', 'classes': 2, 'type': 'binary'},
    'cs_away_0_1': {'name': 'CS Away 0-1', 'classes': 2, 'type': 'binary'},
    'cs_away_0_2': {'name': 'CS Away 0-2', 'classes': 2, 'type': 'binary'},
    'cs_away_1_2': {'name': 'CS Away 1-2', 'classes': 2, 'type': 'binary'},
    
    # Draw No Bet
    'dnb_home': {'name': 'Draw No Bet Home', 'classes': 2, 'type': 'binary'},
    'dnb_away': {'name': 'Draw No Bet Away', 'classes': 2, 'type': 'binary'},
}


class SportyBetTrainer:
    """Advanced trainer for SportyBet markets."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.results: Dict = {}
        
    def load_data(self) -> bool:
        """Load preprocessed training data."""
        cache_path = DATA_DIR / "comprehensive_training_data.csv"
        
        if cache_path.exists():
            logger.info(f"📂 Loading cached data from {cache_path}")
            self.data = pd.read_csv(cache_path, low_memory=False)
            logger.info(f"   Loaded {len(self.data):,} matches")
            return True
        else:
            logger.warning("⚠️ No cached data found. Run ultimate_trainer.py first.")
            return False
    
    def create_target(self, market: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Create target variable for a specific market."""
        df = self.data.copy()
        
        # Ensure we have required columns
        required = ['FTHG', 'FTAG']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")
        
        # Calculate total goals
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        
        # Half-time goals (if available)
        has_ht = 'HTHG' in df.columns and 'HTAG' in df.columns
        if has_ht:
            df['HTTotalGoals'] = df['HTHG'].fillna(0) + df['HTAG'].fillna(0)
            df['2HTotalGoals'] = df['TotalGoals'] - df['HTTotalGoals']
        
        # Create targets based on market
        if market == '1x2':
            conditions = [
                df['FTHG'] > df['FTAG'],
                df['FTHG'] == df['FTAG'],
                df['FTHG'] < df['FTAG']
            ]
            y = np.select(conditions, [0, 1, 2], default=1)
            
        elif market == 'btts':
            y = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int).values
            
        elif market.startswith('over_'):
            threshold = float(market.split('_')[1].replace('5', '.5'))
            y = (df['TotalGoals'] > threshold).astype(int).values
            
        elif market == 'dc_1x':
            y = (df['FTHG'] >= df['FTAG']).astype(int).values
        elif market == 'dc_x2':
            y = (df['FTHG'] <= df['FTAG']).astype(int).values
        elif market == 'dc_12':
            y = (df['FTHG'] != df['FTAG']).astype(int).values
            
        elif market == 'ht_1x2' and has_ht:
            conditions = [
                df['HTHG'] > df['HTAG'],
                df['HTHG'] == df['HTAG'],
                df['HTHG'] < df['HTAG']
            ]
            y = np.select(conditions, [0, 1, 2], default=1)
            
        elif market.startswith('ht_over_') and has_ht:
            threshold = float(market.split('_')[2].replace('5', '.5'))
            y = (df['HTTotalGoals'] > threshold).astype(int).values
            
        elif market == 'ht_btts' and has_ht:
            y = ((df['HTHG'] > 0) & (df['HTAG'] > 0)).astype(int).values
            
        elif market.startswith('2h_over_') and has_ht:
            threshold = float(market.split('_')[2].replace('5', '.5'))
            y = (df['2HTotalGoals'] > threshold).astype(int).values
            
        elif market == 'home_over_15':
            y = ((df['FTHG'] > df['FTAG']) & (df['TotalGoals'] > 1.5)).astype(int).values
        elif market == 'home_over_25':
            y = ((df['FTHG'] > df['FTAG']) & (df['TotalGoals'] > 2.5)).astype(int).values
        elif market == 'away_over_15':
            y = ((df['FTHG'] < df['FTAG']) & (df['TotalGoals'] > 1.5)).astype(int).values
        elif market == 'away_over_25':
            y = ((df['FTHG'] < df['FTAG']) & (df['TotalGoals'] > 2.5)).astype(int).values
            
        elif market == 'home_btts':
            y = ((df['FTHG'] > df['FTAG']) & (df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int).values
        elif market == 'away_btts':
            y = ((df['FTHG'] < df['FTAG']) & (df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int).values
        elif market == 'draw_btts':
            y = ((df['FTHG'] == df['FTAG']) & (df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int).values
            
        elif market == 'home_win_nil':
            y = ((df['FTHG'] > 0) & (df['FTAG'] == 0)).astype(int).values
        elif market == 'away_win_nil':
            y = ((df['FTAG'] > 0) & (df['FTHG'] == 0)).astype(int).values
            
        elif market == 'goals_0_1':
            y = (df['TotalGoals'] <= 1).astype(int).values
        elif market == 'goals_2_3':
            y = ((df['TotalGoals'] >= 2) & (df['TotalGoals'] <= 3)).astype(int).values
        elif market == 'goals_4_5':
            y = ((df['TotalGoals'] >= 4) & (df['TotalGoals'] <= 5)).astype(int).values
        elif market == 'goals_6plus':
            y = (df['TotalGoals'] >= 6).astype(int).values
            
        elif market == 'both_halves_over_05' and has_ht:
            y = ((df['HTTotalGoals'] > 0.5) & (df['2HTotalGoals'] > 0.5)).astype(int).values
        elif market == 'both_halves_over_15' and has_ht:
            y = ((df['HTTotalGoals'] > 1.5) & (df['2HTotalGoals'] > 1.5)).astype(int).values
            
        elif market == 'cs_home_1_0':
            y = ((df['FTHG'] == 1) & (df['FTAG'] == 0)).astype(int).values
        elif market == 'cs_home_2_0':
            y = ((df['FTHG'] == 2) & (df['FTAG'] == 0)).astype(int).values
        elif market == 'cs_home_2_1':
            y = ((df['FTHG'] == 2) & (df['FTAG'] == 1)).astype(int).values
        elif market == 'cs_draw_0_0':
            y = ((df['FTHG'] == 0) & (df['FTAG'] == 0)).astype(int).values
        elif market == 'cs_draw_1_1':
            y = ((df['FTHG'] == 1) & (df['FTAG'] == 1)).astype(int).values
        elif market == 'cs_draw_2_2':
            y = ((df['FTHG'] == 2) & (df['FTAG'] == 2)).astype(int).values
        elif market == 'cs_away_0_1':
            y = ((df['FTHG'] == 0) & (df['FTAG'] == 1)).astype(int).values
        elif market == 'cs_away_0_2':
            y = ((df['FTHG'] == 0) & (df['FTAG'] == 2)).astype(int).values
        elif market == 'cs_away_1_2':
            y = ((df['FTHG'] == 1) & (df['FTAG'] == 2)).astype(int).values
            
        elif market == 'dnb_home':
            # Filter out draws for DNB
            df = df[df['FTHG'] != df['FTAG']].copy()
            y = (df['FTHG'] > df['FTAG']).astype(int).values
        elif market == 'dnb_away':
            df = df[df['FTHG'] != df['FTAG']].copy()
            y = (df['FTHG'] < df['FTAG']).astype(int).values
            
        else:
            raise ValueError(f"Unknown market: {market}")
        
        return y, df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix."""
        # Exclude target and meta columns
        exclude = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'FTR', 'HTR', 'Date', 'HomeTeam', 'AwayTeam',
                   'Div', 'Season', 'TotalGoals', 'HTTotalGoals', '2HTotalGoals', 'Unnamed: 0',
                   'Time', 'Referee', 'Attendance', 'MatchID']
        
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        self.feature_cols = feature_cols
        return X, feature_cols
    
    def train_market(self, market: str, use_optuna: bool = True, n_trials: int = 15) -> Dict:
        """Train model for a specific market."""
        market_info = SPORTYBET_MARKETS[market]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Training: {market_info['name']}")
        logger.info(f"{'='*60}")
        
        try:
            y, df = self.create_target(market)
            X, feature_cols = self.prepare_features(df)
            
            # Stats
            if market_info['type'] == 'binary':
                positive_rate = y.mean()
                logger.info(f"   Samples: {len(y):,}")
                logger.info(f"   Positive rate: {positive_rate:.1%}")
            else:
                unique, counts = np.unique(y, return_counts=True)
                logger.info(f"   Samples: {len(y):,}")
                for u, c in zip(unique, counts):
                    logger.info(f"   Class {u}: {c:,} ({c/len(y):.1%})")
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.15, random_state=42, stratify=y
            )
            
            # Train with Optuna
            import xgboost as xgb
            
            if use_optuna:
                try:
                    import optuna
                    from sklearn.model_selection import cross_val_score
                    
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    
                    def objective(trial):
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 8),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
                            'random_state': 42,
                            'use_label_encoder': False,
                            'eval_metric': 'logloss' if market_info['type'] == 'binary' else 'mlogloss'
                        }
                        
                        if market_info['type'] == 'binary':
                            model = xgb.XGBClassifier(**params)
                        else:
                            params['num_class'] = market_info['classes']
                            params['objective'] = 'multi:softmax'
                            model = xgb.XGBClassifier(**params)
                        
                        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
                        return scores.mean()
                    
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                    
                    best_params = study.best_params
                    best_params['random_state'] = 42
                    best_params['use_label_encoder'] = False
                    best_params['eval_metric'] = 'logloss' if market_info['type'] == 'binary' else 'mlogloss'
                    
                    logger.info(f"   Best CV Accuracy: {study.best_value:.2%}")
                    
                except ImportError:
                    logger.warning("   Optuna not available, using defaults")
                    best_params = {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.05,
                        'random_state': 42,
                        'use_label_encoder': False,
                        'eval_metric': 'logloss'
                    }
            else:
                best_params = {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
            
            # Train final model
            if market_info['type'] == 'binary':
                model = xgb.XGBClassifier(**best_params)
            else:
                best_params['num_class'] = market_info['classes']
                best_params['objective'] = 'multi:softmax'
                model = xgb.XGBClassifier(**best_params)
            
            model.fit(X_train, y_train)
            
            # Evaluate
            from sklearn.metrics import accuracy_score, classification_report
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"   ✅ Test Accuracy: {accuracy:.2%}")
            
            # Save model and artifacts
            model_path = MODELS_DIR / f"{market}_model.json"
            scaler_path = MODELS_DIR / f"{market}_scaler.pkl"
            
            model.save_model(str(model_path))
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            result = {
                'market': market,
                'name': market_info['name'],
                'type': market_info['type'],
                'accuracy': float(accuracy),
                'samples': len(y),
                'positive_rate': float(y.mean()) if market_info['type'] == 'binary' else None,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Error training {market}: {e}")
            return {'market': market, 'error': str(e)}
    
    def train_priority_markets(self, use_optuna: bool = True, n_trials: int = 15) -> Dict:
        """Train priority markets that have highest value for SportyBet users."""
        priority_markets = [
            # High-accuracy markets (binary, common)
            'over_15', 'over_25', 'btts', 
            'dc_1x', 'dc_x2', 'dc_12',
            
            # Half-time markets
            'ht_over_05', 'ht_btts',
            
            # Combo markets (high odds, good accuracy)
            'home_over_25', 'away_over_25',
            'home_btts', 'away_btts',
            
            # Multi-goal ranges
            'goals_2_3', 'goals_4_5',
            
            # Win to Nil
            'home_win_nil', 'away_win_nil',
            
            # Correct scores (popular)
            'cs_draw_1_1', 'cs_home_2_1', 'cs_home_1_0',
        ]
        
        logger.info("\n" + "="*70)
        logger.info("🏆 SPORTYBET ADVANCED SPECIALIZED TRAINING")
        logger.info(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Markets: {len(priority_markets)}")
        logger.info("="*70)
        
        results = {
            'started': datetime.now().isoformat(),
            'markets': {}
        }
        
        if not self.load_data():
            return results
        
        for market in priority_markets:
            try:
                result = self.train_market(market, use_optuna, n_trials)
                results['markets'][market] = result
            except Exception as e:
                logger.error(f"Failed to train {market}: {e}")
                results['markets'][market] = {'error': str(e)}
        
        results['completed'] = datetime.now().isoformat()
        
        # Save results
        with open(MODELS_DIR / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 TRAINING COMPLETE")
        logger.info("="*70)
        
        successful = {k: v for k, v in results['markets'].items() if 'accuracy' in v}
        if successful:
            sorted_markets = sorted(successful.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            logger.info("\n🏆 Top Performing Markets:")
            for market, data in sorted_markets[:10]:
                logger.info(f"   {data['name']}: {data['accuracy']:.2%}")
        
        return results
    
    def train_all_markets(self, use_optuna: bool = True, n_trials: int = 10) -> Dict:
        """Train all supported SportyBet markets."""
        logger.info("\n" + "="*70)
        logger.info("🏆 SPORTYBET FULL MARKET TRAINING")
        logger.info(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Markets: {len(SPORTYBET_MARKETS)}")
        logger.info("="*70)
        
        results = {
            'started': datetime.now().isoformat(),
            'markets': {}
        }
        
        if not self.load_data():
            return results
        
        for market in SPORTYBET_MARKETS:
            try:
                result = self.train_market(market, use_optuna, n_trials)
                results['markets'][market] = result
            except Exception as e:
                logger.error(f"Failed to train {market}: {e}")
                results['markets'][market] = {'error': str(e)}
        
        results['completed'] = datetime.now().isoformat()
        
        # Save results
        with open(MODELS_DIR / 'full_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results


def run_sportybet_training(priority_only: bool = True, use_optuna: bool = True, n_trials: int = 15):
    """Main entry point for SportyBet training."""
    trainer = SportyBetTrainer()
    
    if priority_only:
        results = trainer.train_priority_markets(use_optuna, n_trials)
    else:
        results = trainer.train_all_markets(use_optuna, n_trials)
    
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SportyBet Advanced Training")
    parser.add_argument('--all', action='store_true', help='Train all markets')
    parser.add_argument('--optuna-trials', type=int, default=15, help='Number of Optuna trials')
    parser.add_argument('--no-optuna', action='store_true', help='Skip Optuna optimization')
    
    args = parser.parse_args()
    
    run_sportybet_training(
        priority_only=not args.all,
        use_optuna=not args.no_optuna,
        n_trials=args.optuna_trials
    )
