"""
Backtesting System

Test model accuracy on historical data to validate predictions.
Features:
- Walk-forward validation
- Multiple time periods
- Profit/loss simulation
- Accuracy by league, team, outcome
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "backtest_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class Backtester:
    """Test predictions against historical data"""
    
    def __init__(self):
        self.data = None
        self.results = []
    
    def load_data(self) -> pd.DataFrame:
        """Load historical match data"""
        if self.data is not None:
            return self.data
        
        # Try local cache
        cache_file = DATA_DIR / "training_data.csv"
        if cache_file.exists():
            self.data = pd.read_csv(cache_file)
            self.data['date'] = pd.to_datetime(self.data['date'])
            return self.data
        
        # Download
        try:
            url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
            self.data = pd.read_csv(url)
            self.data['date'] = pd.to_datetime(self.data['date'])
            return self.data
        except:
            return pd.DataFrame()
    
    def calculate_elo(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Elo ratings up to a point in time"""
        elo = {}
        K = 32
        
        df = df.sort_values('date')
        
        for _, row in df.iterrows():
            home, away = row['home_team'], row['away_team']
            h_elo = elo.get(home, 1500)
            a_elo = elo.get(away, 1500)
            
            exp_h = 1 / (1 + 10**((a_elo - h_elo) / 400))
            
            if row['home_score'] > row['away_score']:
                s_h, s_a = 1, 0
            elif row['home_score'] < row['away_score']:
                s_h, s_a = 0, 1
            else:
                s_h, s_a = 0.5, 0.5
            
            elo[home] = h_elo + K * (s_h - exp_h)
            elo[away] = a_elo + K * (s_a - (1 - exp_h))
        
        return elo
    
    def predict_match(self, home_elo: float, away_elo: float, home_advantage: float = 100) -> Dict:
        """Simple Elo-based prediction"""
        h_elo = home_elo + home_advantage
        
        # Expected score
        exp_h = 1 / (1 + 10**((away_elo - h_elo) / 400))
        
        # Convert to 3-way probabilities (rough approximation)
        draw_prob = 0.25 - 0.1 * abs(exp_h - 0.5)  # More likely draw when teams are even
        home_prob = exp_h * (1 - draw_prob)
        away_prob = (1 - exp_h) * (1 - draw_prob)
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        if home_prob > draw_prob and home_prob > away_prob:
            pred = 'H'
        elif away_prob > draw_prob:
            pred = 'A'
        else:
            pred = 'D'
        
        return {
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
            'prediction': pred,
            'confidence': max(home_prob, draw_prob, away_prob)
        }
    
    def run_backtest(self, 
                     start_year: int = 2020, 
                     end_year: int = 2024,
                     min_confidence: float = 0.5) -> Dict:
        """Run backtest over a period"""
        df = self.load_data()
        if df.empty:
            return {'error': 'No data available'}
        
        # Filter date range
        df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)].copy()
        df = df.sort_values('date')
        
        if len(df) < 100:
            return {'error': 'Not enough data for backtest'}
        
        # Split: use first 70% to build Elo, test on last 30%
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Build Elo from training data
        elo = self.calculate_elo(train_df)
        
        # Test predictions
        results = {
            'total': 0,
            'correct': 0,
            'by_outcome': {'H': {'total': 0, 'correct': 0}, 
                          'D': {'total': 0, 'correct': 0}, 
                          'A': {'total': 0, 'correct': 0}},
            'by_confidence': {
                'high': {'total': 0, 'correct': 0},    # > 0.6
                'medium': {'total': 0, 'correct': 0},  # 0.5-0.6
                'low': {'total': 0, 'correct': 0}       # < 0.5
            },
            'profit_loss': 0,  # Assuming $10 flat bets at 1.9 odds
            'predictions': []
        }
        
        for _, row in test_df.iterrows():
            home, away = row['home_team'], row['away_team']
            h_elo = elo.get(home, 1500)
            a_elo = elo.get(away, 1500)
            
            pred = self.predict_match(h_elo, a_elo)
            
            if pred['confidence'] < min_confidence:
                continue
            
            # Actual result
            if row['home_score'] > row['away_score']:
                actual = 'H'
            elif row['home_score'] < row['away_score']:
                actual = 'A'
            else:
                actual = 'D'
            
            correct = pred['prediction'] == actual
            
            results['total'] += 1
            if correct:
                results['correct'] += 1
                results['profit_loss'] += 9  # Win $9 on $10 at 1.9 odds
            else:
                results['profit_loss'] -= 10  # Lose $10
            
            results['by_outcome'][pred['prediction']]['total'] += 1
            if correct:
                results['by_outcome'][pred['prediction']]['correct'] += 1
            
            # Confidence bucket
            if pred['confidence'] > 0.6:
                bucket = 'high'
            elif pred['confidence'] > 0.5:
                bucket = 'medium'
            else:
                bucket = 'low'
            
            results['by_confidence'][bucket]['total'] += 1
            if correct:
                results['by_confidence'][bucket]['correct'] += 1
            
            results['predictions'].append({
                'date': str(row['date'].date()),
                'match': f"{home} vs {away}",
                'predicted': pred['prediction'],
                'actual': actual,
                'correct': correct,
                'confidence': round(pred['confidence'], 3)
            })
            
            # Update Elo
            exp_h = 1 / (1 + 10**((a_elo - h_elo) / 400))
            if actual == 'H': s_h, s_a = 1, 0
            elif actual == 'A': s_h, s_a = 0, 1
            else: s_h, s_a = 0.5, 0.5
            elo[home] = h_elo + 32 * (s_h - exp_h)
            elo[away] = a_elo + 32 * (s_a - (1 - exp_h))
        
        # Calculate summary stats
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
        results['roi'] = results['profit_loss'] / (results['total'] * 10) if results['total'] > 0 else 0
        
        for outcome in results['by_outcome'].values():
            outcome['accuracy'] = outcome['correct'] / outcome['total'] if outcome['total'] > 0 else 0
        
        for conf in results['by_confidence'].values():
            conf['accuracy'] = conf['correct'] / conf['total'] if conf['total'] > 0 else 0
        
        results['period'] = f"{start_year}-{end_year}"
        results['test_matches'] = len(test_df)
        results['predictions'] = results['predictions'][-50:]  # Last 50 only
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(RESULTS_DIR / f'backtest_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary of all backtests"""
        results = []
        for f in RESULTS_DIR.glob('backtest_*.json'):
            with open(f, 'r') as file:
                data = json.load(file)
                results.append({
                    'file': f.name,
                    'period': data.get('period'),
                    'accuracy': data.get('accuracy'),
                    'roi': data.get('roi'),
                    'total_predictions': data.get('total')
                })
        return {'backtests': results}


# Global instance
_backtester: Optional[Backtester] = None

def get_backtester() -> Backtester:
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester

def run_backtest(start_year: int = 2020, end_year: int = 2024, min_confidence: float = 0.5):
    return get_backtester().run_backtest(start_year, end_year, min_confidence)

def get_backtest_summary():
    return get_backtester().get_summary()
