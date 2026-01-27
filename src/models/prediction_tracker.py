"""
Prediction Tracker

Tracks predictions vs actual results for model improvement:
- Logs all predictions with timestamps
- Records actual match results
- Calculates accuracy metrics
- Provides training data for retraining
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


class PredictionTracker:
    """Tracks predictions and actual results for continuous learning"""
    
    def __init__(self, predictions_file: Optional[Path] = None):
        self.predictions_file = predictions_file or PREDICTIONS_DIR / "predictions_log.csv"
        self.results_file = PREDICTIONS_DIR / "results_log.csv"
        self.metrics_file = PREDICTIONS_DIR / "daily_metrics.json"
        
        self._ensure_files()
    
    def _ensure_files(self) -> None:
        """Ensure log files exist with proper headers"""
        if not self.predictions_file.exists():
            pd.DataFrame(columns=[
                'prediction_id', 'timestamp', 'match_date', 'league',
                'home_team', 'away_team', 'market', 'prediction', 
                'confidence', 'home_prob', 'draw_prob', 'away_prob',
                'home_odds', 'draw_odds', 'away_odds', 'model_version'
            ]).to_csv(self.predictions_file, index=False)
        
        if not self.results_file.exists():
            pd.DataFrame(columns=[
                'prediction_id', 'result_timestamp', 'actual_result',
                'home_goals', 'away_goals', 'is_correct'
            ]).to_csv(self.results_file, index=False)
    
    def log_prediction(self, match_id: str, home_team: str, away_team: str,
                      prediction: str, confidence: float,
                      probabilities: Dict[str, float],
                      market: str = '1X2',
                      league: str = 'Unknown',
                      match_date: Optional[str] = None,
                      odds: Optional[Dict[str, float]] = None) -> str:
        """Log a new prediction"""
        prediction_id = f"{match_id}_{market}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'match_date': match_date or datetime.now().strftime('%Y-%m-%d'),
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            'market': market,
            'prediction': prediction,
            'confidence': confidence,
            'home_prob': probabilities.get('home', 0.33),
            'draw_prob': probabilities.get('draw', 0.33),
            'away_prob': probabilities.get('away', 0.34),
            'home_odds': odds.get('home', 0) if odds else 0,
            'draw_odds': odds.get('draw', 0) if odds else 0,
            'away_odds': odds.get('away', 0) if odds else 0,
            'model_version': 'v2.0'
        }
        
        # Append to log
        df = pd.read_csv(self.predictions_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.predictions_file, index=False)
        
        logger.info(f"Logged prediction: {prediction_id}")
        return prediction_id
    
    def log_result(self, prediction_id: str, actual_result: str,
                   home_goals: int = 0, away_goals: int = 0) -> None:
        """Log the actual result for a prediction"""
        # Load prediction to check correctness
        predictions = pd.read_csv(self.predictions_file)
        pred_row = predictions[predictions['prediction_id'] == prediction_id]
        
        is_correct = 0
        if not pred_row.empty:
            predicted = pred_row.iloc[0]['prediction']
            if predicted == actual_result:
                is_correct = 1
        
        record = {
            'prediction_id': prediction_id,
            'result_timestamp': datetime.now().isoformat(),
            'actual_result': actual_result,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'is_correct': is_correct
        }
        
        # Append to log
        df = pd.read_csv(self.results_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.results_file, index=False)
        
        logger.info(f"Logged result for {prediction_id}: {actual_result} (correct: {is_correct})")
    
    def log_batch_results(self, results: List[Dict]) -> None:
        """Log multiple results at once"""
        for result in results:
            self.log_result(
                prediction_id=result['prediction_id'],
                actual_result=result['actual_result'],
                home_goals=result.get('home_goals', 0),
                away_goals=result.get('away_goals', 0)
            )
    
    def get_daily_performance(self, date: Optional[str] = None) -> Dict:
        """Get performance metrics for a specific date"""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        predictions = pd.read_csv(self.predictions_file)
        results = pd.read_csv(self.results_file)
        
        # Filter by date
        predictions['match_date'] = pd.to_datetime(predictions['match_date'])
        day_predictions = predictions[predictions['match_date'].dt.strftime('%Y-%m-%d') == date]
        
        if day_predictions.empty:
            return {'date': date, 'total': 0, 'accuracy': 0}
        
        # Merge with results
        merged = day_predictions.merge(results, on='prediction_id', how='left')
        
        # Calculate metrics
        total = len(merged)
        correct = merged['is_correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        # By market
        by_market = {}
        for market in merged['market'].unique():
            market_preds = merged[merged['market'] == market]
            market_correct = market_preds['is_correct'].sum()
            by_market[market] = {
                'total': len(market_preds),
                'correct': int(market_correct),
                'accuracy': market_correct / len(market_preds) if len(market_preds) > 0 else 0
            }
        
        return {
            'date': date,
            'total': total,
            'correct': int(correct),
            'accuracy': float(accuracy),
            'by_market': by_market
        }
    
    def get_training_data(self, last_n_days: int = 30) -> pd.DataFrame:
        """Get prediction/result data for retraining"""
        predictions = pd.read_csv(self.predictions_file)
        results = pd.read_csv(self.results_file)
        
        # Merge
        merged = predictions.merge(results, on='prediction_id', how='inner')
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=last_n_days)
        merged['timestamp'] = pd.to_datetime(merged['timestamp'])
        merged = merged[merged['timestamp'] >= cutoff]
        
        return merged
    
    def get_accuracy_trend(self, n_days: int = 7) -> List[Dict]:
        """Get accuracy trend over last N days"""
        trend = []
        
        for i in range(n_days):
            date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
            metrics = self.get_daily_performance(date)
            trend.append(metrics)
        
        return trend
    
    def save_daily_metrics(self) -> None:
        """Save daily metrics to JSON for dashboard"""
        metrics = {
            'last_updated': datetime.now().isoformat(),
            'yesterday': self.get_daily_performance(),
            'trend': self.get_accuracy_trend(7)
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved daily metrics to {self.metrics_file}")


class OnlineLearner:
    """Continuous model improvement from daily results"""
    
    def __init__(self, tracker: Optional[PredictionTracker] = None):
        self.tracker = tracker or PredictionTracker()
        self.bias_corrections: Dict[str, float] = {}
        self.feature_importance_updates: Dict[str, float] = {}
    
    def update_from_results(self, days: int = 1) -> Dict:
        """Learn from recent prediction errors"""
        # Get recent data
        training_data = self.tracker.get_training_data(last_n_days=days)
        
        if training_data.empty:
            return {'status': 'no_data'}
        
        # Analyze errors
        training_data['error'] = 1 - training_data['is_correct']
        
        # Calculate systematic biases
        home_pred = training_data[training_data['prediction'] == 'H']
        away_pred = training_data[training_data['prediction'] == 'A']
        draw_pred = training_data[training_data['prediction'] == 'D']
        
        self.bias_corrections = {
            'home_bias': home_pred['error'].mean() if len(home_pred) > 0 else 0,
            'away_bias': away_pred['error'].mean() if len(away_pred) > 0 else 0,
            'draw_bias': draw_pred['error'].mean() if len(draw_pred) > 0 else 0,
        }
        
        # Analyze by confidence level
        high_conf = training_data[training_data['confidence'] >= 0.7]
        low_conf = training_data[training_data['confidence'] < 0.5]
        
        overconfidence = 0
        if len(high_conf) > 0:
            high_conf_acc = high_conf['is_correct'].mean()
            overconfidence = (high_conf['confidence'].mean() - high_conf_acc)
        
        return {
            'status': 'updated',
            'samples': len(training_data),
            'accuracy': float(training_data['is_correct'].mean()),
            'bias_corrections': self.bias_corrections,
            'overconfidence': float(overconfidence)
        }
    
    def adjust_prediction(self, prediction: str, probabilities: Dict[str, float], 
                         confidence: float) -> Tuple[str, Dict[str, float], float]:
        """Adjust prediction based on learned biases"""
        # Apply bias corrections
        home_prob = probabilities.get('home', 0.33)
        draw_prob = probabilities.get('draw', 0.33)
        away_prob = probabilities.get('away', 0.34)
        
        # Reduce probability for outcomes we've been getting wrong
        home_prob *= (1 - self.bias_corrections.get('home_bias', 0) * 0.5)
        draw_prob *= (1 - self.bias_corrections.get('draw_bias', 0) * 0.5)
        away_prob *= (1 - self.bias_corrections.get('away_bias', 0) * 0.5)
        
        # Renormalize
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        
        # Re-determine prediction
        probs = {'H': home_prob, 'D': draw_prob, 'A': away_prob}
        new_prediction = max(probs, key=probs.get)
        
        # Adjust confidence
        adjusted_confidence = confidence * 0.95  # Slightly conservative
        
        return new_prediction, {
            'home': home_prob, 'draw': draw_prob, 'away': away_prob
        }, adjusted_confidence


# Global tracker instance
_tracker = None

def get_tracker() -> PredictionTracker:
    """Get global prediction tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = PredictionTracker()
    return _tracker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tracker = PredictionTracker()
    
    # Test logging
    pred_id = tracker.log_prediction(
        match_id="test_match_1",
        home_team="Arsenal",
        away_team="Chelsea",
        prediction="H",
        confidence=0.72,
        probabilities={'home': 0.45, 'draw': 0.28, 'away': 0.27},
        market="1X2",
        league="EPL"
    )
    
    # Log result
    tracker.log_result(pred_id, actual_result="H", home_goals=2, away_goals=1)
    
    # Get metrics
    metrics = tracker.get_daily_performance()
    print(f"Daily metrics: {metrics}")
