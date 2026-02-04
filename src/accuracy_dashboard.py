"""
Historical Accuracy Dashboard

Track and visualize prediction success rate over time.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"
HISTORY_DIR = DATA_DIR / "prediction_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class AccuracyDashboard:
    """Track prediction accuracy over time"""
    
    def __init__(self):
        self.history_file = HISTORY_DIR / "accuracy_history.json"
        self.predictions_file = HISTORY_DIR / "predictions.json"
        self.history = self._load_history()
        self.predictions = self._load_predictions()
    
    def _load_history(self) -> Dict:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'daily': [], 'weekly': [], 'monthly': []}
    
    def _load_predictions(self) -> List:
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _save_predictions(self):
        # Keep last 5000 predictions
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions[-5000:], f, indent=2)
    
    def record_prediction(self, 
                         match_id: str,
                         home_team: str,
                         away_team: str,
                         predicted: str,
                         confidence: float,
                         probs: Dict[str, float]):
        """Record a new prediction"""
        pred = {
            'id': match_id,
            'timestamp': datetime.now().isoformat(),
            'home_team': home_team,
            'away_team': away_team,
            'predicted': predicted,
            'confidence': confidence,
            'probs': probs,
            'actual': None,
            'correct': None
        }
        self.predictions.append(pred)
        self._save_predictions()
    
    def record_result(self, match_id: str, actual: str):
        """Record actual match result"""
        for pred in reversed(self.predictions):
            if pred['id'] == match_id and pred['actual'] is None:
                pred['actual'] = actual
                pred['correct'] = pred['predicted'] == actual
                self._save_predictions()
                
                # Update daily stats
                self._update_daily_stats()
                return True
        return False
    
    def _update_daily_stats(self):
        """Update daily accuracy statistics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get today's predictions
        today_preds = [p for p in self.predictions 
                       if p['timestamp'].startswith(today) and p['actual'] is not None]
        
        if not today_preds:
            return
        
        correct = sum(1 for p in today_preds if p['correct'])
        total = len(today_preds)
        accuracy = correct / total if total > 0 else 0
        
        # Update or add today's stats
        daily = self.history.get('daily', [])
        updated = False
        for day in daily:
            if day['date'] == today:
                day['correct'] = correct
                day['total'] = total
                day['accuracy'] = accuracy
                updated = True
                break
        
        if not updated:
            daily.append({
                'date': today,
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            })
        
        # Keep last 90 days
        self.history['daily'] = daily[-90:]
        self._save_history()
    
    def get_stats(self, period: str = 'all') -> Dict:
        """Get accuracy statistics"""
        if period == 'today':
            cutoff = datetime.now().strftime('%Y-%m-%d')
            preds = [p for p in self.predictions if p['timestamp'].startswith(cutoff)]
        elif period == 'week':
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            preds = [p for p in self.predictions if p['timestamp'] >= cutoff]
        elif period == 'month':
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            preds = [p for p in self.predictions if p['timestamp'] >= cutoff]
        else:
            preds = self.predictions
        
        # Filter to verified predictions
        verified = [p for p in preds if p.get('actual') is not None]
        
        if not verified:
            return {
                'period': period,
                'total': 0,
                'correct': 0,
                'accuracy': 0,
                'by_outcome': {},
                'by_confidence': {}
            }
        
        correct = sum(1 for p in verified if p['correct'])
        total = len(verified)
        
        # By outcome
        by_outcome = defaultdict(lambda: {'total': 0, 'correct': 0})
        for p in verified:
            by_outcome[p['predicted']]['total'] += 1
            if p['correct']:
                by_outcome[p['predicted']]['correct'] += 1
        
        for outcome in by_outcome.values():
            outcome['accuracy'] = outcome['correct'] / outcome['total'] if outcome['total'] > 0 else 0
        
        # By confidence
        by_confidence = {
            'high_90': {'total': 0, 'correct': 0},
            'strong_80': {'total': 0, 'correct': 0},
            'medium_60': {'total': 0, 'correct': 0},
            'low': {'total': 0, 'correct': 0}
        }
        
        for p in verified:
            conf = p.get('confidence', 0)
            if conf >= 0.9:
                bucket = 'high_90'
            elif conf >= 0.8:
                bucket = 'strong_80'
            elif conf >= 0.6:
                bucket = 'medium_60'
            else:
                bucket = 'low'
            
            by_confidence[bucket]['total'] += 1
            if p['correct']:
                by_confidence[bucket]['correct'] += 1
        
        for bucket in by_confidence.values():
            bucket['accuracy'] = bucket['correct'] / bucket['total'] if bucket['total'] > 0 else 0
        
        return {
            'period': period,
            'total': total,
            'correct': correct,
            'accuracy': correct / total,
            'by_outcome': dict(by_outcome),
            'by_confidence': by_confidence,
            'daily_trend': self.history.get('daily', [])[-30:]
        }
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """Get recent predictions with results"""
        recent = [p for p in self.predictions if p.get('actual') is not None][-limit:]
        return list(reversed(recent))
    
    def get_streak(self) -> Dict:
        """Get current prediction streak"""
        verified = [p for p in self.predictions if p.get('actual') is not None]
        
        if not verified:
            return {'streak': 0, 'type': 'none'}
        
        streak = 0
        streak_type = 'win' if verified[-1]['correct'] else 'loss'
        
        for p in reversed(verified):
            if (streak_type == 'win' and p['correct']) or (streak_type == 'loss' and not p['correct']):
                streak += 1
            else:
                break
        
        return {'streak': streak, 'type': streak_type}


# Global instance
_dashboard = None

def get_dashboard() -> AccuracyDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = AccuracyDashboard()
    return _dashboard

def record_prediction(match_id: str, home: str, away: str, predicted: str, confidence: float, probs: Dict):
    get_dashboard().record_prediction(match_id, home, away, predicted, confidence, probs)

def record_result(match_id: str, actual: str):
    return get_dashboard().record_result(match_id, actual)

def get_accuracy_stats(period: str = 'all'):
    return get_dashboard().get_stats(period)

def get_recent_predictions(limit: int = 50):
    return get_dashboard().get_recent_predictions(limit)
