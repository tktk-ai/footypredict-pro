"""
Real Accuracy Monitor

Tracks predictions vs actual results over time.
Provides live accuracy dashboard data.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MONITOR_DIR = DATA_DIR / "accuracy_monitor"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)


class RealAccuracyMonitor:
    """Monitor and track real prediction accuracy"""
    
    def __init__(self):
        self.predictions_file = MONITOR_DIR / "live_predictions.json"
        self.daily_file = MONITOR_DIR / "daily_accuracy.json"
        self.predictions = self._load_predictions()
        self.daily_stats = self._load_daily()
    
    def _load_predictions(self) -> List:
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_daily(self) -> Dict:
        if self.daily_file.exists():
            with open(self.daily_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_predictions(self):
        # Keep last 10000
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions[-10000:], f, indent=2)
    
    def _save_daily(self):
        with open(self.daily_file, 'w') as f:
            json.dump(self.daily_stats, f, indent=2)
    
    def record_prediction(self, 
                         match_id: str,
                         home_team: str,
                         away_team: str,
                         predicted_outcome: str,
                         confidence: float,
                         probabilities: Dict,
                         version: str = 'v3',
                         odds_used: bool = False) -> Dict:
        """Record a new prediction"""
        pred = {
            'id': match_id,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'predicted': predicted_outcome,
            'confidence': confidence,
            'probabilities': probabilities,
            'version': version,
            'odds_used': odds_used,
            'actual': None,
            'correct': None,
            'verified_at': None
        }
        self.predictions.append(pred)
        self._save_predictions()
        return pred
    
    def record_result(self, match_id: str, actual_outcome: str) -> bool:
        """Record actual match result"""
        for pred in reversed(self.predictions):
            if pred['id'] == match_id and pred['actual'] is None:
                pred['actual'] = actual_outcome
                pred['correct'] = pred['predicted'] == actual_outcome
                pred['verified_at'] = datetime.now().isoformat()
                
                # Update daily stats
                self._update_daily(pred)
                self._save_predictions()
                return True
        return False
    
    def _update_daily(self, pred: Dict):
        """Update daily accuracy statistics"""
        date = pred['date']
        version = pred.get('version', 'v3')
        
        if date not in self.daily_stats:
            self.daily_stats[date] = {
                'total': 0,
                'correct': 0,
                'by_version': {},
                'by_odds': {'with_odds': {'total': 0, 'correct': 0}, 
                           'without_odds': {'total': 0, 'correct': 0}}
            }
        
        day = self.daily_stats[date]
        day['total'] += 1
        if pred['correct']:
            day['correct'] += 1
        
        # By version
        if version not in day['by_version']:
            day['by_version'][version] = {'total': 0, 'correct': 0}
        day['by_version'][version]['total'] += 1
        if pred['correct']:
            day['by_version'][version]['correct'] += 1
        
        # By odds usage
        key = 'with_odds' if pred.get('odds_used') else 'without_odds'
        day['by_odds'][key]['total'] += 1
        if pred['correct']:
            day['by_odds'][key]['correct'] += 1
        
        self._save_daily()
    
    def get_live_stats(self) -> Dict:
        """Get live accuracy statistics"""
        verified = [p for p in self.predictions if p.get('actual') is not None]
        
        if not verified:
            return {
                'total': 0,
                'accuracy': 0,
                'message': 'No verified predictions yet'
            }
        
        total = len(verified)
        correct = sum(1 for p in verified if p['correct'])
        
        # Recent 7 days
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        recent = [p for p in verified if p['date'] >= week_ago]
        recent_correct = sum(1 for p in recent if p['correct'])
        
        # By version
        by_version = {}
        for p in verified:
            v = p.get('version', 'unknown')
            if v not in by_version:
                by_version[v] = {'total': 0, 'correct': 0}
            by_version[v]['total'] += 1
            if p['correct']:
                by_version[v]['correct'] += 1
        
        for v in by_version:
            by_version[v]['accuracy'] = by_version[v]['correct'] / by_version[v]['total']
        
        # By odds usage
        with_odds = [p for p in verified if p.get('odds_used')]
        without_odds = [p for p in verified if not p.get('odds_used')]
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': round(correct / total, 4),
            'accuracy_pct': f"{round(correct / total * 100, 1)}%",
            'recent_7d': {
                'total': len(recent),
                'correct': recent_correct,
                'accuracy': round(recent_correct / len(recent), 4) if recent else 0
            },
            'by_version': by_version,
            'with_odds': {
                'total': len(with_odds),
                'accuracy': sum(1 for p in with_odds if p['correct']) / len(with_odds) if with_odds else 0
            },
            'without_odds': {
                'total': len(without_odds),
                'accuracy': sum(1 for p in without_odds if p['correct']) / len(without_odds) if without_odds else 0
            },
            'pending': len([p for p in self.predictions if p.get('actual') is None])
        }
    
    def get_daily_trend(self, days: int = 30) -> List[Dict]:
        """Get daily accuracy trend"""
        trend = []
        for date, stats in sorted(self.daily_stats.items())[-days:]:
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            trend.append({
                'date': date,
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': round(acc, 4)
            })
        return trend
    
    def get_pending_predictions(self) -> List[Dict]:
        """Get predictions waiting for results"""
        pending = [p for p in self.predictions if p.get('actual') is None]
        return pending[-50:]  # Last 50
    
    def get_recent_results(self, limit: int = 20) -> List[Dict]:
        """Get recent verified results"""
        verified = [p for p in self.predictions if p.get('actual') is not None]
        return list(reversed(verified[-limit:]))


# Global instance
_monitor = None

def get_monitor() -> RealAccuracyMonitor:
    global _monitor
    if _monitor is None:
        _monitor = RealAccuracyMonitor()
    return _monitor

def record_live_prediction(match_id: str, home: str, away: str, 
                           predicted: str, confidence: float, probs: Dict,
                           version: str = 'v3', odds_used: bool = False):
    return get_monitor().record_prediction(match_id, home, away, predicted, 
                                           confidence, probs, version, odds_used)

def record_live_result(match_id: str, actual: str):
    return get_monitor().record_result(match_id, actual)

def get_live_accuracy():
    return get_monitor().get_live_stats()

def get_accuracy_trend(days: int = 30):
    return get_monitor().get_daily_trend(days)

def get_pending():
    return get_monitor().get_pending_predictions()
