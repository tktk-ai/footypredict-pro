"""
Advanced Analytics Dashboard Backend

Provides:
- Detailed accuracy tracking
- ROI calculations
- League performance analysis
- Betting patterns
- Trend analysis
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class PredictionRecord:
    """Single prediction record"""
    id: str
    home: str
    away: str
    league: str
    predicted_outcome: str
    actual_outcome: Optional[str]
    confidence: float
    odds: float
    stake: float
    status: str  # pending, won, lost
    created_at: str
    settled_at: Optional[str] = None
    
    def profit_loss(self) -> float:
        if self.status == 'won':
            return self.stake * (self.odds - 1)
        elif self.status == 'lost':
            return -self.stake
        return 0.0


class AdvancedAnalytics:
    """Advanced analytics engine"""
    
    def __init__(self):
        self.predictions: List[PredictionRecord] = []
        self.daily_stats: Dict[str, Dict] = {}
        self.league_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total': 0, 'correct': 0, 'profit': 0.0
        })
        
    def add_prediction(self, record: PredictionRecord):
        """Add a prediction record"""
        self.predictions.append(record)
        self._update_stats(record)
        
    def _update_stats(self, record: PredictionRecord):
        """Update statistics for a prediction"""
        date_key = record.created_at[:10]
        
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'total': 0, 'correct': 0, 'profit': 0.0,
                'stakes': 0.0, 'high_conf': 0, 'high_conf_correct': 0
            }
        
        stats = self.daily_stats[date_key]
        stats['total'] += 1
        stats['stakes'] += record.stake
        
        if record.confidence >= 0.8:
            stats['high_conf'] += 1
        
        if record.status == 'won':
            stats['correct'] += 1
            stats['profit'] += record.profit_loss()
            if record.confidence >= 0.8:
                stats['high_conf_correct'] += 1
        elif record.status == 'lost':
            stats['profit'] += record.profit_loss()
            
        # League stats
        league_stat = self.league_stats[record.league]
        league_stat['total'] += 1
        if record.status == 'won':
            league_stat['correct'] += 1
        league_stat['profit'] += record.profit_loss()
        
    def get_overall_accuracy(self) -> float:
        """Get overall prediction accuracy"""
        settled = [p for p in self.predictions if p.status in ['won', 'lost']]
        if not settled:
            return 0.0
        correct = len([p for p in settled if p.status == 'won'])
        return correct / len(settled) * 100
    
    def get_roi(self, period_days: int = 30) -> float:
        """Calculate ROI for a period"""
        cutoff = datetime.now() - timedelta(days=period_days)
        cutoff_str = cutoff.strftime('%Y-%m-%d')
        
        total_stakes = 0.0
        total_profit = 0.0
        
        for record in self.predictions:
            if record.created_at >= cutoff_str and record.status != 'pending':
                total_stakes += record.stake
                total_profit += record.profit_loss()
                
        if total_stakes == 0:
            return 0.0
        return (total_profit / total_stakes) * 100
    
    def get_league_performance(self) -> List[Dict]:
        """Get performance breakdown by league"""
        result = []
        for league, stats in self.league_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                result.append({
                    'league': league,
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': round(accuracy, 1),
                    'profit': round(stats['profit'], 2)
                })
        return sorted(result, key=lambda x: x['accuracy'], reverse=True)
    
    def get_confidence_analysis(self) -> Dict:
        """Analyze accuracy by confidence bands"""
        bands = {
            '90-100%': {'total': 0, 'correct': 0},
            '80-90%': {'total': 0, 'correct': 0},
            '70-80%': {'total': 0, 'correct': 0},
            '60-70%': {'total': 0, 'correct': 0},
            '50-60%': {'total': 0, 'correct': 0},
        }
        
        for pred in self.predictions:
            if pred.status == 'pending':
                continue
                
            conf = pred.confidence * 100
            if conf >= 90:
                band = '90-100%'
            elif conf >= 80:
                band = '80-90%'
            elif conf >= 70:
                band = '70-80%'
            elif conf >= 60:
                band = '60-70%'
            else:
                band = '50-60%'
                
            bands[band]['total'] += 1
            if pred.status == 'won':
                bands[band]['correct'] += 1
                
        result = {}
        for band, stats in bands.items():
            if stats['total'] > 0:
                result[band] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': round(stats['correct'] / stats['total'] * 100, 1)
                }
            else:
                result[band] = {'total': 0, 'correct': 0, 'accuracy': 0}
                
        return result
    
    def get_outcome_analysis(self) -> Dict:
        """Analyze accuracy by predicted outcome"""
        outcomes = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in self.predictions:
            if pred.status == 'pending':
                continue
            outcomes[pred.predicted_outcome]['total'] += 1
            if pred.status == 'won':
                outcomes[pred.predicted_outcome]['correct'] += 1
                
        result = {}
        for outcome, stats in outcomes.items():
            if stats['total'] > 0:
                result[outcome] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': round(stats['correct'] / stats['total'] * 100, 1)
                }
        return result
    
    def get_trend_analysis(self, days: int = 30) -> List[Dict]:
        """Get daily accuracy trend"""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime('%Y-%m-%d')
        
        trend = []
        for date, stats in sorted(self.daily_stats.items()):
            if date >= cutoff_str:
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total'] * 100
                else:
                    accuracy = 0
                trend.append({
                    'date': date,
                    'accuracy': round(accuracy, 1),
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'profit': round(stats['profit'], 2)
                })
        return trend
    
    def get_streak_info(self) -> Dict:
        """Get winning/losing streak information"""
        settled = [p for p in self.predictions if p.status in ['won', 'lost']]
        settled.sort(key=lambda x: x.created_at)
        
        if not settled:
            return {'current_streak': 0, 'streak_type': 'none', 'best_streak': 0}
        
        current_streak = 0
        current_type = settled[-1].status
        best_winning_streak = 0
        current_winning = 0
        
        for pred in settled:
            if pred.status == 'won':
                current_winning += 1
                best_winning_streak = max(best_winning_streak, current_winning)
            else:
                current_winning = 0
                
        # Calculate current streak
        for pred in reversed(settled):
            if pred.status == current_type:
                current_streak += 1
            else:
                break
                
        return {
            'current_streak': current_streak,
            'streak_type': 'winning' if current_type == 'won' else 'losing',
            'best_winning_streak': best_winning_streak
        }
    
    def get_value_bet_analysis(self) -> Dict:
        """Analyze value bet performance"""
        value_bets = [p for p in self.predictions if hasattr(p, 'is_value_bet') and p.is_value_bet]
        
        if not value_bets:
            return {'count': 0, 'accuracy': 0, 'avg_edge': 0, 'roi': 0}
        
        settled = [p for p in value_bets if p.status in ['won', 'lost']]
        if not settled:
            return {'count': len(value_bets), 'accuracy': 0, 'avg_edge': 0, 'roi': 0}
        
        correct = len([p for p in settled if p.status == 'won'])
        total_stakes = sum(p.stake for p in settled)
        total_profit = sum(p.profit_loss() for p in settled)
        
        return {
            'count': len(value_bets),
            'settled': len(settled),
            'accuracy': round(correct / len(settled) * 100, 1),
            'roi': round(total_profit / total_stakes * 100, 1) if total_stakes > 0 else 0
        }
    
    def get_sure_wins_analysis(self) -> Dict:
        """Analyze sure wins (91%+ confidence) performance"""
        sure_wins = [p for p in self.predictions if p.confidence >= 0.91]
        
        if not sure_wins:
            return {'count': 0, 'accuracy': 0, 'profit': 0}
        
        settled = [p for p in sure_wins if p.status in ['won', 'lost']]
        if not settled:
            return {'count': len(sure_wins), 'accuracy': 0, 'profit': 0}
        
        correct = len([p for p in settled if p.status == 'won'])
        total_profit = sum(p.profit_loss() for p in settled)
        
        return {
            'count': len(sure_wins),
            'settled': len(settled),
            'correct': correct,
            'accuracy': round(correct / len(settled) * 100, 1),
            'profit': round(total_profit, 2)
        }
    
    def get_dashboard_summary(self) -> Dict:
        """Get complete dashboard summary"""
        return {
            'overall': {
                'total_predictions': len(self.predictions),
                'accuracy': round(self.get_overall_accuracy(), 1),
                'roi_30d': round(self.get_roi(30), 1),
                'roi_7d': round(self.get_roi(7), 1),
            },
            'streak': self.get_streak_info(),
            'by_league': self.get_league_performance()[:5],
            'by_confidence': self.get_confidence_analysis(),
            'by_outcome': self.get_outcome_analysis(),
            'trend': self.get_trend_analysis(14),
            'sure_wins': self.get_sure_wins_analysis(),
            'generated_at': datetime.now().isoformat()
        }


# Global analytics instance
analytics = AdvancedAnalytics()


def record_prediction(
    home: str, away: str, league: str,
    prediction: str, confidence: float,
    odds: float = 1.0, stake: float = 10.0
) -> str:
    """Record a new prediction"""
    record = PredictionRecord(
        id=f"pred_{datetime.now().timestamp()}",
        home=home,
        away=away,
        league=league,
        predicted_outcome=prediction,
        actual_outcome=None,
        confidence=confidence,
        odds=odds,
        stake=stake,
        status='pending',
        created_at=datetime.now().isoformat()
    )
    analytics.add_prediction(record)
    return record.id


def settle_prediction(pred_id: str, actual_outcome: str) -> bool:
    """Settle a prediction as won or lost"""
    for pred in analytics.predictions:
        if pred.id == pred_id:
            pred.actual_outcome = actual_outcome
            pred.status = 'won' if pred.predicted_outcome == actual_outcome else 'lost'
            pred.settled_at = datetime.now().isoformat()
            analytics._update_stats(pred)
            return True
    return False


def get_analytics_summary() -> Dict:
    """Get analytics dashboard summary"""
    return analytics.get_dashboard_summary()


def get_league_accuracy(league: str) -> float:
    """Get accuracy for a specific league"""
    stats = analytics.league_stats.get(league)
    if stats and stats['total'] > 0:
        return stats['correct'] / stats['total'] * 100
    return 0.0
