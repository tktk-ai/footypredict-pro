"""
Prediction Tracking System

Automatically tracks predictions and verifies results:
- Records all predictions made
- Fetches match results automatically
- Calculates accuracy statistics
- Generates performance reports
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
import random


@dataclass
class TrackedPrediction:
    """A tracked prediction"""
    id: str
    home: str
    away: str
    league: str
    predicted_outcome: str  # 'home', 'draw', 'away'
    confidence: float
    predicted_at: str
    match_date: str
    status: str = 'pending'  # 'pending', 'won', 'lost', 'void'
    actual_score: Optional[str] = None
    actual_outcome: Optional[str] = None
    verified_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PredictionTracker:
    """
    Track predictions and verify results automatically
    """
    
    def __init__(self, data_dir: str = "data/predictions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions: List[TrackedPrediction] = []
        self._counter = 0
        self._load()
    
    def _load(self):
        """Load predictions from file"""
        filepath = self.data_dir / "tracked_predictions.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.predictions = [TrackedPrediction(**p) for p in data]
                    if self.predictions:
                        # Get highest counter
                        ids = [int(p.id.split('_')[-1]) for p in self.predictions if p.id.startswith('pred_')]
                        self._counter = max(ids) if ids else 0
            except:
                pass
    
    def _save(self):
        """Save predictions to file"""
        filepath = self.data_dir / "tracked_predictions.json"
        with open(filepath, 'w') as f:
            json.dump([p.to_dict() for p in self.predictions], f, indent=2)
    
    def _generate_id(self) -> str:
        self._counter += 1
        return f"pred_{datetime.now().strftime('%Y%m%d')}_{self._counter:04d}"
    
    def track_prediction(
        self,
        home: str,
        away: str,
        league: str,
        predicted_outcome: str,
        confidence: float,
        match_date: str = None
    ) -> TrackedPrediction:
        """Track a new prediction"""
        pred = TrackedPrediction(
            id=self._generate_id(),
            home=home,
            away=away,
            league=league,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            predicted_at=datetime.now().isoformat(),
            match_date=match_date or datetime.now().strftime('%Y-%m-%d')
        )
        
        self.predictions.append(pred)
        self._save()
        return pred
    
    def verify_prediction(
        self,
        prediction_id: str,
        actual_score: str,
        actual_outcome: str
    ) -> Optional[TrackedPrediction]:
        """Verify a prediction with actual result"""
        for pred in self.predictions:
            if pred.id == prediction_id:
                pred.actual_score = actual_score
                pred.actual_outcome = actual_outcome
                pred.verified_at = datetime.now().isoformat()
                pred.status = 'won' if pred.predicted_outcome == actual_outcome else 'lost'
                self._save()
                return pred
        return None
    
    def verify_by_match(
        self,
        home: str,
        away: str,
        actual_score: str,
        actual_outcome: str
    ) -> Optional[TrackedPrediction]:
        """Verify prediction by match teams"""
        for pred in self.predictions:
            if pred.home.lower() == home.lower() and pred.away.lower() == away.lower() and pred.status == 'pending':
                return self.verify_prediction(pred.id, actual_score, actual_outcome)
        return None
    
    def get_stats(self) -> Dict:
        """Get overall prediction statistics"""
        total = len(self.predictions)
        verified = [p for p in self.predictions if p.status in ['won', 'lost']]
        pending = [p for p in self.predictions if p.status == 'pending']
        won = [p for p in verified if p.status == 'won']
        
        accuracy = len(won) / len(verified) if verified else 0
        
        # Accuracy by league
        by_league = {}
        for pred in verified:
            if pred.league not in by_league:
                by_league[pred.league] = {'won': 0, 'total': 0}
            by_league[pred.league]['total'] += 1
            if pred.status == 'won':
                by_league[pred.league]['won'] += 1
        
        league_accuracy = {
            league: {
                'accuracy': round(data['won'] / data['total'] * 100, 1) if data['total'] > 0 else 0,
                'total': data['total']
            }
            for league, data in by_league.items()
        }
        
        # Accuracy by confidence tier
        high_conf = [p for p in verified if p.confidence >= 0.75]
        med_conf = [p for p in verified if 0.55 <= p.confidence < 0.75]
        low_conf = [p for p in verified if p.confidence < 0.55]
        
        return {
            'total_predictions': total,
            'verified': len(verified),
            'pending': len(pending),
            'won': len(won),
            'lost': len(verified) - len(won),
            'accuracy': round(accuracy * 100, 1),
            'by_league': league_accuracy,
            'by_confidence': {
                'high_75+': {
                    'total': len(high_conf),
                    'won': len([p for p in high_conf if p.status == 'won']),
                    'accuracy': round(len([p for p in high_conf if p.status == 'won']) / len(high_conf) * 100, 1) if high_conf else 0
                },
                'medium_55_75': {
                    'total': len(med_conf),
                    'won': len([p for p in med_conf if p.status == 'won']),
                    'accuracy': round(len([p for p in med_conf if p.status == 'won']) / len(med_conf) * 100, 1) if med_conf else 0
                },
                'low_under_55': {
                    'total': len(low_conf),
                    'won': len([p for p in low_conf if p.status == 'won']),
                    'accuracy': round(len([p for p in low_conf if p.status == 'won']) / len(low_conf) * 100, 1) if low_conf else 0
                }
            }
        }
    
    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get recent predictions"""
        sorted_preds = sorted(self.predictions, key=lambda x: x.predicted_at, reverse=True)
        return [p.to_dict() for p in sorted_preds[:limit]]
    
    def get_pending(self) -> List[Dict]:
        """Get pending predictions"""
        pending = [p for p in self.predictions if p.status == 'pending']
        return [p.to_dict() for p in pending]
    
    def auto_verify_from_results(self, results: List[Dict]) -> List[Dict]:
        """
        Auto-verify predictions from match results.
        Results format: [{'home': 'Team A', 'away': 'Team B', 'score': '2-1'}, ...]
        """
        verified = []
        
        for result in results:
            home = result.get('home', '')
            away = result.get('away', '')
            score = result.get('score', '')
            
            if not score or '-' not in score:
                continue
            
            try:
                home_goals, away_goals = map(int, score.split('-'))
                
                if home_goals > away_goals:
                    outcome = 'home'
                elif away_goals > home_goals:
                    outcome = 'away'
                else:
                    outcome = 'draw'
                
                pred = self.verify_by_match(home, away, score, outcome)
                if pred:
                    verified.append(pred.to_dict())
            except:
                continue
        
        return verified


# Global instance
prediction_tracker = PredictionTracker()


def add_sample_predictions():
    """Add sample historical predictions for demo/testing"""
    samples = [
        # Bundesliga - Past matches (verified)
        {'home': 'Bayern Munich', 'away': 'Borussia Dortmund', 'league': 'bundesliga', 
         'predicted': 'home', 'confidence': 0.94, 'actual': '2-1', 'outcome': 'home', 'days_ago': 7},
        {'home': 'RB Leipzig', 'away': 'Bayer Leverkusen', 'league': 'bundesliga', 
         'predicted': 'home', 'confidence': 0.72, 'actual': '1-1', 'outcome': 'draw', 'days_ago': 6},
        {'home': 'Borussia Dortmund', 'away': 'Eintracht Frankfurt', 'league': 'bundesliga', 
         'predicted': 'home', 'confidence': 0.85, 'actual': '3-0', 'outcome': 'home', 'days_ago': 5},
        {'home': 'Wolfsburg', 'away': 'Hoffenheim', 'league': 'bundesliga', 
         'predicted': 'draw', 'confidence': 0.58, 'actual': '0-0', 'outcome': 'draw', 'days_ago': 4},
        
        # Premier League
        {'home': 'Liverpool', 'away': 'Arsenal', 'league': 'premier_league', 
         'predicted': 'draw', 'confidence': 0.67, 'actual': '1-1', 'outcome': 'draw', 'days_ago': 7},
        {'home': 'Manchester City', 'away': 'Chelsea', 'league': 'premier_league', 
         'predicted': 'home', 'confidence': 0.88, 'actual': '4-0', 'outcome': 'home', 'days_ago': 6},
        {'home': 'Manchester United', 'away': 'Tottenham', 'league': 'premier_league', 
         'predicted': 'home', 'confidence': 0.61, 'actual': '1-2', 'outcome': 'away', 'days_ago': 5},
        {'home': 'Newcastle', 'away': 'Everton', 'league': 'premier_league', 
         'predicted': 'home', 'confidence': 0.76, 'actual': '2-0', 'outcome': 'home', 'days_ago': 4},
        
        # La Liga
        {'home': 'Real Madrid', 'away': 'Barcelona', 'league': 'la_liga', 
         'predicted': 'home', 'confidence': 0.58, 'actual': '2-3', 'outcome': 'away', 'days_ago': 7},
        {'home': 'Atletico Madrid', 'away': 'Sevilla', 'league': 'la_liga', 
         'predicted': 'home', 'confidence': 0.79, 'actual': '2-0', 'outcome': 'home', 'days_ago': 6},
        {'home': 'Valencia', 'away': 'Real Betis', 'league': 'la_liga', 
         'predicted': 'draw', 'confidence': 0.52, 'actual': '1-1', 'outcome': 'draw', 'days_ago': 5},
        
        # Serie A
        {'home': 'Inter Milan', 'away': 'Juventus', 'league': 'serie_a', 
         'predicted': 'home', 'confidence': 0.73, 'actual': '1-0', 'outcome': 'home', 'days_ago': 7},
        {'home': 'AC Milan', 'away': 'Napoli', 'league': 'serie_a', 
         'predicted': 'draw', 'confidence': 0.49, 'actual': '0-1', 'outcome': 'away', 'days_ago': 6},
        {'home': 'Roma', 'away': 'Lazio', 'league': 'serie_a', 
         'predicted': 'home', 'confidence': 0.66, 'actual': '2-1', 'outcome': 'home', 'days_ago': 5},
        
        # Ligue 1
        {'home': 'PSG', 'away': 'Monaco', 'league': 'ligue_1', 
         'predicted': 'home', 'confidence': 0.91, 'actual': '3-1', 'outcome': 'home', 'days_ago': 7},
        {'home': 'Marseille', 'away': 'Lyon', 'league': 'ligue_1', 
         'predicted': 'home', 'confidence': 0.62, 'actual': '2-2', 'outcome': 'draw', 'days_ago': 6},
    ]
    
    tracker = prediction_tracker
    added = []
    
    for sample in samples:
        # Create prediction
        match_date = (datetime.now() - timedelta(days=sample['days_ago'])).strftime('%Y-%m-%d')
        
        pred = tracker.track_prediction(
            home=sample['home'],
            away=sample['away'],
            league=sample['league'],
            predicted_outcome=sample['predicted'],
            confidence=sample['confidence'],
            match_date=match_date
        )
        
        # Verify it immediately
        tracker.verify_prediction(
            pred.id,
            sample['actual'],
            sample['outcome']
        )
        
        added.append(pred.to_dict())
    
    return {
        'added': len(added),
        'predictions': added,
        'stats': tracker.get_stats()
    }


def track_today_predictions(predictions: List[Dict]) -> List[Dict]:
    """Track today's predictions from the app"""
    tracker = prediction_tracker
    tracked = []
    
    for pred in predictions:
        try:
            p = tracker.track_prediction(
                home=pred.get('home', pred.get('home_team', '')),
                away=pred.get('away', pred.get('away_team', '')),
                league=pred.get('league', 'unknown'),
                predicted_outcome=pred.get('outcome', pred.get('prediction', 'home')),
                confidence=pred.get('confidence', 0.5),
                match_date=pred.get('date', datetime.now().strftime('%Y-%m-%d'))
            )
            tracked.append(p.to_dict())
        except Exception as e:
            print(f"Error tracking prediction: {e}")
    
    return tracked


def get_accuracy_stats() -> Dict:
    """Get accuracy statistics"""
    return prediction_tracker.get_stats()


def get_recent_predictions(limit: int = 20) -> List[Dict]:
    """Get recent predictions"""
    return prediction_tracker.get_recent(limit)
