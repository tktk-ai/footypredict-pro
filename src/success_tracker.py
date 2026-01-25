"""
Success Rate Tracker

Track prediction accuracy and generate performance analytics:
- Prediction logging with outcomes
- Accuracy by confidence bracket
- ROI analysis by bet type/section
- Brier score for probability accuracy
- Streak tracking
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import math


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    id: str
    match_id: str
    home_team: str
    away_team: str
    league: str
    predicted_outcome: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float
    section: str  # 'sure_win', 'strong_picks', etc.
    actual_outcome: Optional[str] = None
    is_correct: Optional[bool] = None
    created_at: str = None
    settled_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SuccessRateTracker:
    """
    Track prediction accuracy and generate analytics.
    
    Features:
    - Log predictions before matches
    - Settle with actual results
    - Calculate accuracy by confidence tier
    - Compute Brier scores for probability calibration
    - Track ROI if betting on recommendations
    """
    
    CONFIDENCE_BRACKETS = {
        '90%+': (0.90, 1.01),
        '80-90%': (0.80, 0.90),
        '70-80%': (0.70, 0.80),
        '60-70%': (0.60, 0.70),
        '50-60%': (0.50, 0.60),
        '<50%': (0.00, 0.50)
    }
    
    def __init__(self, data_dir: str = "data/predictions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions: Dict[str, PredictionRecord] = {}
        self._load_predictions()
        self._counter = 0
    
    def _load_predictions(self):
        """Load predictions from file"""
        filepath = self.data_dir / "predictions_log.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for pred_id, pred_data in data.items():
                        self.predictions[pred_id] = PredictionRecord(**pred_data)
            except Exception as e:
                print(f"Error loading predictions: {e}")
    
    def _save_predictions(self):
        """Save predictions to file"""
        filepath = self.data_dir / "predictions_log.json"
        data = {
            pred_id: pred.to_dict()
            for pred_id, pred in self.predictions.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_id(self) -> str:
        self._counter += 1
        return f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._counter:04d}"
    
    def record_prediction(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        predicted_outcome: str,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        confidence: float,
        section: str = 'general'
    ) -> PredictionRecord:
        """
        Log a prediction before match starts.
        
        Args:
            match_id: Unique match identifier
            home_team: Home team name
            away_team: Away team name
            league: League identifier
            predicted_outcome: 'Home', 'Draw', or 'Away'
            home_win_prob: Probability of home win (0-1)
            draw_prob: Probability of draw (0-1)
            away_win_prob: Probability of away win (0-1)
            confidence: Overall confidence (0-1)
            section: Which section this belongs to
            
        Returns:
            PredictionRecord
        """
        # Normalize probabilities
        if home_win_prob > 1:
            home_win_prob /= 100
        if draw_prob > 1:
            draw_prob /= 100
        if away_win_prob > 1:
            away_win_prob /= 100
        if confidence > 1:
            confidence /= 100
        
        pred = PredictionRecord(
            id=self._generate_id(),
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            predicted_outcome=predicted_outcome,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            confidence=confidence,
            section=section,
            created_at=datetime.now().isoformat()
        )
        
        self.predictions[pred.id] = pred
        self._save_predictions()
        
        return pred
    
    def settle_prediction(
        self,
        prediction_id: str,
        actual_outcome: str
    ) -> Optional[PredictionRecord]:
        """
        Update prediction with actual match result.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: 'Home', 'Draw', or 'Away'
            
        Returns:
            Updated PredictionRecord or None
        """
        if prediction_id not in self.predictions:
            return None
        
        pred = self.predictions[prediction_id]
        pred.actual_outcome = actual_outcome
        pred.is_correct = (pred.predicted_outcome == actual_outcome)
        pred.settled_at = datetime.now().isoformat()
        
        self._save_predictions()
        return pred
    
    def settle_by_match_id(
        self,
        match_id: str,
        actual_outcome: str
    ) -> List[PredictionRecord]:
        """Settle all predictions for a match"""
        settled = []
        for pred_id, pred in self.predictions.items():
            if pred.match_id == match_id and pred.actual_outcome is None:
                pred.actual_outcome = actual_outcome
                pred.is_correct = (pred.predicted_outcome == actual_outcome)
                pred.settled_at = datetime.now().isoformat()
                settled.append(pred)
        
        if settled:
            self._save_predictions()
        
        return settled
    
    def get_accuracy_by_confidence(self) -> Dict:
        """
        Calculate hit rate by confidence bracket.
        
        Returns:
            Dict with accuracy stats per bracket
        """
        settled = [p for p in self.predictions.values() if p.is_correct is not None]
        
        if not settled:
            return {
                bracket: {'predictions': 0, 'correct': 0, 'accuracy': 0}
                for bracket in self.CONFIDENCE_BRACKETS
            }
        
        results = {}
        for bracket_name, (min_conf, max_conf) in self.CONFIDENCE_BRACKETS.items():
            bracket_preds = [
                p for p in settled
                if min_conf <= p.confidence < max_conf
            ]
            
            if bracket_preds:
                correct = sum(1 for p in bracket_preds if p.is_correct)
                results[bracket_name] = {
                    'predictions': len(bracket_preds),
                    'correct': correct,
                    'accuracy': round(correct / len(bracket_preds) * 100, 1)
                }
            else:
                results[bracket_name] = {
                    'predictions': 0,
                    'correct': 0,
                    'accuracy': 0
                }
        
        return results
    
    def get_accuracy_by_section(self) -> Dict:
        """Calculate hit rate by prediction section"""
        settled = [p for p in self.predictions.values() if p.is_correct is not None]
        
        sections = {}
        for pred in settled:
            section = pred.section
            if section not in sections:
                sections[section] = {'predictions': 0, 'correct': 0}
            
            sections[section]['predictions'] += 1
            if pred.is_correct:
                sections[section]['correct'] += 1
        
        # Calculate accuracy
        for section in sections:
            total = sections[section]['predictions']
            correct = sections[section]['correct']
            sections[section]['accuracy'] = round(correct / total * 100, 1) if total > 0 else 0
        
        return sections
    
    def get_brier_score(self) -> float:
        """
        Calculate Brier score for probability accuracy.
        
        Brier score measures the accuracy of probabilistic predictions.
        Range: 0 (perfect) to 1 (worst)
        
        Returns:
            Brier score (lower is better)
        """
        settled = [p for p in self.predictions.values() if p.actual_outcome is not None]
        
        if not settled:
            return 0.0
        
        total_score = 0.0
        for pred in settled:
            # Get predicted probabilities
            probs = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
            
            # Create actual outcome vector (1 for correct, 0 for others)
            outcomes = ['Home', 'Draw', 'Away']
            actual = [1 if o == pred.actual_outcome else 0 for o in outcomes]
            
            # Calculate squared error
            for prob, act in zip(probs, actual):
                total_score += (prob - act) ** 2
        
        # Average and normalize
        brier = total_score / (len(settled) * 3)
        return round(brier, 4)
    
    def get_roi_by_section(self, stake: float = 1.0) -> Dict:
        """
        Calculate ROI if user bet flat stakes on each section.
        
        Args:
            stake: Unit stake amount
            
        Returns:
            Dict with ROI per section
        """
        settled = [p for p in self.predictions.values() if p.is_correct is not None]
        
        sections = {}
        for pred in settled:
            section = pred.section
            if section not in sections:
                sections[section] = {
                    'total_staked': 0,
                    'total_return': 0,
                    'bets': 0,
                    'wins': 0
                }
            
            # Estimate odds from probability
            if pred.predicted_outcome == 'Home':
                prob = pred.home_win_prob
            elif pred.predicted_outcome == 'Away':
                prob = pred.away_win_prob
            else:
                prob = pred.draw_prob
            
            odds = 1 / prob if prob > 0 else 2.0
            
            sections[section]['total_staked'] += stake
            sections[section]['bets'] += 1
            
            if pred.is_correct:
                sections[section]['total_return'] += stake * odds
                sections[section]['wins'] += 1
        
        # Calculate ROI
        for section in sections:
            staked = sections[section]['total_staked']
            returns = sections[section]['total_return']
            profit = returns - staked
            
            sections[section]['profit'] = round(profit, 2)
            sections[section]['roi'] = round(profit / staked * 100, 1) if staked > 0 else 0
            sections[section]['win_rate'] = round(
                sections[section]['wins'] / sections[section]['bets'] * 100, 1
            ) if sections[section]['bets'] > 0 else 0
        
        return sections
    
    def get_streak_info(self) -> Dict:
        """
        Get current hot/cold streaks by section.
        
        Returns:
            Dict with streak info per section
        """
        settled = [p for p in self.predictions.values() if p.is_correct is not None]
        
        # Sort by settled time
        settled.sort(key=lambda x: x.settled_at or '', reverse=True)
        
        streaks = {}
        sections_preds = {}
        
        for pred in settled:
            section = pred.section
            if section not in sections_preds:
                sections_preds[section] = []
            sections_preds[section].append(pred)
        
        for section, preds in sections_preds.items():
            if not preds:
                streaks[section] = {'current_streak': 0, 'streak_type': None}
                continue
            
            streak = 0
            streak_type = None
            
            for pred in preds:
                if streak_type is None:
                    streak_type = 'win' if pred.is_correct else 'loss'
                    streak = 1
                elif (pred.is_correct and streak_type == 'win') or \
                     (not pred.is_correct and streak_type == 'loss'):
                    streak += 1
                else:
                    break
            
            streaks[section] = {
                'current_streak': streak,
                'streak_type': streak_type,
                'last_5': [p.is_correct for p in preds[:5]]
            }
        
        return streaks
    
    def get_summary_stats(self) -> Dict:
        """Get overall summary statistics"""
        all_preds = list(self.predictions.values())
        settled = [p for p in all_preds if p.is_correct is not None]
        pending = [p for p in all_preds if p.is_correct is None]
        
        if not settled:
            return {
                'total_predictions': len(all_preds),
                'settled': 0,
                'pending': len(pending),
                'overall_accuracy': 0,
                'brier_score': 0,
                'best_section': None,
                'worst_section': None
            }
        
        correct = sum(1 for p in settled if p.is_correct)
        
        # Find best/worst sections
        section_stats = self.get_accuracy_by_section()
        best_section = max(
            section_stats.items(),
            key=lambda x: x[1]['accuracy'],
            default=(None, {})
        )
        worst_section = min(
            section_stats.items(),
            key=lambda x: x[1]['accuracy'] if x[1]['predictions'] > 0 else 100,
            default=(None, {})
        )
        
        return {
            'total_predictions': len(all_preds),
            'settled': len(settled),
            'pending': len(pending),
            'correct': correct,
            'overall_accuracy': round(correct / len(settled) * 100, 1),
            'brier_score': self.get_brier_score(),
            'best_section': {
                'name': best_section[0],
                'accuracy': best_section[1].get('accuracy', 0)
            } if best_section[0] else None,
            'worst_section': {
                'name': worst_section[0],
                'accuracy': worst_section[1].get('accuracy', 0)
            } if worst_section[0] and worst_section[1].get('predictions', 0) > 0 else None
        }
    
    def get_recent_predictions(self, limit: int = 20) -> List[Dict]:
        """Get most recent predictions"""
        all_preds = list(self.predictions.values())
        all_preds.sort(key=lambda x: x.created_at or '', reverse=True)
        return [p.to_dict() for p in all_preds[:limit]]
    
    def get_predictions_by_date(self, date: str) -> List[Dict]:
        """Get predictions for a specific date (YYYY-MM-DD)"""
        return [
            p.to_dict() for p in self.predictions.values()
            if p.created_at and p.created_at.startswith(date)
        ]


# Global instance
success_tracker = SuccessRateTracker()


def record_prediction(prediction: Dict, section: str = 'general') -> Dict:
    """Record a new prediction"""
    record = success_tracker.record_prediction(
        match_id=prediction.get('match_id', ''),
        home_team=prediction.get('home_team', ''),
        away_team=prediction.get('away_team', ''),
        league=prediction.get('league', ''),
        predicted_outcome=prediction.get('predicted_outcome', ''),
        home_win_prob=prediction.get('home_win_prob', 0),
        draw_prob=prediction.get('draw_prob', 0),
        away_win_prob=prediction.get('away_win_prob', 0),
        confidence=prediction.get('confidence', 0),
        section=section
    )
    return record.to_dict()


def get_success_analytics() -> Dict:
    """Get comprehensive success analytics"""
    return {
        'summary': success_tracker.get_summary_stats(),
        'by_confidence': success_tracker.get_accuracy_by_confidence(),
        'by_section': success_tracker.get_accuracy_by_section(),
        'roi': success_tracker.get_roi_by_section(),
        'streaks': success_tracker.get_streak_info(),
        'brier_score': success_tracker.get_brier_score()
    }
