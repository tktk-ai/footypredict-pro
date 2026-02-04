"""
In-Play Predictions

Real-time prediction adjustments during matches based on:
- Current score
- Time elapsed
- Momentum shifts
- Red cards
"""

import logging
import math
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class InPlayPredictor:
    """Real-time match prediction adjustments"""
    
    def __init__(self):
        self.active_matches: Dict[str, Dict] = {}
    
    def start_tracking(self, match_id: str, home: str, away: str, pre_match: Dict):
        """Start tracking a live match"""
        self.active_matches[match_id] = {
            'home_team': home,
            'away_team': away,
            'pre_match_prediction': pre_match,
            'current_home_score': 0,
            'current_away_score': 0,
            'minute': 0,
            'home_red_cards': 0,
            'away_red_cards': 0,
            'started_at': datetime.now().isoformat()
        }
        return self.active_matches[match_id]
    
    def update_match(self, match_id: str, home_score: int, away_score: int, 
                     minute: int, home_red: int = 0, away_red: int = 0) -> Dict:
        """Update match state and recalculate predictions"""
        if match_id not in self.active_matches:
            return {'error': 'Match not tracked'}
        
        match = self.active_matches[match_id]
        match['current_home_score'] = home_score
        match['current_away_score'] = away_score
        match['minute'] = minute
        match['home_red_cards'] = home_red
        match['away_red_cards'] = away_red
        
        # Recalculate prediction
        match['live_prediction'] = self._calculate_live_prediction(match)
        
        return match
    
    def _calculate_live_prediction(self, match: Dict) -> Dict:
        """Calculate live win probabilities based on current state"""
        home_score = match['current_home_score']
        away_score = match['current_away_score']
        minute = match['minute']
        home_red = match['home_red_cards']
        away_red = match['away_red_cards']
        
        pre = match.get('pre_match_prediction', {}).get('final_prediction', {})
        pre_home = pre.get('home_win_prob', 0.4)
        pre_away = pre.get('away_win_prob', 0.35)
        pre_draw = pre.get('draw_prob', 0.25)
        
        # Time decay factor (less time = current score more predictive)
        time_remaining = max(0, 90 - minute) / 90
        time_weight = 1 - time_remaining
        
        # Score-based probabilities
        goal_diff = home_score - away_score
        
        if goal_diff > 0:
            # Home leading
            if goal_diff >= 2:
                score_home, score_draw, score_away = 0.9, 0.08, 0.02
            else:
                score_home, score_draw, score_away = 0.65, 0.25, 0.10
        elif goal_diff < 0:
            # Away leading
            if goal_diff <= -2:
                score_home, score_draw, score_away = 0.02, 0.08, 0.9
            else:
                score_home, score_draw, score_away = 0.10, 0.25, 0.65
        else:
            # Level
            score_home, score_draw, score_away = 0.35, 0.35, 0.30
        
        # Red card adjustments
        red_diff = away_red - home_red  # Positive = advantage to home
        red_adjustment = red_diff * 0.1  # 10% per red card advantage
        
        # Blend pre-match and score-based predictions weighted by time
        home_prob = (pre_home * (1 - time_weight) + score_home * time_weight) + red_adjustment
        away_prob = (pre_away * (1 - time_weight) + score_away * time_weight) - red_adjustment
        draw_prob = pre_draw * (1 - time_weight) + score_draw * time_weight
        
        # Clamp and normalize
        home_prob = max(0.01, min(0.98, home_prob))
        away_prob = max(0.01, min(0.98, away_prob))
        draw_prob = max(0.01, min(0.98, draw_prob))
        
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        # Predict outcome
        if home_prob > draw_prob and home_prob > away_prob:
            outcome = 'Home Win'
            conf = home_prob
        elif away_prob > draw_prob:
            outcome = 'Away Win'
            conf = away_prob
        else:
            outcome = 'Draw'
            conf = draw_prob
        
        return {
            'home_win_prob': round(home_prob, 4),
            'draw_prob': round(draw_prob, 4),
            'away_win_prob': round(away_prob, 4),
            'predicted_outcome': outcome,
            'confidence': round(conf, 4),
            'minute': minute,
            'score': f"{home_score}-{away_score}"
        }
    
    def get_live_prediction(self, match_id: str) -> Dict:
        """Get current live prediction for a match"""
        if match_id not in self.active_matches:
            return {'error': 'Match not tracked'}
        return self.active_matches[match_id].get('live_prediction', {})
    
    def predict_final_score(self, match_id: str) -> Dict:
        """Predict most likely final score"""
        if match_id not in self.active_matches:
            return {'error': 'Match not tracked'}
        
        match = self.active_matches[match_id]
        home_score = match['current_home_score']
        away_score = match['current_away_score']
        minute = match['minute']
        
        # Expected additional goals based on time remaining
        time_remaining = max(0, 90 - minute) / 90
        avg_total_goals = 2.5  # Average total goals per match
        remaining_goals = avg_total_goals * time_remaining
        
        pred = match.get('live_prediction', {})
        home_prob = pred.get('home_win_prob', 0.4)
        
        # Distribute remaining goals
        exp_home_add = remaining_goals * (home_prob + 0.1)  # Slight home advantage
        exp_away_add = remaining_goals * (1 - home_prob - 0.1)
        
        return {
            'current_score': f"{home_score}-{away_score}",
            'predicted_final': f"{home_score + round(exp_home_add)}-{away_score + round(exp_away_add)}",
            'expected_total_goals': round(home_score + away_score + remaining_goals, 1),
            'minute': minute
        }
    
    def get_all_live_matches(self) -> Dict:
        """Get all tracked live matches"""
        return {
            'matches': list(self.active_matches.values()),
            'count': len(self.active_matches)
        }


# Global instance
_predictor: Optional[InPlayPredictor] = None

def get_inplay_predictor() -> InPlayPredictor:
    global _predictor
    if _predictor is None:
        _predictor = InPlayPredictor()
    return _predictor

def start_live_tracking(match_id: str, home: str, away: str, pre_match: Dict):
    return get_inplay_predictor().start_tracking(match_id, home, away, pre_match)

def update_live_match(match_id: str, home_score: int, away_score: int, 
                      minute: int, home_red: int = 0, away_red: int = 0):
    return get_inplay_predictor().update_match(match_id, home_score, away_score, minute, home_red, away_red)

def get_live_prediction(match_id: str):
    return get_inplay_predictor().get_live_prediction(match_id)

def get_all_live():
    return get_inplay_predictor().get_all_live_matches()
