"""
Live In-Play Betting Predictor

Real-time predictions for live matches:
- Live probability updates
- Momentum detection
- In-play value identification
- Next goal prediction
"""

import math
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MatchPhase(Enum):
    """Match phases"""
    FIRST_HALF = "first_half"
    HALF_TIME = "half_time"
    SECOND_HALF = "second_half"
    EXTRA_TIME = "extra_time"
    FINISHED = "finished"


@dataclass
class LiveMatchState:
    """Current state of a live match"""
    match_id: str
    home: str
    away: str
    home_score: int = 0
    away_score: int = 0
    minute: int = 0
    phase: MatchPhase = MatchPhase.FIRST_HALF
    home_shots: int = 0
    away_shots: int = 0
    home_on_target: int = 0
    away_on_target: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_possession: float = 50.0
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_red_cards: int = 0
    away_red_cards: int = 0
    events: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'match_id': self.match_id,
            'home': self.home,
            'away': self.away,
            'score': f"{self.home_score}-{self.away_score}",
            'minute': self.minute,
            'phase': self.phase.value,
            'stats': {
                'shots': [self.home_shots, self.away_shots],
                'on_target': [self.home_on_target, self.away_on_target],
                'corners': [self.home_corners, self.away_corners],
                'possession': [self.home_possession, 100 - self.home_possession],
                'xg': [round(self.home_xg, 2), round(self.away_xg, 2)],
                'red_cards': [self.home_red_cards, self.away_red_cards]
            },
            'events': self.events[-5:]
        }


@dataclass
class LivePrediction:
    """Live in-play prediction"""
    match_id: str
    timestamp: str
    minute: int
    current_score: str
    
    # Win probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    # Goals predictions
    next_goal_home_prob: float
    next_goal_away_prob: float
    no_more_goals_prob: float
    over_25_prob: float
    btts_prob: float
    
    # Market recommendations
    recommended_bet: Optional[str] = None
    edge: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'match_id': self.match_id,
            'timestamp': self.timestamp,
            'minute': self.minute,
            'current_score': self.current_score,
            'probabilities': {
                'home_win': round(self.home_win_prob * 100, 1),
                'draw': round(self.draw_prob * 100, 1),
                'away_win': round(self.away_win_prob * 100, 1)
            },
            'next_goal': {
                'home': round(self.next_goal_home_prob * 100, 1),
                'away': round(self.next_goal_away_prob * 100, 1),
                'no_goal': round(self.no_more_goals_prob * 100, 1)
            },
            'goals_markets': {
                'over_25': round(self.over_25_prob * 100, 1),
                'btts': round(self.btts_prob * 100, 1)
            },
            'recommendation': {
                'bet': self.recommended_bet,
                'edge': round(self.edge * 100, 1),
                'confidence': round(self.confidence * 100, 1)
            }
        }


class MomentumAnalyzer:
    """Analyze match momentum from live stats"""
    
    def analyze(self, state: LiveMatchState) -> Dict:
        """Calculate momentum indicators"""
        # Calculate momentum score (-1 to 1, negative = away dominating)
        factors = []
        
        # Possession factor
        poss_diff = (state.home_possession - 50) / 50
        factors.append(('possession', poss_diff, 0.3))
        
        # Shots factor
        if state.home_shots + state.away_shots > 0:
            shot_ratio = (state.home_shots - state.away_shots) / (state.home_shots + state.away_shots + 1)
            factors.append(('shots', shot_ratio, 0.25))
        
        # Shots on target factor
        if state.home_on_target + state.away_on_target > 0:
            sot_ratio = (state.home_on_target - state.away_on_target) / (state.home_on_target + state.away_on_target + 1)
            factors.append(('shots_on_target', sot_ratio, 0.25))
        
        # xG factor
        if state.home_xg + state.away_xg > 0:
            xg_ratio = (state.home_xg - state.away_xg) / (state.home_xg + state.away_xg + 0.1)
            factors.append(('xg', xg_ratio, 0.2))
        
        # Red card impact
        if state.home_red_cards > state.away_red_cards:
            factors.append(('red_cards', -0.3, 0.5))
        elif state.away_red_cards > state.home_red_cards:
            factors.append(('red_cards', 0.3, 0.5))
        
        # Calculate weighted momentum
        momentum = sum(f[1] * f[2] for f in factors) / sum(f[2] for f in factors) if factors else 0
        
        # Determine dominant team
        if momentum > 0.2:
            dominant = 'home'
            strength = 'strong' if momentum > 0.5 else 'moderate'
        elif momentum < -0.2:
            dominant = 'away'
            strength = 'strong' if momentum < -0.5 else 'moderate'
        else:
            dominant = 'balanced'
            strength = 'even'
        
        return {
            'momentum_score': round(momentum, 3),
            'dominant_team': dominant,
            'strength': strength,
            'factors': {f[0]: round(f[1], 3) for f in factors},
            'trend': 'improving' if len(state.events) > 2 and momentum > 0 else 'declining' if momentum < 0 else 'stable'
        }


class LiveProbabilityEngine:
    """Calculate live match probabilities"""
    
    def __init__(self):
        self.momentum_analyzer = MomentumAnalyzer()
        
    def calculate_probabilities(
        self,
        state: LiveMatchState,
        pre_match_home_prob: float = 0.45,
        pre_match_draw_prob: float = 0.25,
        pre_match_away_prob: float = 0.30
    ) -> LivePrediction:
        """Calculate live probabilities"""
        
        # Get momentum
        momentum = self.momentum_analyzer.analyze(state)
        mom_score = momentum['momentum_score']
        
        # Time remaining factor
        minutes_remaining = 90 - state.minute
        time_factor = minutes_remaining / 90
        
        # Current score impact
        score_diff = state.home_score - state.away_score
        
        # Calculate win probabilities
        if score_diff > 0:
            # Home leading
            home_lock = 1 - (0.5 * time_factor)  # More locked in as time passes
            home_prob = min(0.95, home_lock + 0.1 * mom_score)
            away_prob = max(0.02, 0.3 * time_factor - 0.1 * score_diff)
            draw_prob = 1 - home_prob - away_prob
        elif score_diff < 0:
            # Away leading
            away_lock = 1 - (0.5 * time_factor)
            away_prob = min(0.95, away_lock - 0.1 * mom_score)
            home_prob = max(0.02, 0.3 * time_factor + 0.1 * score_diff)
            draw_prob = 1 - home_prob - away_prob
        else:
            # Draw
            draw_base = 0.33 + (1 - time_factor) * 0.2
            home_prob = (1 - draw_base) / 2 + 0.15 * mom_score
            away_prob = (1 - draw_base) / 2 - 0.15 * mom_score
            draw_prob = draw_base
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        # Next goal probabilities
        xg_rate_home = (state.home_xg / max(state.minute, 1)) * 90
        xg_rate_away = (state.away_xg / max(state.minute, 1)) * 90
        
        remaining_goals = (xg_rate_home + xg_rate_away) * (minutes_remaining / 90)
        
        if remaining_goals > 0:
            next_home = (xg_rate_home / (xg_rate_home + xg_rate_away + 0.1)) * (1 - math.exp(-remaining_goals))
            next_away = (xg_rate_away / (xg_rate_home + xg_rate_away + 0.1)) * (1 - math.exp(-remaining_goals))
            no_goal = math.exp(-remaining_goals)
        else:
            next_home = 0.35
            next_away = 0.35
            no_goal = 0.30
        
        # Goals markets
        current_goals = state.home_score + state.away_score
        expected_total = current_goals + remaining_goals
        over_25 = 1 - sum(
            (expected_total ** k) * math.exp(-expected_total) / math.factorial(k)
            for k in range(max(0, 3 - current_goals))
        ) if current_goals < 3 else 1.0
        
        # BTTS
        if state.home_score > 0 and state.away_score > 0:
            btts = 1.0
        elif state.home_score > 0:
            btts = 1 - math.exp(-xg_rate_away * time_factor)
        elif state.away_score > 0:
            btts = 1 - math.exp(-xg_rate_home * time_factor)
        else:
            btts = (1 - math.exp(-xg_rate_home * time_factor)) * (1 - math.exp(-xg_rate_away * time_factor))
        
        # Determine recommended bet
        recommended = None
        edge = 0.0
        confidence = 0.0
        
        if home_prob > 0.75 and state.minute > 60:
            recommended = f"{state.home} to win"
            edge = home_prob - 0.65
            confidence = home_prob
        elif away_prob > 0.75 and state.minute > 60:
            recommended = f"{state.away} to win"
            edge = away_prob - 0.65
            confidence = away_prob
        elif over_25 > 0.85 and current_goals < 3:
            recommended = "Over 2.5 goals"
            edge = over_25 - 0.75
            confidence = over_25
        elif btts > 0.80 and (state.home_score == 0 or state.away_score == 0):
            recommended = "Both teams to score"
            edge = btts - 0.70
            confidence = btts
        
        return LivePrediction(
            match_id=state.match_id,
            timestamp=datetime.now().isoformat(),
            minute=state.minute,
            current_score=f"{state.home_score}-{state.away_score}",
            home_win_prob=home_prob,
            draw_prob=draw_prob,
            away_win_prob=away_prob,
            next_goal_home_prob=next_home,
            next_goal_away_prob=next_away,
            no_more_goals_prob=no_goal,
            over_25_prob=over_25,
            btts_prob=btts,
            recommended_bet=recommended,
            edge=edge,
            confidence=confidence
        )


class LiveBettingManager:
    """Manage live betting recommendations"""
    
    def __init__(self):
        self.probability_engine = LiveProbabilityEngine()
        self.live_matches: Dict[str, LiveMatchState] = {}
        self.predictions_history: Dict[str, List[LivePrediction]] = {}
        
    def register_match(self, home: str, away: str) -> str:
        """Register a new live match"""
        match_id = f"{home}_{away}_{datetime.now().strftime('%Y%m%d')}"
        
        self.live_matches[match_id] = LiveMatchState(
            match_id=match_id,
            home=home,
            away=away
        )
        self.predictions_history[match_id] = []
        
        return match_id
    
    def update_match(self, match_id: str, updates: Dict) -> Optional[LivePrediction]:
        """Update match state and get new prediction"""
        if match_id not in self.live_matches:
            return None
        
        state = self.live_matches[match_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        # Add event if goal scored
        if 'home_score' in updates or 'away_score' in updates:
            state.events.append({
                'type': 'goal',
                'minute': state.minute,
                'score': f"{state.home_score}-{state.away_score}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Generate prediction
        prediction = self.probability_engine.calculate_probabilities(state)
        self.predictions_history[match_id].append(prediction)
        
        return prediction
    
    def get_live_prediction(self, match_id: str) -> Optional[Dict]:
        """Get current prediction for a live match"""
        if match_id not in self.live_matches:
            return None
        
        state = self.live_matches[match_id]
        momentum = MomentumAnalyzer().analyze(state)
        prediction = self.probability_engine.calculate_probabilities(state)
        
        return {
            'match': state.to_dict(),
            'momentum': momentum,
            'prediction': prediction.to_dict(),
            'history_count': len(self.predictions_history.get(match_id, []))
        }
    
    def get_all_live_matches(self) -> List[Dict]:
        """Get all live matches with predictions"""
        result = []
        for match_id, state in self.live_matches.items():
            if state.phase != MatchPhase.FINISHED:
                pred = self.get_live_prediction(match_id)
                if pred:
                    result.append(pred)
        return result
    
    def find_live_value_bets(self, min_edge: float = 0.1) -> List[Dict]:
        """Find value betting opportunities in live matches"""
        opportunities = []
        
        for match_id in self.live_matches:
            pred_data = self.get_live_prediction(match_id)
            if pred_data:
                pred = pred_data['prediction']
                if pred['recommendation']['edge'] >= min_edge * 100:
                    opportunities.append({
                        'match_id': match_id,
                        'match': f"{pred_data['match']['home']} vs {pred_data['match']['away']}",
                        'minute': pred_data['match']['minute'],
                        'score': pred_data['match']['score'],
                        'bet': pred['recommendation']['bet'],
                        'edge': pred['recommendation']['edge'],
                        'confidence': pred['recommendation']['confidence']
                    })
        
        return sorted(opportunities, key=lambda x: x['edge'], reverse=True)


# Global instance
live_betting_manager = LiveBettingManager()


def register_live_match(home: str, away: str) -> str:
    """Register a match for live tracking"""
    return live_betting_manager.register_match(home, away)


def update_live_match(match_id: str, **updates) -> Optional[Dict]:
    """Update live match and get prediction"""
    pred = live_betting_manager.update_match(match_id, updates)
    return pred.to_dict() if pred else None


def get_live_prediction(match_id: str) -> Optional[Dict]:
    """Get live prediction for a match"""
    return live_betting_manager.get_live_prediction(match_id)


def get_all_live_predictions() -> List[Dict]:
    """Get all live match predictions"""
    return live_betting_manager.get_all_live_matches()


def find_live_value_bets(min_edge: float = 0.1) -> List[Dict]:
    """Find live value betting opportunities"""
    return live_betting_manager.find_live_value_bets(min_edge)
