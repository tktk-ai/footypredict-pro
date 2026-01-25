"""
WebSocket Real-Time Updates

Provides real-time updates for:
- Live match scores
- Odds changes
- New predictions
- Accumulator updates
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    """WebSocket event types"""
    MATCH_UPDATE = "match_update"
    ODDS_UPDATE = "odds_update"
    PREDICTION = "prediction"
    GOAL = "goal"
    ACCUMULATOR = "accumulator"
    ALERT = "alert"
    SURE_WIN = "sure_win"
    LIVE_SCORE = "live_score"


@dataclass
class RealtimeEvent:
    """Real-time event structure"""
    event_type: str
    data: Dict
    timestamp: str
    channel: str = "global"
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class EventEmitter:
    """Event emitter for real-time updates"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._event_history: List[RealtimeEvent] = []
        self._max_history = 100
        
    def on(self, event_type: str, callback: Callable):
        """Register event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        
    def off(self, event_type: str, callback: Callable):
        """Remove event listener"""
        if event_type in self._listeners:
            self._listeners[event_type] = [
                cb for cb in self._listeners[event_type] if cb != callback
            ]
            
    def emit(self, event_type: str, data: Dict, channel: str = "global"):
        """Emit an event"""
        event = RealtimeEvent(
            event_type=event_type,
            data=data,
            timestamp=datetime.now().isoformat(),
            channel=channel
        )
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # Notify listeners
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Event listener error: {e}")
                    
        # Also notify global listeners
        if "*" in self._listeners:
            for callback in self._listeners["*"]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Global listener error: {e}")
                    
    def get_history(self, event_type: str = None, limit: int = 50) -> List[Dict]:
        """Get event history"""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return [asdict(e) for e in events[-limit:]]


class LiveMatchTracker:
    """Track live matches and emit updates"""
    
    def __init__(self, emitter: EventEmitter):
        self.emitter = emitter
        self.live_matches: Dict[str, Dict] = {}
        self.polling_interval = 30  # seconds
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        
    def add_match(self, match_id: str, match_data: Dict):
        """Add match to tracking"""
        self.live_matches[match_id] = {
            'data': match_data,
            'last_update': time.time(),
            'goals': [],
            'events': []
        }
        
        self.emitter.emit(
            EventType.MATCH_UPDATE.value,
            {'action': 'added', 'match': match_data},
            channel=f"match_{match_id}"
        )
        
    def update_score(self, match_id: str, home_score: int, away_score: int):
        """Update match score"""
        if match_id in self.live_matches:
            match = self.live_matches[match_id]
            old_home = match['data'].get('home_score', 0)
            old_away = match['data'].get('away_score', 0)
            
            match['data']['home_score'] = home_score
            match['data']['away_score'] = away_score
            match['last_update'] = time.time()
            
            # Emit score update
            self.emitter.emit(
                EventType.LIVE_SCORE.value,
                {
                    'match_id': match_id,
                    'home_score': home_score,
                    'away_score': away_score,
                    'match': match['data']
                },
                channel=f"match_{match_id}"
            )
            
            # Check for goals
            if home_score > old_home:
                self.emitter.emit(
                    EventType.GOAL.value,
                    {
                        'match_id': match_id,
                        'team': 'home',
                        'new_score': f"{home_score}-{away_score}",
                        'match': match['data']
                    },
                    channel="goals"
                )
                
            if away_score > old_away:
                self.emitter.emit(
                    EventType.GOAL.value,
                    {
                        'match_id': match_id,
                        'team': 'away',
                        'new_score': f"{home_score}-{away_score}",
                        'match': match['data']
                    },
                    channel="goals"
                )
                
    def get_live_matches(self) -> List[Dict]:
        """Get all live matches"""
        return [m['data'] for m in self.live_matches.values()]
    
    def remove_match(self, match_id: str):
        """Remove match from tracking"""
        if match_id in self.live_matches:
            match_data = self.live_matches.pop(match_id)
            self.emitter.emit(
                EventType.MATCH_UPDATE.value,
                {'action': 'removed', 'match': match_data['data']},
                channel=f"match_{match_id}"
            )


class OddsTracker:
    """Track odds changes and emit updates"""
    
    def __init__(self, emitter: EventEmitter):
        self.emitter = emitter
        self.odds_history: Dict[str, List[Dict]] = {}
        
    def update_odds(self, match_id: str, bookmaker: str, odds: Dict):
        """Update odds for a match"""
        key = f"{match_id}_{bookmaker}"
        
        if key not in self.odds_history:
            self.odds_history[key] = []
            
        # Check for significant changes
        if self.odds_history[key]:
            last = self.odds_history[key][-1]
            changed = False
            
            for outcome, value in odds.items():
                if outcome in last and abs(value - last[outcome]) > 0.05:
                    changed = True
                    break
                    
            if changed:
                self.emitter.emit(
                    EventType.ODDS_UPDATE.value,
                    {
                        'match_id': match_id,
                        'bookmaker': bookmaker,
                        'old_odds': last,
                        'new_odds': odds,
                        'timestamp': datetime.now().isoformat()
                    },
                    channel="odds"
                )
        
        # Store odds
        odds['timestamp'] = time.time()
        self.odds_history[key].append(odds)
        
        # Keep only last 50 entries
        if len(self.odds_history[key]) > 50:
            self.odds_history[key] = self.odds_history[key][-50:]
            
    def get_odds_movement(self, match_id: str, bookmaker: str) -> List[Dict]:
        """Get odds movement history"""
        key = f"{match_id}_{bookmaker}"
        return self.odds_history.get(key, [])


class AlertManager:
    """Manage and emit alerts"""
    
    def __init__(self, emitter: EventEmitter):
        self.emitter = emitter
        self.active_alerts: List[Dict] = []
        
    def send_sure_win_alert(self, prediction: Dict):
        """Send alert for sure win prediction"""
        alert = {
            'type': 'sure_win',
            'title': '🔒 Sure Win Alert!',
            'message': f"{prediction.get('home')} vs {prediction.get('away')}",
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        self.active_alerts.append(alert)
        self.emitter.emit(
            EventType.SURE_WIN.value,
            alert,
            channel="alerts"
        )
        
    def send_value_bet_alert(self, bet: Dict):
        """Send alert for value bet"""
        alert = {
            'type': 'value_bet',
            'title': '💎 Value Bet Found!',
            'message': f"Edge: {bet.get('edge', 0):.1f}%",
            'bet': bet,
            'timestamp': datetime.now().isoformat()
        }
        
        self.active_alerts.append(alert)
        self.emitter.emit(
            EventType.ALERT.value,
            alert,
            channel="alerts"
        )
        
    def get_active_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        return self.active_alerts[-limit:]


# Global instances
event_emitter = EventEmitter()
live_tracker = LiveMatchTracker(event_emitter)
odds_tracker = OddsTracker(event_emitter)
alert_manager = AlertManager(event_emitter)


def emit_prediction(prediction: Dict):
    """Emit a new prediction event"""
    event_emitter.emit(
        EventType.PREDICTION.value,
        prediction,
        channel="predictions"
    )
    
    # Check for sure win
    if prediction.get('confidence', 0) >= 0.91:
        alert_manager.send_sure_win_alert(prediction)


def emit_accumulator_update(acca: Dict):
    """Emit accumulator update"""
    event_emitter.emit(
        EventType.ACCUMULATOR.value,
        acca,
        channel="accumulators"
    )


def get_event_stream() -> Dict:
    """Get current event stream status"""
    return {
        'live_matches': len(live_tracker.live_matches),
        'active_alerts': len(alert_manager.active_alerts),
        'recent_events': event_emitter.get_history(limit=10),
        'channels': list(event_emitter._listeners.keys())
    }
