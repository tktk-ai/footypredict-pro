"""
Stream Processor Module
Real-time data stream processing.

Part of the complete blueprint implementation.
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Processes real-time data streams.
    
    Features:
    - Event buffering
    - Windowed aggregation
    - Real-time triggers
    - State management
    """
    
    def __init__(
        self,
        buffer_size: int = 1000,
        window_seconds: int = 60
    ):
        self.buffer_size = buffer_size
        self.window_seconds = window_seconds
        self.buffers: Dict[str, deque] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.state: Dict[str, Any] = {}
        self.is_running = False
    
    def register_handler(
        self,
        event_type: str,
        handler: Callable
    ):
        """Register an event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for: {event_type}")
    
    def process_event(
        self,
        event_type: str,
        data: Dict
    ):
        """Process a single event."""
        # Add timestamp
        data['_timestamp'] = datetime.now().isoformat()
        data['_event_type'] = event_type
        
        # Buffer event
        if event_type not in self.buffers:
            self.buffers[event_type] = deque(maxlen=self.buffer_size)
        self.buffers[event_type].append(data)
        
        # Trigger handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    handler(data, self.state)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
    
    def get_window(
        self,
        event_type: str,
        seconds: int = None
    ) -> List[Dict]:
        """Get events in time window."""
        seconds = seconds or self.window_seconds
        
        if event_type not in self.buffers:
            return []
        
        cutoff = datetime.now().timestamp() - seconds
        
        return [
            e for e in self.buffers[event_type]
            if datetime.fromisoformat(e['_timestamp']).timestamp() >= cutoff
        ]
    
    def aggregate_window(
        self,
        event_type: str,
        field: str,
        func: str = 'mean',
        seconds: int = None
    ) -> Optional[float]:
        """Aggregate field values in window."""
        events = self.get_window(event_type, seconds)
        
        if not events:
            return None
        
        values = [e.get(field) for e in events if field in e and e[field] is not None]
        
        if not values:
            return None
        
        if func == 'mean':
            return sum(values) / len(values)
        elif func == 'sum':
            return sum(values)
        elif func == 'max':
            return max(values)
        elif func == 'min':
            return min(values)
        elif func == 'count':
            return len(values)
        
        return None
    
    def set_state(self, key: str, value: Any):
        """Set processor state."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get processor state."""
        return self.state.get(key, default)
    
    def clear_buffer(self, event_type: str = None):
        """Clear event buffer."""
        if event_type:
            if event_type in self.buffers:
                self.buffers[event_type].clear()
        else:
            self.buffers.clear()


class LiveMatchProcessor(StreamProcessor):
    """Specialized processor for live match events."""
    
    def __init__(self):
        super().__init__(buffer_size=500, window_seconds=300)
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup default handlers."""
        self.register_handler('goal', self._handle_goal)
        self.register_handler('card', self._handle_card)
        self.register_handler('odds_update', self._handle_odds)
    
    def _handle_goal(self, event: Dict, state: Dict):
        """Handle goal event."""
        match_id = event.get('match_id')
        if match_id:
            key = f"goals_{match_id}"
            current = state.get(key, {'home': 0, 'away': 0})
            
            if event.get('team') == 'home':
                current['home'] += 1
            else:
                current['away'] += 1
            
            state[key] = current
            logger.info(f"Goal scored: {match_id} -> {current}")
    
    def _handle_card(self, event: Dict, state: Dict):
        """Handle card event."""
        match_id = event.get('match_id')
        if match_id:
            key = f"cards_{match_id}"
            cards = state.get(key, [])
            cards.append({
                'player': event.get('player'),
                'card_type': event.get('card_type'),
                'minute': event.get('minute')
            })
            state[key] = cards
    
    def _handle_odds(self, event: Dict, state: Dict):
        """Handle odds update."""
        match_id = event.get('match_id')
        if match_id:
            key = f"odds_{match_id}"
            state[key] = event.get('odds', {})
    
    def get_match_state(self, match_id: str) -> Dict:
        """Get current state for a match."""
        return {
            'goals': self.get_state(f"goals_{match_id}", {'home': 0, 'away': 0}),
            'cards': self.get_state(f"cards_{match_id}", []),
            'odds': self.get_state(f"odds_{match_id}", {}),
            'recent_events': self.get_window(match_id, seconds=60)
        }


_processor: Optional[StreamProcessor] = None
_match_processor: Optional[LiveMatchProcessor] = None

def get_processor() -> StreamProcessor:
    global _processor
    if _processor is None:
        _processor = StreamProcessor()
    return _processor

def get_match_processor() -> LiveMatchProcessor:
    global _match_processor
    if _match_processor is None:
        _match_processor = LiveMatchProcessor()
    return _match_processor
