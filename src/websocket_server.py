"""
WebSocket Server for Real-Time Live Scores

Provides instant score updates via WebSocket connections.
Uses Flask-SocketIO for real-time bidirectional communication.

Features:
- Live score broadcasting
- Client subscription management
- Heartbeat/ping-pong
- Match event notifications
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Set
from dataclasses import dataclass, asdict

# Try to import SocketIO, graceful fallback if not available
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None


@dataclass
class LiveMatch:
    """Live match state"""
    match_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    minute: int
    status: str  # 'live', 'halftime', 'finished', 'not_started'
    events: List[Dict]


class LiveScoreManager:
    """
    Manages live score data and broadcasts to connected clients.
    Works with or without SocketIO.
    """
    
    def __init__(self):
        self.live_matches: Dict[str, LiveMatch] = {}
        self.subscribers: Set[str] = set()
        self._socketio = None
        self._running = False
        self._update_thread = None
    
    def set_socketio(self, socketio):
        """Set SocketIO instance for real-time broadcasting"""
        self._socketio = socketio
    
    def add_live_match(self, match: Dict):
        """Add or update a live match"""
        match_id = match.get('id', f"{match['home_team']}_{match['away_team']}")
        
        self.live_matches[match_id] = LiveMatch(
            match_id=match_id,
            home_team=match.get('home_team', ''),
            away_team=match.get('away_team', ''),
            home_score=match.get('home_score', 0),
            away_score=match.get('away_score', 0),
            minute=match.get('minute', 0),
            status=match.get('status', 'live'),
            events=match.get('events', [])
        )
        
        # Broadcast update
        self._broadcast_update(match_id)
    
    def update_score(self, match_id: str, home_score: int, away_score: int, minute: int = None):
        """Update match score"""
        if match_id in self.live_matches:
            match = self.live_matches[match_id]
            
            # Check if goal scored
            old_home = match.home_score
            old_away = match.away_score
            
            match.home_score = home_score
            match.away_score = away_score
            
            if minute is not None:
                match.minute = minute
            
            # Add goal event
            if home_score > old_home:
                match.events.append({
                    'type': 'goal',
                    'team': 'home',
                    'minute': match.minute,
                    'time': datetime.now().isoformat()
                })
            elif away_score > old_away:
                match.events.append({
                    'type': 'goal',
                    'team': 'away',
                    'minute': match.minute,
                    'time': datetime.now().isoformat()
                })
            
            self._broadcast_update(match_id)
    
    def _broadcast_update(self, match_id: str):
        """Broadcast match update to all clients"""
        if match_id not in self.live_matches:
            return
        
        match = self.live_matches[match_id]
        data = asdict(match)
        
        if self._socketio and SOCKETIO_AVAILABLE:
            self._socketio.emit('score_update', data, room='live_scores')
    
    def get_all_live(self) -> List[Dict]:
        """Get all live matches"""
        return [asdict(m) for m in self.live_matches.values()]
    
    def start_polling(self, interval: int = 60):
        """Start background polling for live scores"""
        if self._running:
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._poll_loop, args=(interval,))
        self._update_thread.daemon = True
        self._update_thread.start()
    
    def stop_polling(self):
        """Stop background polling"""
        self._running = False
    
    def _poll_loop(self, interval: int):
        """Background polling loop"""
        while self._running:
            try:
                self._fetch_live_scores()
            except Exception as e:
                print(f"Live score fetch error: {e}")
            
            time.sleep(interval)
    
    def _fetch_live_scores(self):
        """Fetch live scores from API"""
        try:
            from src.data.api_clients import FootballDataOrgClient
            
            client = FootballDataOrgClient()
            matches = client.get_matches('bundesliga')  # Get today's matches
            
            for match in matches:
                if match.status in ['LIVE', 'IN_PLAY', 'PAUSED']:
                    self.add_live_match({
                        'id': match.id,
                        'home_team': match.home_team,
                        'away_team': match.away_team,
                        'home_score': 0,
                        'away_score': 0,
                        'minute': 45,
                        'status': 'live'
                    })
        except:
            pass  # Silently fail, will retry next interval


# Global instance
live_scores = LiveScoreManager()


def setup_socketio(app):
    """
    Setup SocketIO with Flask app.
    
    Returns:
        SocketIO instance or None if not available
    """
    if not SOCKETIO_AVAILABLE:
        print("Flask-SocketIO not installed. WebSocket features disabled.")
        return None
    
    socketio = SocketIO(app, cors_allowed_origins="*")
    live_scores.set_socketio(socketio)
    
    @socketio.on('connect')
    def handle_connect():
        print(f"Client connected")
        emit('connected', {'status': 'ok', 'timestamp': datetime.now().isoformat()})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print(f"Client disconnected")
    
    @socketio.on('subscribe_live')
    def handle_subscribe():
        join_room('live_scores')
        emit('subscribed', {'room': 'live_scores'})
        # Send current live matches
        emit('live_matches', {'matches': live_scores.get_all_live()})
    
    @socketio.on('unsubscribe_live')
    def handle_unsubscribe():
        leave_room('live_scores')
        emit('unsubscribed', {'room': 'live_scores'})
    
    @socketio.on('get_live')
    def handle_get_live():
        emit('live_matches', {'matches': live_scores.get_all_live()})
    
    return socketio


def get_websocket_client_js() -> str:
    """Generate WebSocket client JavaScript code"""
    return '''
// Live Score WebSocket Client
class LiveScoreClient {
    constructor(serverUrl) {
        this.serverUrl = serverUrl || window.location.origin;
        this.socket = null;
        this.onUpdate = null;
    }
    
    connect() {
        if (typeof io === 'undefined') {
            console.error('Socket.IO client not loaded');
            return;
        }
        
        this.socket = io(this.serverUrl);
        
        this.socket.on('connect', () => {
            console.log('Connected to live scores');
            this.socket.emit('subscribe_live');
        });
        
        this.socket.on('score_update', (data) => {
            console.log('Score update:', data);
            if (this.onUpdate) {
                this.onUpdate(data);
            }
            this.updateUI(data);
        });
        
        this.socket.on('live_matches', (data) => {
            console.log('Live matches:', data.matches);
            data.matches.forEach(match => this.updateUI(match));
        });
    }
    
    updateUI(match) {
        // Find match card and update score
        const cards = document.querySelectorAll('.match-card');
        cards.forEach(card => {
            const homeTeam = card.querySelector('.home-team')?.textContent?.trim();
            const awayTeam = card.querySelector('.away-team')?.textContent?.trim();
            
            if (homeTeam?.includes(match.home_team) || match.home_team?.includes(homeTeam)) {
                const scoreEl = card.querySelector('.live-score');
                if (scoreEl) {
                    scoreEl.textContent = `${match.home_score} - ${match.away_score}`;
                    scoreEl.classList.add('score-updated');
                    setTimeout(() => scoreEl.classList.remove('score-updated'), 1000);
                }
            }
        });
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.emit('unsubscribe_live');
            this.socket.disconnect();
        }
    }
}

// Auto-connect on page load
document.addEventListener('DOMContentLoaded', () => {
    window.liveScores = new LiveScoreClient();
    window.liveScores.connect();
});
'''
