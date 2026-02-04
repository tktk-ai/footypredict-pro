"""
Bet Tracking System

Allows users to:
- Record their bets
- Track profit/loss
- Analyze betting patterns
- View betting history
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Bet:
    """Individual bet record"""
    id: str
    user_id: str
    match_id: str
    home_team: str
    away_team: str
    selection: str
    odds: float
    stake: float
    potential_return: float
    status: str  # 'pending', 'won', 'lost', 'void'
    result: Optional[str]
    profit_loss: float
    created_at: str
    settled_at: Optional[str]
    notes: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BetTracker:
    """
    Track user bets and calculate statistics.
    """
    
    def __init__(self, data_dir: str = "data/bets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bets: Dict[str, List[Bet]] = {}  # user_id -> bets
        self._load_bets()
        self._counter = 0
    
    def _load_bets(self):
        """Load bets from file"""
        filepath = self.data_dir / "bets.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for user_id, bets_data in data.items():
                        self.bets[user_id] = [Bet(**b) for b in bets_data]
            except:
                pass
    
    def _save_bets(self):
        """Save bets to file"""
        filepath = self.data_dir / "bets.json"
        data = {
            user_id: [b.to_dict() for b in bets]
            for user_id, bets in self.bets.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_id(self) -> str:
        self._counter += 1
        return f"bet_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._counter:04d}"
    
    def add_bet(
        self,
        user_id: str,
        match_id: str,
        home_team: str,
        away_team: str,
        selection: str,
        odds: float,
        stake: float,
        notes: str = None
    ) -> Bet:
        """Add a new bet"""
        bet = Bet(
            id=self._generate_id(),
            user_id=user_id,
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            selection=selection,
            odds=odds,
            stake=stake,
            potential_return=stake * odds,
            status='pending',
            result=None,
            profit_loss=0,
            created_at=datetime.now().isoformat(),
            settled_at=None,
            notes=notes
        )
        
        if user_id not in self.bets:
            self.bets[user_id] = []
        
        self.bets[user_id].append(bet)
        self._save_bets()
        
        return bet
    
    def settle_bet(self, bet_id: str, user_id: str, won: bool) -> Optional[Bet]:
        """Settle a pending bet"""
        if user_id not in self.bets:
            return None
        
        for bet in self.bets[user_id]:
            if bet.id == bet_id:
                bet.status = 'won' if won else 'lost'
                bet.settled_at = datetime.now().isoformat()
                
                if won:
                    bet.profit_loss = bet.potential_return - bet.stake
                else:
                    bet.profit_loss = -bet.stake
                
                self._save_bets()
                return bet
        
        return None
    
    def get_user_bets(self, user_id: str, status: str = None, limit: int = 50) -> List[Dict]:
        """Get user's betting history"""
        if user_id not in self.bets:
            return []
        
        bets = self.bets[user_id]
        
        if status:
            bets = [b for b in bets if b.status == status]
        
        # Sort by created_at descending
        bets.sort(key=lambda x: x.created_at, reverse=True)
        
        return [b.to_dict() for b in bets[:limit]]
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Calculate user betting statistics"""
        if user_id not in self.bets:
            return self._empty_stats()
        
        bets = self.bets[user_id]
        
        if not bets:
            return self._empty_stats()
        
        total_bets = len(bets)
        settled_bets = [b for b in bets if b.status in ['won', 'lost']]
        won_bets = [b for b in bets if b.status == 'won']
        pending_bets = [b for b in bets if b.status == 'pending']
        
        total_staked = sum(b.stake for b in settled_bets)
        total_profit = sum(b.profit_loss for b in settled_bets)
        total_pending = sum(b.stake for b in pending_bets)
        
        win_rate = len(won_bets) / len(settled_bets) if settled_bets else 0
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        
        avg_odds = sum(b.odds for b in settled_bets) / len(settled_bets) if settled_bets else 0
        
        # Calculate streak
        streak = 0
        streak_type = None
        for bet in sorted(settled_bets, key=lambda x: x.settled_at or '', reverse=True):
            if streak_type is None:
                streak_type = bet.status
                streak = 1
            elif bet.status == streak_type:
                streak += 1
            else:
                break
        
        return {
            'total_bets': total_bets,
            'settled_bets': len(settled_bets),
            'pending_bets': len(pending_bets),
            'won_bets': len(won_bets),
            'lost_bets': len(settled_bets) - len(won_bets),
            'win_rate': round(win_rate * 100, 1),
            'total_staked': round(total_staked, 2),
            'total_profit': round(total_profit, 2),
            'total_pending': round(total_pending, 2),
            'roi': round(roi, 1),
            'avg_odds': round(avg_odds, 2),
            'current_streak': streak,
            'streak_type': streak_type
        }
    
    def _empty_stats(self) -> Dict:
        return {
            'total_bets': 0,
            'settled_bets': 0,
            'pending_bets': 0,
            'won_bets': 0,
            'lost_bets': 0,
            'win_rate': 0,
            'total_staked': 0,
            'total_profit': 0,
            'total_pending': 0,
            'roi': 0,
            'avg_odds': 0,
            'current_streak': 0,
            'streak_type': None
        }
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top users by profit"""
        user_stats = []
        
        for user_id in self.bets.keys():
            stats = self.get_user_stats(user_id)
            if stats['settled_bets'] >= 5:  # Minimum 5 bets to qualify
                user_stats.append({
                    'user_id': user_id,
                    'profit': stats['total_profit'],
                    'roi': stats['roi'],
                    'win_rate': stats['win_rate'],
                    'total_bets': stats['total_bets']
                })
        
        # Sort by profit
        user_stats.sort(key=lambda x: x['profit'], reverse=True)
        
        # Add rank
        for i, stats in enumerate(user_stats[:limit]):
            stats['rank'] = i + 1
        
        return user_stats[:limit]


# Global instance
bet_tracker = BetTracker()
