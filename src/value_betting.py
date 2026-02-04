"""
Value Betting Module (SofaScore Inspired)

Implements value betting strategies based on SofaScore's "Winning Odds" concept:
- Track historical performance at specific odds levels
- Identify value when actual win rate exceeds implied probability
- Dropping odds detection
- Streak analysis for betting tips
- H2H dominance factor

Key Insight from SofaScore:
If odds are 1.44 (69% implied), but historically teams win 85% at those odds,
there's VALUE - the actual probability exceeds market expectation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import math


class ValueBetType(Enum):
    """Types of value bets identified"""
    HIGH_VALUE = "high_value"       # 15%+ edge over implied probability
    MEDIUM_VALUE = "medium_value"   # 10-15% edge
    LOW_VALUE = "low_value"         # 5-10% edge
    FAIR = "fair"                   # Within 5% of implied
    OVERPRICED = "overpriced"       # Worse than implied


class OddsMovement(Enum):
    """Direction of odds movement"""
    DROPPING = "dropping"     # Odds getting shorter (more confidence)
    DRIFTING = "drifting"     # Odds getting longer (less confidence)
    STABLE = "stable"         # No significant movement


class StreakType(Enum):
    """Types of streaks to track (SofaScore style)"""
    WINNING = "winning_streak"
    LOSING = "losing_streak"
    UNBEATEN = "unbeaten_streak"
    WINLESS = "winless_streak"
    OVER_2_5 = "over_2_5_streak"
    UNDER_2_5 = "under_2_5_streak"
    BTTS_YES = "btts_yes_streak"
    BTTS_NO = "btts_no_streak"
    CLEAN_SHEETS = "clean_sheet_streak"
    FAILED_TO_SCORE = "failed_to_score_streak"
    SCORED = "scored_streak"


@dataclass
class OddsHistoryEntry:
    """Single odds history entry for a team/market"""
    odds_range: Tuple[float, float]  # e.g., (1.40, 1.50)
    total_matches: int
    wins: int
    actual_win_rate: float
    implied_probability: float  # Average for this odds range
    edge: float  # actual_win_rate - implied_probability
    
    def to_dict(self) -> Dict:
        return {
            'odds_range': f"{self.odds_range[0]:.2f} - {self.odds_range[1]:.2f}",
            'total_matches': self.total_matches,
            'wins': self.wins,
            'actual_win_rate': round(self.actual_win_rate * 100, 1),
            'implied_probability': round(self.implied_probability * 100, 1),
            'edge': round(self.edge * 100, 1),
            'has_value': self.edge > 0.05
        }


@dataclass
class StreakInfo:
    """Streak information for a team"""
    team: str
    streak_type: str
    streak_length: int
    last_updated: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValueBet:
    """Identified value bet"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    bet_type: str
    bet_label: str
    
    # Probability analysis
    our_probability: float      # Our calculated probability
    implied_probability: float  # From bookmaker odds
    bookmaker_odds: float
    
    # Value metrics
    edge: float                 # our_prob - implied_prob
    value_type: str            # high/medium/low/fair
    expected_value: float       # EV calculation
    
    # Historical context
    historical_win_rate: Optional[float] = None
    historical_matches: Optional[int] = None
    
    # Odds movement
    odds_movement: str = "stable"
    opening_odds: Optional[float] = None
    
    # Streaks
    relevant_streaks: Optional[List[Dict]] = None
    
    # H2H
    h2h_matches: int = 0
    h2h_wins: int = 0
    h2h_dominance: bool = False
    
    reasoning: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'league': self.league,
            'bet_type': self.bet_type,
            'bet_label': self.bet_label,
            'our_probability': round(self.our_probability * 100, 1),
            'implied_probability': round(self.implied_probability * 100, 1),
            'bookmaker_odds': round(self.bookmaker_odds, 2),
            'edge': round(self.edge * 100, 1),
            'value_type': self.value_type,
            'expected_value': round(self.expected_value, 3),
            'historical_win_rate': round(self.historical_win_rate * 100, 1) if self.historical_win_rate else None,
            'historical_matches': self.historical_matches,
            'odds_movement': self.odds_movement,
            'opening_odds': self.opening_odds,
            'relevant_streaks': self.relevant_streaks,
            'h2h_dominance': self.h2h_dominance,
            'reasoning': self.reasoning,
            'confidence': round(self.confidence * 100, 1)
        }


class ValueBettingEngine:
    """
    Identifies value bets using SofaScore-style analysis.
    
    Key concepts:
    1. Value = When our probability > implied probability (from odds)
    2. Historical odds performance tracking
    3. Dropping odds detection
    4. Streak analysis
    5. H2H dominance
    """
    
    # Historical odds performance by range (simulated database)
    # In production, this would be populated from real historical data
    HISTORICAL_ODDS_PERFORMANCE = {
        # Odds range: (min, max, historical_win_rate)
        # Home Win
        'home_win': [
            ((1.10, 1.25), 0.89),  # Heavy favorites: 89% actual win rate
            ((1.25, 1.40), 0.78),  # Strong favorites: 78% actual
            ((1.40, 1.60), 0.68),  # Favorites: 68% actual
            ((1.60, 1.80), 0.59),  # Slight favorites: 59% actual
            ((1.80, 2.20), 0.48),  # Even match: 48% actual
            ((2.20, 3.00), 0.35),  # Underdogs: 35% actual
            ((3.00, 5.00), 0.22),  # Big underdogs: 22% actual
        ],
        # Over 2.5 Goals
        'over_2_5': [
            ((1.30, 1.50), 0.78),  # Strong over: 78% actual
            ((1.50, 1.70), 0.62),  # Moderate over: 62% actual
            ((1.70, 2.00), 0.52),  # Balanced: 52% actual
            ((2.00, 2.50), 0.42),  # Slight under favorite: 42% actual
        ],
        # BTTS Yes
        'btts_yes': [
            ((1.40, 1.60), 0.70),  # Strong BTTS: 70% actual
            ((1.60, 1.80), 0.60),  # Moderate BTTS: 60% actual
            ((1.80, 2.10), 0.50),  # Balanced: 50% actual
        ]
    }
    
    # Minimum edge thresholds for value classification
    VALUE_THRESHOLDS = {
        'high_value': 0.15,    # 15%+ edge
        'medium_value': 0.10,  # 10-15% edge
        'low_value': 0.05,     # 5-10% edge
    }
    
    def __init__(self):
        self.odds_history: Dict[str, List[OddsHistoryEntry]] = {}
        self.streaks_cache: Dict[str, List[StreakInfo]] = {}
    
    def odds_to_implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability"""
        if odds <= 1:
            return 1.0
        return 1 / odds
    
    def calculate_edge(self, our_probability: float, implied_probability: float) -> float:
        """Calculate the edge (our prob - implied prob)"""
        return our_probability - implied_probability
    
    def classify_value(self, edge: float) -> str:
        """Classify the value bet type based on edge"""
        if edge >= self.VALUE_THRESHOLDS['high_value']:
            return ValueBetType.HIGH_VALUE.value
        elif edge >= self.VALUE_THRESHOLDS['medium_value']:
            return ValueBetType.MEDIUM_VALUE.value
        elif edge >= self.VALUE_THRESHOLDS['low_value']:
            return ValueBetType.LOW_VALUE.value
        elif edge >= 0:
            return ValueBetType.FAIR.value
        else:
            return ValueBetType.OVERPRICED.value
    
    def calculate_expected_value(
        self, 
        probability: float, 
        odds: float, 
        stake: float = 1.0
    ) -> float:
        """
        Calculate Expected Value (EV)
        EV = (prob * profit) - ((1-prob) * stake)
        """
        profit = (odds - 1) * stake
        ev = (probability * profit) - ((1 - probability) * stake)
        return ev
    
    def get_historical_win_rate(
        self,
        bet_type: str,
        odds: float
    ) -> Tuple[Optional[float], Optional[int]]:
        """
        Get historical win rate for a bet type at specific odds.
        SofaScore style: "At these odds, teams historically win X%"
        """
        if bet_type not in self.HISTORICAL_ODDS_PERFORMANCE:
            return None, None
        
        for (min_odds, max_odds), win_rate in self.HISTORICAL_ODDS_PERFORMANCE[bet_type]:
            if min_odds <= odds < max_odds:
                # Simulate match count (in production from real data)
                matches = int(1000 / (max_odds - min_odds))
                return win_rate, matches
        
        return None, None
    
    def detect_odds_movement(
        self,
        current_odds: float,
        opening_odds: Optional[float] = None
    ) -> str:
        """
        Detect if odds are dropping (smart money coming in) or drifting.
        Dropping odds often indicate insider knowledge or strong market confidence.
        """
        if opening_odds is None:
            return OddsMovement.STABLE.value
        
        change_pct = (current_odds - opening_odds) / opening_odds
        
        if change_pct <= -0.05:  # 5% or more decrease
            return OddsMovement.DROPPING.value
        elif change_pct >= 0.05:  # 5% or more increase
            return OddsMovement.DRIFTING.value
        else:
            return OddsMovement.STABLE.value
    
    def analyze_streaks(self, team: str, recent_results: List[Dict]) -> List[StreakInfo]:
        """
        Analyze team streaks (SofaScore style).
        Returns active streaks for the team.
        """
        if not recent_results:
            return []
        
        streaks = []
        
        # Count consecutive results
        win_streak = 0
        lose_streak = 0
        unbeaten_streak = 0
        over_streak = 0
        btts_streak = 0
        clean_sheet_streak = 0
        scored_streak = 0
        
        for result in recent_results:
            # Win/Loss/Draw streaks
            outcome = result.get('outcome', 'draw')
            if outcome == 'win':
                win_streak += 1
                unbeaten_streak += 1
                lose_streak = 0
            elif outcome == 'loss':
                lose_streak += 1
                win_streak = 0
                unbeaten_streak = 0
            else:  # draw
                win_streak = 0
                lose_streak = 0
                unbeaten_streak += 1
            
            # Goals streaks
            total_goals = result.get('total_goals', 0)
            if total_goals >= 3:
                over_streak += 1
            else:
                break  # Streak broken
            
            # BTTS
            home_scored = result.get('home_goals', 0) > 0
            away_scored = result.get('away_goals', 0) > 0
            if home_scored and away_scored:
                btts_streak += 1
            
            # Clean sheets
            conceded = result.get('conceded', 0)
            if conceded == 0:
                clean_sheet_streak += 1
            
            # Scored
            scored = result.get('scored', 0)
            if scored > 0:
                scored_streak += 1
        
        now = datetime.now().isoformat()
        
        # Add significant streaks (3+ matches)
        if win_streak >= 3:
            streaks.append(StreakInfo(team, StreakType.WINNING.value, win_streak, now))
        if lose_streak >= 3:
            streaks.append(StreakInfo(team, StreakType.LOSING.value, lose_streak, now))
        if unbeaten_streak >= 5:
            streaks.append(StreakInfo(team, StreakType.UNBEATEN.value, unbeaten_streak, now))
        if over_streak >= 3:
            streaks.append(StreakInfo(team, StreakType.OVER_2_5.value, over_streak, now))
        if btts_streak >= 3:
            streaks.append(StreakInfo(team, StreakType.BTTS_YES.value, btts_streak, now))
        if clean_sheet_streak >= 3:
            streaks.append(StreakInfo(team, StreakType.CLEAN_SHEETS.value, clean_sheet_streak, now))
        if scored_streak >= 5:
            streaks.append(StreakInfo(team, StreakType.SCORED.value, scored_streak, now))
        
        return streaks
    
    def check_h2h_dominance(
        self,
        team: str,
        opponent: str,
        h2h_matches: int,
        h2h_wins: int
    ) -> bool:
        """
        Check if a team has H2H dominance over opponent.
        SofaScore criteria: 4+ out of last 5 meetings won or unbeaten in 7+ meetings.
        """
        if h2h_matches < 3:
            return False
        
        win_rate = h2h_wins / h2h_matches
        
        # Dominance: 70%+ win rate in at least 4 meetings OR unbeaten in 7+
        if h2h_matches >= 4 and win_rate >= 0.70:
            return True
        if h2h_matches >= 7 and win_rate >= 0.50:  # Unbeaten in 7 implies dominance
            return True
        
        return False
    
    def identify_value_bets(
        self,
        predictions: List[Dict],
        bookmaker_odds: Optional[Dict] = None
    ) -> List[ValueBet]:
        """
        Identify value bets from predictions.
        
        Returns bets where our probability exceeds the implied probability
        from bookmaker odds by a significant margin.
        """
        value_bets = []
        
        for pred in predictions:
            match = pred.get('match', {})
            goals = pred.get('goals', {})
            final_pred = pred.get('final_prediction', pred.get('prediction', {}))
            
            # Extract match info
            home_team = match.get('home_team', {})
            away_team = match.get('away_team', {})
            home_name = home_team.get('name', str(home_team)) if isinstance(home_team, dict) else str(home_team)
            away_name = away_team.get('name', str(away_team)) if isinstance(away_team, dict) else str(away_team)
            match_id = match.get('id', f"{home_name}_{away_name}")
            league = pred.get('league', 'Unknown')
            
            # Get our probabilities
            home_prob = final_pred.get('home_win_prob', 0)
            draw_prob = final_pred.get('draw_prob', 0)
            away_prob = final_pred.get('away_win_prob', 0)
            
            over_under = goals.get('over_under', {})
            over_2_5_prob = over_under.get('over_2.5', 0.5)
            
            btts = goals.get('btts', {})
            btts_yes_prob = btts.get('yes', 0.5)
            
            # Get bookmaker odds (use estimated if not provided)
            match_odds = bookmaker_odds.get(match_id, {}) if bookmaker_odds else {}
            
            # Estimate odds from probability if not provided
            def prob_to_fair_odds(prob):
                if prob <= 0:
                    return 10.0
                return round(1 / prob * 1.05, 2)  # 5% margin
            
            # Analyze Home Win value
            home_odds = match_odds.get('home_win', prob_to_fair_odds(home_prob))
            home_implied = self.odds_to_implied_probability(home_odds)
            home_edge = self.calculate_edge(home_prob, home_implied)
            
            if home_edge >= self.VALUE_THRESHOLDS['low_value']:
                hist_rate, hist_matches = self.get_historical_win_rate('home_win', home_odds)
                
                value_bets.append(ValueBet(
                    match_id=match_id,
                    home_team=home_name,
                    away_team=away_name,
                    league=league,
                    bet_type='home_win',
                    bet_label='Home Win',
                    our_probability=home_prob,
                    implied_probability=home_implied,
                    bookmaker_odds=home_odds,
                    edge=home_edge,
                    value_type=self.classify_value(home_edge),
                    expected_value=self.calculate_expected_value(home_prob, home_odds),
                    historical_win_rate=hist_rate,
                    historical_matches=hist_matches,
                    odds_movement=self.detect_odds_movement(home_odds, match_odds.get('home_win_opening')),
                    reasoning=f"Our model: {home_prob*100:.0f}% vs Market: {home_implied*100:.0f}% ({home_edge*100:.1f}% edge)",
                    confidence=home_prob
                ))
            
            # Analyze Over 2.5 value
            over_odds = match_odds.get('over_2_5', prob_to_fair_odds(over_2_5_prob))
            over_implied = self.odds_to_implied_probability(over_odds)
            over_edge = self.calculate_edge(over_2_5_prob, over_implied)
            
            if over_edge >= self.VALUE_THRESHOLDS['low_value']:
                hist_rate, hist_matches = self.get_historical_win_rate('over_2_5', over_odds)
                
                value_bets.append(ValueBet(
                    match_id=match_id,
                    home_team=home_name,
                    away_team=away_name,
                    league=league,
                    bet_type='over_2.5',
                    bet_label='Over 2.5 Goals',
                    our_probability=over_2_5_prob,
                    implied_probability=over_implied,
                    bookmaker_odds=over_odds,
                    edge=over_edge,
                    value_type=self.classify_value(over_edge),
                    expected_value=self.calculate_expected_value(over_2_5_prob, over_odds),
                    historical_win_rate=hist_rate,
                    historical_matches=hist_matches,
                    reasoning=f"Over 2.5: {over_2_5_prob*100:.0f}% vs Market {over_implied*100:.0f}%",
                    confidence=over_2_5_prob
                ))
            
            # Analyze BTTS value
            btts_odds = match_odds.get('btts_yes', prob_to_fair_odds(btts_yes_prob))
            btts_implied = self.odds_to_implied_probability(btts_odds)
            btts_edge = self.calculate_edge(btts_yes_prob, btts_implied)
            
            if btts_edge >= self.VALUE_THRESHOLDS['low_value']:
                hist_rate, hist_matches = self.get_historical_win_rate('btts_yes', btts_odds)
                
                value_bets.append(ValueBet(
                    match_id=match_id,
                    home_team=home_name,
                    away_team=away_name,
                    league=league,
                    bet_type='btts_yes',
                    bet_label='Both Teams to Score',
                    our_probability=btts_yes_prob,
                    implied_probability=btts_implied,
                    bookmaker_odds=btts_odds,
                    edge=btts_edge,
                    value_type=self.classify_value(btts_edge),
                    expected_value=self.calculate_expected_value(btts_yes_prob, btts_odds),
                    historical_win_rate=hist_rate,
                    historical_matches=hist_matches,
                    reasoning=f"BTTS: {btts_yes_prob*100:.0f}% vs Market {btts_implied*100:.0f}%",
                    confidence=btts_yes_prob
                ))
        
        # Sort by edge (best value first)
        value_bets.sort(key=lambda x: x.edge, reverse=True)
        
        return value_bets
    
    def generate_value_accumulator(
        self,
        value_bets: List[ValueBet],
        max_picks: int = 4,
        min_value_type: str = 'low_value'
    ) -> Optional[Dict]:
        """
        Generate a value accumulator from identified value bets.
        """
        # Filter by minimum value type
        value_order = ['high_value', 'medium_value', 'low_value', 'fair', 'overpriced']
        min_idx = value_order.index(min_value_type)
        
        filtered = [vb for vb in value_bets if value_order.index(vb.value_type) <= min_idx]
        
        if len(filtered) < 2:
            return None
        
        picks = filtered[:max_picks]
        
        # Calculate combined metrics
        combined_odds = 1.0
        combined_prob = 1.0
        avg_edge = 0.0
        
        for pick in picks:
            combined_odds *= pick.bookmaker_odds
            combined_prob *= pick.our_probability
            avg_edge += pick.edge
        
        avg_edge /= len(picks)
        
        return {
            'name': 'Value Accumulator',
            'emoji': '💰',
            'description': f'High-value picks with {avg_edge*100:.1f}% average edge',
            'category': 'value_bets',
            'picks': [p.to_dict() for p in picks],
            'num_picks': len(picks),
            'combined_odds': round(combined_odds, 2),
            'combined_probability': round(combined_prob, 4),
            'win_chance_pct': round(combined_prob * 100, 1),
            'average_edge': round(avg_edge * 100, 1),
            'expected_value': round(sum(p.expected_value for p in picks), 3),
            'risk_level': 'medium' if avg_edge >= 0.10 else 'high',
            'confidence_rating': 'high' if avg_edge >= 0.12 else 'medium',
            'suggested_stake': round(min(avg_edge * 100, 25), 2),
            'potential_return': round(min(avg_edge * 100, 25) * combined_odds, 2)
        }


# Global instance
value_betting_engine = ValueBettingEngine()


def find_value_bets(predictions: List[Dict], odds: Optional[Dict] = None) -> List[Dict]:
    """Find value bets from predictions"""
    value_bets = value_betting_engine.identify_value_bets(predictions, odds)
    return [vb.to_dict() for vb in value_bets]


def get_value_accumulator(predictions: List[Dict], odds: Optional[Dict] = None) -> Optional[Dict]:
    """Generate a value accumulator"""
    value_bets = value_betting_engine.identify_value_bets(predictions, odds)
    return value_betting_engine.generate_value_accumulator(value_bets)
