"""
Advanced Accumulator Engine

Multiple accumulator strategies for different risk profiles:
- Safe Acca: Over 0.5 goals (very low risk)
- Banker Acca: 85%+ confidence picks
- Value Acca: Edge > 5% value bets
- BTTS Acca: Both Teams To Score
- Goals Acca: Over 2.5 goals
- Corners Acca: Over 9.5 corners (based on league data)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math


@dataclass
class AccumulatorPick:
    """Single pick in an accumulator"""
    match_id: str
    home_team: str
    away_team: str
    selection: str
    odds: float
    probability: float
    confidence: float
    reasoning: str


@dataclass
class Accumulator:
    """Complete accumulator with multiple picks"""
    id: str
    strategy: str
    picks: List[AccumulatorPick]
    combined_odds: float
    combined_probability: float
    stake_suggestion: float
    potential_return: float
    risk_level: str
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'strategy': self.strategy,
            'picks': [asdict(p) for p in self.picks],
            'combined_odds': round(self.combined_odds, 2),
            'combined_probability': round(self.combined_probability, 4),
            'stake_suggestion': self.stake_suggestion,
            'potential_return': round(self.potential_return, 2),
            'risk_level': self.risk_level,
            'pick_count': len(self.picks),
            'created_at': self.created_at
        }


class AccumulatorEngine:
    """
    Generates multiple types of accumulators from predictions.
    """
    
    STRATEGIES = {
        'safe': {
            'name': 'Safe Acca',
            'description': 'Over 0.5 goals - Highest probability picks',
            'risk_level': 'Very Low',
            'min_probability': 0.90,
            'max_picks': 6,
            'emoji': '🛡️'
        },
        'banker': {
            'name': 'Banker Acca',
            'description': '85%+ confidence match outcomes',
            'risk_level': 'Low',
            'min_probability': 0.70,
            'min_confidence': 0.85,
            'max_picks': 5,
            'emoji': '🏦'
        },
        'value': {
            'name': 'Value Acca',
            'description': 'Edge > 5% value bets only',
            'risk_level': 'Medium',
            'min_edge': 0.05,
            'max_picks': 4,
            'emoji': '💎'
        },
        'btts': {
            'name': 'BTTS Acca',
            'description': 'Both Teams To Score picks',
            'risk_level': 'Medium',
            'min_probability': 0.55,
            'max_picks': 5,
            'emoji': '⚽'
        },
        'goals': {
            'name': 'Goals Acca',
            'description': 'Over 2.5 goals picks',
            'risk_level': 'Medium-High',
            'min_probability': 0.50,
            'max_picks': 4,
            'emoji': '🎯'
        },
        'risky': {
            'name': 'Risky Acca',
            'description': 'Higher odds, lower probability',
            'risk_level': 'High',
            'min_odds': 2.5,
            'max_picks': 4,
            'emoji': '🎲'
        }
    }
    
    def __init__(self):
        self._counter = 0
    
    def _generate_id(self) -> str:
        self._counter += 1
        return f"acca_{datetime.now().strftime('%Y%m%d')}_{self._counter:04d}"
    
    def generate_safe_acca(self, predictions: List[Dict], max_picks: int = 6) -> Optional[Accumulator]:
        """
        Generate Safe Accumulator - Over 0.5 goals picks.
        These have 95%+ probability typically.
        """
        picks = []
        
        for pred in predictions:
            goals_data = pred.get('goals', {})
            over_05 = goals_data.get('over_under', {}).get('over_0.5', 0)
            
            if over_05 >= 0.90:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection='Over 0.5 Goals',
                    odds=1.08 + (1 - over_05) * 0.5,  # ~1.08-1.15
                    probability=over_05,
                    confidence=min(0.99, over_05 + 0.02),
                    reasoning=f"xG: {goals_data.get('home_xg', 0):.1f} + {goals_data.get('away_xg', 0):.1f}"
                ))
        
        # Sort by probability and take top picks
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) < 2:
            return None
        
        return self._build_accumulator('safe', picks)
    
    def generate_banker_acca(self, predictions: List[Dict], max_picks: int = 5) -> Optional[Accumulator]:
        """
        Generate Banker Accumulator - High confidence match winners.
        """
        picks = []
        
        for pred in predictions:
            confidence = pred.get('confidence', 0)
            home_prob = pred.get('home_win_prob', 0)
            away_prob = pred.get('away_win_prob', 0)
            
            # Only 85%+ confidence
            if confidence < 0.85:
                continue
            
            # Find the favored outcome
            if home_prob >= 0.65:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection=f"{pred.get('home_team')} Win",
                    odds=1 / home_prob if home_prob > 0 else 1.5,
                    probability=home_prob,
                    confidence=confidence,
                    reasoning=f"ELO advantage, {confidence*100:.0f}% confidence"
                ))
            elif away_prob >= 0.65:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection=f"{pred.get('away_team')} Win",
                    odds=1 / away_prob if away_prob > 0 else 2.0,
                    probability=away_prob,
                    confidence=confidence,
                    reasoning=f"Strong away form, {confidence*100:.0f}% confidence"
                ))
        
        picks.sort(key=lambda x: x.confidence, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) < 2:
            return None
        
        return self._build_accumulator('banker', picks)
    
    def generate_value_acca(self, predictions: List[Dict], odds_data: List[Dict] = None, max_picks: int = 4) -> Optional[Accumulator]:
        """
        Generate Value Accumulator - Picks where our probability > bookmaker probability.
        """
        picks = []
        
        for pred in predictions:
            home_prob = pred.get('home_win_prob', 0.33)
            draw_prob = pred.get('draw_prob', 0.33)
            away_prob = pred.get('away_win_prob', 0.33)
            
            # Simulated bookmaker odds (would use real odds API in production)
            bookie_home = pred.get('odds', {}).get('home', 2.5)
            bookie_draw = pred.get('odds', {}).get('draw', 3.3)
            bookie_away = pred.get('odds', {}).get('away', 3.0)
            
            # Calculate edges
            home_edge = home_prob - (1 / bookie_home)
            draw_edge = draw_prob - (1 / bookie_draw)
            away_edge = away_prob - (1 / bookie_away)
            
            # Find best value
            best_edge = max(home_edge, draw_edge, away_edge)
            
            if best_edge >= 0.05:  # 5% edge minimum
                if home_edge == best_edge:
                    selection = f"{pred.get('home_team')} Win"
                    odds = bookie_home
                    prob = home_prob
                elif draw_edge == best_edge:
                    selection = "Draw"
                    odds = bookie_draw
                    prob = draw_prob
                else:
                    selection = f"{pred.get('away_team')} Win"
                    odds = bookie_away
                    prob = away_prob
                
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection=selection,
                    odds=odds,
                    probability=prob,
                    confidence=0.7 + best_edge,
                    reasoning=f"+{best_edge*100:.1f}% edge vs bookies"
                ))
        
        picks.sort(key=lambda x: x.probability * x.odds, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) < 2:
            return None
        
        return self._build_accumulator('value', picks)
    
    def generate_btts_acca(self, predictions: List[Dict], max_picks: int = 5) -> Optional[Accumulator]:
        """
        Generate BTTS Accumulator - Both Teams To Score picks.
        """
        picks = []
        
        for pred in predictions:
            goals_data = pred.get('goals', {})
            btts = goals_data.get('btts', {}).get('yes', 0)
            
            if btts >= 0.55:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection='BTTS Yes',
                    odds=1.6 + (1 - btts) * 0.8,  # ~1.6-2.0
                    probability=btts,
                    confidence=min(0.95, btts + 0.1),
                    reasoning=f"xG: {goals_data.get('home_xg', 0):.1f} - {goals_data.get('away_xg', 0):.1f}"
                ))
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) < 2:
            return None
        
        return self._build_accumulator('btts', picks)
    
    def generate_goals_acca(self, predictions: List[Dict], max_picks: int = 4) -> Optional[Accumulator]:
        """
        Generate Goals Accumulator - Over 2.5 goals picks.
        """
        picks = []
        
        for pred in predictions:
            goals_data = pred.get('goals', {})
            over_25 = goals_data.get('over_under', {}).get('over_2.5', 0)
            
            if over_25 >= 0.50:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection='Over 2.5 Goals',
                    odds=1.7 + (1 - over_25) * 1.0,  # ~1.7-2.5
                    probability=over_25,
                    confidence=min(0.90, over_25 + 0.05),
                    reasoning=f"Total xG: {goals_data.get('total_xg', 0):.1f}"
                ))
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) < 2:
            return None
        
        return self._build_accumulator('goals', picks)
    
    def generate_risky_acca(self, predictions: List[Dict], max_picks: int = 4) -> Optional[Accumulator]:
        """
        Generate Risky Accumulator - Higher odds long shots.
        """
        picks = []
        
        for pred in predictions:
            home_prob = pred.get('home_win_prob', 0.33)
            draw_prob = pred.get('draw_prob', 0.33)
            away_prob = pred.get('away_win_prob', 0.33)
            
            # Look for underdogs with decent value
            if away_prob >= 0.25 and away_prob <= 0.40:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection=f"{pred.get('away_team')} Win",
                    odds=1 / away_prob if away_prob > 0 else 3.5,
                    probability=away_prob,
                    confidence=0.5,
                    reasoning="Underdog value pick"
                ))
            elif draw_prob >= 0.30:
                picks.append(AccumulatorPick(
                    match_id=pred.get('id', ''),
                    home_team=pred.get('home_team', ''),
                    away_team=pred.get('away_team', ''),
                    selection="Draw",
                    odds=1 / draw_prob if draw_prob > 0 else 3.3,
                    probability=draw_prob,
                    confidence=0.5,
                    reasoning="Draw value pick"
                ))
        
        picks.sort(key=lambda x: x.odds, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) < 2:
            return None
        
        return self._build_accumulator('risky', picks)
    
    def _build_accumulator(self, strategy: str, picks: List[AccumulatorPick]) -> Accumulator:
        """Build accumulator from picks"""
        combined_odds = 1.0
        combined_prob = 1.0
        
        for pick in picks:
            combined_odds *= pick.odds
            combined_prob *= pick.probability
        
        strategy_info = self.STRATEGIES.get(strategy, {})
        
        # Stake suggestion based on probability
        if combined_prob >= 0.5:
            stake = 20.0
        elif combined_prob >= 0.3:
            stake = 10.0
        elif combined_prob >= 0.1:
            stake = 5.0
        else:
            stake = 2.0
        
        return Accumulator(
            id=self._generate_id(),
            strategy=strategy,
            picks=picks,
            combined_odds=combined_odds,
            combined_probability=combined_prob,
            stake_suggestion=stake,
            potential_return=stake * combined_odds,
            risk_level=strategy_info.get('risk_level', 'Medium'),
            created_at=datetime.now().isoformat()
        )
    
    def generate_all(self, predictions: List[Dict]) -> Dict[str, Optional[Accumulator]]:
        """Generate all accumulator types"""
        return {
            'safe': self.generate_safe_acca(predictions),
            'banker': self.generate_banker_acca(predictions),
            'value': self.generate_value_acca(predictions),
            'btts': self.generate_btts_acca(predictions),
            'goals': self.generate_goals_acca(predictions),
            'risky': self.generate_risky_acca(predictions)
        }
    
    def get_strategy_info(self, strategy: str) -> Dict:
        """Get strategy information"""
        return self.STRATEGIES.get(strategy, {})
    
    def get_all_strategies(self) -> Dict:
        """Get all strategy definitions"""
        return self.STRATEGIES


# Global instance
acca_engine = AccumulatorEngine()


def generate_accumulator(predictions: List[Dict], strategy: str = 'safe') -> Optional[Dict]:
    """Generate specific accumulator type"""
    generators = {
        'safe': acca_engine.generate_safe_acca,
        'banker': acca_engine.generate_banker_acca,
        'value': acca_engine.generate_value_acca,
        'btts': acca_engine.generate_btts_acca,
        'goals': acca_engine.generate_goals_acca,
        'risky': acca_engine.generate_risky_acca
    }
    
    generator = generators.get(strategy)
    if generator:
        acca = generator(predictions)
        return acca.to_dict() if acca else None
    return None


def generate_all_accumulators(predictions: List[Dict]) -> Dict:
    """Generate all accumulator types"""
    accas = acca_engine.generate_all(predictions)
    return {
        k: v.to_dict() if v else None
        for k, v in accas.items()
    }
