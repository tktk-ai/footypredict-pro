"""
Accumulator Builder Module

Builds multi-match betting accumulators with combined odds
and win probability calculations.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass 
class AccumulatorLeg:
    """Single leg of an accumulator"""
    match_id: str
    home_team: str
    away_team: str
    selection: str  # 'home', 'draw', 'away', 'over_2.5', 'btts_yes'
    probability: float
    odds: float  # Approximate decimal odds


@dataclass
class Accumulator:
    """Complete accumulator bet"""
    legs: List[AccumulatorLeg]
    combined_probability: float
    combined_odds: float
    stake: float
    potential_return: float
    
    def to_dict(self) -> Dict:
        return {
            'legs': [
                {
                    'match': f"{leg.home_team} vs {leg.away_team}",
                    'selection': leg.selection,
                    'probability': round(leg.probability, 3),
                    'odds': round(leg.odds, 2)
                }
                for leg in self.legs
            ],
            'num_legs': len(self.legs),
            'combined_probability': round(self.combined_probability, 4),
            'combined_odds': round(self.combined_odds, 2),
            'stake': self.stake,
            'potential_return': round(self.potential_return, 2)
        }


class AccumulatorBuilder:
    """
    Builds and analyzes accumulators from predictions
    
    Features:
    - Combine multiple selections
    - Calculate combined probability
    - Suggest value accumulators
    """
    
    def __init__(self):
        pass
    
    def prob_to_odds(self, probability: float) -> float:
        """Convert probability to approximate decimal odds (with 5% margin)"""
        if probability <= 0:
            return 100.0
        raw_odds = 1 / probability
        # Apply ~5% bookmaker margin
        return round(raw_odds * 0.95, 2)
    
    def create_leg(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        selection: str,
        probability: float
    ) -> AccumulatorLeg:
        """Create a single accumulator leg"""
        return AccumulatorLeg(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            selection=selection,
            probability=probability,
            odds=self.prob_to_odds(probability)
        )
    
    def build_accumulator(
        self,
        legs: List[AccumulatorLeg],
        stake: float = 10.0
    ) -> Accumulator:
        """Build complete accumulator from legs"""
        if not legs:
            return None
        
        # Calculate combined probability (multiply individual probs)
        combined_prob = 1.0
        combined_odds = 1.0
        
        for leg in legs:
            combined_prob *= leg.probability
            combined_odds *= leg.odds
        
        potential_return = stake * combined_odds
        
        return Accumulator(
            legs=legs,
            combined_probability=combined_prob,
            combined_odds=combined_odds,
            stake=stake,
            potential_return=potential_return
        )
    
    def suggest_value_acca(
        self,
        predictions: List[Dict],
        num_legs: int = 3,
        min_prob: float = 0.5
    ) -> Optional[Accumulator]:
        """
        Suggest a value accumulator from predictions
        
        Selects high-probability selections that offer good value
        """
        candidates = []
        
        for pred in predictions:
            match = pred.get('match', {})
            prediction = pred.get('prediction', {})
            goals = pred.get('goals', {})
            
            if not prediction:
                continue
            
            home_team = match.get('home_team', {}).get('name', 'Home')
            away_team = match.get('away_team', {}).get('name', 'Away')
            match_id = match.get('id', '')
            
            # Check match result selections
            home_prob = prediction.get('home_win_prob', 0)
            draw_prob = prediction.get('draw_prob', 0)
            away_prob = prediction.get('away_win_prob', 0)
            
            if home_prob >= min_prob:
                candidates.append({
                    'match_id': match_id,
                    'home': home_team,
                    'away': away_team,
                    'selection': 'home_win',
                    'probability': home_prob,
                    'value': home_prob - 0.33  # Edge over random
                })
            
            if away_prob >= min_prob:
                candidates.append({
                    'match_id': match_id,
                    'home': home_team,
                    'away': away_team,
                    'selection': 'away_win',
                    'probability': away_prob,
                    'value': away_prob - 0.33
                })
            
            # Check goals markets
            if goals:
                over_25 = goals.get('over_under', {}).get('over_2.5', 0)
                btts = goals.get('btts', {}).get('yes', 0)
                
                if over_25 >= min_prob:
                    candidates.append({
                        'match_id': match_id,
                        'home': home_team,
                        'away': away_team,
                        'selection': 'over_2.5',
                        'probability': over_25,
                        'value': over_25 - 0.5
                    })
                
                if btts >= min_prob:
                    candidates.append({
                        'match_id': match_id,
                        'home': home_team,
                        'away': away_team,
                        'selection': 'btts_yes',
                        'probability': btts,
                        'value': btts - 0.5
                    })
        
        # Sort by value (probability edge)
        candidates.sort(key=lambda x: x['value'], reverse=True)
        
        # Take top candidates, ensuring different matches
        selected = []
        used_matches = set()
        
        for c in candidates:
            if c['match_id'] not in used_matches and len(selected) < num_legs:
                leg = self.create_leg(
                    c['match_id'],
                    c['home'],
                    c['away'],
                    c['selection'],
                    c['probability']
                )
                selected.append(leg)
                used_matches.add(c['match_id'])
        
        if len(selected) >= 2:
            return self.build_accumulator(selected)
        
        return None


# Global instance
accumulator_builder = AccumulatorBuilder()
