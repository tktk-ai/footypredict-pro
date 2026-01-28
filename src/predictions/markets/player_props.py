"""
Player Props Predictor
======================
Predicts player-specific betting markets: goals, assists, cards, shots.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from scipy.stats import poisson, nbinom

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Player statistics for predictions."""
    name: str
    team: str
    position: str
    goals_per_90: float = 0.0
    assists_per_90: float = 0.0
    shots_per_90: float = 0.0
    shots_on_target_per_90: float = 0.0
    yellow_cards_per_90: float = 0.0
    red_cards_per_90: float = 0.0
    minutes_played: int = 0
    avg_minutes: float = 0.0
    xG_per_90: float = 0.0
    xA_per_90: float = 0.0


@dataclass
class PlayerPropPrediction:
    """Prediction for a player prop market."""
    player: str
    market: str
    line: float
    over_prob: float
    under_prob: float
    expected_value: float
    confidence: float


class PlayerPropsPredictor:
    """
    Predicts player-specific betting markets.
    
    Supports:
    - Anytime goalscorer
    - First goalscorer
    - 2+ goals
    - Player assists
    - Player shots
    - Player cards
    """
    
    def __init__(self):
        self.player_data: Dict[str, PlayerStats] = {}
        
    def add_player(self, player: PlayerStats) -> None:
        """Add player statistics."""
        self.player_data[player.name] = player
        
    def predict_goals(
        self,
        player_name: str,
        opponent_defense: float = 1.0,
        expected_minutes: float = 90,
        team_xg: float = 1.5
    ) -> Dict[str, float]:
        """
        Predict goal-related props for a player.
        
        Returns:
            Probabilities for goal markets
        """
        player = self.player_data.get(player_name)
        
        if player:
            base_xg = player.xG_per_90 if player.xG_per_90 > 0 else player.goals_per_90
        else:
            # Default for unknown player
            base_xg = 0.15
        
        # Adjust for minutes and opponent
        minutes_factor = expected_minutes / 90
        xg_adjusted = base_xg * minutes_factor * opponent_defense
        
        # Share of team xG
        team_share = min(0.5, xg_adjusted / team_xg) if team_xg > 0 else 0.2
        final_xg = team_xg * team_share
        
        # Poisson probabilities
        no_goal = poisson.pmf(0, final_xg)
        one_goal = poisson.pmf(1, final_xg)
        two_plus = 1 - poisson.cdf(1, final_xg)
        
        return {
            'anytime_scorer': round(1 - no_goal, 4),
            'no_goal': round(no_goal, 4),
            '2_or_more': round(two_plus, 4),
            'exactly_1': round(one_goal, 4),
            'expected_goals': round(final_xg, 3),
            'first_scorer_share': round(team_share, 3)
        }
    
    def predict_first_goalscorer(
        self,
        players: List[str],
        team_xg: float = 1.5
    ) -> List[Dict]:
        """
        Predict first goalscorer probabilities.
        
        Returns:
            Sorted list of players with first scorer probability
        """
        scorer_probs = []
        
        for player_name in players:
            player = self.player_data.get(player_name)
            
            if player:
                # Base probability from xG share
                base_xg = player.xG_per_90 if player.xG_per_90 > 0 else player.goals_per_90
                
                # Position weighting
                pos_weight = {
                    'Forward': 1.5,
                    'Midfielder': 1.0,
                    'Defender': 0.4,
                    'Goalkeeper': 0.02
                }.get(player.position, 1.0)
                
                weighted_xg = base_xg * pos_weight
            else:
                weighted_xg = 0.1
            
            scorer_probs.append({
                'player': player_name,
                'weighted_xg': weighted_xg
            })
        
        # Normalize to get probabilities
        total_xg = sum(p['weighted_xg'] for p in scorer_probs)
        
        if total_xg > 0:
            for p in scorer_probs:
                # First scorer prob = share of goals * prob of scoring first
                p['first_scorer_prob'] = round(p['weighted_xg'] / total_xg * 0.8, 4)
        else:
            for p in scorer_probs:
                p['first_scorer_prob'] = round(1 / len(players) * 0.8, 4)
        
        return sorted(scorer_probs, key=lambda x: x['first_scorer_prob'], reverse=True)
    
    def predict_shots(
        self,
        player_name: str,
        line: float = 1.5,
        expected_minutes: float = 90
    ) -> Dict[str, float]:
        """
        Predict shots on target prop.
        """
        player = self.player_data.get(player_name)
        
        if player:
            sot_per_90 = player.shots_on_target_per_90
        else:
            sot_per_90 = 0.8  # Default
        
        expected_sot = sot_per_90 * (expected_minutes / 90)
        
        # Use Poisson for shots
        over_prob = 1 - poisson.cdf(line, expected_sot)
        under_prob = poisson.cdf(line - 0.5, expected_sot)
        
        return {
            'line': line,
            'over_prob': round(over_prob, 4),
            'under_prob': round(under_prob, 4),
            'expected_sot': round(expected_sot, 2)
        }
    
    def predict_assists(
        self,
        player_name: str,
        team_xg: float = 1.5,
        expected_minutes: float = 90
    ) -> Dict[str, float]:
        """
        Predict assist probabilities.
        """
        player = self.player_data.get(player_name)
        
        if player:
            xa_per_90 = player.xA_per_90 if player.xA_per_90 > 0 else player.assists_per_90
        else:
            xa_per_90 = 0.1  # Default
        
        expected_assists = xa_per_90 * (expected_minutes / 90) * (team_xg / 1.5)
        
        no_assist = poisson.pmf(0, expected_assists)
        
        return {
            'anytime_assist': round(1 - no_assist, 4),
            'no_assist': round(no_assist, 4),
            'expected_assists': round(expected_assists, 3)
        }
    
    def predict_cards(
        self,
        player_name: str,
        match_intensity: float = 1.0,
        expected_minutes: float = 90
    ) -> Dict[str, float]:
        """
        Predict card probabilities.
        """
        player = self.player_data.get(player_name)
        
        if player:
            yellow_rate = player.yellow_cards_per_90
            red_rate = player.red_cards_per_90
        else:
            yellow_rate = 0.12  # Default
            red_rate = 0.01
        
        # Adjust for match intensity
        adjusted_yellow = yellow_rate * match_intensity * (expected_minutes / 90)
        adjusted_red = red_rate * match_intensity * (expected_minutes / 90)
        
        return {
            'yellow_card': round(1 - poisson.pmf(0, adjusted_yellow), 4),
            'red_card': round(1 - poisson.pmf(0, adjusted_red), 4),
            'any_card': round(1 - poisson.pmf(0, adjusted_yellow + adjusted_red), 4),
            'no_card': round(poisson.pmf(0, adjusted_yellow + adjusted_red), 4)
        }
    
    def get_all_props(
        self,
        player_name: str,
        team_xg: float = 1.5,
        opponent_defense: float = 1.0,
        expected_minutes: float = 90
    ) -> Dict[str, Any]:
        """
        Get all player props predictions.
        """
        return {
            'player': player_name,
            'goals': self.predict_goals(player_name, opponent_defense, expected_minutes, team_xg),
            'shots': self.predict_shots(player_name, 1.5, expected_minutes),
            'assists': self.predict_assists(player_name, team_xg, expected_minutes),
            'cards': self.predict_cards(player_name, 1.0, expected_minutes)
        }
    
    def find_value_props(
        self,
        player_name: str,
        odds: Dict[str, float],
        team_xg: float = 1.5
    ) -> List[Dict]:
        """
        Find value player props based on odds.
        
        Args:
            player_name: Player to analyze
            odds: Dict mapping market to decimal odds
            team_xg: Expected team goals
            
        Returns:
            List of value prop opportunities
        """
        predictions = self.get_all_props(player_name, team_xg)
        value_props = []
        
        market_mapping = {
            'anytime_scorer': predictions['goals']['anytime_scorer'],
            '2_or_more_goals': predictions['goals']['2_or_more'],
            'anytime_assist': predictions['assists']['anytime_assist'],
            'yellow_card': predictions['cards']['yellow_card'],
        }
        
        for market, pred_prob in market_mapping.items():
            if market in odds and pred_prob > 0:
                implied_prob = 1 / odds[market]
                edge = pred_prob - implied_prob
                
                if edge > 0.05:  # 5% edge threshold
                    value_props.append({
                        'market': market,
                        'odds': odds[market],
                        'predicted_prob': round(pred_prob, 4),
                        'implied_prob': round(implied_prob, 4),
                        'edge': round(edge * 100, 2),
                        'rating': 'STRONG' if edge > 0.10 else 'MODERATE'
                    })
        
        return sorted(value_props, key=lambda x: x['edge'], reverse=True)


# Global instance
_predictor: Optional[PlayerPropsPredictor] = None


def get_predictor() -> PlayerPropsPredictor:
    """Get or create player props predictor."""
    global _predictor
    if _predictor is None:
        _predictor = PlayerPropsPredictor()
    return _predictor


def predict_player_goals(
    player: str,
    team_xg: float = 1.5,
    opponent_defense: float = 1.0
) -> Dict:
    """Quick function to predict player goals."""
    return get_predictor().predict_goals(player, opponent_defense, 90, team_xg)
