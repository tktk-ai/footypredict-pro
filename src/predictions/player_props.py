"""
Player Props Prediction Module V3.0
Predicts individual player statistics and props

Features:
- Goals/assists prediction
- Shots on target
- Anytime scorer probability
- Card predictions
- Per-90 normalized statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import poisson, norm
import logging

logger = logging.getLogger(__name__)


class PlayerPropsPredictor:
    """
    Predicts player props including:
    - Goals/assists
    - Shots on target
    - Passes completed
    - Cards
    - Minutes played
    """
    
    PROP_TYPES = [
        'goals',
        'assists', 
        'shots',
        'shots_on_target',
        'passes',
        'key_passes',
        'tackles',
        'interceptions',
        'fouls',
        'yellow_cards',
        'minutes'
    ]
    
    def __init__(self):
        self.player_stats = {}
        self.position_multipliers = {
            'FW': {'goals': 1.5, 'assists': 1.2, 'shots': 1.4, 'yellow_cards': 0.8},
            'MF': {'goals': 0.8, 'assists': 1.3, 'shots': 0.9, 'key_passes': 1.4, 'yellow_cards': 1.2},
            'DF': {'goals': 0.3, 'assists': 0.5, 'tackles': 1.5, 'interceptions': 1.4, 'yellow_cards': 1.3},
            'GK': {'goals': 0.01, 'assists': 0.1, 'yellow_cards': 0.5}
        }
        
    def create_player_features(
        self,
        player_stats: pd.DataFrame,
        match_context: Dict
    ) -> Dict:
        """
        Create features for player prop prediction.
        
        Args:
            player_stats: Historical player statistics
            match_context: Context about upcoming match
        """
        features = {}
        
        if player_stats.empty:
            return self._default_features(match_context)
        
        # Rolling averages for different windows
        windows = [3, 5, 10, 20]
        
        for window in windows:
            for stat in self.PROP_TYPES:
                if stat in player_stats.columns:
                    # Average
                    col = player_stats[stat].tail(window)
                    features[f'{stat}_avg_{window}'] = col.mean() if len(col) > 0 else 0
                    features[f'{stat}_std_{window}'] = col.std() if len(col) > 1 else 0
                    features[f'{stat}_max_{window}'] = col.max() if len(col) > 0 else 0
        
        # Minutes played ratio
        if 'minutes' in player_stats.columns:
            recent_mins = player_stats['minutes'].tail(5).mean()
            features['minutes_ratio'] = min(recent_mins / 90, 1.0) if recent_mins > 0 else 0.5
        else:
            features['minutes_ratio'] = 0.8  # Assume regular starter
        
        # Per 90 statistics
        minutes_avg = features.get('minutes_avg_10', 70)
        for stat in ['goals', 'assists', 'shots', 'key_passes']:
            avg_stat = features.get(f'{stat}_avg_10', 0)
            if minutes_avg > 0:
                features[f'{stat}_per90'] = (avg_stat / minutes_avg) * 90
        
        # Match context features
        features['is_home'] = match_context.get('is_home', 0)
        features['opponent_strength'] = match_context.get('opponent_defense_rating', 1.0)
        features['match_importance'] = match_context.get('importance', 0.5)
        features['days_rest'] = match_context.get('days_rest', 7)
        
        return features
    
    def _default_features(self, match_context: Dict) -> Dict:
        """Return default features for unknown players."""
        return {
            'goals_avg_5': 0.2,
            'assists_avg_5': 0.1,
            'shots_avg_5': 1.5,
            'shots_on_target_avg_5': 0.8,
            'minutes_ratio': 0.8,
            'goals_per90': 0.25,
            'is_home': match_context.get('is_home', 0),
            'opponent_strength': match_context.get('opponent_defense_rating', 1.0)
        }
    
    def predict_goals(
        self,
        player_features: Dict,
        position: str = 'FW',
        line: float = 0.5
    ) -> Dict:
        """
        Predict player goals.
        
        Args:
            player_features: Player feature dictionary
            position: Player position (FW, MF, DF, GK)
            line: Betting line
        """
        # Base expected goals from features
        base_xg = player_features.get('goals_avg_5', 0.2)
        
        # Position multiplier
        pos_mult = self.position_multipliers.get(position, {}).get('goals', 1.0)
        
        # Context adjustments
        home_boost = 1.1 if player_features.get('is_home', 0) else 1.0
        opp_adjust = 1.0 / max(player_features.get('opponent_strength', 1.0), 0.5)
        
        # Final expected goals
        expected_goals = base_xg * pos_mult * home_boost * opp_adjust
        expected_goals = max(0.05, min(expected_goals, 2.0))  # Clamp
        
        # Poisson probabilities
        probs = {k: float(poisson.pmf(k, expected_goals)) for k in range(6)}
        
        return {
            'expected_goals': round(expected_goals, 3),
            'distribution': probs,
            'prob_0': probs[0],
            'prob_1plus': 1 - probs[0],
            'prob_2plus': 1 - probs[0] - probs[1],
            'over_line_prob': float(1 - poisson.cdf(line - 0.01, expected_goals)),
            'under_line_prob': float(poisson.cdf(line - 0.01, expected_goals))
        }
    
    def predict_assists(
        self,
        player_features: Dict,
        position: str = 'MF',
        line: float = 0.5
    ) -> Dict:
        """Predict player assists."""
        base_xa = player_features.get('assists_avg_5', 0.15)
        
        pos_mult = self.position_multipliers.get(position, {}).get('assists', 1.0)
        home_boost = 1.1 if player_features.get('is_home', 0) else 1.0
        
        expected_assists = base_xa * pos_mult * home_boost
        expected_assists = max(0.02, min(expected_assists, 1.5))
        
        probs = {k: float(poisson.pmf(k, expected_assists)) for k in range(5)}
        
        return {
            'expected_assists': round(expected_assists, 3),
            'distribution': probs,
            'prob_1plus': 1 - probs[0],
            'over_line_prob': float(1 - poisson.cdf(line - 0.01, expected_assists))
        }
    
    def predict_shots(
        self,
        player_features: Dict,
        position: str = 'FW',
        line: float = 2.5
    ) -> Dict:
        """Predict player shots."""
        base_shots = player_features.get('shots_avg_5', 1.5)
        
        pos_mult = self.position_multipliers.get(position, {}).get('shots', 1.0)
        minutes_ratio = player_features.get('minutes_ratio', 0.8)
        
        expected_shots = base_shots * pos_mult * minutes_ratio
        expected_shots = max(0.5, min(expected_shots, 8.0))
        
        probs = {k: float(poisson.pmf(k, expected_shots)) for k in range(10)}
        
        return {
            'expected_shots': round(expected_shots, 2),
            'distribution': probs,
            'over_line_prob': float(1 - poisson.cdf(line - 0.01, expected_shots)),
            'under_line_prob': float(poisson.cdf(line - 0.01, expected_shots))
        }
    
    def predict_shots_on_target(
        self,
        player_features: Dict,
        position: str = 'FW',
        line: float = 1.5
    ) -> Dict:
        """Predict player shots on target."""
        base_sot = player_features.get('shots_on_target_avg_5', 0.8)
        
        pos_mult = self.position_multipliers.get(position, {}).get('shots', 1.0)
        minutes_ratio = player_features.get('minutes_ratio', 0.8)
        
        expected_sot = base_sot * pos_mult * minutes_ratio
        expected_sot = max(0.2, min(expected_sot, 4.0))
        
        return {
            'expected_sot': round(expected_sot, 2),
            'over_line_prob': float(1 - poisson.cdf(line - 0.01, expected_sot)),
            'under_line_prob': float(poisson.cdf(line - 0.01, expected_sot))
        }
    
    def predict_anytime_scorer(
        self,
        player_features: Dict,
        position: str = 'FW',
        odds: float = None
    ) -> Dict:
        """
        Predict probability of player scoring at least one goal.
        """
        goals_pred = self.predict_goals(player_features, position)
        
        prob_to_score = goals_pred['prob_1plus']
        
        result = {
            'probability': round(prob_to_score, 4),
            'expected_goals': goals_pred['expected_goals'],
            'prob_2plus_goals': goals_pred['prob_2plus'],
            'fair_odds': round(1 / prob_to_score, 2) if prob_to_score > 0 else 99.0
        }
        
        if odds is not None:
            implied_prob = 1 / odds
            result['edge'] = round(prob_to_score - implied_prob, 4)
            result['value'] = result['edge'] > 0.05
            result['odds'] = odds
        
        return result
    
    def predict_cards(
        self,
        player_features: Dict,
        position: str = 'MF'
    ) -> Dict:
        """
        Predict card probabilities.
        """
        # Base fouls per game
        avg_fouls = player_features.get('fouls_avg_5', 1.5)
        
        # Position adjustment
        pos_mult = self.position_multipliers.get(position, {}).get('yellow_cards', 1.0)
        
        # Approximate card rate: ~1 yellow per 4-5 fouls
        yellow_rate_per_foul = 0.22
        
        expected_fouls = avg_fouls * pos_mult
        yellow_prob = 1 - np.exp(-expected_fouls * yellow_rate_per_foul)
        
        return {
            'yellow_card_prob': round(yellow_prob, 4),
            'no_card_prob': round(1 - yellow_prob, 4),
            'expected_fouls': round(expected_fouls, 2),
            'fair_odds_yellow': round(1 / yellow_prob, 2) if yellow_prob > 0 else 99.0
        }
    
    def predict_all_props(
        self,
        player_features: Dict,
        position: str = 'FW',
        available_props: Dict[str, float] = None
    ) -> Dict:
        """
        Predict all available props for a player.
        
        Args:
            player_features: Player features dictionary
            position: Player position
            available_props: Dict of prop_type -> line
        """
        available_props = available_props or {}
        
        predictions = {
            'goals': self.predict_goals(
                player_features, 
                position, 
                available_props.get('goals', 0.5)
            ),
            'assists': self.predict_assists(
                player_features, 
                position, 
                available_props.get('assists', 0.5)
            ),
            'shots': self.predict_shots(
                player_features, 
                position, 
                available_props.get('shots', 2.5)
            ),
            'shots_on_target': self.predict_shots_on_target(
                player_features, 
                position, 
                available_props.get('shots_on_target', 1.5)
            ),
            'anytime_scorer': self.predict_anytime_scorer(
                player_features, 
                position,
                available_props.get('anytime_scorer_odds')
            ),
            'cards': self.predict_cards(player_features, position)
        }
        
        return predictions


class FootballPlayerPropsEngine:
    """
    Complete engine for football player props prediction.
    """
    
    def __init__(self):
        self.predictor = PlayerPropsPredictor()
        self.player_database = {}
        
    def add_player_stats(self, player_id: str, stats: pd.DataFrame):
        """Add or update player statistics."""
        self.player_database[player_id] = stats
        
    def predict_all_props(
        self,
        player_id: str,
        player_name: str = None,
        position: str = 'FW',
        match_context: Dict = None,
        available_props: Dict[str, float] = None
    ) -> Dict:
        """
        Predict all available props for a player.
        
        Args:
            player_id: Unique player identifier
            player_name: Player name for display
            position: Player position
            match_context: Match context information
            available_props: Dict of prop_type -> line
        """
        match_context = match_context or {}
        
        # Get player stats
        player_stats = self.player_database.get(player_id, pd.DataFrame())
        
        # Create features
        features = self.predictor.create_player_features(player_stats, match_context)
        
        # Get predictions
        predictions = self.predictor.predict_all_props(
            features, position, available_props
        )
        
        predictions['player_id'] = player_id
        predictions['player_name'] = player_name or player_id
        predictions['position'] = position
        
        return predictions
    
    def find_value_props(
        self,
        player_id: str,
        position: str,
        match_context: Dict,
        prop_odds: Dict[str, float],
        min_edge: float = 0.05
    ) -> List[Dict]:
        """
        Find value betting opportunities in player props.
        
        Args:
            player_id: Player identifier
            position: Player position
            match_context: Match context
            prop_odds: Dict of prop_type -> bookmaker odds
            min_edge: Minimum edge required
        """
        predictions = self.predict_all_props(
            player_id, None, position, match_context
        )
        
        value_bets = []
        
        # Check anytime scorer
        if 'anytime_scorer_odds' in prop_odds:
            ats = predictions['anytime_scorer']
            implied = 1 / prop_odds['anytime_scorer_odds']
            edge = ats['probability'] - implied
            
            if edge >= min_edge:
                value_bets.append({
                    'market': 'Anytime Scorer',
                    'probability': ats['probability'],
                    'odds': prop_odds['anytime_scorer_odds'],
                    'edge': round(edge, 4),
                    'expected_value': round(edge * prop_odds['anytime_scorer_odds'], 4)
                })
        
        # Check shots over
        if 'shots_over_odds' in prop_odds and 'shots_line' in prop_odds:
            shots = predictions['shots']
            implied = 1 / prop_odds['shots_over_odds']
            edge = shots['over_line_prob'] - implied
            
            if edge >= min_edge:
                value_bets.append({
                    'market': f"Shots Over {prop_odds['shots_line']}",
                    'probability': shots['over_line_prob'],
                    'odds': prop_odds['shots_over_odds'],
                    'edge': round(edge, 4)
                })
        
        return value_bets


# Convenience function
def predict_player_goals(
    goals_avg: float,
    position: str = 'FW',
    is_home: bool = True,
    opponent_strength: float = 1.0
) -> Dict:
    """
    Quick player goals prediction.
    
    Args:
        goals_avg: Player's average goals per game
        position: Player position
        is_home: Playing at home
        opponent_strength: Opponent defensive rating (1.0 = average)
    """
    predictor = PlayerPropsPredictor()
    features = {
        'goals_avg_5': goals_avg,
        'is_home': 1 if is_home else 0,
        'opponent_strength': opponent_strength,
        'minutes_ratio': 0.9
    }
    return predictor.predict_goals(features, position)
