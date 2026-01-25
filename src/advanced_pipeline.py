"""
Advanced Prediction Pipeline

Complete end-to-end prediction system integrating:
- Dixon-Coles Model (correct score, draws)
- Bivariate Poisson (enhanced draw prediction)
- Pi-ratings (team strength)
- Kelly Criterion (value betting)
- All betting markets (1X2, O/U, BTTS, CS, HT/FT)

This is the gold-standard implementation based on academic research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class AdvancedPredictionPipeline:
    """
    Complete prediction pipeline integrating all advanced models.
    
    Models:
    - Dixon-Coles: Correct score prediction with rho correction
    - Bivariate Poisson: Enhanced draw prediction
    - Diagonal-Inflated BP: Best for draw-heavy leagues
    - Pi-ratings: Team strength ratings
    
    Markets:
    - 1X2 (Match Result)
    - Asian Handicap
    - Over/Under Goals (0.5 to 4.5)
    - BTTS (Both Teams to Score)
    - Correct Score (top 15)
    - HT/FT (all 9 combinations)
    - Double Chance
    - Draw No Bet
    """
    
    def __init__(self):
        """Initialize all prediction components."""
        # Import models
        from .dixon_coles import DixonColesModel
        from .bivariate_poisson import (
            BivariatePoissonModel, 
            DiagonalInflatedBivariatePoissonModel
        )
        from .pi_ratings import PiRatingSystem
        from .kelly_criterion import ValueBettingSystem, KellyCriterion
        
        # Initialize models
        self.dixon_coles = DixonColesModel(xi=0.0018)
        self.bivariate = BivariatePoissonModel(correlation=0.08)
        self.diagonal_inflated = DiagonalInflatedBivariatePoissonModel(
            correlation=0.08,
            inflation_factor=0.12
        )
        self.pi_ratings = PiRatingSystem()
        
        # Initialize betting system
        self.kelly = KellyCriterion(
            kelly_fraction=0.25,
            min_edge=0.05,
            max_stake_pct=5.0
        )
        self.value_system = ValueBettingSystem()
        
        # Model weights for ensemble
        self.weights = {
            'dixon_coles': 0.45,
            'bivariate': 0.30,
            'diagonal_inflated': 0.15,
            'pi_ratings': 0.10
        }
    
    def predict(self, home_team: str, away_team: str,
                odds: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive match prediction.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            odds: Optional bookmaker odds for value betting
            
        Returns:
            Complete prediction dictionary with all markets
        """
        logger.info(f"Predicting: {home_team} vs {away_team}")
        
        # Get individual predictions
        dc_pred = self.dixon_coles.predict(home_team, away_team)
        bp_pred = self.bivariate.predict(home_team, away_team)
        di_pred = self.diagonal_inflated.predict(home_team, away_team)
        pi_pred = self.pi_ratings.predict(home_team, away_team)
        
        # Ensemble 1X2 probabilities
        home_win = (
            self.weights['dixon_coles'] * dc_pred.home_win +
            self.weights['bivariate'] * bp_pred.home_win +
            self.weights['diagonal_inflated'] * di_pred['home_win'] +
            self.weights['pi_ratings'] * pi_pred['probabilities']['home_win']
        )
        
        draw = (
            self.weights['dixon_coles'] * dc_pred.draw +
            self.weights['bivariate'] * bp_pred.draw +
            self.weights['diagonal_inflated'] * di_pred['draw'] +
            self.weights['pi_ratings'] * pi_pred['probabilities']['draw']
        )
        
        away_win = (
            self.weights['dixon_coles'] * dc_pred.away_win +
            self.weights['bivariate'] * bp_pred.away_win +
            self.weights['diagonal_inflated'] * di_pred['away_win'] +
            self.weights['pi_ratings'] * pi_pred['probabilities']['away_win']
        )
        
        # Normalize
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        # Determine recommendation
        if home_win > draw and home_win > away_win:
            recommendation = 'Home Win'
            confidence = home_win
        elif away_win > draw and away_win > home_win:
            recommendation = 'Away Win'
            confidence = away_win
        else:
            recommendation = 'Draw'
            confidence = draw
        
        # Confidence level
        if confidence >= 0.60:
            confidence_level = 'HIGH'
        elif confidence >= 0.45:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'LOW'
        
        # Get HT/FT predictions
        htft = self.dixon_coles.predict_htft(home_team, away_team)
        
        # Build prediction result
        prediction = {
            'match': {
                'home_team': home_team,
                'away_team': away_team,
                'timestamp': datetime.now().isoformat()
            },
            
            # 1X2 Market
            '1x2': {
                'home_win': round(home_win, 4),
                'draw': round(draw, 4),
                'away_win': round(away_win, 4)
            },
            'recommendation': recommendation,
            'confidence': round(confidence, 4),
            'confidence_level': confidence_level,
            
            # Expected Goals
            'expected_goals': {
                'home': dc_pred.home_xg,
                'away': dc_pred.away_xg,
                'total': round(dc_pred.home_xg + dc_pred.away_xg, 2)
            },
            
            # Correct Score (from Dixon-Coles - best for this market)
            'correct_scores': dc_pred.correct_scores,
            'most_likely_score': max(dc_pred.correct_scores.items(), 
                                     key=lambda x: x[1])[0],
            
            # Over/Under Goals
            'over_under': {
                'over_0.5': dc_pred.over_0_5,
                'over_1.5': dc_pred.over_1_5,
                'over_2.5': dc_pred.over_2_5,
                'over_3.5': dc_pred.over_3_5,
                'over_4.5': dc_pred.over_4_5,
                'under_0.5': round(1 - dc_pred.over_0_5, 4),
                'under_1.5': round(1 - dc_pred.over_1_5, 4),
                'under_2.5': round(1 - dc_pred.over_2_5, 4),
                'under_3.5': round(1 - dc_pred.over_3_5, 4),
                'under_4.5': round(1 - dc_pred.over_4_5, 4)
            },
            
            # BTTS
            'btts': {
                'yes': dc_pred.btts_yes,
                'no': dc_pred.btts_no
            },
            'btts_recommendation': 'Yes' if dc_pred.btts_yes > 0.5 else 'No',
            
            # Double Chance
            'double_chance': {
                '1X': dc_pred.dc_1x,
                '12': dc_pred.dc_12,
                'X2': dc_pred.dc_x2
            },
            
            # Draw No Bet
            'draw_no_bet': {
                'home': dc_pred.dnb_home,
                'away': dc_pred.dnb_away
            },
            
            # HT/FT
            'htft': htft,
            'htft_recommendation': max(htft.items(), key=lambda x: x[1])[0],
            
            # Team Ratings (from Pi-ratings)
            'ratings': pi_pred.get('ratings', {}),
            
            # Model Breakdown
            'model_breakdown': {
                'dixon_coles': {
                    'home': dc_pred.home_win,
                    'draw': dc_pred.draw,
                    'away': dc_pred.away_win,
                    'rho': dc_pred.rho
                },
                'bivariate_poisson': {
                    'home': bp_pred.home_win,
                    'draw': bp_pred.draw,
                    'away': bp_pred.away_win,
                    'correlation': bp_pred.lambda3
                },
                'diagonal_inflated': {
                    'home': di_pred['home_win'],
                    'draw': di_pred['draw'],
                    'away': di_pred['away_win'],
                    'inflation': di_pred.get('inflation', 0.12)
                },
                'pi_ratings': {
                    'home': pi_pred['probabilities']['home_win'],
                    'draw': pi_pred['probabilities']['draw'],
                    'away': pi_pred['probabilities']['away_win']
                }
            },
            
            # Ensemble weights
            'ensemble_weights': self.weights,
            
            # Model info
            'models_used': ['Dixon-Coles', 'Bivariate Poisson', 
                           'Diagonal-Inflated BP', 'Pi-Ratings']
        }
        
        # Add value betting if odds provided
        if odds:
            prediction['value_betting'] = self._analyze_value(
                home_team, away_team, prediction, odds
            )
        
        return prediction
    
    def _analyze_value(self, home_team: str, away_team: str,
                       prediction: Dict, odds: Dict) -> Dict:
        """Analyze value betting opportunities."""
        value_bets = []
        
        # Check 1X2
        markets = [
            ('1X2', 'Home Win', prediction['1x2']['home_win'], 'odds_home'),
            ('1X2', 'Draw', prediction['1x2']['draw'], 'odds_draw'),
            ('1X2', 'Away Win', prediction['1x2']['away_win'], 'odds_away'),
            ('BTTS', 'Yes', prediction['btts']['yes'], 'odds_btts_yes'),
            ('BTTS', 'No', prediction['btts']['no'], 'odds_btts_no'),
            ('Goals', 'Over 2.5', prediction['over_under']['over_2.5'], 'odds_over_2.5'),
            ('Goals', 'Under 2.5', prediction['over_under']['under_2.5'], 'odds_under_2.5'),
        ]
        
        for market, selection, our_prob, odds_key in markets:
            if odds_key not in odds:
                continue
            
            decimal_odds = odds[odds_key]
            edge = (our_prob * decimal_odds) - 1
            
            if edge >= 0.05:  # 5% minimum edge
                implied = 1 / decimal_odds
                kelly = max(0, ((decimal_odds - 1) * our_prob - (1 - our_prob)) / (decimal_odds - 1))
                
                value_bets.append({
                    'market': market,
                    'selection': selection,
                    'our_probability': round(our_prob, 4),
                    'implied_probability': round(implied, 4),
                    'odds': decimal_odds,
                    'edge': round(edge * 100, 2),
                    'kelly_stake_pct': round(kelly * 25, 2),  # 25% fractional Kelly
                    'confidence': 'HIGH' if edge > 0.15 else 'MEDIUM' if edge > 0.10 else 'LOW'
                })
        
        # Sort by edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return {
            'value_bets': value_bets,
            'best_bet': value_bets[0] if value_bets else None,
            'total_value_bets': len(value_bets)
        }
    
    def predict_correct_score_detailed(self, home_team: str, away_team: str) -> Dict:
        """Get detailed correct score probabilities."""
        dc_pred = self.dixon_coles.predict(home_team, away_team)
        
        return {
            'match': f'{home_team} vs {away_team}',
            'score_probabilities': dc_pred.correct_scores,
            'expected_goals': {
                'home': dc_pred.home_xg,
                'away': dc_pred.away_xg
            },
            'rho_correction': dc_pred.rho,
            'model': 'Dixon-Coles'
        }
    
    def predict_btts_detailed(self, home_team: str, away_team: str) -> Dict:
        """Get detailed BTTS prediction with reasoning."""
        dc_pred = self.dixon_coles.predict(home_team, away_team)
        
        # Calculate probabilities
        home_scores_prob = 1 - np.exp(-dc_pred.home_xg)  # P(X >= 1)
        away_scores_prob = 1 - np.exp(-dc_pred.away_xg)  # P(Y >= 1)
        
        return {
            'match': f'{home_team} vs {away_team}',
            'btts_yes': dc_pred.btts_yes,
            'btts_no': dc_pred.btts_no,
            'recommendation': 'Yes' if dc_pred.btts_yes > 0.5 else 'No',
            'reasoning': {
                'home_scoring_prob': round(home_scores_prob, 4),
                'away_scoring_prob': round(away_scores_prob, 4),
                'home_xg': dc_pred.home_xg,
                'away_xg': dc_pred.away_xg
            },
            'confidence': 'HIGH' if abs(dc_pred.btts_yes - 0.5) > 0.2 else 'MEDIUM'
        }
    
    def predict_htft_detailed(self, home_team: str, away_team: str) -> Dict:
        """Get detailed HT/FT prediction with time breakdown."""
        htft = self.dixon_coles.predict_htft(home_team, away_team)
        dc_pred = self.dixon_coles.predict(home_team, away_team)
        
        # First half probabilities
        first_half_goals = dc_pred.home_xg * 0.42 + dc_pred.away_xg * 0.42
        
        # Most likely combinations
        sorted_htft = sorted(htft.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'match': f'{home_team} vs {away_team}',
            'htft_probabilities': htft,
            'top_3': sorted_htft[:3],
            'recommendation': sorted_htft[0][0],
            'expected_1h_goals': round(first_half_goals, 2),
            'strategy_tip': self._get_htft_strategy(htft)
        }
    
    def _get_htft_strategy(self, htft: Dict) -> str:
        """Get strategy recommendation based on HT/FT probabilities."""
        # Check for value in X/1 or X/2 (research-backed strategy)
        if htft.get('D/H', 0) > 0.15:
            return 'Consider X/1: Draw at HT, Home wins FT - good value in low-scoring games'
        elif htft.get('D/A', 0) > 0.12:
            return 'Consider X/2: Draw at HT, Away wins FT - value for late-goal teams'
        elif htft.get('H/H', 0) > 0.30:
            return 'H/H looks strong: Back home team to lead at both HT and FT'
        else:
            return 'No clear HT/FT edge identified'
    
    def compare_models(self, home_team: str, away_team: str) -> Dict:
        """Compare predictions from all models."""
        dc_pred = self.dixon_coles.predict(home_team, away_team)
        bp_pred = self.bivariate.predict(home_team, away_team)
        di_pred = self.diagonal_inflated.predict(home_team, away_team)
        pi_pred = self.pi_ratings.predict(home_team, away_team)
        
        return {
            'match': f'{home_team} vs {away_team}',
            'models': {
                'Dixon-Coles': {
                    'home': dc_pred.home_win,
                    'draw': dc_pred.draw,
                    'away': dc_pred.away_win,
                    'best_for': 'Correct score'
                },
                'Bivariate Poisson': {
                    'home': bp_pred.home_win,
                    'draw': bp_pred.draw,
                    'away': bp_pred.away_win,
                    'best_for': 'Draw prediction'
                },
                'Diagonal-Inflated BP': {
                    'home': di_pred['home_win'],
                    'draw': di_pred['draw'],
                    'away': di_pred['away_win'],
                    'best_for': 'Draw-heavy leagues'
                },
                'Pi-Ratings': {
                    'home': pi_pred['probabilities']['home_win'],
                    'draw': pi_pred['probabilities']['draw'],
                    'away': pi_pred['probabilities']['away_win'],
                    'best_for': 'Team strength'
                }
            },
            'model_agreement': self._calculate_agreement(
                [dc_pred.home_win, bp_pred.home_win, di_pred['home_win']],
                [dc_pred.draw, bp_pred.draw, di_pred['draw']],
                [dc_pred.away_win, bp_pred.away_win, di_pred['away_win']]
            )
        }
    
    def _calculate_agreement(self, home_probs, draw_probs, away_probs) -> str:
        """Calculate how much models agree."""
        home_std = np.std(home_probs)
        draw_std = np.std(draw_probs)
        away_std = np.std(away_probs)
        
        avg_std = (home_std + draw_std + away_std) / 3
        
        if avg_std < 0.03:
            return 'HIGH (models strongly agree)'
        elif avg_std < 0.06:
            return 'MEDIUM (models mostly agree)'
        else:
            return 'LOW (significant model disagreement)'


# Global instance
advanced_pipeline = AdvancedPredictionPipeline()


def get_advanced_prediction(home_team: str, away_team: str, 
                            odds: Optional[Dict] = None) -> Dict:
    """Get comprehensive prediction using all models."""
    return advanced_pipeline.predict(home_team, away_team, odds)


def get_correct_score_prediction(home_team: str, away_team: str) -> Dict:
    """Get correct score prediction."""
    return advanced_pipeline.predict_correct_score_detailed(home_team, away_team)


def get_btts_prediction(home_team: str, away_team: str) -> Dict:
    """Get BTTS prediction."""
    return advanced_pipeline.predict_btts_detailed(home_team, away_team)


def get_htft_prediction(home_team: str, away_team: str) -> Dict:
    """Get HT/FT prediction."""
    return advanced_pipeline.predict_htft_detailed(home_team, away_team)


def compare_all_models(home_team: str, away_team: str) -> Dict:
    """Compare predictions from all models."""
    return advanced_pipeline.compare_models(home_team, away_team)
