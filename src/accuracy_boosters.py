"""
Advanced Accuracy Boosters

Techniques to push accuracy beyond 70%:
1. Odds-as-features (bookmaker implied probabilities)
2. Time-decay weighting
3. Stacking ensemble (meta-learner)
4. Calibrated probabilities
5. League-specific adjustments
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class OddsAsFeatures:
    """
    Use bookmaker odds as features.
    Bookmakers have massive analytical resources - their odds contain
    valuable information about match outcomes.
    """
    
    def __init__(self):
        self.avg_margin = 0.05  # Typical 5% margin
    
    def odds_to_probabilities(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict:
        """Convert decimal odds to implied probabilities, removing margin"""
        # Raw implied probabilities
        raw_home = 1 / home_odds if home_odds > 1 else 0.33
        raw_draw = 1 / draw_odds if draw_odds > 1 else 0.33
        raw_away = 1 / away_odds if away_odds > 1 else 0.33
        
        # Total (includes margin)
        total = raw_home + raw_draw + raw_away
        
        # Normalize to remove margin
        return {
            'home_prob': raw_home / total,
            'draw_prob': raw_draw / total,
            'away_prob': raw_away / total,
            'margin': total - 1,
            'odds_confidence': 1 - (total - 1) / 0.2  # Higher margin = less confident
        }
    
    def blend_with_model(self, model_pred: Dict, odds_pred: Dict, 
                         model_weight: float = 0.4) -> Dict:
        """
        Blend model predictions with bookmaker-implied probabilities.
        Bookmaker odds are often more reliable, so they get higher weight.
        """
        odds_weight = 1 - model_weight
        
        home = model_pred.get('home_prob', 0.4) * model_weight + \
               odds_pred.get('home_prob', 0.4) * odds_weight
        draw = model_pred.get('draw_prob', 0.25) * model_weight + \
               odds_pred.get('draw_prob', 0.25) * odds_weight
        away = model_pred.get('away_prob', 0.35) * model_weight + \
               odds_pred.get('away_prob', 0.35) * odds_weight
        
        # Normalize
        total = home + draw + away
        home /= total
        draw /= total
        away /= total
        
        # Determine prediction
        if home > draw and home > away:
            pred, conf = 'Home Win', home
        elif away > draw:
            pred, conf = 'Away Win', away
        else:
            pred, conf = 'Draw', draw
        
        return {
            'home_win_prob': round(home, 4),
            'draw_prob': round(draw, 4),
            'away_win_prob': round(away, 4),
            'predicted_outcome': pred,
            'confidence': round(conf, 4),
            'blend': {
                'model_weight': model_weight,
                'odds_weight': odds_weight
            }
        }


class TimeDecayWeighting:
    """
    Weight recent matches more heavily than older ones.
    A team's form from last week matters more than form from 3 months ago.
    """
    
    def __init__(self, half_life_days: int = 30):
        self.half_life = half_life_days
    
    def calculate_weight(self, match_date: str, reference_date: str = None) -> float:
        """Calculate time-decay weight for a match"""
        try:
            if isinstance(match_date, str):
                match_dt = datetime.fromisoformat(match_date.replace('Z', ''))
            else:
                match_dt = match_date
            
            ref_dt = datetime.fromisoformat(reference_date) if reference_date else datetime.now()
            
            days_ago = (ref_dt - match_dt).days
            
            # Exponential decay
            weight = math.exp(-0.693 * days_ago / self.half_life)  # 0.693 = ln(2)
            return max(0.01, weight)
        except:
            return 0.5
    
    def weighted_form(self, matches: List[Dict], team: str) -> Dict:
        """Calculate time-weighted form"""
        if not matches:
            return {'weighted_points': 0, 'weighted_gf': 0, 'weighted_ga': 0}
        
        total_weight = 0
        weighted_points = 0
        weighted_gf = 0
        weighted_ga = 0
        
        for match in matches:
            weight = self.calculate_weight(match.get('date', ''))
            
            home = match.get('home_team')
            h_score = match.get('home_score', 0)
            a_score = match.get('away_score', 0)
            
            if team == home:
                gf, ga = h_score, a_score
                if h_score > a_score: pts = 3
                elif h_score < a_score: pts = 0
                else: pts = 1
            else:
                gf, ga = a_score, h_score
                if a_score > h_score: pts = 3
                elif a_score < h_score: pts = 0
                else: pts = 1
            
            weighted_points += pts * weight
            weighted_gf += gf * weight
            weighted_ga += ga * weight
            total_weight += weight
        
        if total_weight > 0:
            return {
                'weighted_points': round(weighted_points / total_weight, 3),
                'weighted_gf': round(weighted_gf / total_weight, 3),
                'weighted_ga': round(weighted_ga / total_weight, 3),
                'total_weight': round(total_weight, 3)
            }
        return {'weighted_points': 0, 'weighted_gf': 0, 'weighted_ga': 0}


class StackingEnsemble:
    """
    Meta-learner that combines multiple base model predictions.
    Learns optimal weights for combining different models.
    """
    
    def __init__(self):
        self.meta_weights = {
            'xgb': 0.25,
            'lgb': 0.25, 
            'cat': 0.20,
            'nn': 0.10,
            'odds': 0.20  # Bookmaker odds
        }
        self.learned = False
    
    def set_weights(self, weights: Dict):
        """Set custom weights"""
        self.meta_weights = weights
        self.learned = True
    
    def learn_weights(self, predictions: List[Dict], actuals: List[str]):
        """
        Learn optimal weights from historical predictions.
        Uses simple accuracy-based weighting.
        """
        if len(predictions) < 50:
            return  # Need enough data
        
        model_accuracy = {}
        
        for model in self.meta_weights.keys():
            correct = 0
            total = 0
            
            for pred, actual in zip(predictions, actuals):
                if model in pred:
                    model_pred = pred[model].get('predicted_outcome')
                    if model_pred == actual:
                        correct += 1
                    total += 1
            
            if total > 0:
                model_accuracy[model] = correct / total
        
        # Convert accuracy to weights
        if model_accuracy:
            total_acc = sum(model_accuracy.values())
            for model in model_accuracy:
                self.meta_weights[model] = model_accuracy[model] / total_acc
            self.learned = True
    
    def predict(self, model_predictions: Dict) -> Dict:
        """
        Combine predictions from multiple models using learned weights.
        """
        home_prob = 0
        draw_prob = 0
        away_prob = 0
        total_weight = 0
        
        for model, weight in self.meta_weights.items():
            if model in model_predictions:
                pred = model_predictions[model]
                home_prob += pred.get('home_prob', 0.33) * weight
                draw_prob += pred.get('draw_prob', 0.33) * weight
                away_prob += pred.get('away_prob', 0.33) * weight
                total_weight += weight
        
        if total_weight > 0:
            home_prob /= total_weight
            draw_prob /= total_weight
            away_prob /= total_weight
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        
        if home_prob > draw_prob and home_prob > away_prob:
            pred, conf = 'Home Win', home_prob
        elif away_prob > draw_prob:
            pred, conf = 'Away Win', away_prob
        else:
            pred, conf = 'Draw', draw_prob
        
        return {
            'home_win_prob': round(home_prob, 4),
            'draw_prob': round(draw_prob, 4),
            'away_win_prob': round(away_prob, 4),
            'predicted_outcome': pred,
            'confidence': round(conf, 4),
            'weights_used': self.meta_weights,
            'is_learned': self.learned
        }


class LeagueSpecificAdjustments:
    """
    Apply league-specific adjustments based on known patterns.
    Different leagues have different characteristics.
    """
    
    # League characteristics based on historical data
    LEAGUE_PATTERNS = {
        'Premier League': {
            'home_advantage': 0.08,  # 8% extra for home
            'draw_rate': 0.22,
            'avg_goals': 2.8,
            'unpredictability': 0.15
        },
        'Bundesliga': {
            'home_advantage': 0.10,
            'draw_rate': 0.20,
            'avg_goals': 3.1,
            'unpredictability': 0.12
        },
        'La Liga': {
            'home_advantage': 0.12,
            'draw_rate': 0.25,
            'avg_goals': 2.6,
            'unpredictability': 0.10
        },
        'Serie A': {
            'home_advantage': 0.10,
            'draw_rate': 0.28,
            'avg_goals': 2.7,
            'unpredictability': 0.13
        },
        'Ligue 1': {
            'home_advantage': 0.09,
            'draw_rate': 0.24,
            'avg_goals': 2.5,
            'unpredictability': 0.14
        },
        'Champions League': {
            'home_advantage': 0.06,
            'draw_rate': 0.20,
            'avg_goals': 2.9,
            'unpredictability': 0.18
        },
        'default': {
            'home_advantage': 0.08,
            'draw_rate': 0.25,
            'avg_goals': 2.5,
            'unpredictability': 0.15
        }
    }
    
    def get_league_pattern(self, league: str) -> Dict:
        """Get pattern for a league"""
        for name, pattern in self.LEAGUE_PATTERNS.items():
            if name.lower() in league.lower():
                return pattern
        return self.LEAGUE_PATTERNS['default']
    
    def adjust_prediction(self, prediction: Dict, league: str) -> Dict:
        """Adjust prediction based on league characteristics"""
        pattern = self.get_league_pattern(league)
        
        home = prediction.get('home_win_prob', 0.4)
        draw = prediction.get('draw_prob', 0.25)
        away = prediction.get('away_win_prob', 0.35)
        
        # Apply home advantage adjustment
        home_boost = pattern['home_advantage'] * 0.5  # Partial application
        home += home_boost
        away -= home_boost * 0.5
        draw -= home_boost * 0.5
        
        # Adjust draw probability toward league average
        draw_target = pattern['draw_rate']
        draw = draw * 0.7 + draw_target * 0.3
        
        # Normalize
        total = home + draw + away
        home /= total
        draw /= total
        away /= total
        
        # Apply unpredictability (reduce confidence)
        unpred = pattern['unpredictability']
        confidence = prediction.get('confidence', 0.5) * (1 - unpred)
        
        if home > draw and home > away:
            pred = 'Home Win'
            conf = home
        elif away > draw:
            pred = 'Away Win'
            conf = away
        else:
            pred = 'Draw'
            conf = draw
        
        return {
            'home_win_prob': round(home, 4),
            'draw_prob': round(draw, 4),
            'away_win_prob': round(away, 4),
            'predicted_outcome': pred,
            'confidence': round(conf * (1 - unpred * 0.5), 4),
            'league_pattern': pattern
        }


class ProbabilityCalibrator:
    """
    Calibrate probabilities so a 70% prediction wins 70% of the time.
    Uses isotonic regression or Platt scaling.
    """
    
    def __init__(self):
        self.calibration_map = {}
        self.is_calibrated = False
    
    def calibrate(self, predictions: List[float], outcomes: List[bool]):
        """
        Learn calibration from historical data.
        predictions: list of predicted probabilities
        outcomes: list of whether prediction was correct (True/False)
        """
        if len(predictions) < 100:
            return
        
        # Bin predictions and calculate actual rates
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            bin_preds = []
            bin_outcomes = []
            
            for p, o in zip(predictions, outcomes):
                if low <= p < high:
                    bin_preds.append(p)
                    bin_outcomes.append(o)
            
            if bin_preds:
                actual_rate = sum(bin_outcomes) / len(bin_outcomes)
                avg_pred = sum(bin_preds) / len(bin_preds)
                self.calibration_map[(low, high)] = actual_rate / avg_pred if avg_pred > 0 else 1
        
        self.is_calibrated = True
    
    def apply(self, prob: float) -> float:
        """Apply calibration to a probability"""
        if not self.is_calibrated:
            return prob
        
        for (low, high), factor in self.calibration_map.items():
            if low <= prob < high:
                calibrated = prob * factor
                return max(0.01, min(0.99, calibrated))
        
        return prob


# Global instances
_odds_features = None
_time_decay = None
_stacking = None
_league_adj = None
_calibrator = None

def get_odds_features() -> OddsAsFeatures:
    global _odds_features
    if _odds_features is None:
        _odds_features = OddsAsFeatures()
    return _odds_features

def get_time_decay() -> TimeDecayWeighting:
    global _time_decay
    if _time_decay is None:
        _time_decay = TimeDecayWeighting()
    return _time_decay

def get_stacking() -> StackingEnsemble:
    global _stacking
    if _stacking is None:
        _stacking = StackingEnsemble()
    return _stacking

def get_league_adj() -> LeagueSpecificAdjustments:
    global _league_adj
    if _league_adj is None:
        _league_adj = LeagueSpecificAdjustments()
    return _league_adj

def get_calibrator() -> ProbabilityCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = ProbabilityCalibrator()
    return _calibrator
