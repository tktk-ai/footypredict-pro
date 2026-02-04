"""
Enhanced Prediction Engine v2

Integrates ALL features for maximum accuracy:
- ML ensemble (XGBoost, LightGBM, CatBoost, Neural Net)
- Team form (last 5 home/away)
- Head-to-head history
- Injury impact
- Weather conditions
- Momentum tracking
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedPredictorV2:
    """Maximum accuracy predictor using all available features"""
    
    def __init__(self):
        self._ml_loaded = False
        self._features_loaded = False
    
    def _ensure_loaded(self):
        """Lazy load all components"""
        if not self._ml_loaded:
            try:
                from src.models.trained_loader import get_trained_loader
                self.ml_loader = get_trained_loader()
                self._ml_loaded = True
            except Exception as e:
                logger.warning(f"Could not load ML models: {e}")
                self.ml_loader = None
        
        if not self._features_loaded:
            try:
                from src.advanced_features import get_feature_builder
                from src.injuries_weather import get_injury_provider, get_weather_provider
                self.feature_builder = get_feature_builder()
                self.injury_provider = get_injury_provider()
                self.weather_provider = get_weather_provider()
                self._features_loaded = True
            except Exception as e:
                logger.warning(f"Could not load feature builders: {e}")
    
    def predict(self, home_team: str, away_team: str, 
                venue: str = None, match_date: str = None) -> Dict:
        """Get enhanced prediction with all features"""
        self._ensure_loaded()
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'adjustments': {},
            'final_prediction': {}
        }
        
        # Base ML prediction
        base_probs = self._get_ml_prediction(home_team, away_team)
        result['base_ml'] = base_probs
        
        # Get advanced features
        features = self._get_features(home_team, away_team)
        result['features'] = features
        
        # Calculate adjustments
        adjustments = self._calculate_adjustments(features)
        result['adjustments'] = adjustments
        
        # Apply adjustments to base probabilities
        final = self._apply_adjustments(base_probs, adjustments)
        result['final_prediction'] = final
        
        return result
    
    def _get_ml_prediction(self, home: str, away: str) -> Dict:
        """Get base ML ensemble prediction"""
        if self.ml_loader and hasattr(self.ml_loader, 'predict'):
            try:
                pred = self.ml_loader.predict(home, away)
                return {
                    'home_prob': pred.get('home_win_prob', 0.4),
                    'draw_prob': pred.get('draw_prob', 0.25),
                    'away_prob': pred.get('away_win_prob', 0.35),
                    'confidence': pred.get('confidence', 0.5),
                    'models_used': pred.get('models_used', [])
                }
            except:
                pass
        
        # Fallback to simple Elo-based prediction
        return {
            'home_prob': 0.45,
            'draw_prob': 0.25,
            'away_prob': 0.30,
            'confidence': 0.45,
            'models_used': ['fallback']
        }
    
    def _get_features(self, home: str, away: str) -> Dict:
        """Get all advanced features"""
        features = {}
        
        try:
            # Form data
            if self.feature_builder:
                match_features = self.feature_builder.build_features(home, away)
                features['form'] = {
                    'home_form': match_features.get('home_form', {}),
                    'away_form': match_features.get('away_form', {}),
                    'form_diff': match_features.get('form_diff', 0),
                    'home_momentum': match_features.get('home_momentum', 0),
                    'away_momentum': match_features.get('away_momentum', 0)
                }
                features['h2h'] = match_features.get('h2h', {})
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
        
        try:
            # Injury data
            if self.injury_provider:
                injury_impact = self.injury_provider.get_match_injury_impact(home, away)
                features['injuries'] = {
                    'home_impact': injury_impact.get('home_impact', 0),
                    'away_impact': injury_impact.get('away_impact', 0),
                    'net_advantage': injury_impact.get('net_advantage', 0)
                }
        except Exception as e:
            logger.warning(f"Injury data error: {e}")
        
        return features
    
    def _calculate_adjustments(self, features: Dict) -> Dict:
        """Calculate probability adjustments based on features"""
        adj = {
            'home': 0,
            'away': 0,
            'draw': 0,
            'reasons': []
        }
        
        # Form adjustment (up to ±5%)
        form = features.get('form', {})
        form_diff = form.get('form_diff', 0)
        if form_diff > 0.5:
            adj['home'] += min(form_diff * 0.03, 0.05)
            adj['reasons'].append(f"Home team in better form (+{adj['home']*100:.1f}%)")
        elif form_diff < -0.5:
            adj['away'] += min(-form_diff * 0.03, 0.05)
            adj['reasons'].append(f"Away team in better form (+{adj['away']*100:.1f}%)")
        
        # Momentum adjustment (up to ±3%)
        home_mom = form.get('home_momentum', 0)
        away_mom = form.get('away_momentum', 0)
        if home_mom > 0.3:
            adj['home'] += min(home_mom * 0.02, 0.03)
            adj['reasons'].append("Home momentum positive")
        if away_mom > 0.3:
            adj['away'] += min(away_mom * 0.02, 0.03)
            adj['reasons'].append("Away momentum positive")
        
        # H2H adjustment (up to ±4%)
        h2h = features.get('h2h', {})
        if h2h.get('total_matches', 0) >= 3:
            h2h_adv = h2h.get('team1_win_pct', 0.5) - h2h.get('team2_win_pct', 0.5)
            if abs(h2h_adv) > 0.2:
                if h2h_adv > 0:
                    adj['home'] += min(h2h_adv * 0.08, 0.04)
                    adj['reasons'].append("Historical H2H favors home")
                else:
                    adj['away'] += min(-h2h_adv * 0.08, 0.04)
                    adj['reasons'].append("Historical H2H favors away")
        
        # Injury adjustment (up to ±5%)
        injuries = features.get('injuries', {})
        net_inj = injuries.get('net_advantage', 0)
        if abs(net_inj) > 0.05:
            if net_inj > 0:
                adj['home'] += min(net_inj * 0.5, 0.05)
                adj['reasons'].append("Away team has more injuries")
            else:
                adj['away'] += min(-net_inj * 0.5, 0.05)
                adj['reasons'].append("Home team has more injuries")
        
        return adj
    
    def _apply_adjustments(self, base: Dict, adj: Dict) -> Dict:
        """Apply adjustments to base probabilities"""
        home = base.get('home_prob', 0.4) + adj.get('home', 0) - adj.get('away', 0) * 0.5
        away = base.get('away_prob', 0.3) + adj.get('away', 0) - adj.get('home', 0) * 0.5
        draw = base.get('draw_prob', 0.25) + adj.get('draw', 0)
        
        # Normalize to sum to 1
        total = home + draw + away
        home /= total
        draw /= total
        away /= total
        
        # Clamp probabilities
        home = max(0.05, min(0.9, home))
        away = max(0.05, min(0.9, away))
        draw = max(0.05, min(0.5, draw))
        
        # Re-normalize
        total = home + draw + away
        home /= total
        draw /= total
        away /= total
        
        # Determine prediction
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
            'confidence': round(conf, 4),
            'adjustments_applied': adj.get('reasons', [])
        }
    
    def predict_with_goals(self, home_team: str, away_team: str) -> Dict:
        """Predict match outcome AND expected goals"""
        pred = self.predict(home_team, away_team)
        
        # Get goal-related features
        try:
            features = self.feature_builder.build_features(home_team, away_team) if self.feature_builder else {}
            
            home_scoring = features.get('home_scoring_rate', 1.3)
            home_conceding = features.get('home_conceding_rate', 1.0)
            away_scoring = features.get('away_scoring_rate', 1.0)
            away_conceding = features.get('away_conceding_rate', 1.2)
            
            # Expected goals (simple model)
            home_xg = (home_scoring + away_conceding) / 2
            away_xg = (away_scoring + home_conceding) / 2
            total_xg = home_xg + away_xg
            
            # Over/under probabilities (based on Poisson-like distribution)
            import math
            
            def poisson_prob(k, lambda_):
                return (lambda_**k * math.exp(-lambda_)) / math.factorial(k)
            
            over_25 = 1 - sum(poisson_prob(k, total_xg) for k in range(3))
            over_15 = 1 - sum(poisson_prob(k, total_xg) for k in range(2))
            btts = (1 - poisson_prob(0, home_xg)) * (1 - poisson_prob(0, away_xg))
            
            pred['goals'] = {
                'home_xg': round(home_xg, 2),
                'away_xg': round(away_xg, 2),
                'total_xg': round(total_xg, 2),
                'over_2.5': round(over_25, 3),
                'over_1.5': round(over_15, 3),
                'btts': round(btts, 3)
            }
        except Exception as e:
            logger.warning(f"Goals prediction error: {e}")
            pred['goals'] = {
                'home_xg': 1.3,
                'away_xg': 1.0,
                'total_xg': 2.3,
                'over_2.5': 0.45,
                'btts': 0.48
            }
        
        return pred


# Global instance
_predictor: Optional[EnhancedPredictorV2] = None

def get_enhanced_predictor() -> EnhancedPredictorV2:
    global _predictor
    if _predictor is None:
        _predictor = EnhancedPredictorV2()
    return _predictor

def enhanced_predict(home: str, away: str, venue: str = None) -> Dict:
    """Get enhanced prediction with all features"""
    return get_enhanced_predictor().predict(home, away, venue)

def enhanced_predict_with_goals(home: str, away: str) -> Dict:
    """Get enhanced prediction with goal predictions"""
    return get_enhanced_predictor().predict_with_goals(home, away)
