"""
Unified Ensemble Predictor
===========================

Combines predictions from multiple model sources:
- SportyBet specialized models (highest accuracy)
- V4 Enhanced models
- Correct Score, Halftime, Corner predictors

Uses weighted averaging based on historical accuracy.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


@dataclass
class ModelWeight:
    """Weight configuration for a model source."""
    name: str
    weight: float
    accuracy: float


class UnifiedEnsemblePredictor:
    """
    Unified predictor that combines multiple model sources.
    """
    
    # Default weights based on historical accuracy
    MODEL_WEIGHTS = {
        'sportybet': {'weight': 0.6, 'accuracy': 0.72},  # Highest accuracy
        'v4_enhanced': {'weight': 0.25, 'accuracy': 0.67},
        'v4_fixed': {'weight': 0.15, 'accuracy': 0.57},
    }
    
    def __init__(self):
        self.sportybet_predictor = None
        self.v4_enhanced = None
        self.v4_fixed = None
        self.correct_score = None
        self.halftime = None
        self.corners = None
        
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available model sources."""
        # Load SportyBet predictor
        try:
            from src.models.sportybet_predictor import get_sportybet_predictor
            self.sportybet_predictor = get_sportybet_predictor()
            logger.info(f"✅ Loaded SportyBet predictor with {len(self.sportybet_predictor.models)} markets")
        except Exception as e:
            logger.warning(f"⚠️ SportyBet predictor not available: {e}")
        
        # Load V4 Enhanced
        try:
            v4_enhanced_dir = MODELS_DIR / "v4_enhanced"
            if (v4_enhanced_dir / "result_model.joblib").exists():
                self.v4_enhanced = {
                    'models': {},
                    'scalers': joblib.load(v4_enhanced_dir / "scalers.joblib"),
                    'encoders': joblib.load(v4_enhanced_dir / "encoders.joblib"),
                    'feature_cols': joblib.load(v4_enhanced_dir / "feature_cols.joblib"),
                }
                for market in ['result', 'over25', 'over15', 'btts', 'dc_1x', 'dc_x2', 'dc_12']:
                    path = v4_enhanced_dir / f"{market}_model.joblib"
                    if path.exists():
                        self.v4_enhanced['models'][market] = joblib.load(path)
                logger.info(f"✅ Loaded V4 Enhanced with {len(self.v4_enhanced['models'])} markets")
        except Exception as e:
            logger.warning(f"⚠️ V4 Enhanced not available: {e}")
        
        # Load V4 Fixed
        try:
            v4_fixed_dir = MODELS_DIR / "v4_fixed"
            if (v4_fixed_dir / "result_model.joblib").exists():
                self.v4_fixed = {
                    'models': {},
                    'scalers': joblib.load(v4_fixed_dir / "scalers.joblib"),
                    'encoders': joblib.load(v4_fixed_dir / "encoders.joblib"),
                    'feature_cols': joblib.load(v4_fixed_dir / "feature_cols.joblib"),
                }
                for market in ['result', 'over25', 'over15', 'btts']:
                    path = v4_fixed_dir / f"{market}_model.joblib"
                    if path.exists():
                        self.v4_fixed['models'][market] = joblib.load(path)
                logger.info(f"✅ Loaded V4 Fixed with {len(self.v4_fixed['models'])} markets")
        except Exception as e:
            logger.warning(f"⚠️ V4 Fixed not available: {e}")
        
        # Load specialized predictors
        try:
            from src.models.correct_score_predictor import get_correct_score_predictor
            self.correct_score = get_correct_score_predictor()
            logger.info("✅ Loaded Correct Score predictor")
        except Exception as e:
            logger.warning(f"⚠️ Correct Score not available: {e}")
        
        try:
            from src.models.halftime_predictor import get_halftime_predictor
            self.halftime = get_halftime_predictor()
            logger.info("✅ Loaded Halftime predictor")
        except Exception as e:
            logger.warning(f"⚠️ Halftime not available: {e}")
        
        try:
            from src.models.corner_predictor import get_corner_predictor
            self.corners = get_corner_predictor()
            logger.info("✅ Loaded Corner predictor")
        except Exception as e:
            logger.warning(f"⚠️ Corner not available: {e}")
    
    def predict(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = '',
        features: Optional[Dict] = None
    ) -> Dict:
        """
        Make predictions using all available models and combine.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            features: Optional feature dictionary for V4 models
        
        Returns:
            Combined predictions for all markets
        """
        predictions = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'sources': [],
            'markets': {},
            'special': {},
        }
        
        # ==========================================
        # 1. SportyBet predictions (primary source)
        # ==========================================
        if self.sportybet_predictor:
            try:
                sportybet_result = self.sportybet_predictor.predict_all(home_team, away_team, league)
                predictions['sources'].append('sportybet')
                
                # SportyBetMultiPrediction has .predictions dict of SportyBetPrediction objects
                for market, pred in sportybet_result.predictions.items():
                    if market not in predictions['markets']:
                        predictions['markets'][market] = []
                    predictions['markets'][market].append({
                        'source': 'sportybet',
                        'prediction': pred.prediction,
                        'probability': pred.probability * 100,  # Convert to percentage
                        'weight': self.MODEL_WEIGHTS['sportybet']['weight'],
                    })
            except Exception as e:
                logger.warning(f"SportyBet prediction error: {e}")
        
        # ==========================================
        # 2. Correct Score predictions
        # ==========================================
        if self.correct_score:
            try:
                cs_pred = self.correct_score.predict(home_team, away_team, league)
                predictions['special']['correct_score'] = cs_pred['top_scores'][:5]
                predictions['special']['cs_result'] = cs_pred['result_probabilities']
            except Exception as e:
                logger.warning(f"Correct Score error: {e}")
        
        # ==========================================
        # 3. Halftime predictions
        # ==========================================
        if self.halftime:
            try:
                ht_pred = self.halftime.predict(home_team, away_team)
                predictions['special']['halftime'] = {
                    'result': ht_pred['ht_result'],
                    'over_under': ht_pred['ht_over_under'],
                    'btts': ht_pred['ht_btts'],
                    'best_picks': ht_pred.get('best_picks', []),
                }
            except Exception as e:
                logger.warning(f"Halftime error: {e}")
        
        # ==========================================
        # 4. Corner predictions
        # ==========================================
        if self.corners:
            try:
                corner_pred = self.corners.predict(home_team, away_team, league)
                predictions['special']['corners'] = {
                    'total': corner_pred['total_corners'],
                    'team': corner_pred['team_corners'],
                    'best_picks': corner_pred.get('best_picks', []),
                }
            except Exception as e:
                logger.warning(f"Corner error: {e}")
        
        # ==========================================
        # 5. Combine predictions using weighted average
        # ==========================================
        predictions['combined'] = self._combine_predictions(predictions['markets'])
        
        # ==========================================
        # 6. Generate best picks
        # ==========================================
        predictions['best_picks'] = self._generate_best_picks(predictions)
        
        return predictions
    
    def _combine_predictions(self, markets: Dict) -> Dict:
        """
        Combine predictions from multiple sources using weighted average.
        """
        combined = {}
        
        for market, preds in markets.items():
            if not preds:
                continue
            
            # Calculate weighted average probability
            total_weight = sum(p['weight'] for p in preds)
            weighted_prob = sum(p['probability'] * p['weight'] for p in preds) / total_weight
            
            # Determine consensus prediction
            pred_votes = {}
            for p in preds:
                pred = p['prediction']
                if pred:
                    pred_votes[pred] = pred_votes.get(pred, 0) + p['weight']
            
            consensus = max(pred_votes, key=pred_votes.get) if pred_votes else preds[0]['prediction']
            
            combined[market] = {
                'prediction': consensus,
                'probability': weighted_prob,
                'confidence': 'high' if weighted_prob > 70 else ('medium' if weighted_prob > 55 else 'low'),
                'sources': len(preds),
                'agreement': len(set(p['prediction'] for p in preds if p['prediction'])) == 1,
            }
        
        return combined
    
    def _generate_best_picks(self, predictions: Dict) -> List[Dict]:
        """
        Generate best picks across all markets.
        """
        best_picks = []
        
        # From combined predictions
        for market, pred in predictions.get('combined', {}).items():
            if pred['probability'] > 65 and pred['agreement']:
                best_picks.append({
                    'market': market,
                    'prediction': pred['prediction'],
                    'probability': pred['probability'],
                    'confidence': pred['confidence'],
                    'type': 'combined',
                })
        
        # From halftime
        if 'halftime' in predictions.get('special', {}):
            for pick in predictions['special']['halftime'].get('best_picks', []):
                if pick.get('probability', 0) > 60:
                    best_picks.append({
                        'market': pick['market'],
                        'prediction': pick['prediction'],
                        'probability': pick['probability'],
                        'confidence': 'medium',
                        'type': 'halftime',
                    })
        
        # From corners
        if 'corners' in predictions.get('special', {}):
            for pick in predictions['special']['corners'].get('best_picks', []):
                if pick.get('probability', 0) > 60:
                    best_picks.append({
                        'market': pick['market'],
                        'prediction': pick['prediction'],
                        'probability': pick['probability'],
                        'confidence': 'medium',
                        'type': 'corners',
                    })
        
        # Sort by probability
        best_picks.sort(key=lambda x: x['probability'], reverse=True)
        
        return best_picks[:10]  # Top 10 picks
    
    def predict_match_summary(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = ''
    ) -> Dict:
        """
        Get a summary of predictions for a match.
        """
        full_pred = self.predict(home_team, away_team, league)
        
        summary = {
            'match': f"{home_team} vs {away_team}",
            'league': league,
            'result': None,
            'goals': {},
            'best_picks': full_pred.get('best_picks', [])[:5],
        }
        
        # Result prediction
        if 'result' in full_pred.get('combined', {}):
            result = full_pred['combined']['result']
            summary['result'] = {
                'prediction': result['prediction'],
                'confidence': f"{result['probability']:.1f}%",
            }
        elif '1x2' in full_pred.get('combined', {}):
            result = full_pred['combined']['1x2']
            summary['result'] = {
                'prediction': result['prediction'],
                'confidence': f"{result['probability']:.1f}%",
            }
        
        # Goals predictions
        for market in ['over_15', 'over_25', 'btts']:
            if market in full_pred.get('combined', {}):
                summary['goals'][market] = {
                    'prediction': full_pred['combined'][market]['prediction'],
                    'probability': f"{full_pred['combined'][market]['probability']:.1f}%",
                }
        
        return summary


# Singleton instance
_predictor = None


def get_unified_predictor() -> UnifiedEnsemblePredictor:
    """Get singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = UnifiedEnsemblePredictor()
    return _predictor


if __name__ == "__main__":
    predictor = UnifiedEnsemblePredictor()
    
    result = predictor.predict('Bayern München', 'Borussia Dortmund', 'Bundesliga')
    
    print(f"\n⚽ {result['home_team']} vs {result['away_team']}")
    print(f"📊 Sources: {result['sources']}")
    print()
    
    print("🎯 Best Picks:")
    for pick in result['best_picks'][:5]:
        print(f"  {pick['market']}: {pick['prediction']} ({pick['probability']:.1f}%)")
    
    print()
    print("📊 Combined Markets:")
    for market, pred in result.get('combined', {}).items():
        print(f"  {market}: {pred['prediction']} ({pred['probability']:.1f}%)")
