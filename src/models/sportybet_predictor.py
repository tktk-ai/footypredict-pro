"""
SportyBet Markets Predictor
============================
Inference module for SportyBet specialized models.

Loads trained XGBoost models from models/trained/sportybet/ and provides
prediction functions for all available markets.

Usage:
    from src.models.sportybet_predictor import SportyBetPredictor, sportybet_predict
    
    # Quick prediction
    predictor = SportyBetPredictor()
    result = predictor.predict_all('Bayern', 'Dortmund')
    
    # Specific market
    btts = predictor.predict_market('btts', home='Bayern', away='Dortmund')
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "trained" / "sportybet"
DATA_DIR = BASE_DIR / "data"

# Market display names
MARKET_NAMES = {
    'over_15': 'Over 1.5 Goals',
    'over_25': 'Over 2.5 Goals',
    'over_35': 'Over 3.5 Goals',
    'btts': 'Both Teams to Score',
    'dc_1x': 'Double Chance 1X',
    'dc_x2': 'Double Chance X2',
    'dc_12': 'Double Chance 12',
    'ht_over_05': 'HT Over 0.5',
    'ht_over_15': 'HT Over 1.5',
    'ht_btts': 'HT Both Teams Score',
    'home_over_25': 'Home & Over 2.5',
    'away_over_25': 'Away & Over 2.5',
    'home_btts': 'Home & BTTS',
    'away_btts': 'Away & BTTS',
    'home_win_nil': 'Home Win to Nil',
    'away_win_nil': 'Away Win to Nil',
    'goals_2_3': '2-3 Goals',
    'goals_4_5': '4-5 Goals',
    'cs_home_1_0': 'CS Home 1-0',
    'cs_home_2_1': 'CS Home 2-1',
    'cs_draw_1_1': 'CS Draw 1-1',
    '1x2': 'Match Result',
    'ht_1x2': 'Half-Time Result',
    'dnb_home': 'Draw No Bet Home',
    'dnb_away': 'Draw No Bet Away',
}


@dataclass
class SportyBetPrediction:
    """Single market prediction result."""
    market: str
    market_name: str
    probability: float
    confidence: float
    prediction: str  # 'Yes', 'No', 'Home', 'Draw', 'Away'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SportyBetMultiPrediction:
    """Multi-market prediction result."""
    home_team: str
    away_team: str
    predictions: Dict[str, SportyBetPrediction]
    best_picks: List[Dict]  # Top 5 most confident picks
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'home_team': self.home_team,
            'away_team': self.away_team,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'best_picks': self.best_picks,
            'timestamp': self.timestamp
        }


class SportyBetPredictor:
    """
    Predictor for SportyBet specialized markets.
    
    Loads trained XGBoost models and provides inference methods.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self._load_models()
        self._load_feature_cols()
    
    def _load_models(self) -> None:
        """Load all available trained models."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        import xgboost as xgb
        
        for model_file in self.models_dir.glob("*_model.json"):
            market = model_file.stem.replace("_model", "")
            scaler_file = self.models_dir / f"{market}_scaler.pkl"
            
            try:
                model = xgb.XGBClassifier()
                model.load_model(str(model_file))
                self.models[market] = model
                
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.scalers[market] = pickle.load(f)
                
                logger.debug(f"Loaded model: {market}")
            except Exception as e:
                logger.error(f"Failed to load {market}: {e}")
        
        logger.info(f"Loaded {len(self.models)} SportyBet models")
    
    def _load_feature_cols(self) -> None:
        """Load feature column names from training data."""
        # Try to load from cached features file
        features_file = self.models_dir / "feature_cols.json"
        if features_file.exists():
            with open(features_file) as f:
                self.feature_cols = json.load(f)
            return
        
        # Try training data
        data_file = DATA_DIR / "comprehensive_training_data.csv"
        if data_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(data_file, nrows=1)
                exclude = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'FTR', 'HTR', 'Date', 
                          'HomeTeam', 'AwayTeam', 'Div', 'Season', 'TotalGoals',
                          'HTTotalGoals', '2HTotalGoals', 'Unnamed: 0', 'Time',
                          'Referee', 'Attendance', 'MatchID']
                self.feature_cols = [c for c in df.columns if c not in exclude 
                                    and df[c].dtype in ['int64', 'float64']]
            except Exception as e:
                logger.warning(f"Could not load feature cols: {e}")
    
    def get_available_markets(self) -> List[Dict]:
        """Get list of available trained markets."""
        return [
            {
                'id': market,
                'name': MARKET_NAMES.get(market, market.replace('_', ' ').title()),
                'type': 'binary' if market not in ['1x2', 'ht_1x2'] else 'multiclass'
            }
            for market in sorted(self.models.keys())
        ]
    
    def _get_team_features(self, home_team: str, away_team: str, league: str = '', 
                            market: str = None) -> np.ndarray:
        """
        Generate feature vector for a match.
        
        Uses historical data or creates prediction-time features.
        The feature count is determined from the scaler's trained dimensions.
        """
        # For now, create mock features based on team name hashes
        # In production, this would pull from live data sources
        np.random.seed(hash(f"{home_team}{away_team}") % 2**32)
        
        # Get expected feature count from scaler if available
        n_features = 162  # Default based on training
        if market and market in self.scalers:
            # Get from specific market scaler
            n_features = self.scalers[market].n_features_in_
        elif self.scalers:
            # Get from first available scaler
            first_scaler = next(iter(self.scalers.values()))
            n_features = first_scaler.n_features_in_
        elif self.feature_cols:
            n_features = len(self.feature_cols)
        
        features = np.random.randn(n_features) * 0.5
        
        # Add some structure based on typical football stats
        # These would be replaced with real feature extraction
        features[0] = np.random.uniform(0.3, 0.6)  # Home win base
        features[1] = np.random.uniform(0.2, 0.35)  # Draw base
        features[2] = np.random.uniform(0.2, 0.5)  # Away win base
        features[3] = np.random.uniform(1.5, 3.5)  # Expected goals
        features[4] = np.random.uniform(0.4, 0.7)  # BTTS rate
        
        return features.reshape(1, -1)
    
    def predict_market(self, market: str, home_team: str, away_team: str, 
                       league: str = '', features: Optional[np.ndarray] = None) -> SportyBetPrediction:
        """
        Predict a specific market for a match.
        
        Args:
            market: Market ID (e.g., 'btts', 'over_25')
            home_team: Home team name
            away_team: Away team name
            league: Optional league name
            features: Optional pre-computed features
        
        Returns:
            SportyBetPrediction with probability and confidence
        """
        if market not in self.models:
            raise ValueError(f"Model not found for market: {market}. "
                           f"Available: {list(self.models.keys())}")
        
        model = self.models[market]
        
        # Get features
        if features is None:
            features = self._get_team_features(home_team, away_team, league, market)
        
        # Scale if scaler available
        if market in self.scalers:
            features = self.scalers[market].transform(features)
        
        # Predict
        try:
            proba = model.predict_proba(features)[0]
            pred_class = model.predict(features)[0]
            
            if market in ['1x2', 'ht_1x2']:
                # Multiclass
                labels = ['Home', 'Draw', 'Away']
                prediction = labels[pred_class]
                probability = float(proba[pred_class])
            else:
                # Binary
                prediction = 'Yes' if pred_class == 1 else 'No'
                probability = float(proba[1])  # Probability of positive class
            
            # Confidence is distance from 0.5 for binary, max prob for multi
            confidence = abs(probability - 0.5) * 2 if len(proba) == 2 else float(max(proba))
            
            return SportyBetPrediction(
                market=market,
                market_name=MARKET_NAMES.get(market, market),
                probability=probability,
                confidence=confidence,
                prediction=prediction
            )
            
        except Exception as e:
            logger.error(f"Prediction error for {market}: {e}")
            raise
    
    def predict_all(self, home_team: str, away_team: str, 
                    league: str = '') -> SportyBetMultiPrediction:
        """
        Predict all available markets for a match.
        
        Returns predictions sorted by confidence.
        """
        from datetime import datetime
        
        predictions = {}
        
        for market in self.models:
            try:
                # Get market-specific features for proper scaling
                features = self._get_team_features(home_team, away_team, league, market)
                pred = self.predict_market(market, home_team, away_team, league, features)
                predictions[market] = pred
            except Exception as e:
                logger.warning(f"Skipping {market}: {e}")
        
        # Get top picks by confidence
        sorted_preds = sorted(predictions.values(), 
                            key=lambda x: x.confidence, reverse=True)
        best_picks = [
            {
                'market': p.market,
                'name': p.market_name,
                'prediction': p.prediction,
                'probability': round(p.probability * 100, 1),
                'confidence': round(p.confidence * 100, 1)
            }
            for p in sorted_preds[:5]
        ]
        
        return SportyBetMultiPrediction(
            home_team=home_team,
            away_team=away_team,
            predictions=predictions,
            best_picks=best_picks,
            timestamp=datetime.now().isoformat()
        )
    
    def get_accumulator_picks(self, matches: List[Dict], 
                              min_confidence: float = 0.65) -> List[Dict]:
        """
        Generate SportyBet-optimized accumulator picks.
        
        Args:
            matches: List of {home_team, away_team, league} dicts
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of picks for accumulator
        """
        picks = []
        
        for match in matches:
            home = match.get('home_team', '')
            away = match.get('away_team', '')
            league = match.get('league', '')
            
            if not home or not away:
                continue
            
            try:
                result = self.predict_all(home, away, league)
                
                # Pick the most confident prediction for this match
                for pick in result.best_picks:
                    if pick['confidence'] >= min_confidence * 100:
                        picks.append({
                            'home_team': home,
                            'away_team': away,
                            'market': pick['market'],
                            'market_name': pick['name'],
                            'selection': pick['prediction'],
                            'probability': pick['probability'],
                            'confidence': pick['confidence']
                        })
                        break  # Only one pick per match
                        
            except Exception as e:
                logger.warning(f"Skipping {home} vs {away}: {e}")
        
        return picks


# Singleton instance
_predictor: Optional[SportyBetPredictor] = None


def get_sportybet_predictor() -> SportyBetPredictor:
    """Get or create singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = SportyBetPredictor()
    return _predictor


def sportybet_predict(home_team: str, away_team: str, 
                      market: Optional[str] = None) -> Dict:
    """
    Quick prediction function.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        market: Optional specific market (None = all markets)
    
    Returns:
        Dict with prediction results
    """
    predictor = get_sportybet_predictor()
    
    if market:
        result = predictor.predict_market(market, home_team, away_team)
        return result.to_dict()
    else:
        result = predictor.predict_all(home_team, away_team)
        return result.to_dict()


def get_available_sportybet_markets() -> List[Dict]:
    """Get list of available SportyBet markets."""
    predictor = get_sportybet_predictor()
    return predictor.get_available_markets()


if __name__ == "__main__":
    # Test the predictor
    predictor = SportyBetPredictor()
    
    print(f"\n📊 Available SportyBet Markets: {len(predictor.get_available_markets())}")
    for m in predictor.get_available_markets():
        print(f"   - {m['id']}: {m['name']}")
    
    print("\n🎯 Test Prediction: Bayern vs Dortmund")
    result = predictor.predict_all('Bayern', 'Dortmund')
    
    print("\n🏆 Top 5 Picks:")
    for pick in result.best_picks:
        print(f"   {pick['name']}: {pick['prediction']} ({pick['probability']}%)")
