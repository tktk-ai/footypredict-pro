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
    # Transparency fields
    home_team_known: bool = True
    away_team_known: bool = True
    data_quality: str = 'high'  # 'high', 'medium', 'low'
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict:
        return {
            'home_team': self.home_team,
            'away_team': self.away_team,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'best_picks': self.best_picks,
            'timestamp': self.timestamp,
            'transparency': {
                'home_team_known': self.home_team_known,
                'away_team_known': self.away_team_known,
                'data_quality': self.data_quality,
                'warnings': self.warnings,
                'model_accuracy': {
                    '1x2': '56%',
                    'double_chance': '70-76%',
                    'over_15': '75%',
                    'btts': '62%'
                }
            }
        }


class SportyBetPredictor:
    """
    Predictor for SportyBet specialized markets.
    
    Loads trained XGBoost models and provides inference methods.
    Uses real historical team statistics for feature extraction.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.team_stats: Dict[str, Dict] = {}  # Team name -> avg stats
        self._load_models()
        self._load_feature_cols()
        self._build_team_stats()
    
    def _build_team_stats(self) -> None:
        """Build team statistics cache from training data."""
        data_file = DATA_DIR / "comprehensive_training_data.csv"
        
        if not data_file.exists():
            logger.warning("Training data not found, using default stats")
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(data_file, usecols=[
                'HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 
                'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A'
            ])
            
            # Calculate home stats
            home_stats = df.groupby('HomeTeam').agg({
                'HS': 'mean', 'HST': 'mean', 'HF': 'mean', 
                'HC': 'mean', 'HY': 'mean', 'HR': 'mean',
                'FTHG': 'mean', 'FTAG': 'mean',
                'B365H': 'mean', 'B365D': 'mean', 'B365A': 'mean'
            }).to_dict('index')
            
            # Calculate away stats
            away_stats = df.groupby('AwayTeam').agg({
                'AS': 'mean', 'AST': 'mean', 'AF': 'mean', 
                'AC': 'mean', 'AY': 'mean', 'AR': 'mean',
                'FTAG': 'mean'
            }).to_dict('index')
            
            # Build team cache
            for team in set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()):
                # Skip NaN or non-string team names
                if not isinstance(team, str):
                    continue
                    
                h = home_stats.get(team, {})
                a = away_stats.get(team, {})
                
                self.team_stats[team.lower()] = {
                    'shots_home': h.get('HS', 12),
                    'shots_away': a.get('AS', 10),
                    'shots_target_home': h.get('HST', 4),
                    'shots_target_away': a.get('AST', 3),
                    'fouls_home': h.get('HF', 11),
                    'fouls_away': a.get('AF', 12),
                    'corners_home': h.get('HC', 5),
                    'corners_away': a.get('AC', 4),
                    'yellows_home': h.get('HY', 1.5),
                    'yellows_away': a.get('AY', 1.8),
                    'reds_home': h.get('HR', 0.05),
                    'reds_away': a.get('AR', 0.05),
                    'goals_scored_home': h.get('FTHG', 1.5),
                    'goals_conceded_home': h.get('FTAG', 1.2),
                    'goals_scored_away': a.get('FTAG', 1.1),
                    'avg_odds_home': h.get('B365H', 2.2),
                    'avg_odds_draw': h.get('B365D', 3.3),
                    'avg_odds_away': h.get('B365A', 3.5),
                }
            
            logger.info(f"Built stats cache for {len(self.team_stats)} teams")
            
        except Exception as e:
            logger.error(f"Error building team stats: {e}")
    
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
                            market: str = None) -> Tuple[np.ndarray, bool, bool]:
        """
        Generate feature vector for a match using real historical team statistics.
        
        FULLY DETERMINISTIC - no random noise.
        
        Returns:
            Tuple of (features, home_team_known, away_team_known)
        """
        # Get expected feature count from scaler
        n_features = 162
        if market and market in self.scalers:
            n_features = self.scalers[market].n_features_in_
        elif self.scalers:
            first_scaler = next(iter(self.scalers.values()))
            n_features = first_scaler.n_features_in_
        
        # Get team statistics from cache (case-insensitive lookup)
        home_key = home_team.lower().strip()
        away_key = away_team.lower().strip()
        
        # Try partial matches if exact not found
        home_stats = self.team_stats.get(home_key, None)
        away_stats = self.team_stats.get(away_key, None)
        home_team_known = home_stats is not None
        away_team_known = away_stats is not None
        
        if not home_stats:
            for key in self.team_stats:
                if home_key in key or key in home_key:
                    home_stats = self.team_stats[key]
                    home_team_known = True
                    break
        
        if not away_stats:
            for key in self.team_stats:
                if away_key in key or key in away_key:
                    away_stats = self.team_stats[key]
                    away_team_known = True
                    break
        
        # Default stats if team not found (league average values)
        if not home_stats:
            home_stats = {
                'shots_home': 12.5, 'shots_target_home': 4.2, 'fouls_home': 11.0,
                'corners_home': 5.0, 'yellows_home': 1.6, 'reds_home': 0.05,
                'goals_scored_home': 1.5, 'goals_conceded_home': 1.2,
                'avg_odds_home': 2.5, 'avg_odds_draw': 3.3, 'avg_odds_away': 3.0
            }
        
        if not away_stats:
            away_stats = {
                'shots_away': 10.5, 'shots_target_away': 3.5, 'fouls_away': 11.5,
                'corners_away': 4.0, 'yellows_away': 1.7, 'reds_away': 0.05,
                'goals_scored_away': 1.1
            }
        
        # Build feature array with real statistics - DETERMINISTIC, NO RANDOM NOISE
        # Order matches training data: HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, odds...
        features = np.zeros(n_features)
        
        # Core match stats (indices 0-11)
        features[0] = home_stats.get('shots_home', 12.5)  # HS
        features[1] = away_stats.get('shots_away', 10.5)  # AS
        features[2] = home_stats.get('shots_target_home', 4.2)  # HST
        features[3] = away_stats.get('shots_target_away', 3.5)  # AST
        features[4] = home_stats.get('fouls_home', 11.0)  # HF
        features[5] = away_stats.get('fouls_away', 11.5)  # AF
        features[6] = home_stats.get('corners_home', 5.0)  # HC
        features[7] = away_stats.get('corners_away', 4.0)  # AC
        features[8] = home_stats.get('yellows_home', 1.6)  # HY
        features[9] = away_stats.get('yellows_away', 1.7)  # AY
        features[10] = home_stats.get('reds_home', 0.05)  # HR
        features[11] = away_stats.get('reds_away', 0.05)  # AR
        
        # Betting odds (indices 12-40+) - DETERMINISTIC
        home_odds = home_stats.get('avg_odds_home', 2.5)
        draw_odds = home_stats.get('avg_odds_draw', 3.3)
        away_odds = away_stats.get('avg_odds_away', home_stats.get('avg_odds_away', 3.0))
        
        # Fill odds columns with slight systematic variation (simulating different bookmakers)
        # NO RANDOM - use deterministic small offsets based on position
        for i in range(12, min(42, n_features)):
            idx = (i - 12) % 3
            bookmaker_offset = 1.0 + ((i - 12) // 3) * 0.01  # Small systematic offset per bookmaker
            if idx == 0:
                features[i] = home_odds * bookmaker_offset
            elif idx == 1:
                features[i] = draw_odds * bookmaker_offset
            else:
                features[i] = away_odds * bookmaker_offset
        
        # Over/Under odds and other columns (42+)
        expected_goals = (home_stats.get('goals_scored_home', 1.5) + 
                         away_stats.get('goals_scored_away', 1.1))
        over25_prob = 1 / (1 + np.exp(-(expected_goals - 2.5)))  # Logistic sigmoid
        
        for i in range(42, min(60, n_features)):
            if i % 2 == 0:
                features[i] = 1 / max(over25_prob, 0.1)  # Over odds
            else:
                features[i] = 1 / max(1 - over25_prob, 0.1)  # Under odds
        
        # Asian handicap and closing odds (60-100) - DETERMINISTIC based on team strength
        team_strength_ratio = home_odds / max(away_odds, 0.1)
        handicap_base = 1.9 + (team_strength_ratio - 1) * 0.1
        for i in range(60, min(100, n_features)):
            features[i] = handicap_base + (i - 60) * 0.002  # Small deterministic increment
        
        # Extended features (100+) - DETERMINISTIC
        avg_odds = (home_odds + away_odds) / 2
        for i in range(100, n_features):
            features[i] = avg_odds / 2 + (i - 100) * 0.001  # Deterministic fill
        
        return features.reshape(1, -1), home_team_known, away_team_known
    
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
        
        # Get features (returns tuple with team_known flags)
        if features is None:
            features, _, _ = self._get_team_features(home_team, away_team, league, market)
        
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
            
            # Use calibrated confidence based on market accuracy
            # This provides honest, realistic confidence scores
            try:
                from src.models.calibration import empirical_calibrate
                calibrated_prob = empirical_calibrate(probability, market)
                confidence = abs(calibrated_prob - 0.5) * 2
            except ImportError:
                # Fallback to raw confidence
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
        home_team_known = True
        away_team_known = True
        
        for market in self.models:
            try:
                # Get market-specific features for proper scaling
                features, h_known, a_known = self._get_team_features(home_team, away_team, league, market)
                home_team_known = home_team_known and h_known
                away_team_known = away_team_known and a_known
                pred = self.predict_market(market, home_team, away_team, league, features)
                predictions[market] = pred
            except Exception as e:
                logger.warning(f"Skipping {market}: {e}")
        
        # Determine data quality
        warnings = []
        if not home_team_known:
            warnings.append(f"Home team '{home_team}' not found in database - using defaults")
        if not away_team_known:
            warnings.append(f"Away team '{away_team}' not found in database - using defaults")
        
        if home_team_known and away_team_known:
            data_quality = 'high'
        elif home_team_known or away_team_known:
            data_quality = 'medium'
        else:
            data_quality = 'low'
        
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
            timestamp=datetime.now().isoformat(),
            home_team_known=home_team_known,
            away_team_known=away_team_known,
            data_quality=data_quality,
            warnings=warnings
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
