"""
Blueprint Module Integration
============================
Unified integration layer connecting all blueprint modules to the main app.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ML MODELS INTEGRATION
# ============================================================================

def get_ml_models() -> Dict[str, Any]:
    """Load all ML models from blueprint."""
    models = {}
    
    # XGBoost
    try:
        from src.models.machine_learning.xgboost_model import XGBoostPredictor, get_model as get_xgb
        models['xgboost'] = get_xgb()
        logger.info("✅ XGBoost model loaded")
    except ImportError as e:
        logger.warning(f"XGBoost not available: {e}")
    
    # LightGBM
    try:
        from src.models.machine_learning.lightgbm_model import LightGBMPredictor, get_model as get_lgb
        models['lightgbm'] = get_lgb()
        logger.info("✅ LightGBM model loaded")
    except ImportError as e:
        logger.warning(f"LightGBM not available: {e}")
    
    # CatBoost
    try:
        from src.models.machine_learning.catboost_model import CatBoostPredictor, get_model as get_cat
        models['catboost'] = get_cat()
        logger.info("✅ CatBoost model loaded")
    except ImportError as e:
        logger.warning(f"CatBoost not available: {e}")
    
    return models


def get_deep_learning_models() -> Dict[str, Any]:
    """Load deep learning models from blueprint."""
    models = {}
    
    # CNN-BiLSTM-Attention
    try:
        from src.models.deep_learning.cnn_bilstm_attention import CNNBiLSTMAttention
        models['cnn_bilstm'] = CNNBiLSTMAttention
        logger.info("✅ CNN-BiLSTM-Attention loaded")
    except ImportError as e:
        logger.warning(f"CNN-BiLSTM not available: {e}")
    
    # Graph Transformer
    try:
        from src.models.deep_learning.graph_transformer import GraphTransformer
        models['graph_transformer'] = GraphTransformer
        logger.info("✅ Graph Transformer loaded")
    except ImportError as e:
        logger.warning(f"Graph Transformer not available: {e}")
    
    # Transformer-LSTM
    try:
        from src.models.deep_learning.transformer_lstm import TransformerLSTM
        models['transformer_lstm'] = TransformerLSTM
        logger.info("✅ Transformer-LSTM loaded")
    except ImportError as e:
        logger.warning(f"Transformer-LSTM not available: {e}")
    
    return models


# ============================================================================
# ENSEMBLE INTEGRATION
# ============================================================================

def get_ensemble_components() -> Dict[str, Any]:
    """Load ensemble components from blueprint."""
    components = {}
    
    # Model Combiner
    try:
        from src.models.ensemble.model_combiner import ModelCombiner, get_combiner
        components['combiner'] = get_combiner()
        logger.info("✅ Model Combiner loaded")
    except ImportError as e:
        logger.warning(f"Model Combiner not available: {e}")
    
    # Meta Learner
    try:
        from src.models.ensemble.meta_learner import MetaLearner, get_learner
        components['meta_learner'] = get_learner()
        logger.info("✅ Meta Learner loaded")
    except ImportError as e:
        logger.warning(f"Meta Learner not available: {e}")
    
    return components


# ============================================================================
# MARKET PREDICTORS INTEGRATION
# ============================================================================

def get_market_predictors() -> Dict[str, Any]:
    """Load all market predictors from blueprint."""
    predictors = {}
    
    # Match Result (1X2)
    try:
        from src.predictions.markets.match_result import MatchResultPredictor, get_predictor
        predictors['match_result'] = get_predictor()
        logger.info("✅ Match Result Predictor loaded")
    except ImportError as e:
        logger.warning(f"Match Result not available: {e}")
    
    # Over/Under
    try:
        from src.predictions.markets.over_under import OverUnderPredictor, get_predictor as get_ou
        predictors['over_under'] = get_ou()
        logger.info("✅ Over/Under Predictor loaded")
    except ImportError as e:
        logger.warning(f"Over/Under not available: {e}")
    
    # BTTS
    try:
        from src.predictions.markets.btts import BTTSPredictor, get_predictor as get_btts
        predictors['btts'] = get_btts()
        logger.info("✅ BTTS Predictor loaded")
    except ImportError as e:
        logger.warning(f"BTTS not available: {e}")
    
    # Correct Score
    try:
        from src.predictions.markets.correct_score import CorrectScorePredictor, get_predictor as get_cs
        predictors['correct_score'] = get_cs()
        logger.info("✅ Correct Score Predictor loaded")
    except ImportError as e:
        logger.warning(f"Correct Score not available: {e}")
    
    # Asian Handicap
    try:
        from src.predictions.markets.asian_handicap import AsianHandicapPredictor, get_predictor as get_ah
        predictors['asian_handicap'] = get_ah()
        logger.info("✅ Asian Handicap Predictor loaded")
    except ImportError as e:
        logger.warning(f"Asian Handicap not available: {e}")
    
    # HTFT
    try:
        from src.predictions.markets.htft import HTFTPredictor, get_predictor as get_htft
        predictors['htft'] = get_htft()
        logger.info("✅ HTFT Predictor loaded")
    except ImportError as e:
        logger.warning(f"HTFT not available: {e}")
    
    return predictors


# ============================================================================
# BETTING MODULES INTEGRATION
# ============================================================================

def get_betting_modules() -> Dict[str, Any]:
    """Load betting strategy modules from blueprint."""
    modules = {}
    
    # Value Detection
    try:
        from src.betting.value_detection import ValueDetector, get_detector
        modules['value_detector'] = get_detector()
        logger.info("✅ Value Detection loaded")
    except ImportError as e:
        logger.warning(f"Value Detection not available: {e}")
    
    # Portfolio Optimization
    try:
        from src.betting.portfolio_optimization import PortfolioOptimizer, get_optimizer
        modules['portfolio'] = get_optimizer()
        logger.info("✅ Portfolio Optimization loaded")
    except ImportError as e:
        logger.warning(f"Portfolio Optimization not available: {e}")
    
    # Bankroll Management
    try:
        from src.betting.bankroll_management import BankrollManager, get_manager
        modules['bankroll'] = get_manager()
        logger.info("✅ Bankroll Management loaded")
    except ImportError as e:
        logger.warning(f"Bankroll Management not available: {e}")
    
    # Kelly Criterion
    try:
        from src.betting.kelly_criterion import KellyCriterion, get_kelly
        modules['kelly'] = get_kelly()
        logger.info("✅ Kelly Criterion loaded")
    except ImportError as e:
        logger.warning(f"Kelly Criterion not available: {e}")
    
    # Risk Management
    try:
        from src.betting.risk_management import RiskManager, get_risk_manager
        modules['risk'] = get_risk_manager()
        logger.info("✅ Risk Management loaded")
    except ImportError as e:
        logger.warning(f"Risk Management not available: {e}")
    
    return modules


# ============================================================================
# LIVE BETTING INTEGRATION
# ============================================================================

def get_live_modules() -> Dict[str, Any]:
    """Load live betting modules from blueprint."""
    modules = {}
    
    # Odds Integration
    try:
        from src.live.odds_integration import OddsIntegrator, get_integrator
        modules['odds'] = get_integrator()
        logger.info("✅ Odds Integration loaded")
    except ImportError as e:
        logger.warning(f"Odds Integration not available: {e}")
    
    # Stream Processor
    try:
        from src.live.stream_processor import StreamProcessor, get_processor
        modules['stream'] = get_processor()
        logger.info("✅ Stream Processor loaded")
    except ImportError as e:
        logger.warning(f"Stream Processor not available: {e}")
    
    # Arbitrage Detector
    try:
        from src.live.arbitrage_detector import ArbitrageDetector, get_detector as get_arb
        modules['arbitrage'] = get_arb()
        logger.info("✅ Arbitrage Detector loaded")
    except ImportError as e:
        logger.warning(f"Arbitrage Detector not available: {e}")
    
    return modules


# ============================================================================
# EXPLAINABILITY INTEGRATION
# ============================================================================

def get_explainability_modules() -> Dict[str, Any]:
    """Load explainability modules from blueprint."""
    modules = {}
    
    # SHAP Explainer
    try:
        from src.explainability.shap_explainer import SHAPExplainer, get_explainer
        modules['shap'] = get_explainer()
        logger.info("✅ SHAP Explainer loaded")
    except ImportError as e:
        logger.warning(f"SHAP Explainer not available: {e}")
    
    # LIME Explainer
    try:
        from src.explainability.lime_explainer import LIMEExplainer, get_explainer as get_lime
        modules['lime'] = get_lime()
        logger.info("✅ LIME Explainer loaded")
    except ImportError as e:
        logger.warning(f"LIME Explainer not available: {e}")
    
    return modules


# ============================================================================
# DATA INTEGRATION
# ============================================================================

def get_data_modules() -> Dict[str, Any]:
    """Load data collection and processing modules."""
    modules = {}
    
    # Data Integration Manager
    try:
        from src.data.integration import get_data_manager
        modules['data_manager'] = get_data_manager()
        logger.info("✅ Data Manager loaded")
    except ImportError as e:
        logger.warning(f"Data Manager not available: {e}")
    
    # Feature Store
    try:
        from src.features.selection.store import get_store
        modules['feature_store'] = get_store()
        logger.info("✅ Feature Store loaded")
    except ImportError as e:
        logger.warning(f"Feature Store not available: {e}")
    
    # Evaluation Metrics
    try:
        from src.evaluation.metrics import get_metrics
        modules['metrics'] = get_metrics()
        logger.info("✅ Evaluation Metrics loaded")
    except ImportError as e:
        logger.warning(f"Evaluation Metrics not available: {e}")
    
    return modules


# ============================================================================
# UNIFIED BLUEPRINT MANAGER
# ============================================================================

class BlueprintManager:
    """
    Unified manager for all blueprint modules.
    
    Provides single point of access to:
    - ML Models (XGBoost, LightGBM, CatBoost)
    - Deep Learning (CNN-BiLSTM, Graph Transformer)
    - Ensemble (Combiner, Meta Learner)
    - Market Predictors (1X2, O/U, BTTS, CS, AH, HTFT)
    - Betting Strategy (Value, Portfolio, Bankroll, Kelly, Risk)
    - Live Betting (Odds, Stream, Arbitrage)
    - Explainability (SHAP, LIME)
    - Data (Manager, Feature Store, Metrics)
    """
    
    def __init__(self):
        self._ml_models = None
        self._dl_models = None
        self._ensemble = None
        self._markets = None
        self._betting = None
        self._live = None
        self._explain = None
        self._data = None
        
    @property
    def ml_models(self) -> Dict[str, Any]:
        if self._ml_models is None:
            self._ml_models = get_ml_models()
        return self._ml_models
    
    @property
    def deep_learning(self) -> Dict[str, Any]:
        if self._dl_models is None:
            self._dl_models = get_deep_learning_models()
        return self._dl_models
    
    @property
    def ensemble(self) -> Dict[str, Any]:
        if self._ensemble is None:
            self._ensemble = get_ensemble_components()
        return self._ensemble
    
    @property
    def markets(self) -> Dict[str, Any]:
        if self._markets is None:
            self._markets = get_market_predictors()
        return self._markets
    
    @property
    def betting(self) -> Dict[str, Any]:
        if self._betting is None:
            self._betting = get_betting_modules()
        return self._betting
    
    @property
    def live(self) -> Dict[str, Any]:
        if self._live is None:
            self._live = get_live_modules()
        return self._live
    
    @property
    def explainability(self) -> Dict[str, Any]:
        if self._explain is None:
            self._explain = get_explainability_modules()
        return self._explain
    
    @property
    def data(self) -> Dict[str, Any]:
        if self._data is None:
            self._data = get_data_modules()
        return self._data
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all modules."""
        return {
            'ml_models': list(self.ml_models.keys()),
            'deep_learning': list(self.deep_learning.keys()),
            'ensemble': list(self.ensemble.keys()),
            'markets': list(self.markets.keys()),
            'betting': list(self.betting.keys()),
            'live': list(self.live.keys()),
            'explainability': list(self.explainability.keys()),
            'data': list(self.data.keys()),
            'total_modules': sum([
                len(self.ml_models),
                len(self.deep_learning),
                len(self.ensemble),
                len(self.markets),
                len(self.betting),
                len(self.live),
                len(self.explainability),
                len(self.data),
            ])
        }
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        home_xg: float = 1.5,
        away_xg: float = 1.2,
        odds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive prediction using all available modules.
        """
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'markets': {},
            'betting': {},
            'confidence': 0.0
        }
        
        # Market predictions
        if 'match_result' in self.markets:
            try:
                mr = self.markets['match_result'].predict(home_xg, away_xg)
                result['markets']['1x2'] = mr
            except Exception as e:
                logger.error(f"Match result prediction error: {e}")
        
        if 'over_under' in self.markets:
            try:
                ou = self.markets['over_under'].predict(home_xg, away_xg)
                result['markets']['over_under'] = ou
            except Exception as e:
                logger.error(f"Over/Under prediction error: {e}")
        
        if 'btts' in self.markets:
            try:
                btts = self.markets['btts'].predict(home_xg, away_xg)
                result['markets']['btts'] = btts
            except Exception as e:
                logger.error(f"BTTS prediction error: {e}")
        
        if 'correct_score' in self.markets:
            try:
                cs = self.markets['correct_score'].predict(home_xg, away_xg)
                result['markets']['correct_score'] = cs
            except Exception as e:
                logger.error(f"Correct score prediction error: {e}")
        
        if 'asian_handicap' in self.markets:
            try:
                ah = self.markets['asian_handicap'].predict(home_xg, away_xg)
                result['markets']['asian_handicap'] = ah
            except Exception as e:
                logger.error(f"Asian Handicap prediction error: {e}")
        
        # Betting analysis
        if odds and 'value_detector' in self.betting:
            try:
                value = self.betting['value_detector'].find_value(
                    predicted_prob=result.get('markets', {}).get('1x2', {}).get('home_win', 0.5),
                    odds=odds.get('home', 2.0)
                )
                result['betting']['value'] = value
            except Exception as e:
                logger.error(f"Value detection error: {e}")
        
        return result
    
    def get_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Get upcoming matches from data sources."""
        if 'data_manager' in self.data:
            try:
                fixtures = self.data['data_manager'].fetch_upcoming_fixtures(days_ahead=days)
                if fixtures is not None and len(fixtures) > 0:
                    return fixtures.to_dict('records')
            except Exception as e:
                logger.error(f"Error fetching fixtures: {e}")
        return []


# Global instance
_blueprint: Optional[BlueprintManager] = None


def get_blueprint() -> BlueprintManager:
    """Get or create the blueprint manager."""
    global _blueprint
    if _blueprint is None:
        _blueprint = BlueprintManager()
    return _blueprint


def get_blueprint_status() -> Dict[str, Any]:
    """Get status of all blueprint modules."""
    return get_blueprint().get_status()


def predict_with_blueprint(
    home_team: str,
    away_team: str,
    **kwargs
) -> Dict[str, Any]:
    """Make predictions using all blueprint modules."""
    return get_blueprint().predict_match(home_team, away_team, **kwargs)
