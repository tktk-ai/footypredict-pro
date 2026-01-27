"""Market Predictors Package."""

from .match_result import MatchResultPredictor, get_predictor as get_match_result
from .correct_score import CorrectScorePredictor, get_predictor as get_correct_score
from .over_under import OverUnderPredictor, get_predictor as get_over_under
from .btts import BTTSPredictor, get_predictor as get_btts
from .htft import HTFTPredictor, get_predictor as get_htft
from .asian_handicap import AsianHandicapPredictor, get_predictor as get_asian_handicap

# Import existing player props if available
try:
    from src.player_props import PlayerPropsPredictor
except ImportError:
    PlayerPropsPredictor = None

__all__ = [
    'MatchResultPredictor', 'get_match_result',
    'CorrectScorePredictor', 'get_correct_score',
    'OverUnderPredictor', 'get_over_under',
    'BTTSPredictor', 'get_btts',
    'HTFTPredictor', 'get_htft',
    'AsianHandicapPredictor', 'get_asian_handicap',
    'PlayerPropsPredictor'
]
