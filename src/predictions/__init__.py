"""Predictions Package."""

from .confidence import ConfidenceCalculator, get_calculator, calculate_confidence

# Import markets
from .markets import (
    MatchResultPredictor, get_match_result,
    CorrectScorePredictor, get_correct_score,
    OverUnderPredictor, get_over_under,
    BTTSPredictor, get_btts,
    HTFTPredictor, get_htft,
    AsianHandicapPredictor, get_asian_handicap
)

__all__ = [
    'ConfidenceCalculator', 'get_calculator', 'calculate_confidence',
    'MatchResultPredictor', 'get_match_result',
    'CorrectScorePredictor', 'get_correct_score',
    'OverUnderPredictor', 'get_over_under',
    'BTTSPredictor', 'get_btts',
    'HTFTPredictor', 'get_htft',
    'AsianHandicapPredictor', 'get_asian_handicap'
]
