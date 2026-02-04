"""Live Betting Module Package."""

from .arbitrage_detector import ArbitrageDetector

# Import new modules
try:
    from .odds_integration import OddsIntegration, get_integration
except ImportError:
    OddsIntegration = None
    get_integration = None

try:
    from .stream_processor import (
        StreamProcessor, LiveMatchProcessor,
        get_processor, get_match_processor
    )
except ImportError:
    StreamProcessor = None
    LiveMatchProcessor = None
    get_processor = None
    get_match_processor = None

# Import existing live predictor if available
try:
    from src.live_predictor import LivePredictor
except ImportError:
    LivePredictor = None

__all__ = [
    'ArbitrageDetector',
    'OddsIntegration', 'get_integration',
    'StreamProcessor', 'LiveMatchProcessor',
    'get_processor', 'get_match_processor',
    'LivePredictor'
]
