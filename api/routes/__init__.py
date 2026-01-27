"""API Routes Package."""

from .predictions import router as predictions_router
from .live import router as live_router
from .betting import router as betting_router
from .analytics import router as analytics_router

__all__ = [
    'predictions_router',
    'live_router',
    'betting_router',
    'analytics_router'
]
