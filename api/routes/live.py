"""
Live Betting API Routes
FastAPI routes for live betting endpoints.

Part of the complete blueprint implementation.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"])


@router.get("/matches")
async def get_live_matches() -> Dict:
    """Get all currently live matches."""
    try:
        from src.live import get_match_processor
        
        processor = get_match_processor()
        
        # Would fetch from live data source
        return {
            'matches': [],
            'count': 0,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Live matches error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/match/{match_id}")
async def get_live_match(match_id: str) -> Dict:
    """Get live match state and predictions."""
    try:
        from src.live import get_match_processor
        
        processor = get_match_processor()
        state = processor.get_match_state(match_id)
        
        return {
            'match_id': match_id,
            'state': state,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Live match error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/odds/{match_id}")
async def get_live_odds(match_id: str) -> Dict:
    """Get live odds for a match."""
    try:
        from src.live import get_integration
        
        integration = get_integration()
        best_odds = integration.get_best_odds(match_id)
        consensus = integration.get_consensus_probability(match_id)
        movement = integration.detect_odds_movement(match_id)
        
        return {
            'match_id': match_id,
            'best_odds': best_odds,
            'consensus': consensus,
            'movement': movement,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Live odds error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/arbitrage")
async def find_arbitrage() -> Dict:
    """Find arbitrage opportunities."""
    try:
        from src.live import ArbitrageDetector
        
        detector = ArbitrageDetector()
        opportunities = detector.find_opportunities()
        
        return {
            'opportunities': opportunities,
            'count': len(opportunities),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Arbitrage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/event")
async def process_event(event: Dict) -> Dict:
    """Process a live match event."""
    try:
        from src.live import get_match_processor
        
        processor = get_match_processor()
        event_type = event.get('type', 'unknown')
        
        processor.process_event(event_type, event)
        
        return {
            'processed': True,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Event processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{match_id}")
async def get_event_stream(
    match_id: str,
    seconds: int = Query(60, ge=10, le=300)
) -> Dict:
    """Get recent event stream for a match."""
    try:
        from src.live import get_match_processor
        
        processor = get_match_processor()
        events = processor.get_window(match_id, seconds)
        
        return {
            'match_id': match_id,
            'events': events,
            'count': len(events),
            'window_seconds': seconds,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
