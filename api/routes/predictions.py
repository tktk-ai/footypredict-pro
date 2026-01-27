"""
Predictions API Routes
FastAPI routes for prediction endpoints.

Part of the complete blueprint implementation.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/match/{match_id}")
async def get_match_prediction(
    match_id: str,
    markets: str = Query("1x2,btts,over_under", description="Comma-separated markets")
) -> Dict:
    """Get predictions for a specific match."""
    try:
        from src.predictions import (
            get_match_result, get_btts, get_over_under,
            get_correct_score, get_htft, get_asian_handicap,
            calculate_confidence
        )
        
        result = {'match_id': match_id, 'predictions': {}, 'timestamp': datetime.now().isoformat()}
        
        # Mock features - in production would come from feature store
        features = {'home_xg': 1.5, 'away_xg': 1.2}
        
        market_list = [m.strip() for m in markets.split(',')]
        
        if '1x2' in market_list:
            predictor = get_match_result()
            result['predictions']['1x2'] = predictor.predict(
                "Home Team", "Away Team", features
            )
        
        if 'btts' in market_list:
            predictor = get_btts()
            result['predictions']['btts'] = predictor.predict_from_xg(
                features['home_xg'], features['away_xg']
            )
        
        if 'over_under' in market_list:
            predictor = get_over_under()
            result['predictions']['over_under'] = predictor.predict(
                features['home_xg'], features['away_xg']
            )
        
        if 'correct_score' in market_list:
            predictor = get_correct_score()
            result['predictions']['correct_score'] = predictor.predict(
                features['home_xg'], features['away_xg']
            )
        
        if 'htft' in market_list:
            predictor = get_htft()
            result['predictions']['htft'] = predictor.predict(
                features['home_xg'], features['away_xg']
            )
        
        if 'asian_handicap' in market_list:
            predictor = get_asian_handicap()
            result['predictions']['asian_handicap'] = predictor.predict(
                features['home_xg'], features['away_xg']
            )
        
        # Add confidence
        result['confidence'] = calculate_confidence(result['predictions'])
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming")
async def get_upcoming_predictions(
    league: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50)
) -> Dict:
    """Get predictions for upcoming matches."""
    # Placeholder - would fetch from scheduler/live data
    return {
        'matches': [],
        'league_filter': league,
        'limit': limit,
        'timestamp': datetime.now().isoformat()
    }


@router.get("/accuracy")
async def get_prediction_accuracy(
    days: int = Query(7, ge=1, le=90),
    market: Optional[str] = None
) -> Dict:
    """Get historical prediction accuracy."""
    try:
        from src.models.prediction_tracker import PredictionTracker
        
        tracker = PredictionTracker()
        accuracy = tracker.get_accuracy(days=days, market=market)
        
        return {
            'period_days': days,
            'market': market or 'all',
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Accuracy error: {e}")
        return {'error': str(e), 'accuracy': {}}


@router.post("/batch")
async def batch_predictions(
    matches: List[Dict]
) -> Dict:
    """Get predictions for multiple matches."""
    results = []
    
    for match in matches[:20]:  # Limit to 20
        try:
            pred = await get_match_prediction(
                match.get('match_id', ''),
                match.get('markets', '1x2')
            )
            results.append(pred)
        except Exception as e:
            results.append({'match_id': match.get('match_id'), 'error': str(e)})
    
    return {
        'predictions': results,
        'total': len(results),
        'timestamp': datetime.now().isoformat()
    }
