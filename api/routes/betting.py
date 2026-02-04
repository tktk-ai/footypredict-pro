"""
Betting API Routes
FastAPI routes for betting strategy endpoints.

Part of the complete blueprint implementation.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/betting", tags=["betting"])


@router.post("/value")
async def find_value_bets(
    predictions: Dict,
    odds: Dict
) -> Dict:
    """Find value bets given predictions and odds."""
    try:
        from src.betting import find_value_bets
        
        value_bets = find_value_bets(predictions, odds)
        
        return {
            'value_bets': value_bets,
            'count': len(value_bets),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Value detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stake")
async def calculate_stake(
    probability: float,
    odds: float,
    bankroll: float = 1000
) -> Dict:
    """Calculate optimal stake using Kelly criterion."""
    try:
        from src.betting import calculate_stake
        
        stake_info = calculate_stake(probability, odds, bankroll)
        
        return {
            'stake_info': stake_info,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stake calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio")
async def optimize_portfolio(
    bets: List[Dict],
    bankroll: float = 1000
) -> Dict:
    """Optimize betting portfolio allocation."""
    try:
        from src.betting import get_optimizer
        
        optimizer = get_optimizer()
        portfolio = optimizer.optimize_mean_variance(bets, bankroll)
        
        return {
            'portfolio': portfolio,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bankroll")
async def get_bankroll_status() -> Dict:
    """Get current bankroll status."""
    try:
        from src.betting import get_bankroll_manager
        
        manager = get_bankroll_manager()
        stats = manager.get_stats()
        drawdown = manager.get_drawdown()
        should_stop = manager.should_stop()
        
        return {
            'stats': stats,
            'drawdown': drawdown,
            'should_stop': should_stop,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Bankroll status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-check")
async def check_bet_risk(
    bet: Dict,
    bankroll: float = 1000
) -> Dict:
    """Check if a bet passes risk controls."""
    try:
        from src.betting import get_risk_manager
        
        manager = get_risk_manager()
        check = manager.check_bet(bet, bankroll)
        
        return {
            'risk_check': check,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Risk check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place")
async def place_bet(
    stake: float,
    odds: float,
    match_id: str,
    market: str,
    outcome: str
) -> Dict:
    """Record placing a bet."""
    try:
        from src.betting import get_bankroll_manager, get_risk_manager
        
        bankroll_manager = get_bankroll_manager()
        risk_manager = get_risk_manager()
        
        bet = {
            'stake': stake,
            'odds': odds,
            'match_id': match_id,
            'market': market,
            'outcome': outcome
        }
        
        # Risk check
        risk_check = risk_manager.check_bet(bet, bankroll_manager.current_bankroll)
        
        if not risk_check['approved']:
            return {
                'success': False,
                'reason': 'Risk check failed',
                'issues': risk_check['issues']
            }
        
        # Place bet
        result = bankroll_manager.place_bet(stake, odds, f"{match_id}:{market}:{outcome}")
        
        if result['success']:
            risk_manager.add_bet(bet)
        
        return {
            'bet_result': result,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Place bet error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/settle")
async def settle_bet(
    stake: float,
    odds: float,
    won: bool,
    match_id: str
) -> Dict:
    """Settle a bet result."""
    try:
        from src.betting import get_bankroll_manager
        
        manager = get_bankroll_manager()
        result = manager.settle_bet(stake, odds, won, match_id)
        
        return {
            'settlement': result,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Settle bet error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
