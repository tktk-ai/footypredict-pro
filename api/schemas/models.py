"""
Pydantic Schemas for API
Request and response models.

Part of the complete blueprint implementation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


# Request Models

class MatchPredictionRequest(BaseModel):
    """Request for match prediction."""
    home_team: str
    away_team: str
    league: Optional[str] = None
    match_date: Optional[str] = None
    markets: List[str] = Field(default=['1x2', 'btts', 'over_under'])


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    matches: List[MatchPredictionRequest]


class ValueBetRequest(BaseModel):
    """Request for value bet detection."""
    predictions: Dict
    odds: Dict
    min_edge: float = Field(default=0.05, ge=0, le=0.5)


class StakeCalculationRequest(BaseModel):
    """Request for stake calculation."""
    probability: float = Field(ge=0, le=1)
    odds: float = Field(gt=1)
    bankroll: float = Field(default=1000, gt=0)


class PortfolioRequest(BaseModel):
    """Request for portfolio optimization."""
    bets: List[Dict]
    bankroll: float = Field(default=1000, gt=0)
    strategy: str = Field(default='mean_variance')


class PlaceBetRequest(BaseModel):
    """Request to place a bet."""
    stake: float = Field(gt=0)
    odds: float = Field(gt=1)
    match_id: str
    market: str
    outcome: str


class SettleBetRequest(BaseModel):
    """Request to settle a bet."""
    stake: float = Field(gt=0)
    odds: float = Field(gt=1)
    won: bool
    match_id: str


class BacktestRequest(BaseModel):
    """Request for backtest."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategy: str = Field(default='kelly')
    initial_bankroll: float = Field(default=1000, gt=0)


# Response Models

class PredictionResponse(BaseModel):
    """Prediction response."""
    match_id: str
    predictions: Dict
    confidence: Optional[Dict] = None
    timestamp: str


class ValueBetResponse(BaseModel):
    """Value bet response."""
    value_bets: List[Dict]
    count: int
    timestamp: str


class StakeResponse(BaseModel):
    """Stake calculation response."""
    stake_info: Dict
    timestamp: str


class BankrollStatusResponse(BaseModel):
    """Bankroll status response."""
    stats: Dict
    drawdown: Dict
    should_stop: Dict
    timestamp: str


class PerformanceResponse(BaseModel):
    """Performance metrics response."""
    period_days: int
    market: str
    metrics: Dict
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Live Match Models

class LiveMatchEvent(BaseModel):
    """Live match event."""
    type: str
    match_id: str
    minute: Optional[int] = None
    team: Optional[str] = None
    player: Optional[str] = None
    data: Optional[Dict] = None


class LiveOddsUpdate(BaseModel):
    """Live odds update."""
    match_id: str
    bookmaker: str
    market: str
    odds: Dict
