"""
Complete Production FastAPI for Football Prediction System V3.0

Features:
- REST endpoints for all predictions
- WebSocket for real-time updates
- Monte Carlo simulation
- Value betting detection
- Player props
- RL strategy recommendations
"""

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.simulation.monte_carlo import MonteCarloSimulator, run_monte_carlo
from src.predictions.player_props import PlayerPropsPredictor, predict_player_goals
from src.betting.reinforcement_learning import BettingEnvironment, DQNBettingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with all features
app = FastAPI(
    title="Ultimate Football Prediction API V3.0",
    description="""
    Complete football prediction system with:
    - Monte Carlo simulation (100k iterations)
    - Deep learning predictions
    - Reinforcement learning betting strategy
    - Player props prediction
    - Real-time odds integration
    - Value betting detection
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    league: Optional[str] = None
    include_simulation: bool = True
    simulation_count: int = 100000
    odds: Optional[Dict[str, float]] = None


class MonteCarloRequest(BaseModel):
    home_xg: float
    away_xg: float
    n_simulations: int = 100000
    include_htft: bool = False


class PlayerPropsRequest(BaseModel):
    player_id: str
    player_name: Optional[str] = None
    position: str = "FW"
    goals_avg: float = 0.3
    is_home: bool = True
    opponent_strength: float = 1.0
    props: Optional[Dict[str, float]] = None


class SeasonSimulationRequest(BaseModel):
    league: str
    remaining_fixtures: List[Dict]
    team_strengths: Dict[str, Dict]
    n_simulations: int = 10000


class ValueBetRequest(BaseModel):
    predictions: Dict[str, float]
    odds: Dict[str, float]
    min_edge: float = 0.03


# Global services
simulator = MonteCarloSimulator(n_simulations=100000)
player_predictor = PlayerPropsPredictor()
rl_agent = DQNBettingAgent()


# Health check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "features": [
            "monte_carlo_simulation",
            "player_props",
            "rl_betting",
            "value_detection"
        ]
    }


# Main prediction endpoint
@app.post("/api/v3/predict")
async def predict_match(request: MatchPredictionRequest):
    """
    Generate comprehensive match prediction using Monte Carlo simulation.
    
    Includes:
    - 1X2 probabilities
    - Correct score probabilities
    - Over/Under probabilities
    - BTTS probabilities
    - Expected goals
    - Asian Handicap probabilities
    """
    try:
        # Get xG values
        home_xg = request.home_xg or 1.5
        away_xg = request.away_xg or 1.2
        
        # Run simulation
        if request.include_simulation:
            result = simulator.simulate_match(
                home_xg=home_xg,
                away_xg=away_xg,
                home_xg_std=0.3,
                away_xg_std=0.3
            )
            
            simulation = result.to_dict()
        else:
            simulation = None
        
        # Find value bets if odds provided
        value_bets = []
        if request.odds and simulation:
            for market, prob in [
                ('home_win', simulation['1x2']['home_win']),
                ('draw', simulation['1x2']['draw']),
                ('away_win', simulation['1x2']['away_win']),
                ('over_2.5', simulation['over_under']['over_2.5']),
                ('btts_yes', simulation['btts']['yes'])
            ]:
                if market in request.odds:
                    implied = 1 / request.odds[market]
                    edge = prob - implied
                    if edge > 0.03:
                        value_bets.append({
                            'market': market,
                            'probability': round(prob, 4),
                            'odds': request.odds[market],
                            'edge': round(edge, 4),
                            'expected_value': round(edge * request.odds[market], 4)
                        })
        
        return {
            'success': True,
            'match': f"{request.home_team} vs {request.away_team}",
            'timestamp': datetime.now().isoformat(),
            'simulation': simulation,
            'value_bets': value_bets if value_bets else None,
            'methodology': f'Monte Carlo simulation with {request.simulation_count:,} iterations'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Monte Carlo simulation endpoint
@app.post("/api/v3/simulate")
async def monte_carlo_simulate(request: MonteCarloRequest):
    """
    Run Monte Carlo simulation for a match.
    
    Returns detailed probabilities for all markets.
    """
    try:
        result = run_monte_carlo(
            home_xg=request.home_xg,
            away_xg=request.away_xg,
            n_simulations=request.n_simulations,
            include_htft=request.include_htft
        )
        
        return {
            'success': True,
            'result': result,
            'simulations': request.n_simulations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# HT/FT simulation endpoint
@app.get("/api/v3/simulate-htft")
async def simulate_htft(
    home_xg: float = Query(..., description="Home team expected goals"),
    away_xg: float = Query(..., description="Away team expected goals"),
    n_simulations: int = Query(100000, description="Number of simulations")
):
    """
    Simulate match with HT/FT breakdown.
    
    Uses time-segmented Poisson (42% goals in 1st half).
    """
    try:
        result = simulator.simulate_match_with_htft(
            home_xg_1h=home_xg * 0.42,
            away_xg_1h=away_xg * 0.42,
            home_xg_2h=home_xg * 0.58,
            away_xg_2h=away_xg * 0.58
        )
        
        return {
            'success': True,
            '1x2': {
                'home_win': result.home_win_prob,
                'draw': result.draw_prob,
                'away_win': result.away_win_prob
            },
            'htft': result.htft_probs,
            'correct_scores': result.correct_score_probs,
            'btts': result.btts_prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Season simulation endpoint
@app.post("/api/v3/simulate-season")
async def simulate_season(request: SeasonSimulationRequest):
    """
    Simulate remaining season to predict final standings.
    """
    try:
        result = simulator.simulate_season(
            fixtures=request.remaining_fixtures,
            team_strengths=request.team_strengths,
            n_simulations=request.n_simulations
        )
        
        return {
            'success': True,
            'league': request.league,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Player props endpoint
@app.post("/api/v3/player-props")
async def predict_player_props(request: PlayerPropsRequest):
    """
    Predict player props for a match.
    
    Includes:
    - Goals probability
    - Assists probability
    - Shots
    - Anytime scorer
    - Card probability
    """
    try:
        # Create features
        features = {
            'goals_avg_5': request.goals_avg,
            'assists_avg_5': request.goals_avg * 0.5,
            'shots_avg_5': request.goals_avg * 5,
            'shots_on_target_avg_5': request.goals_avg * 2.5,
            'is_home': 1 if request.is_home else 0,
            'opponent_strength': request.opponent_strength,
            'minutes_ratio': 0.9
        }
        
        predictions = player_predictor.predict_all_props(
            features, 
            request.position,
            request.props
        )
        
        predictions['player_id'] = request.player_id
        predictions['player_name'] = request.player_name or request.player_id
        
        return {
            'success': True,
            'predictions': predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Anytime scorer endpoint
@app.get("/api/v3/anytime-scorer")
async def anytime_scorer(
    goals_avg: float = Query(..., description="Player's average goals per game"),
    position: str = Query("FW", description="Player position"),
    is_home: bool = Query(True, description="Playing at home"),
    odds: Optional[float] = Query(None, description="Bookmaker odds")
):
    """
    Calculate anytime scorer probability.
    """
    try:
        result = predict_player_goals(
            goals_avg=goals_avg,
            position=position,
            is_home=is_home
        )
        
        prob = result['prob_1plus']
        fair_odds = 1 / prob if prob > 0 else 99
        
        response = {
            'success': True,
            'probability': round(prob, 4),
            'fair_odds': round(fair_odds, 2),
            'expected_goals': result['expected_goals']
        }
        
        if odds:
            implied = 1 / odds
            edge = prob - implied
            response['bookmaker_odds'] = odds
            response['edge'] = round(edge, 4)
            response['value_bet'] = edge > 0.05
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Value betting endpoint
@app.post("/api/v3/value-bets")
async def find_value_bets(request: ValueBetRequest):
    """
    Find value betting opportunities.
    """
    try:
        value_bets = []
        
        for market, prob in request.predictions.items():
            if market in request.odds:
                implied = 1 / request.odds[market]
                edge = prob - implied
                
                if edge >= request.min_edge:
                    # Calculate Kelly stake
                    kelly = (prob * request.odds[market] - 1) / (request.odds[market] - 1)
                    kelly = max(0, min(kelly, 0.25))  # Cap at 25%
                    
                    value_bets.append({
                        'market': market,
                        'probability': round(prob, 4),
                        'odds': request.odds[market],
                        'implied_probability': round(implied, 4),
                        'edge': round(edge, 4),
                        'expected_value': round(edge * request.odds[market], 4),
                        'kelly_stake': round(kelly * 100, 1)
                    })
        
        # Sort by edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return {
            'success': True,
            'value_bets': value_bets,
            'total_opportunities': len(value_bets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# RL betting strategy endpoint
@app.post("/api/v3/rl-strategy")
async def get_rl_betting_strategy(
    probability: float = Query(..., description="Model probability"),
    odds: float = Query(..., description="Bookmaker odds"),
    confidence: float = Query(0.5, description="Model confidence")
):
    """
    Get betting action from RL agent.
    """
    try:
        result = rl_agent.get_optimal_bet_size(
            model_probability=probability,
            odds=odds,
            confidence=confidence
        )
        
        return {
            'success': True,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for real-time predictions
@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    WebSocket for real-time match predictions.
    
    Send JSON with action and parameters, receive predictions.
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            action = request.get('action')
            
            if action == 'simulate':
                result = run_monte_carlo(
                    home_xg=request.get('home_xg', 1.5),
                    away_xg=request.get('away_xg', 1.2),
                    n_simulations=request.get('n_simulations', 100000)
                )
                
                await websocket.send_json({
                    'type': 'simulation',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
            
            elif action == 'player_props':
                goals_pred = predict_player_goals(
                    goals_avg=request.get('goals_avg', 0.3),
                    position=request.get('position', 'FW'),
                    is_home=request.get('is_home', True)
                )
                
                await websocket.send_json({
                    'type': 'player_props',
                    'result': goals_pred,
                    'timestamp': datetime.now().isoformat()
                })
            
            elif action == 'ping':
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
