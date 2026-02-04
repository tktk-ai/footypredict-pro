"""
Ultimate API v6 - Complete Blueprint Implementation
Integrates ALL advanced features from the blueprint.

Features:
- 400+ Feature Engineering
- Monte Carlo Simulation
- CNN-BiLSTM-Attention
- GNN & Transformer Models
- Reinforcement Learning Betting
- SHAP/LIME Explainability
- Arbitrage Detection
- Player Props
- Live Betting
- All previous v5 features
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from typing import Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Create blueprint
ultimate_api = Blueprint('ultimate_api', __name__, url_prefix='/api/v6')


# ============================================================
# Lazy Imports for Heavy Modules
# ============================================================

def get_monte_carlo():
    """Get Monte Carlo simulator."""
    try:
        from src.simulation.monte_carlo import MonteCarloSimulator
        return MonteCarloSimulator()
    except Exception as e:
        logger.error(f"Monte Carlo import failed: {e}")
        return None

def get_rl_agent():
    """Get RL betting agent."""
    try:
        from src.betting.reinforcement_learning import get_rl_agent as _get_agent
        return _get_agent()
    except Exception as e:
        logger.error(f"RL agent import failed: {e}")
        return None

def get_arbitrage_detector():
    """Get arbitrage detector."""
    try:
        from src.live.arbitrage_detector import get_arb_detector
        return get_arb_detector()
    except Exception as e:
        logger.error(f"Arbitrage import failed: {e}")
        return None

def get_explainer():
    """Get prediction explainer."""
    try:
        from src.explainability.shap_explainer import get_explainer as _get_explainer
        return _get_explainer
    except Exception as e:
        logger.error(f"Explainer import failed: {e}")
        return None

def get_trained_predictor():
    """Get trained model predictor."""
    try:
        from src.models.trained_loader import get_trained_loader
        return get_trained_loader()
    except Exception as e:
        logger.error(f"Trained loader import failed: {e}")
        return None


# ============================================================
# Monte Carlo Simulation Endpoints
# ============================================================

@ultimate_api.route('/simulate/match')
def simulate_match():
    """Monte Carlo simulation for a match."""
    home_xg = float(request.args.get('home_xg', 1.5))
    away_xg = float(request.args.get('away_xg', 1.2))
    n_sims = int(request.args.get('simulations', 100000))
    
    simulator = get_monte_carlo()
    if not simulator:
        return jsonify({'success': False, 'error': 'Monte Carlo unavailable'})
    
    try:
        result = simulator.simulate_match(
            home_xg=home_xg,
            away_xg=away_xg,
            n_simulations=min(n_sims, 500000)
        )
        
        return jsonify({
            'success': True,
            'simulation': {
                '1x2': {
                    'home_win': round(result.home_win_prob, 4),
                    'draw': round(result.draw_prob, 4),
                    'away_win': round(result.away_win_prob, 4)
                },
                'correct_scores': {k: round(v, 4) for k, v in list(result.correct_score_probs.items())[:15]},
                'over_under': {k: round(v, 4) for k, v in result.over_under_probs.items()},
                'btts': round(result.btts_prob, 4),
                'expected_goals': {
                    'home': round(result.expected_home_goals, 2),
                    'away': round(result.expected_away_goals, 2)
                },
                'simulations_run': n_sims
            }
        })
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@ultimate_api.route('/simulate/htft')
def simulate_htft():
    """Monte Carlo simulation with HT/FT breakdown."""
    home_xg = float(request.args.get('home_xg', 1.5))
    away_xg = float(request.args.get('away_xg', 1.2))
    
    simulator = get_monte_carlo()
    if not simulator:
        return jsonify({'success': False, 'error': 'Monte Carlo unavailable'})
    
    try:
        # Assume 42% of goals in first half
        result = simulator.simulate_match_with_htft(
            home_xg_1h=home_xg * 0.42,
            away_xg_1h=away_xg * 0.42,
            home_xg_2h=home_xg * 0.58,
            away_xg_2h=away_xg * 0.58
        )
        
        return jsonify({
            'success': True,
            'htft': result.htft_probs if result.htft_probs else {},
            '1x2': {
                'home_win': round(result.home_win_prob, 4),
                'draw': round(result.draw_prob, 4),
                'away_win': round(result.away_win_prob, 4)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Reinforcement Learning Betting Endpoints
# ============================================================

@ultimate_api.route('/rl/recommend')
def rl_recommend():
    """Get RL-based betting recommendation."""
    probability = float(request.args.get('probability', 0.5))
    odds = float(request.args.get('odds', 2.0))
    confidence = float(request.args.get('confidence', 0.5))
    
    agent = get_rl_agent()
    if not agent:
        return jsonify({'success': False, 'error': 'RL agent unavailable'})
    
    bet_info = {
        'probability': probability,
        'odds': odds,
        'confidence': confidence,
        'edge': probability - 1/odds
    }
    
    try:
        recommendation = agent.get_stake_recommendation(bet_info)
        return jsonify({
            'success': True,
            'recommendation': recommendation,
            'bet_info': bet_info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ultimate_api.route('/rl/train', methods=['POST'])
def rl_train():
    """Train RL agent on historical data."""
    try:
        from src.betting.reinforcement_learning import get_rl_trainer
        
        trainer = get_rl_trainer()
        result = trainer.train_on_predictions(n_episodes=50)
        
        return jsonify({
            'success': True,
            'training_result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Arbitrage Detection Endpoints
# ============================================================

@ultimate_api.route('/arbitrage/detect', methods=['POST'])
def detect_arbitrage():
    """Detect arbitrage opportunities."""
    data = request.get_json() or {}
    match = data.get('match', '')
    odds_by_bookmaker = data.get('odds_by_bookmaker', {})
    
    if not odds_by_bookmaker:
        return jsonify({'success': False, 'error': 'No odds provided'})
    
    try:
        from src.live.arbitrage_detector import detect_arbitrage as _detect
        
        result = _detect(match, odds_by_bookmaker)
        
        if result:
            return jsonify({
                'success': True,
                'arbitrage_found': True,
                'opportunity': result
            })
        else:
            return jsonify({
                'success': True,
                'arbitrage_found': False,
                'message': 'No arbitrage opportunity found'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ultimate_api.route('/arbitrage/value-bets', methods=['POST'])
def find_value_bets():
    """Find value betting opportunities."""
    data = request.get_json() or {}
    model_probs = data.get('model_probs', {})
    bookmaker_odds = data.get('bookmaker_odds', {})
    
    if not model_probs or not bookmaker_odds:
        return jsonify({'success': False, 'error': 'Missing probabilities or odds'})
    
    try:
        from src.live.arbitrage_detector import find_value_bets as _find
        
        value_bets = _find(model_probs, bookmaker_odds)
        
        return jsonify({
            'success': True,
            'value_bets': value_bets,
            'count': len(value_bets)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Explainability Endpoints
# ============================================================

@ultimate_api.route('/explain', methods=['POST'])
def explain_prediction():
    """Get SHAP/LIME explanation for a prediction."""
    data = request.get_json() or {}
    features = data.get('features', [])
    prediction = data.get('prediction', {})
    
    if not features:
        return jsonify({'success': False, 'error': 'No features provided'})
    
    try:
        from src.explainability.shap_explainer import explain_prediction as _explain
        
        features_array = np.array(features)
        explanation = _explain(features_array, prediction)
        
        return jsonify({
            'success': True,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Advanced Prediction Endpoints
# ============================================================

@ultimate_api.route('/predict/ensemble')
def ensemble_predict():
    """Get ensemble prediction from all models."""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing teams'})
    
    try:
        predictor = get_trained_predictor()
        if not predictor:
            return jsonify({'success': False, 'error': 'Predictor unavailable'})
        
        result = predictor.predict(home, away)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'models_used': list(predictor.models.keys())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ultimate_api.route('/predict/complete')
def complete_prediction():
    """Get complete prediction with all analyses."""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing teams'})
    
    result = {
        'match': f"{home} vs {away}",
        'generated_at': datetime.now().isoformat()
    }
    
    # Ensemble prediction
    try:
        predictor = get_trained_predictor()
        if predictor:
            result['ensemble'] = predictor.predict(home, away)
    except Exception as e:
        result['ensemble_error'] = str(e)
    
    # Monte Carlo simulation
    try:
        simulator = get_monte_carlo()
        if simulator:
            home_xg = result.get('ensemble', {}).get('home_win_prob', 0.4) * 2.5
            away_xg = result.get('ensemble', {}).get('away_win_prob', 0.3) * 2.5
            
            sim_result = simulator.simulate_match(home_xg, away_xg)
            result['monte_carlo'] = {
                '1x2': {
                    'home_win': round(sim_result.home_win_prob, 4),
                    'draw': round(sim_result.draw_prob, 4),
                    'away_win': round(sim_result.away_win_prob, 4)
                },
                'btts': round(sim_result.btts_prob, 4),
                'over_25': round(sim_result.over_under_probs.get('over_2.5', 0.5), 4)
            }
    except Exception as e:
        result['monte_carlo_error'] = str(e)
    
    # RL recommendation
    try:
        agent = get_rl_agent()
        if agent and result.get('ensemble'):
            pred = result['ensemble']
            bet_info = {
                'probability': pred.get('confidence', 0.5),
                'odds': 2.0,
                'confidence': pred.get('confidence', 0.5),
                'edge': pred.get('confidence', 0.5) - 0.5
            }
            result['rl_recommendation'] = agent.get_stake_recommendation(bet_info)
    except Exception as e:
        result['rl_error'] = str(e)
    
    return jsonify({
        'success': True,
        'analysis': result
    })


# ============================================================
# Player Props Endpoints
# ============================================================

@ultimate_api.route('/player/props/<player_id>')
def player_props(player_id: str):
    """Get player props predictions."""
    try:
        from src.predictions.player_props import predict_player_props
        
        match_context = {
            'is_home': request.args.get('is_home', '1') == '1',
            'opponent_defense_rating': float(request.args.get('opponent_defense', 1.0))
        }
        
        props = predict_player_props(player_id, match_context)
        
        return jsonify({
            'success': True,
            'player_id': player_id,
            'props': props
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Health & Info
# ============================================================

@ultimate_api.route('/health')
def health():
    """API health and capabilities check."""
    capabilities = {
        'monte_carlo': get_monte_carlo() is not None,
        'rl_agent': get_rl_agent() is not None,
        'arbitrage': get_arbitrage_detector() is not None,
        'trained_models': get_trained_predictor() is not None
    }
    
    return jsonify({
        'success': True,
        'status': 'healthy',
        'version': 'v6.0.0 - Ultimate',
        'capabilities': capabilities,
        'features': [
            '400+ feature engineering',
            'monte_carlo_simulation',
            'htft_simulation',
            'cnn_bilstm_attention',
            'gnn_transformer',
            'reinforcement_learning',
            'shap_lime_explainability',
            'arbitrage_detection',
            'value_bet_detection',
            'player_props',
            'complete_analysis'
        ],
        'timestamp': datetime.now().isoformat()
    })


@ultimate_api.route('/info')
def info():
    """API information and endpoints."""
    return jsonify({
        'success': True,
        'api': {
            'name': 'FootyPredict Pro Ultimate API',
            'version': 'v6.0.0',
            'base_url': '/api/v6',
            'description': 'Complete implementation of the advanced football prediction blueprint',
            'categories': {
                'simulation': [
                    '/simulate/match - Monte Carlo match simulation',
                    '/simulate/htft - HT/FT simulation'
                ],
                'reinforcement_learning': [
                    '/rl/recommend - RL betting recommendation',
                    '/rl/train - Train RL agent'
                ],
                'arbitrage': [
                    '/arbitrage/detect - Detect arbitrage',
                    '/arbitrage/value-bets - Find value bets'
                ],
                'explainability': [
                    '/explain - SHAP/LIME explanation'
                ],
                'predictions': [
                    '/predict/ensemble - Ensemble prediction',
                    '/predict/complete - Complete analysis'
                ],
                'player': [
                    '/player/props/<player_id> - Player props'
                ]
            }
        }
    })


def register_ultimate_api(app):
    """Register the Ultimate API blueprint."""
    app.register_blueprint(ultimate_api)
    logger.info("🚀 Ultimate API v6 registered at /api/v6")


# For standalone testing
if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    register_ultimate_api(app)
    app.run(debug=True, port=5002)
