"""
V3.0 API Routes for Flask Integration

Provides Flask Blueprint with Monte Carlo, Player Props, RL Strategy endpoints.
Integrates with existing app.py cleanly.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import V3.0 modules
from src.simulation.monte_carlo import MonteCarloSimulator, run_monte_carlo
from src.predictions.player_props import PlayerPropsPredictor, predict_player_goals
from src.betting.reinforcement_learning import BettingEnvironment, DQNBettingAgent
from src.features.engineering.advanced_features import AdvancedFeatureEngineer

# Create Blueprint
v3_api = Blueprint('v3_api', __name__, url_prefix='/api/v3')

# Initialize components
monte_carlo = MonteCarloSimulator(n_simulations=100000)
player_predictor = PlayerPropsPredictor()
rl_agent = DQNBettingAgent()


@v3_api.route('/health')
def v3_health():
    """V3.0 API health check."""
    return jsonify({
        'status': 'healthy',
        'version': '3.0.0',
        'modules': ['monte_carlo', 'player_props', 'rl_betting', 'deep_learning'],
        'timestamp': datetime.now().isoformat()
    })


@v3_api.route('/simulate', methods=['POST', 'GET'])
def simulate_match():
    """
    Run Monte Carlo simulation for a match.
    
    POST/GET params:
    - home_xg: float (required) - Expected goals for home team
    - away_xg: float (required) - Expected goals for away team
    - n_simulations: int (optional, default 100000)
    - include_htft: bool (optional, default false)
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
        else:
            data = request.args
        
        home_xg = float(data.get('home_xg', 1.5))
        away_xg = float(data.get('away_xg', 1.2))
        n_simulations = int(data.get('n_simulations', 100000))
        include_htft = str(data.get('include_htft', 'false')).lower() == 'true'
        
        result = run_monte_carlo(
            home_xg=home_xg,
            away_xg=away_xg,
            n_simulations=n_simulations,
            include_htft=include_htft
        )
        
        return jsonify({
            'success': True,
            'input': {'home_xg': home_xg, 'away_xg': away_xg},
            'simulations': n_simulations,
            'predictions': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/simulate-htft', methods=['GET'])
def simulate_htft():
    """Simulate match with HT/FT breakdown."""
    try:
        home_xg = float(request.args.get('home_xg', 1.5))
        away_xg = float(request.args.get('away_xg', 1.2))
        
        # Simulate with 42% of goals in first half
        result = monte_carlo.simulate_match_with_htft(
            home_xg_1h=home_xg * 0.42,
            away_xg_1h=away_xg * 0.42,
            home_xg_2h=home_xg * 0.58,
            away_xg_2h=away_xg * 0.58
        )
        
        return jsonify({
            'success': True,
            'input': {'home_xg': home_xg, 'away_xg': away_xg},
            '1x2': {
                'home_win': round(result.home_win_prob, 4),
                'draw': round(result.draw_prob, 4),
                'away_win': round(result.away_win_prob, 4)
            },
            'htft': result.htft_probs,
            'correct_scores': result.correct_score_probs,
            'btts': round(result.btts_prob, 4)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/correct-scores', methods=['GET'])
def get_correct_scores():
    """Get correct score probabilities from simulation."""
    try:
        home_xg = float(request.args.get('home_xg', 1.5))
        away_xg = float(request.args.get('away_xg', 1.2))
        
        result = monte_carlo.simulate_match(home_xg, away_xg)
        
        return jsonify({
            'success': True,
            'correct_scores': result.correct_score_probs,
            'expected_goals': {
                'home': round(result.expected_home_goals, 2),
                'away': round(result.expected_away_goals, 2)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/player-props', methods=['POST', 'GET'])
def player_props():
    """
    Predict player props (goals, assists, shots, cards).
    
    Params:
    - goals_avg: float - Player's average goals per game
    - position: str - FW, MF, DF, GK
    - is_home: bool - Playing at home
    - opponent_strength: float - Opponent defensive rating (1.0 = avg)
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
        else:
            data = request.args
        
        goals_avg = float(data.get('goals_avg', 0.3))
        position = data.get('position', 'FW')
        is_home = str(data.get('is_home', 'true')).lower() == 'true'
        opponent_strength = float(data.get('opponent_strength', 1.0))
        
        # Create features
        features = {
            'goals_avg_5': goals_avg,
            'assists_avg_5': goals_avg * 0.5,
            'shots_avg_5': goals_avg * 5,
            'shots_on_target_avg_5': goals_avg * 2.5,
            'fouls_avg_5': 1.5,
            'is_home': 1 if is_home else 0,
            'opponent_strength': opponent_strength,
            'minutes_ratio': 0.9
        }
        
        # Get all predictions
        predictions = player_predictor.predict_all_props(features, position)
        
        return jsonify({
            'success': True,
            'player_info': {
                'goals_avg': goals_avg,
                'position': position,
                'is_home': is_home
            },
            'predictions': {
                'anytime_scorer': {
                    'probability': round(predictions['goals']['prob_1plus'], 4),
                    'fair_odds': round(1 / predictions['goals']['prob_1plus'], 2) if predictions['goals']['prob_1plus'] > 0 else 99,
                    'expected_goals': predictions['goals']['expected_goals']
                },
                '2_plus_goals': {
                    'probability': round(predictions['goals']['prob_2plus'], 4),
                    'fair_odds': round(1 / predictions['goals']['prob_2plus'], 2) if predictions['goals']['prob_2plus'] > 0 else 99
                },
                'shots': predictions['shots'],
                'shots_on_target': predictions['shots_on_target'],
                'cards': predictions['cards']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/anytime-scorer', methods=['GET'])
def anytime_scorer():
    """Quick anytime scorer probability check."""
    try:
        goals_avg = float(request.args.get('goals_avg', 0.3))
        position = request.args.get('position', 'FW')
        is_home = request.args.get('is_home', 'true').lower() == 'true'
        odds = request.args.get('odds')
        
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
            odds_float = float(odds)
            implied = 1 / odds_float
            edge = prob - implied
            response['bookmaker_odds'] = odds_float
            response['edge'] = round(edge, 4)
            response['value_bet'] = edge > 0.05
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/value-bets', methods=['POST'])
def find_value_bets():
    """
    Find value betting opportunities.
    
    POST body:
    {
        "predictions": {"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
        "odds": {"home_win": 2.1, "draw": 3.5, "away_win": 3.8},
        "min_edge": 0.03
    }
    """
    try:
        data = request.json or {}
        predictions = data.get('predictions', {})
        odds = data.get('odds', {})
        min_edge = float(data.get('min_edge', 0.03))
        
        value_bets = []
        
        for market, prob in predictions.items():
            if market in odds:
                implied = 1 / odds[market]
                edge = prob - implied
                
                if edge >= min_edge:
                    # Kelly criterion
                    kelly = (prob * odds[market] - 1) / (odds[market] - 1)
                    kelly = max(0, min(kelly, 0.25))  # Cap at 25%
                    
                    value_bets.append({
                        'market': market,
                        'probability': round(prob, 4),
                        'odds': odds[market],
                        'implied_prob': round(implied, 4),
                        'edge': round(edge, 4),
                        'expected_value': round(edge * odds[market], 4),
                        'kelly_stake_pct': round(kelly * 100, 1)
                    })
        
        # Sort by edge descending
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return jsonify({
            'success': True,
            'value_bets': value_bets,
            'count': len(value_bets)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/rl-strategy', methods=['GET', 'POST'])
def rl_betting_strategy():
    """
    Get betting recommendation from RL agent.
    
    Params:
    - probability: float - Model-predicted probability
    - odds: float - Bookmaker odds
    - confidence: float - Model confidence (0-1)
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
        else:
            data = request.args
        
        probability = float(data.get('probability', 0.5))
        odds = float(data.get('odds', 2.0))
        confidence = float(data.get('confidence', 0.5))
        
        result = rl_agent.get_optimal_bet_size(
            model_probability=probability,
            odds=odds,
            confidence=confidence
        )
        
        return jsonify({
            'success': True,
            'input': {
                'probability': probability,
                'odds': odds,
                'confidence': confidence
            },
            'recommendation': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@v3_api.route('/full-prediction', methods=['POST'])
def full_prediction():
    """
    Complete match prediction using all V3.0 models.
    
    POST body:
    {
        "home_team": "Manchester City",
        "away_team": "Arsenal",
        "home_xg": 1.8,
        "away_xg": 1.3,
        "odds": {"home_win": 1.8, "draw": 3.5, "away_win": 4.5}
    }
    """
    try:
        data = request.json or {}
        
        home_team = data.get('home_team', 'Home')
        away_team = data.get('away_team', 'Away')
        home_xg = float(data.get('home_xg', 1.5))
        away_xg = float(data.get('away_xg', 1.2))
        odds = data.get('odds', {})
        
        # Monte Carlo simulation
        mc_result = monte_carlo.simulate_match(
            home_xg=home_xg,
            away_xg=away_xg,
            home_xg_std=0.3,
            away_xg_std=0.3
        )
        
        simulation = mc_result.to_dict()
        
        # Find value bets
        value_bets = []
        predictions = {
            'home_win': simulation['1x2']['home_win'],
            'draw': simulation['1x2']['draw'],
            'away_win': simulation['1x2']['away_win'],
            'over_2.5': simulation['over_under']['over_2.5'],
            'btts_yes': simulation['btts']['yes']
        }
        
        for market, prob in predictions.items():
            if market in odds:
                implied = 1 / odds[market]
                edge = prob - implied
                if edge > 0.03:
                    kelly = (prob * odds[market] - 1) / (odds[market] - 1)
                    kelly = max(0, min(kelly, 0.25))
                    value_bets.append({
                        'market': market,
                        'probability': round(prob, 4),
                        'odds': odds[market],
                        'edge': round(edge, 4),
                        'kelly_stake_pct': round(kelly * 100, 1)
                    })
        
        return jsonify({
            'success': True,
            'match': f"{home_team} vs {away_team}",
            'timestamp': datetime.now().isoformat(),
            'simulation': simulation,
            'value_bets': value_bets if value_bets else None,
            'methodology': 'Monte Carlo simulation with 100,000 iterations'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def register_v3_api(app):
    """Register V3.0 API Blueprint with Flask app."""
    app.register_blueprint(v3_api)
    print("✅ V3.0 API registered at /api/v3/")
