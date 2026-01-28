"""
Blueprint API Routes
====================
Additional API endpoints for blueprint modules including matches, markets, models, and betting.
"""

from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

# Create blueprint for API routes
blueprint_api = Blueprint('blueprint_api', __name__, url_prefix='/api')


def register_blueprint_api(app):
    """Register the blueprint API routes with the Flask app."""
    
    # Import BlueprintManager
    try:
        from src.blueprint_integration import get_blueprint, get_blueprint_status, predict_with_blueprint
        manager = get_blueprint()
        logger.info("✅ Blueprint Manager loaded for API")
    except Exception as e:
        logger.error(f"Failed to load Blueprint Manager: {e}")
        manager = None
    
    # =========================================================================
    # BLUEPRINT STATUS ENDPOINT
    # =========================================================================
    
    @blueprint_api.route('/blueprint/status', methods=['GET'])
    def blueprint_status():
        """Get status of all blueprint modules."""
        try:
            status = get_blueprint_status()
            return jsonify({
                'success': True,
                'data': status
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # =========================================================================
    # MATCHES ENDPOINTS (from data sources)
    # =========================================================================
    
    @blueprint_api.route('/matches', methods=['GET'])
    @blueprint_api.route('/matches/', methods=['GET'])
    def get_matches():
        """Get upcoming matches from all data sources."""
        days = request.args.get('days', 7, type=int)
        league = request.args.get('league', None)
        
        try:
            if manager and 'data_manager' in manager.data:
                fixtures = manager.data['data_manager'].fetch_upcoming_fixtures(
                    days_ahead=days,
                    leagues=[league] if league else None
                )
                if fixtures is not None and len(fixtures) > 0:
                    matches = fixtures.head(50).to_dict('records')
                else:
                    matches = []
            else:
                # Fallback to sample matches
                matches = _get_sample_matches()
            
            return jsonify({
                'success': True,
                'count': len(matches),
                'data': matches
            })
        except Exception as e:
            logger.error(f"Error fetching matches: {e}")
            return jsonify({
                'success': True,
                'count': 0,
                'data': _get_sample_matches(),
                'note': 'Using sample data'
            })
    
    @blueprint_api.route('/matches/today', methods=['GET'])
    def get_matches_today():
        """Get today's matches."""
        try:
            if manager and 'data_manager' in manager.data:
                fixtures = manager.data['data_manager'].fetch_upcoming_fixtures(days_ahead=1)
                if fixtures is not None:
                    matches = fixtures.to_dict('records')
                else:
                    matches = _get_sample_matches()
            else:
                matches = _get_sample_matches()
            
            return jsonify({
                'success': True,
                'count': len(matches),
                'data': matches
            })
        except Exception as e:
            return jsonify({
                'success': True,
                'data': _get_sample_matches(),
                'note': str(e)
            })
    
    # =========================================================================
    # MARKETS ENDPOINTS (prediction markets)
    # =========================================================================
    
    @blueprint_api.route('/markets', methods=['GET'])
    @blueprint_api.route('/markets/', methods=['GET'])
    def get_markets():
        """List available prediction markets."""
        markets = [
            {'id': 'match_result', 'name': '1X2 Match Result', 'description': 'Home/Draw/Away prediction'},
            {'id': 'over_under', 'name': 'Over/Under Goals', 'description': 'Total goals O/U 0.5-5.5'},
            {'id': 'btts', 'name': 'Both Teams To Score', 'description': 'BTTS Yes/No prediction'},
            {'id': 'correct_score', 'name': 'Correct Score', 'description': 'Exact match score prediction'},
            {'id': 'asian_handicap', 'name': 'Asian Handicap', 'description': 'Handicap betting markets'},
            {'id': 'htft', 'name': 'Half-Time/Full-Time', 'description': 'HT/FT result prediction'},
            {'id': 'first_goal', 'name': 'First Goal', 'description': 'Team to score first'},
            {'id': 'double_chance', 'name': 'Double Chance', 'description': '1X, 12, X2 markets'},
        ]
        
        if manager:
            available = list(manager.markets.keys())
            for m in markets:
                m['available'] = m['id'] in available
        
        return jsonify({
            'success': True,
            'markets': markets
        })
    
    @blueprint_api.route('/markets/<market_id>/predict', methods=['POST'])
    def predict_market(market_id):
        """Get prediction for a specific market."""
        data = request.get_json() or {}
        
        home_team = data.get('home_team', 'Home')
        away_team = data.get('away_team', 'Away')
        home_xg = data.get('home_xg', 1.5)
        away_xg = data.get('away_xg', 1.2)
        
        try:
            if manager and market_id in manager.markets:
                predictor = manager.markets[market_id]
                if hasattr(predictor, 'predict'):
                    prediction = predictor.predict(home_xg, away_xg)
                else:
                    prediction = {'error': 'Predictor has no predict method'}
            else:
                # Fallback predictions
                prediction = _get_fallback_prediction(market_id, home_xg, away_xg)
            
            return jsonify({
                'success': True,
                'market': market_id,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @blueprint_api.route('/markets/all', methods=['POST'])
    def predict_all_markets():
        """Get predictions for all markets for a match."""
        data = request.get_json() or {}
        
        home_team = data.get('home_team', 'Home')
        away_team = data.get('away_team', 'Away')
        home_xg = data.get('home_xg', 1.5)
        away_xg = data.get('away_xg', 1.2)
        odds = data.get('odds', None)
        
        try:
            result = predict_with_blueprint(
                home_team=home_team,
                away_team=away_team,
                home_xg=home_xg,
                away_xg=away_xg,
                odds=odds
            )
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # =========================================================================
    # MODELS ENDPOINTS
    # =========================================================================
    
    @blueprint_api.route('/models', methods=['GET'])
    def list_models():
        """List available ML models."""
        models = {
            'ml': [],
            'deep_learning': [],
            'ensemble': []
        }
        
        if manager:
            models['ml'] = list(manager.ml_models.keys())
            models['deep_learning'] = list(manager.deep_learning.keys())
            models['ensemble'] = list(manager.ensemble.keys())
        
        return jsonify({
            'success': True,
            'models': models
        })
    
    @blueprint_api.route('/models/<model_name>/predict', methods=['POST'])
    def model_predict(model_name):
        """Get prediction from a specific model."""
        data = request.get_json() or {}
        
        try:
            if manager:
                # Check all model categories
                model = None
                if model_name in manager.ml_models:
                    model = manager.ml_models[model_name]
                elif model_name in manager.deep_learning:
                    model = manager.deep_learning[model_name]
                
                if model and hasattr(model, 'predict'):
                    prediction = model.predict(**data)
                    return jsonify({
                        'success': True,
                        'model': model_name,
                        'prediction': prediction
                    })
            
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not found'
            }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # =========================================================================
    # BETTING ENDPOINTS
    # =========================================================================
    
    @blueprint_api.route('/betting/value', methods=['POST'])
    def find_value_bet():
        """Find value betting opportunities."""
        data = request.get_json() or {}
        
        predicted_prob = data.get('predicted_prob', 0.5)
        odds = data.get('odds', 2.0)
        
        try:
            if manager and 'value_detector' in manager.betting:
                result = manager.betting['value_detector'].find_value(
                    predicted_prob=predicted_prob,
                    odds=odds
                )
            else:
                # Fallback value calculation
                fair_odds = 1 / predicted_prob if predicted_prob > 0 else 100
                edge = (odds - fair_odds) / fair_odds * 100
                result = {
                    'is_value': edge > 5,
                    'edge': round(edge, 2),
                    'recommendation': 'BACK' if edge > 10 else 'CONSIDER' if edge > 5 else 'AVOID'
                }
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @blueprint_api.route('/betting/kelly', methods=['POST'])
    def calculate_kelly():
        """Calculate Kelly criterion stake."""
        data = request.get_json() or {}
        
        prob = data.get('probability', 0.5)
        odds = data.get('odds', 2.0)
        bankroll = data.get('bankroll', 1000)
        
        # Kelly formula: f* = (bp - q) / b
        # where b = odds - 1, p = prob, q = 1 - p
        b = odds - 1
        q = 1 - prob
        kelly_fraction = max(0, (b * prob - q) / b) if b > 0 else 0
        
        stake = bankroll * kelly_fraction * 0.25  # Quarter Kelly
        
        return jsonify({
            'success': True,
            'kelly_fraction': round(kelly_fraction, 4),
            'recommended_stake': round(stake, 2),
            'bankroll': bankroll,
            'probability': prob,
            'odds': odds
        })
    
    # =========================================================================
    # EXPLAINABILITY ENDPOINTS
    # =========================================================================
    
    @blueprint_api.route('/explain/<model_name>', methods=['POST'])
    def explain_prediction(model_name):
        """Get explanation for a prediction."""
        data = request.get_json() or {}
        
        try:
            if manager and 'shap' in manager.explainability:
                explanation = manager.explainability['shap'].explain(
                    model=model_name,
                    features=data.get('features', {})
                )
            else:
                explanation = {
                    'method': 'feature_importance',
                    'top_features': [
                        {'name': 'home_form', 'importance': 0.25},
                        {'name': 'away_form', 'importance': 0.20},
                        {'name': 'h2h', 'importance': 0.15},
                        {'name': 'home_goals_scored', 'importance': 0.12},
                        {'name': 'away_goals_conceded', 'importance': 0.10},
                    ]
                }
            
            return jsonify({
                'success': True,
                'explanation': explanation
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def _get_sample_matches():
        """Return sample matches for testing."""
        from datetime import datetime, timedelta
        today = datetime.now()
        
        return [
            {
                'home_team': 'Liverpool',
                'away_team': 'Manchester United',
                'league': 'Premier League',
                'date': today.strftime('%Y-%m-%d'),
                'time': '15:00'
            },
            {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'league': 'Premier League',
                'date': today.strftime('%Y-%m-%d'),
                'time': '17:30'
            },
            {
                'home_team': 'Bayern Munich',
                'away_team': 'Borussia Dortmund',
                'league': 'Bundesliga',
                'date': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '18:30'
            },
            {
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'league': 'La Liga',
                'date': (today + timedelta(days=2)).strftime('%Y-%m-%d'),
                'time': '21:00'
            },
        ]
    
    def _get_fallback_prediction(market_id, home_xg, away_xg):
        """Generate fallback predictions when modules unavailable."""
        import math
        total_xg = home_xg + away_xg
        home_strength = home_xg / total_xg if total_xg > 0 else 0.5
        
        if market_id == 'match_result':
            return {
                'home_win': round(home_strength * 0.9, 3),
                'draw': round(0.25, 3),
                'away_win': round((1 - home_strength) * 0.75, 3),
            }
        elif market_id == 'over_under':
            return {
                'over_2.5': round(1 - math.exp(-total_xg) * (1 + total_xg + total_xg**2/2), 3),
                'under_2.5': round(math.exp(-total_xg) * (1 + total_xg + total_xg**2/2), 3),
            }
        elif market_id == 'btts':
            btts_yes = round((1 - math.exp(-home_xg)) * (1 - math.exp(-away_xg)), 3)
            return {'yes': btts_yes, 'no': round(1 - btts_yes, 3)}
        else:
            return {'prediction': 'unavailable', 'market': market_id}
    
    # Register the blueprint
    app.register_blueprint(blueprint_api)
    logger.info("✅ Blueprint API registered at /api/*")
    
    return app
