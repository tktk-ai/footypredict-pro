"""
Football Prediction Web Application

Flask-based web interface for the prediction system.
Now with advanced accumulators, monetization, and user management.
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.api_clients import DataAggregator, OpenLigaDBClient
from src.predictor import PredictionEngine
from src.goals_predictor import PoissonGoalPredictor
from src.accumulator import AccumulatorBuilder
from src.risk_management import AccuracyTracker, PatternDetector, KellyCriterion, Backtester
from src.telegram_bot import TelegramBot
from src.live_data import LiveDataClient
from src.ml_predictor import EnsemblePredictor
from src.betting_intel import OddsComparer, ArbitrageFinder, ValueBetFinder
from src.user_manager import UserManager
from src.whatsapp_bot import WhatsAppBot
from src.advanced_predictions import AdvancedPredictor, HeadToHeadAnalyzer

# New Phase 7-10 imports
from src.accumulators import AccumulatorEngine, generate_all_accumulators
from src.monetization import MonetizationManager, get_pricing
from src.bet_tracker import BetTracker

app = Flask(__name__)

# Initialize components
data_aggregator = DataAggregator()
predictor = PredictionEngine()
goals_predictor = PoissonGoalPredictor()
acca_builder = AccumulatorBuilder()
accuracy_tracker = AccuracyTracker()
pattern_detector = PatternDetector(accuracy_tracker)
kelly_calc = KellyCriterion()
backtester = Backtester()
live_data = LiveDataClient()
ml_predictor = EnsemblePredictor()
odds_comparer = OddsComparer()
arbitrage_finder = ArbitrageFinder()
value_finder = ValueBetFinder()
user_manager = UserManager()
whatsapp_bot = WhatsAppBot()
advanced_predictor = AdvancedPredictor()
h2h_analyzer = HeadToHeadAnalyzer()

# New Phase 7-10 components
acca_engine = AccumulatorEngine()
monetization = MonetizationManager()
bet_tracker = BetTracker()


@app.route('/')
def index():
    """Main page with upcoming fixtures and predictions"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    return render_template('dashboard.html')


@app.route('/login')
def login_page():
    """Login/Register page"""
    return render_template('login.html')


@app.route('/pricing')
def pricing_page():
    """Pricing page with subscription tiers"""
    return render_template('pricing.html')


@app.route('/accumulators')
def accumulators_page():
    """Accumulators page with all 6 strategies"""
    return render_template('accumulators.html')


# ============================================================
# User Authentication API
# ============================================================

@app.route('/api/user/register', methods=['POST'])
def register_user():
    """Register new user"""
    data = request.get_json()
    result = user_manager.register(
        email=data.get('email', ''),
        username=data.get('username', ''),
        password=data.get('password', ''),
        tier=data.get('tier', 'free')
    )
    return jsonify(result)


@app.route('/api/user/login', methods=['POST'])
def login_user():
    """Login user"""
    data = request.get_json()
    result = user_manager.login(
        email=data.get('email', ''),
        password=data.get('password', '')
    )
    return jsonify(result)


@app.route('/api/user/profile')
def get_user_profile():
    """Get current user profile"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = user_manager.validate_session(token)
    
    if user:
        return jsonify({'success': True, 'user': user.to_public_dict()})
    return jsonify({'success': False, 'error': 'Invalid session'}), 401


# ============================================================
# Advanced Accumulators API
# ============================================================

@app.route('/api/accumulators')
def get_all_accumulators():
    """Get all accumulator types for today's fixtures"""
    league = request.args.get('league', 'bundesliga')
    
    # Get predictions for accumulator generation
    try:
        matches = data_aggregator.get_upcoming_matches([league], days=3)
        predictions = []
        
        for match in matches[:10]:  # Limit for performance
            home_name = match.home_team.name if hasattr(match.home_team, 'name') else str(match.home_team)
            away_name = match.away_team.name if hasattr(match.away_team, 'name') else str(match.away_team)
            
            pred = predictor.predict_match(home_name, away_name)
            goals = goals_predictor.predict_goals(home_name, away_name)
            
            predictions.append({
                'id': match.id,
                'home_team': home_name,
                'away_team': away_name,
                'home_win_prob': pred.home_win_prob,
                'draw_prob': pred.draw_prob,
                'away_win_prob': pred.away_win_prob,
                'confidence': pred.confidence,
                'goals': {
                    'home_xg': goals.home_xg,
                    'away_xg': goals.away_xg,
                    'total_xg': goals.total_xg,
                    'over_under': {
                        'over_0.5': goals.over_0_5,
                        'over_2.5': goals.over_2_5,
                    },
                    'btts': {'yes': goals.btts_yes}
                }
            })
        
        accas = generate_all_accumulators(predictions)
        
        return jsonify({
            'success': True,
            'strategies': acca_engine.get_all_strategies(),
            'accumulators': accas
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/accumulators/<strategy>')
def get_accumulator_by_strategy(strategy):
    """Get specific accumulator type"""
    from src.accumulators import generate_accumulator
    
    league = request.args.get('league', 'bundesliga')
    
    try:
        matches = data_aggregator.get_upcoming_matches([league], days=3)
        predictions = []
        
        for match in matches[:10]:
            home_name = match.home_team.name if hasattr(match.home_team, 'name') else str(match.home_team)
            away_name = match.away_team.name if hasattr(match.away_team, 'name') else str(match.away_team)
            
            pred = predictor.predict_match(home_name, away_name)
            goals = goals_predictor.predict_goals(home_name, away_name)
            
            predictions.append({
                'id': match.id,
                'home_team': home_name,
                'away_team': away_name,
                'home_win_prob': pred.home_win_prob,
                'draw_prob': pred.draw_prob,
                'away_win_prob': pred.away_win_prob,
                'confidence': pred.confidence,
                'goals': {
                    'home_xg': goals.home_xg,
                    'away_xg': goals.away_xg,
                    'over_under': {
                        'over_0.5': goals.over_0_5,
                        'over_2.5': goals.over_2_5,
                    },
                    'btts': {'yes': goals.btts_yes}
                }
            })
        
        acca = generate_accumulator(predictions, strategy)
        
        if acca:
            return jsonify({'success': True, 'accumulator': acca})
        return jsonify({'success': False, 'error': 'Could not generate accumulator'}), 404
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# Monetization API
# ============================================================

@app.route('/api/pricing')
def get_pricing_info():
    """Get subscription pricing information"""
    return jsonify(get_pricing())


@app.route('/api/checkout', methods=['POST'])
def create_checkout():
    """Create checkout session for subscription"""
    data = request.get_json()
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = user_manager.validate_session(token)
    
    if not user:
        return jsonify({'success': False, 'error': 'Login required'}), 401
    
    session = monetization.generate_checkout_session(
        user_id=user.id,
        tier_id=data.get('tier', 'pro'),
        period=data.get('period', 'monthly')
    )
    
    return jsonify({'success': True, **session})


# ============================================================
# Bet Tracking API
# ============================================================

@app.route('/api/bets', methods=['GET', 'POST'])
def handle_bets():
    """Get or add user bets"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = user_manager.validate_session(token)
    
    if not user:
        return jsonify({'success': False, 'error': 'Login required'}), 401
    
    if request.method == 'POST':
        data = request.get_json()
        bet = bet_tracker.add_bet(
            user_id=user.id,
            match_id=data.get('match_id', ''),
            home_team=data.get('home_team', ''),
            away_team=data.get('away_team', ''),
            selection=data.get('selection', ''),
            odds=float(data.get('odds', 1.0)),
            stake=float(data.get('stake', 0)),
            notes=data.get('notes')
        )
        return jsonify({'success': True, 'bet': bet.to_dict()})
    
    else:
        status = request.args.get('status')
        bets = bet_tracker.get_user_bets(user.id, status=status)
        stats = bet_tracker.get_user_stats(user.id)
        return jsonify({'success': True, 'bets': bets, 'stats': stats})


@app.route('/api/bets/<bet_id>/settle', methods=['POST'])
def settle_bet(bet_id):
    """Settle a bet as won or lost"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = user_manager.validate_session(token)
    
    if not user:
        return jsonify({'success': False, 'error': 'Login required'}), 401
    
    data = request.get_json()
    won = data.get('won', False)
    
    bet = bet_tracker.settle_bet(bet_id, user.id, won)
    
    if bet:
        return jsonify({'success': True, 'bet': bet.to_dict()})
    return jsonify({'success': False, 'error': 'Bet not found'}), 404


@app.route('/api/leaderboard')
def get_leaderboard():
    """Get betting leaderboard"""
    limit = int(request.args.get('limit', 10))
    leaderboard = bet_tracker.get_leaderboard(limit)
    return jsonify({'success': True, 'leaderboard': leaderboard})


@app.route('/api/fixtures')
def get_fixtures():
    """API endpoint to get upcoming fixtures with predictions"""
    leagues = request.args.getlist('league') or ['bundesliga']
    days = int(request.args.get('days', 7))
    
    try:
        # Get upcoming matches
        matches = data_aggregator.get_upcoming_matches(leagues, days)
        
        # Generate predictions for each match
        results = []
        for match in matches:
            if match.status == 'scheduled':
                prediction = predictor.predict_match(
                    home_team=match.home_team.name,
                    away_team=match.away_team.name,
                    league=match.league
                )
                goals = goals_predictor.predict_goals(
                    home_team=match.home_team.name,
                    away_team=match.away_team.name,
                    league=match.league
                )
                
                results.append({
                    'match': match.to_dict(),
                    'prediction': prediction.to_dict(),
                    'goals': goals.to_dict()
                })
            else:
                results.append({
                    'match': match.to_dict(),
                    'prediction': None,
                    'goals': None
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'fixtures': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict')
def predict_match():
    """API endpoint for custom match prediction"""
    home = request.args.get('home')
    away = request.args.get('away')
    league = request.args.get('league', 'default')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    # Parse market odds if provided
    market_odds = None
    home_odds = request.args.get('home_odds')
    draw_odds = request.args.get('draw_odds')
    away_odds = request.args.get('away_odds')
    
    if home_odds and draw_odds and away_odds:
        try:
            market_odds = (float(home_odds), float(draw_odds), float(away_odds))
        except:
            pass
    
    prediction = predictor.predict_match(home, away, league, market_odds)
    goals = goals_predictor.predict_goals(home, away, league)
    
    return jsonify({
        'success': True,
        'home_team': home,
        'away_team': away,
        'league': league,
        'prediction': prediction.to_dict(),
        'goals': goals.to_dict()
    })


@app.route('/api/goals')
def predict_goals_endpoint():
    """API endpoint for goal predictions"""
    home = request.args.get('home')
    away = request.args.get('away')
    league = request.args.get('league', 'default')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    goals = goals_predictor.predict_goals(home, away, league)
    
    return jsonify({
        'success': True,
        'home_team': home,
        'away_team': away,
        'league': league,
        'goals': goals.to_dict()
    })


@app.route('/api/standings')
def get_standings():
    """API endpoint for league standings"""
    league = request.args.get('league', 'bundesliga')
    
    try:
        openliga = OpenLigaDBClient()
        standings = openliga.get_standings(league)
        
        return jsonify({
            'success': True,
            'league': league,
            'standings': standings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/accumulator')
def get_accumulator():
    """API endpoint to get suggested accumulator"""
    leagues = request.args.getlist('league') or ['bundesliga']
    num_legs = int(request.args.get('legs', 3))
    min_prob = float(request.args.get('min_prob', 0.55))
    
    try:
        # Get fixtures for selected leagues
        matches = data_aggregator.get_upcoming_matches(leagues, days=7)
        
        # Generate predictions
        predictions = []
        for match in matches[:20]:  # Limit to avoid overload
            if match.status == 'scheduled':
                pred = predictor.predict_match(
                    match.home_team.name, match.away_team.name, match.league
                )
                goals = goals_predictor.predict_goals(
                    match.home_team.name, match.away_team.name, match.league
                )
                predictions.append({
                    'match': match.to_dict(),
                    'prediction': pred.to_dict(),
                    'goals': goals.to_dict()
                })
        
        # Build value accumulator
        acca = acca_builder.suggest_value_acca(predictions, num_legs, min_prob)
        
        if acca:
            return jsonify({
                'success': True,
                'accumulator': acca.to_dict()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Not enough value selections found'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/leagues')
def get_leagues():
    """API endpoint to list available leagues"""
    leagues = [
        {'id': 'bundesliga', 'name': 'Bundesliga', 'country': '🇩🇪', 'source': 'Free', 'active': True},
        {'id': 'bundesliga2', 'name': '2. Bundesliga', 'country': '🇩🇪', 'source': 'Free', 'active': True},
        {'id': '3liga', 'name': '3. Liga', 'country': '🇩🇪', 'source': 'Free', 'active': True},
        {'id': 'dfb_pokal', 'name': 'DFB-Pokal', 'country': '🇩🇪', 'source': 'Free', 'active': True},
        {'id': 'champions_league', 'name': 'Champions League', 'country': '🌍', 'source': 'Free', 'active': True},
        {'id': 'europa_league', 'name': 'Europa League', 'country': '🌍', 'source': 'Free', 'active': True},
        {'id': 'premier_league', 'name': 'Premier League', 'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'source': 'API Key ✅', 'active': True},
        {'id': 'la_liga', 'name': 'La Liga', 'country': '🇪🇸', 'source': 'API Key ✅', 'active': True},
        {'id': 'serie_a', 'name': 'Serie A', 'country': '🇮🇹', 'source': 'API Key ✅', 'active': True},
        {'id': 'ligue_1', 'name': 'Ligue 1', 'country': '🇫🇷', 'source': 'API Key ✅', 'active': True},
        {'id': 'eredivisie', 'name': 'Eredivisie', 'country': '🇳🇱', 'source': 'API Key ✅', 'active': True},
    ]
    return jsonify({'leagues': leagues})


@app.route('/api/accuracy')
def get_accuracy_stats():
    """Get prediction accuracy statistics"""
    stats = accuracy_tracker.get_accuracy_stats()
    return jsonify({
        'success': True,
        'stats': stats
    })


@app.route('/api/live')
def get_live_scores():
    """Get live match scores"""
    league = request.args.get('league', 'bundesliga')
    scores = live_data.get_live_scores(league)
    return jsonify({
        'success': True,
        'count': len(scores),
        'live_matches': scores
    })


@app.route('/api/injuries')
def get_injuries():
    """Get player injuries for a match"""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    injuries = live_data.get_key_absences(home, away)
    return jsonify({
        'success': True,
        'injuries': injuries
    })


@app.route('/api/weather')
def get_weather():
    """Get weather for match venue"""
    team = request.args.get('team', '')
    
    if not team:
        return jsonify({'error': 'Missing team parameter'}), 400
    
    weather = live_data.get_weather_for_match(team)
    
    if weather:
        return jsonify({
            'success': True,
            'weather': {
                'temperature': weather.temperature,
                'condition': weather.condition,
                'humidity': weather.humidity,
                'wind_speed': weather.wind_speed,
                'affects_play': weather.affects_play
            }
        })
    
    return jsonify({
        'success': False,
        'error': 'Weather data not available'
    })


@app.route('/api/patterns')
def get_patterns():
    """Get high-confidence betting patterns"""
    patterns = pattern_detector.find_high_confidence_patterns()
    return jsonify({
        'success': True,
        'patterns': patterns
    })


@app.route('/api/ml-predict')
def ml_prediction():
    """Get ML ensemble prediction"""
    home = request.args.get('home')
    away = request.args.get('away')
    league = request.args.get('league', 'default')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    result = ml_predictor.predict(home, away, league)
    
    return jsonify({
        'success': True,
        'home_team': home,
        'away_team': away,
        'prediction': result
    })


@app.route('/api/advanced-predict')
def advanced_prediction():
    """Get advanced prediction with all factors"""
    home = request.args.get('home')
    away = request.args.get('away')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    # Get base prediction from ML model
    ml_result = ml_predictor.predict(home, away)
    base_pred = ml_result.get('ensemble', {})
    
    # Get advanced prediction
    pred = advanced_predictor.predict(home, away, base_pred)
    
    return jsonify({
        'success': True,
        'home_team': home,
        'away_team': away,
        'prediction': {
            'home_win_prob': pred.home_win_prob,
            'draw_prob': pred.draw_prob,
            'away_win_prob': pred.away_win_prob,
            'predicted_outcome': pred.predicted_outcome,
            'raw_confidence': pred.raw_confidence,
            'calibrated_confidence': pred.calibrated_confidence,
        },
        'factors': pred.factors,
        'recommendations': pred.recommendations,
        'value_bets': pred.value_bets
    })


@app.route('/api/odds')
def get_odds_comparison():
    """Compare odds across bookmakers"""
    home = request.args.get('home')
    away = request.args.get('away')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    result = odds_comparer.get_best_odds(home, away)
    
    return jsonify({
        'success': True,
        'comparison': result
    })


@app.route('/api/arbitrage')
def check_arbitrage():
    """Check for arbitrage opportunities"""
    home = request.args.get('home')
    away = request.args.get('away')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    arb = arbitrage_finder.check_arbitrage(home, away)
    
    if arb:
        return jsonify({
            'success': True,
            'arbitrage_found': True,
            'opportunity': {
                'match': arb.match,
                'profit_percent': arb.profit_percent,
                'stakes': arb.stakes,
                'bookmakers': arb.bookmakers,
                'odds': arb.odds,
                'total_stake': arb.total_stake,
                'guaranteed_return': arb.guaranteed_return
            }
        })
    
    return jsonify({
        'success': True,
        'arbitrage_found': False,
        'message': 'No arbitrage opportunity found'
    })


@app.route('/api/value-bets')
def find_value_bets():
    """Find value betting opportunities"""
    home = request.args.get('home')
    away = request.args.get('away')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    # Get our probability estimates
    ml_result = ml_predictor.predict(home, away)
    our_probs = {
        'home': ml_result['ensemble']['home_win_prob'],
        'draw': ml_result['ensemble']['draw_prob'],
        'away': ml_result['ensemble']['away_win_prob']
    }
    
    value_bets = value_finder.find_value_bets(home, away, our_probs)
    
    return jsonify({
        'success': True,
        'our_probabilities': our_probs,
        'value_bets': [
            {
                'selection': b.selection,
                'our_probability': b.our_probability,
                'bookmaker_probability': b.bookmaker_probability,
                'edge': b.edge,
                'odds': b.odds,
                'bookmaker': b.bookmaker,
                'kelly_stake': b.kelly_stake
            }
            for b in value_bets
        ]
    })


@app.route('/api/kelly')
def calculate_kelly():
    """Calculate Kelly Criterion bet size"""
    prob = float(request.args.get('prob', 0.6))
    odds = float(request.args.get('odds', 1.9))
    bankroll = float(request.args.get('bankroll', 100))
    
    kelly_calc.bankroll = bankroll
    result = kelly_calc.calculate_kelly(prob, odds)
    
    return jsonify({
        'success': True,
        'kelly': result
    })


@app.route('/api/backtest')
def run_backtest():
    """Run backtest on historical predictions"""
    # Get completed predictions
    completed = [p for p in accuracy_tracker.predictions if p.is_correct is not None]
    
    if len(completed) < 5:
        return jsonify({
            'success': False,
            'error': 'Not enough historical data for backtest (need 5+ completed predictions)',
            'tip': 'Track predictions first using /api/track endpoint'
        })
    
    # Convert to dict format for backtester
    predictions = [
        {
            'predicted_outcome': p.predicted_outcome,
            'predicted_prob': p.predicted_prob,
            'actual_outcome': p.actual_outcome,
            'odds': 1 / p.predicted_prob * 0.95  # Approximate odds
        }
        for p in completed
    ]
    
    result = backtester.simulate_strategies(predictions)
    
    return jsonify({
        'success': True,
        'backtest': result
    })


@app.route('/api/track', methods=['POST'])
def track_prediction():
    """Track a prediction for accuracy logging"""
    data = request.get_json() or {}
    
    match_id = data.get('match_id', f"manual_{datetime.now().timestamp()}")
    home_team = data.get('home_team', 'Unknown')
    away_team = data.get('away_team', 'Unknown')
    league = data.get('league', 'unknown')
    predicted_outcome = data.get('predicted', 'H')
    predicted_prob = float(data.get('probability', 0.5))
    confidence = float(data.get('confidence', 0.5))
    bet_type = data.get('bet_type', 'match')
    
    record = accuracy_tracker.record_prediction(
        match_id, home_team, away_team, league,
        predicted_outcome, predicted_prob, confidence, bet_type
    )
    
    return jsonify({
        'success': True,
        'message': 'Prediction tracked',
        'record_id': match_id
    })


@app.route('/api/result', methods=['POST'])
def record_result():
    """Record the actual result of a tracked prediction"""
    data = request.get_json() or {}
    
    match_id = data.get('match_id')
    actual_outcome = data.get('actual')
    
    if not match_id or not actual_outcome:
        return jsonify({'error': 'Missing match_id or actual outcome'}), 400
    
    accuracy_tracker.record_result(match_id, actual_outcome)
    
    return jsonify({
        'success': True,
        'message': f'Result recorded for {match_id}'
    })


@app.route('/api/h2h')
def get_h2h():
    """Get head-to-head data between two teams"""
    home = request.args.get('home')
    away = request.args.get('away')
    
    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    
    h2h_data = h2h_analyzer.get_full_h2h(home, away)
    
    return jsonify({
        'success': True,
        **h2h_data
    })


if __name__ == '__main__':
    print("=" * 60)
    print("⚽ Football Prediction System - Complete Edition")
    print("=" * 60)
    print()
    print("Starting server at http://localhost:5000")
    print()
    print("Core Features:")
    print("  ✅ 11 Leagues | ✅ ML Predictions | ✅ Goal Predictions")
    print("  ✅ Accumulators | ✅ Kelly Criterion | ✅ Value Bets")
    print("  ✅ Odds Comparison | ✅ Arbitrage Finder")
    print("  ✅ Dashboard | ✅ PWA Mobile App")
    print("  ✅ Telegram Bot | ✅ WhatsApp Bot")
    print()
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    print("=" * 60)
    print()
    print("Starting server at http://localhost:5000")
    print()
    print("API Endpoints:")
    print("  GET /api/fixtures?league=bundesliga&days=7")
    print("  GET /api/predict?home=Bayern&away=Dortmund&league=bundesliga")
    print("  GET /api/standings?league=bundesliga")
    print("  GET /api/leagues")
    print()
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
