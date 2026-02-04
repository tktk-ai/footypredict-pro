"""
Advanced API v5 - Cutting Edge Features

Integrates all advanced features:
- AI Sentiment Analysis
- Pattern Recognition
- Smart Bankroll
- AI Assistant
- Live Betting
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from typing import Dict, List

# Import advanced modules
from src.ai_sentiment import (
    analyze_match_sentiment, get_smart_advice,
    sentiment_analyzer, market_tracker
)
from src.pattern_recognition import (
    detect_patterns, check_anomalies, predict_exact_score,
    pattern_engine, anomaly_detector
)
from src.smart_bankroll import (
    calculate_optimal_stake, get_bankroll_status,
    record_bet_result, default_bankroll
)
from src.ai_assistant import chat, get_chat_history, ai_assistant
from src.live_predictor import (
    register_live_match, update_live_match,
    get_live_prediction, get_all_live_predictions,
    find_live_value_bets, live_betting_manager
)


# Create blueprint
advanced_api = Blueprint('advanced_api', __name__, url_prefix='/api/v5')


# ============================================================
# AI Sentiment Analysis Endpoints
# ============================================================

@advanced_api.route('/sentiment/match')
def match_sentiment():
    """Get sentiment analysis for a match"""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing home or away team'})
    
    # For demo, use sample news
    sample_news = [
        f"{home} looking confident ahead of big match",
        f"{home} key players fit and available",
        f"Manager praises {home} form in training",
        f"{away} struggling with injuries",
        f"Questions over {away} morale after poor results",
        f"{away} fans concerned about team performance"
    ]
    
    sentiment = analyze_match_sentiment(home, away, sample_news)
    
    return jsonify({
        'success': True,
        'sentiment': sentiment,
        'news_analyzed': len(sample_news)
    })


@advanced_api.route('/sentiment/advice')
def smart_advice():
    """Get AI-powered betting advice"""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing teams'})
    
    try:
        from src.enhanced_predictor_v2 import enhanced_predict
        
        prediction = enhanced_predict(home, away, 'bundesliga')
        advice = get_smart_advice(
            {'home': home, 'away': away},
            prediction
        )
        
        return jsonify({
            'success': True,
            'advice': advice
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Pattern Recognition Endpoints
# ============================================================

@advanced_api.route('/patterns/<team>')
def team_patterns(team: str):
    """Get detected patterns for a team"""
    patterns = detect_patterns(team)
    
    return jsonify({
        'success': True,
        'team': team,
        'patterns': patterns,
        'count': len(patterns)
    })


@advanced_api.route('/patterns/anomalies')
def check_anomalies_endpoint():
    """Check for anomalies in odds"""
    match_id = request.args.get('match_id', '')
    home_odds = float(request.args.get('home', 2.0))
    draw_odds = float(request.args.get('draw', 3.5))
    away_odds = float(request.args.get('away', 3.0))
    
    odds = {'home': home_odds, 'draw': draw_odds, 'away': away_odds}
    anomaly = check_anomalies(match_id, odds)
    
    return jsonify({
        'success': True,
        'match_id': match_id,
        'anomaly': anomaly,
        'has_anomaly': anomaly is not None
    })


@advanced_api.route('/patterns/score')
def exact_score():
    """Predict exact score"""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing teams'})
    
    prediction = predict_exact_score(home, away)
    
    return jsonify({
        'success': True,
        'prediction': prediction
    })


# ============================================================
# Smart Bankroll Endpoints
# ============================================================

@advanced_api.route('/bankroll/status')
def bankroll_status():
    """Get bankroll status"""
    status = get_bankroll_status()
    return jsonify({
        'success': True,
        'status': status
    })


@advanced_api.route('/bankroll/stake')
def calculate_stake():
    """Calculate optimal stake for a bet"""
    probability = float(request.args.get('probability', 0.5))
    odds = float(request.args.get('odds', 2.0))
    bankroll = float(request.args.get('bankroll', 0)) or None
    risk_level = request.args.get('risk', 'moderate')
    
    recommendation = calculate_optimal_stake(probability, odds, bankroll, risk_level)
    
    return jsonify({
        'success': True,
        'recommendation': recommendation
    })


@advanced_api.route('/bankroll/record', methods=['POST'])
def record_result():
    """Record a bet result"""
    data = request.get_json() or {}
    stake = float(data.get('stake', 10))
    odds = float(data.get('odds', 2.0))
    won = data.get('won', False)
    
    record_bet_result(stake, odds, won)
    
    return jsonify({
        'success': True,
        'new_status': get_bankroll_status()
    })


@advanced_api.route('/bankroll/portfolio', methods=['POST'])
def optimize_portfolio():
    """Optimize stake allocation across multiple bets"""
    data = request.get_json() or {}
    bets = data.get('bets', [])
    
    if not bets:
        return jsonify({'success': False, 'error': 'No bets provided'})
    
    allocation = default_bankroll.get_portfolio_allocation(bets)
    
    return jsonify({
        'success': True,
        'allocation': allocation,
        'total_stake': sum(b['recommended_stake'] for b in allocation)
    })


# ============================================================
# AI Assistant Endpoints
# ============================================================

@advanced_api.route('/chat', methods=['POST'])
def chat_endpoint():
    """Chat with AI betting assistant"""
    data = request.get_json() or {}
    message = data.get('message', '')
    user_id = data.get('user_id', 'default')
    
    if not message:
        return jsonify({'success': False, 'error': 'No message provided'})
    
    response = chat(message, user_id)
    
    return jsonify({
        'success': True,
        'response': response
    })


@advanced_api.route('/chat/history')
def chat_history():
    """Get chat history"""
    user_id = request.args.get('user_id', 'default')
    limit = int(request.args.get('limit', 10))
    
    history = get_chat_history(user_id, limit)
    
    return jsonify({
        'success': True,
        'history': history
    })


# ============================================================
# Live Betting Endpoints
# ============================================================

@advanced_api.route('/live/register', methods=['POST'])
def register_match():
    """Register a match for live tracking"""
    data = request.get_json() or {}
    home = data.get('home', '')
    away = data.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing teams'})
    
    match_id = register_live_match(home, away)
    
    return jsonify({
        'success': True,
        'match_id': match_id
    })


@advanced_api.route('/live/update', methods=['POST'])
def update_match():
    """Update live match state"""
    data = request.get_json() or {}
    match_id = data.get('match_id', '')
    updates = data.get('updates', {})
    
    if not match_id:
        return jsonify({'success': False, 'error': 'Missing match_id'})
    
    prediction = update_live_match(match_id, **updates)
    
    if prediction:
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    else:
        return jsonify({'success': False, 'error': 'Match not found'})


@advanced_api.route('/live/prediction/<match_id>')
def live_prediction(match_id: str):
    """Get live prediction for a match"""
    prediction = get_live_prediction(match_id)
    
    if prediction:
        return jsonify({
            'success': True,
            **prediction
        })
    else:
        return jsonify({'success': False, 'error': 'Match not found'})


@advanced_api.route('/live/all')
def all_live():
    """Get all live matches with predictions"""
    matches = get_all_live_predictions()
    
    return jsonify({
        'success': True,
        'matches': matches,
        'count': len(matches)
    })


@advanced_api.route('/live/value')
def live_value_bets():
    """Find live value betting opportunities"""
    min_edge = float(request.args.get('min_edge', 0.1))
    
    opportunities = find_live_value_bets(min_edge)
    
    return jsonify({
        'success': True,
        'opportunities': opportunities,
        'count': len(opportunities)
    })


# ============================================================
# Combined Analysis Endpoints
# ============================================================

@advanced_api.route('/analyze')
def full_analysis():
    """Get complete analysis for a match"""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    
    if not home or not away:
        return jsonify({'success': False, 'error': 'Missing teams'})
    
    try:
        from src.enhanced_predictor_v2 import enhanced_predict_with_goals
        
        # Get prediction
        prediction = enhanced_predict_with_goals(home, away, 'bundesliga')
        
        # Get patterns
        home_patterns = detect_patterns(home)
        away_patterns = detect_patterns(away)
        
        # Get score prediction
        score = predict_exact_score(home, away)
        
        # Get sentiment
        sentiment = analyze_match_sentiment(home, away)
        
        # Get advice
        advice = get_smart_advice({'home': home, 'away': away}, prediction)
        
        # Calculate stake
        stake = calculate_optimal_stake(
            prediction.get('confidence', 0.5),
            prediction.get('odds', {}).get('home', 2.0)
        )
        
        return jsonify({
            'success': True,
            'analysis': {
                'match': f"{home} vs {away}",
                'prediction': prediction,
                'patterns': {
                    'home': home_patterns,
                    'away': away_patterns
                },
                'exact_score': score,
                'sentiment': sentiment,
                'advice': advice,
                'stake_recommendation': stake,
                'generated_at': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# Health & Info
# ============================================================

@advanced_api.route('/health')
def health():
    """API health check"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'version': 'v5',
        'features': [
            'ai_sentiment_analysis',
            'pattern_recognition',
            'anomaly_detection',
            'smart_bankroll',
            'kelly_criterion',
            'ai_assistant',
            'live_betting',
            'momentum_analysis',
            'exact_score_prediction',
            'portfolio_optimization'
        ],
        'timestamp': datetime.now().isoformat()
    })


@advanced_api.route('/info')
def info():
    """API information"""
    return jsonify({
        'success': True,
        'api': {
            'name': 'FootyPredict Pro Advanced API',
            'version': 'v5.0.0',
            'base_url': '/api/v5',
            'categories': {
                'sentiment': ['/sentiment/match', '/sentiment/advice'],
                'patterns': ['/patterns/<team>', '/patterns/anomalies', '/patterns/score'],
                'bankroll': ['/bankroll/status', '/bankroll/stake', '/bankroll/portfolio'],
                'assistant': ['/chat', '/chat/history'],
                'live': ['/live/register', '/live/update', '/live/prediction', '/live/value'],
                'analysis': ['/analyze']
            }
        }
    })


def register_advanced_api(app):
    """Register the advanced API blueprint"""
    app.register_blueprint(advanced_api)
    print("🚀 Advanced API v5 registered at /api/v5")
