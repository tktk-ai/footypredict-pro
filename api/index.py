"""
Vercel Serverless Function Entry Point

This file adapts Flask to work with Vercel's serverless functions.
Uses proper WSGI adapter for Vercel's Python runtime.
"""

from flask import Flask, render_template, jsonify
import os
import sys

# Simple Flask app for Vercel (minimal version to avoid crashes)
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Basic routes that work in serverless
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'platform': 'vercel'})

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/accumulators')
def accumulators():
    return render_template('accumulators.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/tracker')
def tracker():
    return render_template('tracker.html')

@app.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# API endpoints (simplified for serverless)
@app.route('/api/leagues')
def get_leagues():
    """Get available leagues"""
    leagues = [
        {'id': 'bundesliga', 'name': 'Bundesliga', 'country': 'Germany'},
        {'id': 'premier_league', 'name': 'Premier League', 'country': 'England'},
        {'id': 'la_liga', 'name': 'La Liga', 'country': 'Spain'},
        {'id': 'serie_a', 'name': 'Serie A', 'country': 'Italy'},
        {'id': 'ligue_1', 'name': 'Ligue 1', 'country': 'France'},
    ]
    return jsonify({'success': True, 'leagues': leagues})

@app.route('/api/fixtures')
def get_fixtures():
    """Get fixtures - returns empty for serverless, use Cloudflare Worker API for real data"""
    return jsonify({
        'success': True, 
        'fixtures': [],
        'message': 'Use Cloudflare Worker API at https://footypredict-api.tirene857.workers.dev/predictions for live predictions'
    })

@app.route('/api/pricing')
def get_pricing():
    """Get pricing info"""
    return jsonify({
        'success': True,
        'tiers': [
            {'id': 'free', 'name': 'Free', 'price': 0},
            {'id': 'pro', 'name': 'Pro', 'price': 9.99},
            {'id': 'premium', 'name': 'Premium', 'price': 24.99}
        ]
    })

@app.route('/api/accumulators')
def get_accumulators():
    """Get accumulators - returns empty for serverless, use Cloudflare Worker API for real data"""
    return jsonify({
        'success': True,
        'accumulators': {},
        'message': 'Use Cloudflare Worker API at https://footypredict-api.tirene857.workers.dev/accumulators for live accumulators'
    })

# Vercel needs the app to handle requests
if __name__ == '__main__':
    app.run(debug=True)
