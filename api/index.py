"""
Vercel Serverless Function Entry Point

This file adapts Flask to work with Vercel's serverless functions.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app

# Vercel expects the WSGI app to be named 'app' or 'handler'
# For Flask, we export the app directly

def handler(request):
    """Handle incoming requests for Vercel"""
    return app(request.environ, request.start_response)

# Export for Vercel
app = app
