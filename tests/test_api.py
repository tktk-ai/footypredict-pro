"""
API endpoint tests for the Football Prediction System

Tests for Flask API endpoints.
"""

import pytest
import sys
import os

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the Flask app
from app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    """Tests for API endpoints"""
    
    def test_home_page_loads(self, client):
        """Home page should return 200"""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_dashboard_page_loads(self, client):
        """Dashboard page should return 200"""
        response = client.get('/dashboard')
        assert response.status_code == 200
    
    def test_leagues_endpoint(self, client):
        """Leagues endpoint should return list of leagues"""
        response = client.get('/api/leagues')
        data = response.get_json()
        
        assert response.status_code == 200
        assert 'leagues' in data
        assert len(data['leagues']) > 0
        
        # Check league structure
        league = data['leagues'][0]
        assert 'id' in league
        assert 'name' in league
        assert 'country' in league
    
    def test_predict_endpoint_without_params(self, client):
        """Predict endpoint should require home and away params"""
        response = client.get('/api/predict')
        
        assert response.status_code == 400
        assert 'error' in response.get_json()
    
    def test_predict_endpoint_with_params(self, client):
        """Predict endpoint should return valid prediction"""
        response = client.get('/api/predict?home=Bayern&away=Dortmund')
        data = response.get_json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert 'prediction' in data
        assert 'home_win_prob' in data['prediction']
        assert 'draw_prob' in data['prediction']
        assert 'away_win_prob' in data['prediction']


class TestH2HEndpoint:
    """Tests for the H2H API endpoint"""
    
    def test_h2h_without_params(self, client):
        """H2H endpoint should require home and away params"""
        response = client.get('/api/h2h')
        
        assert response.status_code == 400
        assert 'error' in response.get_json()
    
    def test_h2h_known_matchup(self, client):
        """H2H should return data for known matchup"""
        response = client.get('/api/h2h?home=Bayern&away=Dortmund')
        data = response.get_json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert data['found'] == True
        assert 'total_matches' in data
        assert 'record' in data
        assert 'goals' in data
        assert 'last_5_matches' in data
    
    def test_h2h_unknown_matchup(self, client):
        """H2H should return found=False for unknown matchup"""
        response = client.get('/api/h2h?home=UnknownTeam1&away=UnknownTeam2')
        data = response.get_json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert data['found'] == False
    
    def test_h2h_record_structure(self, client):
        """H2H record should have correct structure"""
        response = client.get('/api/h2h?home=Liverpool&away=Manchester%20City')
        data = response.get_json()
        
        if data.get('found'):
            record = data['record']
            assert 'home_wins' in record
            assert 'draws' in record
            assert 'away_wins' in record
            assert 'home_win_pct' in record


class TestGoalsEndpoint:
    """Tests for goal prediction endpoint"""
    
    def test_goals_endpoint_without_params(self, client):
        """Goals endpoint should require home and away params"""
        response = client.get('/api/goals')
        
        assert response.status_code == 400
    
    def test_goals_endpoint_with_params(self, client):
        """Goals endpoint should return valid prediction"""
        response = client.get('/api/goals?home=Bayern&away=Dortmund')
        data = response.get_json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert 'goals' in data


class TestAdvancedEndpoints:
    """Tests for advanced prediction endpoints"""
    
    def test_ml_predict_endpoint(self, client):
        """ML predict endpoint should work"""
        response = client.get('/api/ml-predict?home=Bayern&away=Dortmund')
        data = response.get_json()
        
        assert response.status_code == 200
        assert data['success'] == True
    
    def test_advanced_predict_endpoint(self, client):
        """Advanced predict endpoint should return comprehensive prediction"""
        response = client.get('/api/advanced-predict?home=Bayern&away=Dortmund')
        data = response.get_json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert 'prediction' in data
        assert 'factors' in data
        assert 'recommendations' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
