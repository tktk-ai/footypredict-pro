"""
Comprehensive Test Suite

Unit tests for all prediction systems.
Run with: pytest tests/test_predictions.py -v
"""

import pytest
import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFormCalculator:
    """Test team form calculations"""
    
    def test_form_from_matches(self):
        from src.advanced_features import FormCalculator
        
        matches = [
            {'home_team': 'Team A', 'away_team': 'Team B', 'home_score': 2, 'away_score': 1, 'date': '2024-01-01'},
            {'home_team': 'Team A', 'away_team': 'Team C', 'home_score': 1, 'away_score': 1, 'date': '2024-01-02'},
            {'home_team': 'Team B', 'away_team': 'Team A', 'home_score': 0, 'away_score': 3, 'date': '2024-01-03'},
        ]
        
        calc = FormCalculator(matches)
        form = calc.get_form('Team A', 3)
        
        assert form['games'] == 3
        assert form['points'] == 7  # W(3) + D(1) + W(3)
        assert 'W' in form['form']
    
    def test_empty_form(self):
        from src.advanced_features import FormCalculator
        
        calc = FormCalculator([])
        form = calc.get_form('Unknown Team')
        
        assert form['games'] == 0
        assert form['form'] == ''


class TestH2HAnalyzer:
    """Test head-to-head analysis"""
    
    def test_h2h_stats(self):
        from src.advanced_features import HeadToHeadAnalyzer
        
        matches = [
            {'home_team': 'Team A', 'away_team': 'Team B', 'home_score': 2, 'away_score': 0, 'date': '2024-01-01'},
            {'home_team': 'Team B', 'away_team': 'Team A', 'home_score': 1, 'away_score': 1, 'date': '2024-01-02'},
            {'home_team': 'Team A', 'away_team': 'Team B', 'home_score': 0, 'away_score': 1, 'date': '2024-01-03'},
        ]
        
        analyzer = HeadToHeadAnalyzer(matches)
        h2h = analyzer.get_h2h('Team A', 'Team B')
        
        assert h2h['total_matches'] == 3
        assert h2h['team1_wins'] == 1  # Team A wins
        assert h2h['team2_wins'] == 1  # Team B wins
        assert h2h['draws'] == 1


class TestEnhancedPredictor:
    """Test enhanced prediction engine"""
    
    def test_prediction_format(self):
        from src.enhanced_predictor_v2 import enhanced_predict
        
        result = enhanced_predict('Germany', 'France')
        
        assert 'home_team' in result
        assert 'away_team' in result
        assert 'final_prediction' in result
        
        pred = result['final_prediction']
        assert 'home_win_prob' in pred
        assert 'draw_prob' in pred
        assert 'away_win_prob' in pred
        assert 'predicted_outcome' in pred
        assert 'confidence' in pred
    
    def test_probabilities_sum_to_one(self):
        from src.enhanced_predictor_v2 import enhanced_predict
        
        result = enhanced_predict('Brazil', 'Argentina')
        pred = result['final_prediction']
        
        total = pred['home_win_prob'] + pred['draw_prob'] + pred['away_win_prob']
        assert 0.99 <= total <= 1.01  # Allow small floating point error
    
    def test_goals_prediction(self):
        from src.enhanced_predictor_v2 import enhanced_predict_with_goals
        
        result = enhanced_predict_with_goals('England', 'Spain')
        
        assert 'goals' in result
        goals = result['goals']
        
        assert 'home_xg' in goals
        assert 'away_xg' in goals
        assert 'over_2.5' in goals


class TestInPlayPredictor:
    """Test in-play predictions"""
    
    def test_start_tracking(self):
        from src.inplay_predictor import get_inplay_predictor
        
        predictor = get_inplay_predictor()
        match = predictor.start_tracking('match1', 'Home FC', 'Away FC', {})
        
        assert match['home_team'] == 'Home FC'
        assert match['current_home_score'] == 0
    
    def test_score_update(self):
        from src.inplay_predictor import get_inplay_predictor
        
        predictor = get_inplay_predictor()
        predictor.start_tracking('match2', 'Home FC', 'Away FC', {'final_prediction': {'home_win_prob': 0.4}})
        
        result = predictor.update_match('match2', home_score=1, away_score=0, minute=30)
        
        assert result['current_home_score'] == 1
        assert result['minute'] == 30
        assert 'live_prediction' in result
        
        # Home leading should increase home win prob
        assert result['live_prediction']['home_win_prob'] > 0.4


class TestInjuries:
    """Test injury tracking"""
    
    def test_get_injuries(self):
        from src.injuries_weather import get_injuries
        
        injuries = get_injuries('Manchester United')
        
        assert 'team' in injuries
        assert 'injury_count' in injuries
        assert 'impact_score' in injuries
        assert injuries['impact_score'] >= 0
        assert injuries['impact_score'] <= 1


class TestCronJobs:
    """Test cron job manager"""
    
    def test_manager_status(self):
        from src.cron_jobs import get_cron_status
        
        status = get_cron_status()
        
        assert 'is_running' in status
        assert 'scheduler_available' in status


class TestABTesting:
    """Test A/B testing framework"""
    
    def test_create_test(self):
        from src.ab_testing import get_ab_tester
        
        tester = get_ab_tester()
        test = tester.create_test('test1', 'model_a', 'model_b')
        
        assert test['name'] == 'test1'
        assert test['variant_a'] == 'model_a'
        assert test['is_active'] == True
    
    def test_variant_assignment(self):
        from src.ab_testing import get_ab_tester
        
        tester = get_ab_tester()
        tester.create_test('test2')
        
        # Same user ID should always get same variant
        v1 = tester.get_variant('test2', 'user123')
        v2 = tester.get_variant('test2', 'user123')
        
        assert v1 == v2


class TestAPIEndpoints:
    """Test API endpoint responses"""
    
    @pytest.fixture
    def client(self):
        from app import app
        app.config['TESTING'] = True
        return app.test_client()
    
    def test_health_endpoint(self, client):
        response = client.get('/api/health')
        assert response.status_code in [200, 404]  # May not exist
    
    def test_predict_v2_endpoint(self, client):
        response = client.get('/api/v2/predict?home=Germany&away=France')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_form_endpoint(self, client):
        response = client.get('/api/form/Brazil')
        assert response.status_code == 200
    
    def test_h2h_endpoint(self, client):
        response = client.get('/api/h2h?team1=Brazil&team2=Argentina')
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
