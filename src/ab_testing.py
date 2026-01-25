"""
A/B Testing Framework

Compare v1 vs v2 predictor accuracy on historical data.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import random

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
AB_RESULTS_DIR = DATA_DIR / "ab_tests"
AB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ABTester:
    """A/B testing for prediction models"""
    
    def __init__(self):
        self.tests: Dict[str, Dict] = {}
        self.current_variant = 'A'  # Default to v1
    
    def create_test(self, test_name: str, variant_a: str = 'v1', variant_b: str = 'v2'):
        """Create a new A/B test"""
        self.tests[test_name] = {
            'name': test_name,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'results_a': {'predictions': 0, 'correct': 0},
            'results_b': {'predictions': 0, 'correct': 0},
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }
        return self.tests[test_name]
    
    def get_variant(self, test_name: str, user_id: str = None) -> str:
        """Get variant for a user (deterministic based on user ID)"""
        if test_name not in self.tests:
            return 'A'
        
        if user_id:
            # Consistent assignment based on user ID
            return 'A' if hash(user_id) % 2 == 0 else 'B'
        else:
            # Random assignment
            return random.choice(['A', 'B'])
    
    def record_prediction(self, test_name: str, variant: str, correct: bool):
        """Record a prediction result"""
        if test_name not in self.tests:
            return
        
        key = 'results_a' if variant == 'A' else 'results_b'
        self.tests[test_name][key]['predictions'] += 1
        if correct:
            self.tests[test_name][key]['correct'] += 1
    
    def get_results(self, test_name: str) -> Dict:
        """Get A/B test results"""
        if test_name not in self.tests:
            return {'error': 'Test not found'}
        
        test = self.tests[test_name]
        
        a_preds = test['results_a']['predictions']
        a_correct = test['results_a']['correct']
        b_preds = test['results_b']['predictions']
        b_correct = test['results_b']['correct']
        
        a_acc = a_correct / a_preds if a_preds > 0 else 0
        b_acc = b_correct / b_preds if b_preds > 0 else 0
        
        # Simple significance test (needs more data in practice)
        winner = None
        if a_preds >= 100 and b_preds >= 100:
            diff = abs(a_acc - b_acc)
            if diff > 0.05:  # 5% difference threshold
                winner = 'A' if a_acc > b_acc else 'B'
        
        return {
            'test_name': test_name,
            'variant_a': test['variant_a'],
            'variant_b': test['variant_b'],
            'results': {
                'A': {'predictions': a_preds, 'correct': a_correct, 'accuracy': round(a_acc, 4)},
                'B': {'predictions': b_preds, 'correct': b_correct, 'accuracy': round(b_acc, 4)}
            },
            'winner': winner,
            'improvement': round((b_acc - a_acc) * 100, 2) if a_acc > 0 else None,
            'is_significant': winner is not None
        }
    
    def run_historical_test(self, test_name: str = 'v1_vs_v2') -> Dict:
        """Run A/B test on historical data"""
        self.create_test(test_name, 'predictor_v1', 'enhanced_predictor_v2')
        
        # Load historical matches
        try:
            import pandas as pd
            data_file = DATA_DIR / "training_data.csv"
            if not data_file.exists():
                return {'error': 'No training data available'}
            
            df = pd.read_csv(data_file)
            df = df.dropna(subset=['home_score', 'away_score'])
            
            # Use last 1000 matches for testing
            test_df = df.tail(1000)
            
            # Import predictors
            try:
                from src.predictor import PredictionEngine
                from src.enhanced_predictor_v2 import get_enhanced_predictor
                
                v1 = PredictionEngine()
                v2 = get_enhanced_predictor()
            except Exception as e:
                logger.error(f"Could not load predictors: {e}")
                return {'error': str(e)}
            
            for _, row in test_df.iterrows():
                home = row['home_team']
                away = row['away_team']
                
                # Actual result
                if row['home_score'] > row['away_score']:
                    actual = 'Home Win'
                elif row['home_score'] < row['away_score']:
                    actual = 'Away Win'
                else:
                    actual = 'Draw'
                
                # V1 prediction
                try:
                    v1_pred = v1.predict_match({'home_team': {'name': home}, 'away_team': {'name': away}})
                    v1_outcome = v1_pred.get('prediction', {}).get('predicted_outcome', '')
                    self.record_prediction(test_name, 'A', v1_outcome == actual)
                except:
                    pass
                
                # V2 prediction  
                try:
                    v2_pred = v2.predict(home, away)
                    v2_outcome = v2_pred.get('final_prediction', {}).get('predicted_outcome', '')
                    self.record_prediction(test_name, 'B', v2_outcome == actual)
                except:
                    pass
            
            # Save results
            results = self.get_results(test_name)
            with open(AB_RESULTS_DIR / f"{test_name}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            logger.error(f"A/B test error: {e}")
            return {'error': str(e)}


# Global instance
_tester = None

def get_ab_tester() -> ABTester:
    global _tester
    if _tester is None:
        _tester = ABTester()
    return _tester

def run_ab_test(test_name: str = 'v1_vs_v2'):
    return get_ab_tester().run_historical_test(test_name)

def get_ab_results(test_name: str):
    return get_ab_tester().get_results(test_name)
