"""
Model Combiner
Combines predictions from multiple models.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ModelCombiner:
    """
    Combines predictions from multiple models.
    
    Strategies:
    - Simple averaging
    - Weighted averaging
    - Stacking ensemble
    - Voting
    """
    
    def __init__(
        self,
        strategy: str = 'weighted',
        weights: Dict[str, float] = None
    ):
        self.strategy = strategy
        self.weights = weights or {}
        self.models: Dict[str, Callable] = {}
        self.performance_history: Dict[str, List[float]] = {}
    
    def register_model(
        self,
        name: str,
        model: Callable,
        weight: float = 1.0
    ):
        """Register a model for the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        self.performance_history[name] = []
        logger.info(f"Registered model: {name} with weight {weight}")
    
    def combine_predictions(
        self,
        predictions: Dict[str, Dict],
        market: str = '1x2'
    ) -> Dict:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: Dict of model_name -> prediction_dict
            market: Market to combine (1x2, btts, over_under)
        """
        if not predictions:
            return {}
        
        if self.strategy == 'simple':
            return self._simple_average(predictions, market)
        elif self.strategy == 'weighted':
            return self._weighted_average(predictions, market)
        elif self.strategy == 'voting':
            return self._voting(predictions, market)
        else:
            return self._simple_average(predictions, market)
    
    def _simple_average(
        self,
        predictions: Dict[str, Dict],
        market: str
    ) -> Dict:
        """Simple average of predictions."""
        if market == '1x2':
            home = draw = away = 0
            count = 0
            
            for name, pred in predictions.items():
                if '1x2' in pred:
                    home += pred['1x2'].get('home', 0)
                    draw += pred['1x2'].get('draw', 0)
                    away += pred['1x2'].get('away', 0)
                    count += 1
            
            if count == 0:
                return {}
            
            return {
                '1x2': {
                    'home': round(home / count, 4),
                    'draw': round(draw / count, 4),
                    'away': round(away / count, 4)
                },
                'method': 'simple_average',
                'models_used': list(predictions.keys())
            }
        
        elif market == 'btts':
            btts_sum = 0
            count = 0
            
            for name, pred in predictions.items():
                if 'btts' in pred:
                    btts_sum += pred['btts']
                    count += 1
            
            return {
                'btts': round(btts_sum / count, 4) if count > 0 else 0.5,
                'method': 'simple_average'
            }
        
        return {}
    
    def _weighted_average(
        self,
        predictions: Dict[str, Dict],
        market: str
    ) -> Dict:
        """Weighted average of predictions."""
        # Normalize weights for available models
        available_weights = {
            name: self.weights.get(name, 1.0)
            for name in predictions.keys()
        }
        total_weight = sum(available_weights.values())
        
        if total_weight == 0:
            return self._simple_average(predictions, market)
        
        if market == '1x2':
            home = draw = away = 0
            
            for name, pred in predictions.items():
                if '1x2' in pred:
                    w = available_weights[name] / total_weight
                    home += pred['1x2'].get('home', 0) * w
                    draw += pred['1x2'].get('draw', 0) * w
                    away += pred['1x2'].get('away', 0) * w
            
            return {
                '1x2': {
                    'home': round(home, 4),
                    'draw': round(draw, 4),
                    'away': round(away, 4)
                },
                'method': 'weighted_average',
                'weights': available_weights,
                'models_used': list(predictions.keys())
            }
        
        elif market == 'btts':
            btts_weighted = 0
            
            for name, pred in predictions.items():
                if 'btts' in pred:
                    w = available_weights[name] / total_weight
                    btts_weighted += pred['btts'] * w
            
            return {
                'btts': round(btts_weighted, 4),
                'method': 'weighted_average'
            }
        
        return {}
    
    def _voting(
        self,
        predictions: Dict[str, Dict],
        market: str
    ) -> Dict:
        """Majority voting for classification."""
        if market == '1x2':
            votes = {'home': 0, 'draw': 0, 'away': 0}
            
            for name, pred in predictions.items():
                if '1x2' in pred:
                    probs = pred['1x2']
                    winner = max(probs, key=probs.get)
                    votes[winner] += self.weights.get(name, 1.0)
            
            total_votes = sum(votes.values())
            
            return {
                '1x2': {
                    'home': round(votes['home'] / total_votes, 4) if total_votes > 0 else 0.33,
                    'draw': round(votes['draw'] / total_votes, 4) if total_votes > 0 else 0.33,
                    'away': round(votes['away'] / total_votes, 4) if total_votes > 0 else 0.34
                },
                'method': 'voting',
                'votes': votes
            }
        
        return {}
    
    def update_weights_from_performance(
        self,
        model_name: str,
        accuracy: float
    ):
        """Update model weight based on performance."""
        self.performance_history[model_name].append(accuracy)
        
        # Use recent performance for weighting
        if len(self.performance_history[model_name]) >= 5:
            recent = self.performance_history[model_name][-10:]
            self.weights[model_name] = np.mean(recent)
        
        logger.info(f"Updated weight for {model_name}: {self.weights.get(model_name, 1.0):.3f}")
    
    def predict_with_all_models(
        self,
        features: Dict
    ) -> Dict:
        """Run all registered models and combine."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model(features)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        return self.combine_predictions(predictions)


_combiner: Optional[ModelCombiner] = None

def get_combiner() -> ModelCombiner:
    global _combiner
    if _combiner is None:
        _combiner = ModelCombiner()
    return _combiner
