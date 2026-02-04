"""
Model Ensemble - Combines multiple ML models for robust predictions

Weighted averaging of predictions from:
- Podos Transformer (30%)
- XGBoost (35%)
- FootballerModel (20%)
- LSTM Form (15%)

Includes confidence calibration and model disagreement detection.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Output from ensemble prediction"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str
    confidence: float
    model_agreement: float
    individual_predictions: Dict[str, Dict]
    calibrated: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelEnsemble:
    """
    Ensemble predictor combining multiple ML models.
    
    Uses weighted averaging with confidence calibration.
    """
    
    # Default model weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'podos': 0.30,
        'xgboost': 0.35,
        'footballer': 0.20,
        'lstm': 0.15
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble.
        
        Args:
            weights: Custom model weights. Default weights used if not provided.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.models: Dict[str, Any] = {}
        self._calibration_params: Dict[str, float] = {}
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration parameters if available"""
        config_path = Path(__file__).parent.parent.parent / "models" / "config" / "calibration.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self._calibration_params = json.load(f)
    
    def register_model(self, name: str, model: Any, weight: Optional[float] = None):
        """
        Register a model with the ensemble.
        
        Args:
            name: Model identifier
            model: Model object with predict() method
            weight: Model weight (optional, uses default if not specified)
        """
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
            self._normalize_weights()
        logger.info(f"Registered model: {name} (weight: {self.weights.get(name, 0)})")
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        # Only consider weights for registered models
        active_weights = {k: v for k, v in self.weights.items() if k in self.models}
        total = sum(active_weights.values())
        if total > 0:
            for k in active_weights:
                self.weights[k] = active_weights[k] / total
    
    def predict(self, home_team: str, away_team: str, **features) -> EnsemblePrediction:
        """
        Get ensemble prediction from all models.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            **features: Additional features (form, odds, etc.)
            
        Returns:
            EnsemblePrediction with combined probabilities
        """
        if not self.models:
            raise ValueError("No models registered in ensemble")
        
        individual_preds = {}
        weighted_probs = np.array([0.0, 0.0, 0.0])  # home, draw, away
        total_weight = 0.0
        
        # Get prediction from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(home_team, away_team, **features)
                
                # Extract probabilities
                if hasattr(pred, 'home_win_prob'):
                    probs = np.array([
                        pred.home_win_prob,
                        pred.draw_prob,
                        pred.away_win_prob
                    ])
                elif isinstance(pred, dict):
                    probs = np.array([
                        pred.get('home_win_prob', 0.33),
                        pred.get('draw_prob', 0.33),
                        pred.get('away_win_prob', 0.34)
                    ])
                else:
                    continue
                
                # Store individual prediction
                individual_preds[name] = {
                    'home_win_prob': float(probs[0]),
                    'draw_prob': float(probs[1]),
                    'away_win_prob': float(probs[2]),
                    'confidence': getattr(pred, 'confidence', 0.5)
                }
                
                # Add weighted contribution
                weight = self.weights.get(name, 0.1)
                weighted_probs += probs * weight
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                continue
        
        if total_weight == 0:
            raise ValueError("All model predictions failed")
        
        # Normalize weighted probabilities
        final_probs = weighted_probs / total_weight
        
        # Ensure probabilities sum to 1
        final_probs = final_probs / final_probs.sum()
        
        # Calculate model agreement (measure of consensus)
        agreement = self._calculate_agreement(individual_preds)
        
        # Determine predicted outcome
        outcome_idx = np.argmax(final_probs)
        outcomes = ['Home Win', 'Draw', 'Away Win']
        predicted_outcome = outcomes[outcome_idx]
        
        # Calculate confidence (base + agreement bonus)
        base_confidence = float(final_probs[outcome_idx])
        confidence = self._calibrate_confidence(base_confidence, agreement)
        
        return EnsemblePrediction(
            home_win_prob=float(final_probs[0]),
            draw_prob=float(final_probs[1]),
            away_win_prob=float(final_probs[2]),
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            model_agreement=agreement,
            individual_predictions=individual_preds,
            calibrated=True
        )
    
    def _calculate_agreement(self, predictions: Dict[str, Dict]) -> float:
        """
        Calculate how much models agree on the outcome.
        
        Returns:
            Agreement score from 0 (complete disagreement) to 1 (full agreement)
        """
        if len(predictions) < 2:
            return 1.0
        
        # Get predicted outcome from each model
        outcomes = []
        for pred in predictions.values():
            probs = [pred['home_win_prob'], pred['draw_prob'], pred['away_win_prob']]
            outcomes.append(np.argmax(probs))
        
        # Calculate agreement as percentage of models agreeing with majority
        from collections import Counter
        outcome_counts = Counter(outcomes)
        most_common_count = outcome_counts.most_common(1)[0][1]
        
        return most_common_count / len(outcomes)
    
    def _calibrate_confidence(self, raw_confidence: float, agreement: float) -> float:
        """
        Calibrate confidence based on model agreement.
        
        High agreement → boost confidence
        Low agreement → reduce confidence
        """
        # Agreement multiplier
        if agreement >= 0.8:
            multiplier = 1.1  # Boost for high agreement
        elif agreement >= 0.6:
            multiplier = 1.0  # No change
        elif agreement >= 0.4:
            multiplier = 0.9  # Slight reduction
        else:
            multiplier = 0.8  # Significant reduction for disagreement
        
        calibrated = raw_confidence * multiplier
        
        # Clamp to valid range
        return max(0.3, min(0.95, calibrated))
    
    def get_model_contributions(self, home_team: str, away_team: str, 
                                  **features) -> Dict[str, Dict]:
        """
        Get detailed breakdown of each model's contribution.
        
        Useful for debugging and understanding predictions.
        """
        contributions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(home_team, away_team, **features)
                weight = self.weights.get(name, 0)
                
                contributions[name] = {
                    'weight': weight,
                    'prediction': pred.to_dict() if hasattr(pred, 'to_dict') else pred,
                    'weighted_contribution': {
                        'home': getattr(pred, 'home_win_prob', 0.33) * weight,
                        'draw': getattr(pred, 'draw_prob', 0.33) * weight,
                        'away': getattr(pred, 'away_win_prob', 0.34) * weight
                    }
                }
            except Exception as e:
                contributions[name] = {'error': str(e)}
        
        return contributions
    
    def save_weights(self, path: Optional[Path] = None):
        """Save current weights to config file"""
        path = path or Path(__file__).parent.parent.parent / "models" / "config" / "ensemble_weights.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.weights, f, indent=2)
        
        logger.info(f"Saved weights to {path}")
    
    def load_weights(self, path: Optional[Path] = None):
        """Load weights from config file"""
        path = path or Path(__file__).parent.parent.parent / "models" / "config" / "ensemble_weights.json"
        
        if path.exists():
            with open(path, 'r') as f:
                self.weights = json.load(f)
            logger.info(f"Loaded weights from {path}")
        else:
            logger.warning(f"No weights file found at {path}, using defaults")


class SimpleVotingEnsemble:
    """
    Simple voting ensemble - each model gets one vote.
    Good for when you want equal weighting.
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
    
    def register_model(self, name: str, model: Any):
        self.models[name] = model
    
    def predict(self, home_team: str, away_team: str, **features) -> str:
        """Get majority vote for outcome"""
        votes = {'home': 0, 'draw': 0, 'away': 0}
        
        for model in self.models.values():
            pred = model.predict(home_team, away_team, **features)
            probs = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
            outcome_idx = np.argmax(probs)
            vote_keys = ['home', 'draw', 'away']
            votes[vote_keys[outcome_idx]] += 1
        
        winner = max(votes, key=votes.get)
        return {'home': 'Home Win', 'draw': 'Draw', 'away': 'Away Win'}[winner]
