"""
Prediction Transparency Module
==============================

Provides detailed explanations and transparency for predictions,
helping users understand:
- How predictions are made
- What factors contribute most
- Model limitations and accuracy
- Data quality indicators

This module promotes honest, trustworthy predictions.
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# Model accuracy metadata
MODEL_INFO = {
    'sportybet_v1': {
        'name': 'SportyBet XGBoost v1.0',
        'trained_date': '2026-01-27',
        'training_samples': 105142,
        'markets': {
            'over_15': {'accuracy': 0.75, 'samples': 95000},
            'over_25': {'accuracy': 0.60, 'samples': 95000},
            'btts': {'accuracy': 0.62, 'samples': 95000},
            'dc_1x': {'accuracy': 0.72, 'samples': 95000},
            'dc_x2': {'accuracy': 0.70, 'samples': 95000},
            'dc_12': {'accuracy': 0.73, 'samples': 95000},
            'ht_over_05': {'accuracy': 0.68, 'samples': 90000},
            'ht_btts': {'accuracy': 0.58, 'samples': 90000},
            'home_over_25': {'accuracy': 0.52, 'samples': 85000},
        },
        'default_accuracy': 0.56
    }
}


@dataclass
class PredictionExplanation:
    """Detailed explanation of how a prediction was made."""
    
    # Core prediction info
    market: str
    market_name: str
    prediction: str
    raw_probability: float
    calibrated_probability: float
    confidence_level: str  # 'very_low', 'low', 'medium', 'high', 'very_high'
    
    # Model info
    model_name: str = 'SportyBet XGBoost v1.0'
    model_accuracy: float = 0.56
    
    # Data quality
    home_team_known: bool = True
    away_team_known: bool = True
    has_live_odds: bool = False
    data_quality: str = 'high'  # 'high', 'medium', 'low'
    data_freshness: str = 'historical'  # 'live', 'cached', 'historical'
    
    # Top factors contributing to prediction
    top_factors: List[Dict] = field(default_factory=list)
    
    # Warnings and limitations
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # When was this generated
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        pct = round(self.calibrated_probability * 100, 1)
        
        summary = f"{self.market_name}: {self.prediction} ({pct}%)"
        
        if self.warnings:
            summary += f" ⚠️ {len(self.warnings)} warning(s)"
        
        return summary


def generate_explanation(
    market: str,
    market_name: str,
    prediction: str,
    raw_probability: float,
    home_team: str,
    away_team: str,
    home_known: bool = True,
    away_known: bool = True,
    has_live_odds: bool = False,
    home_stats: Optional[Dict] = None,
    away_stats: Optional[Dict] = None,
) -> PredictionExplanation:
    """
    Generate a detailed explanation for a prediction.
    
    Args:
        market: Market ID
        market_name: Display name
        prediction: The prediction outcome
        raw_probability: Raw model probability
        home_team: Home team name
        away_team: Away team name
        home_known: Whether home team is in database
        away_known: Whether away team is in database
        has_live_odds: Whether live odds were used
        home_stats: Home team statistics (for factors)
        away_stats: Away team statistics (for factors)
        
    Returns:
        PredictionExplanation with full transparency info
    """
    from src.models.calibration import empirical_calibrate, get_honest_confidence
    
    # Get calibrated confidence
    confidence_info = get_honest_confidence(raw_probability, market)
    calibrated = confidence_info['calibrated_probability'] / 100
    
    # Get model info
    model_info = MODEL_INFO.get('sportybet_v1', {})
    market_info = model_info.get('markets', {}).get(market, {})
    accuracy = market_info.get('accuracy', model_info.get('default_accuracy', 0.56))
    
    # Determine data quality
    if home_known and away_known:
        data_quality = 'high'
    elif home_known or away_known:
        data_quality = 'medium'
    else:
        data_quality = 'low'
    
    data_freshness = 'live' if has_live_odds else 'historical'
    
    # Generate top factors
    top_factors = _generate_factors(
        market, prediction, home_team, away_team,
        home_stats, away_stats
    )
    
    # Generate warnings
    warnings = []
    if not home_known:
        warnings.append(f"'{home_team}' not found in database - using average stats")
    if not away_known:
        warnings.append(f"'{away_team}' not found in database - using average stats")
    if not has_live_odds:
        warnings.append("No live odds available - using historical averages")
    if accuracy < 0.55:
        warnings.append(f"This market has lower historical accuracy ({round(accuracy*100)}%)")
    
    # Standard limitations
    limitations = [
        "Predictions are based on historical patterns",
        "Past performance does not guarantee future results",
        "External factors (injuries, weather, motivation) may not be fully captured"
    ]
    
    if data_quality == 'low':
        limitations.append("Limited data available for these teams")
    
    return PredictionExplanation(
        market=market,
        market_name=market_name,
        prediction=prediction,
        raw_probability=raw_probability,
        calibrated_probability=calibrated,
        confidence_level=confidence_info['confidence_level'],
        model_name=model_info.get('name', 'Unknown'),
        model_accuracy=accuracy,
        home_team_known=home_known,
        away_team_known=away_known,
        has_live_odds=has_live_odds,
        data_quality=data_quality,
        data_freshness=data_freshness,
        top_factors=top_factors,
        warnings=warnings,
        limitations=limitations
    )


def _generate_factors(
    market: str,
    prediction: str,
    home_team: str,
    away_team: str,
    home_stats: Optional[Dict],
    away_stats: Optional[Dict]
) -> List[Dict]:
    """Generate human-readable factors that contributed to prediction."""
    factors = []
    
    if not home_stats and not away_stats:
        factors.append({
            'factor': 'League averages',
            'description': 'Using typical league statistics',
            'impact': 'neutral'
        })
        return factors
    
    # Goals-related factors
    if market in ['over_15', 'over_25', 'over_35', 'btts']:
        home_goals = home_stats.get('goals_scored_home', 1.5) if home_stats else 1.5
        away_goals = away_stats.get('goals_scored_away', 1.1) if away_stats else 1.1
        expected_total = home_goals + away_goals
        
        if expected_total > 2.5:
            factors.append({
                'factor': 'High-scoring teams',
                'description': f'Expected {expected_total:.1f} goals',
                'impact': 'positive' if prediction == 'Yes' else 'negative'
            })
        else:
            factors.append({
                'factor': 'Lower-scoring matchup',
                'description': f'Expected {expected_total:.1f} goals',
                'impact': 'negative' if prediction == 'Yes' else 'positive'
            })
    
    # Home advantage
    if home_stats:
        home_odds = home_stats.get('avg_odds_home', 2.5)
        if home_odds < 2.0:
            factors.append({
                'factor': 'Strong home team',
                'description': f'{home_team} typically favored at home',
                'impact': 'positive'
            })
    
    # Historical patterns
    if market in ['dc_1x', 'dc_x2', 'dc_12']:
        factors.append({
            'factor': 'Double chance stability',
            'description': 'Double chance markets have higher accuracy',
            'impact': 'positive'
        })
    
    return factors[:3]  # Top 3 factors


def get_transparency_summary(predictions: Dict, home_team: str, away_team: str) -> Dict:
    """
    Get a summary transparency block for multiple predictions.
    
    This is designed to be added to API responses.
    """
    # Find data quality from predictions
    if hasattr(predictions, 'data_quality'):
        data_quality = predictions.data_quality
        home_known = predictions.home_team_known
        away_known = predictions.away_team_known
    else:
        data_quality = 'high'
        home_known = True
        away_known = True
    
    return {
        'data_quality': data_quality,
        'home_team_recognized': home_known,
        'away_team_recognized': away_known,
        'model_version': 'SportyBet XGBoost v1.0',
        'base_accuracies': {
            '1x2': '56%',
            'double_chance': '70-76%',
            'over_15': '75%',
            'over_25': '60%',
            'btts': '62%'
        },
        'disclaimer': 'Predictions are probabilistic estimates based on historical data. Use responsibly.',
        'generated_at': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test transparency
    print("\n📊 Testing Transparency Module")
    print("=" * 50)
    
    explanation = generate_explanation(
        market='over_25',
        market_name='Over 2.5 Goals',
        prediction='Yes',
        raw_probability=0.75,
        home_team='Arsenal',
        away_team='Liverpool',
        home_known=True,
        away_known=True,
        has_live_odds=False
    )
    
    print(f"\nPrediction: {explanation.get_summary()}")
    print(f"Model: {explanation.model_name}")
    print(f"Accuracy: {explanation.model_accuracy*100:.0f}%")
    print(f"Data Quality: {explanation.data_quality}")
    
    if explanation.top_factors:
        print("\nTop Factors:")
        for f in explanation.top_factors:
            print(f"  - {f['factor']}: {f['description']}")
    
    if explanation.warnings:
        print("\n⚠️ Warnings:")
        for w in explanation.warnings:
            print(f"  - {w}")
    
    print("\n✅ Transparency module working!")
