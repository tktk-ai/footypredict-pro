"""
Confidence-Based Prediction Sections

Categorizes predictions by confidence level:
- Sure Win: 91%+ confidence (ultra-high probability picks)
- Strong Picks: 80-90% confidence (reliable selections)
- Value Hunters: 5%+ edge vs bookmaker odds
- Upset Watch: Underdog potential picks
- Daily Banker: Single safest pick of the day
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


@dataclass
class SectionConfig:
    """Configuration for a confidence section"""
    name: str
    description: str
    icon: str
    color: str
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    min_edge: Optional[float] = None
    max_picks: Optional[int] = None


class ConfidenceSectionsManager:
    """
    Categorize predictions into confidence-based sections.
    
    Sections:
    - Sure Win: 91%+ confidence picks (very rare, very reliable)
    - Strong Picks: 80-90% confidence (solid selections)
    - Value Hunters: Good edge vs market odds (5%+ value)
    - Upset Watch: Underdog potential (25%+ probability)
    - Daily Banker: Single highest confidence pick
    """
    
    SECTIONS = {
        'sure_win': SectionConfig(
            name='🔒 Sure Win',
            description='Ultra-high confidence picks (91%+ probability)',
            icon='🔒',
            color='#10B981',  # Emerald green
            min_confidence=0.91
        ),
        'strong_picks': SectionConfig(
            name='💪 Strong Picks',
            description='High confidence selections (80-90%)',
            icon='💪',
            color='#3B82F6',  # Blue
            min_confidence=0.80,
            max_confidence=0.90
        ),
        'value_hunters': SectionConfig(
            name='💎 Value Hunters',
            description='Great edge vs bookmaker odds (5%+ value)',
            icon='💎',
            color='#8B5CF6',  # Purple
            min_edge=0.05
        ),
        'upset_watch': SectionConfig(
            name='⚡ Upset Watch',
            description='Potential upsets worth watching',
            icon='⚡',
            color='#F59E0B',  # Amber
        ),
        'daily_banker': SectionConfig(
            name='🎯 Daily Banker',
            description='Single safest pick of the day',
            icon='🎯',
            color='#EF4444',  # Red
            max_picks=1
        ),
        'risky_rewards': SectionConfig(
            name='🎲 Risky Rewards',
            description='Long shots with high potential returns',
            icon='🎲',
            color='#EC4899',  # Pink
            min_confidence=0.30,
            max_confidence=0.50
        )
    }
    
    def __init__(self):
        self.prediction_history = []
    
    def categorize(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Sort predictions into confidence-based sections.
        
        Args:
            predictions: List of prediction dicts with 'confidence', 'value_edge', etc.
            
        Returns:
            Dict mapping section name to list of predictions
        """
        sections = {key: [] for key in self.SECTIONS}
        
        if not predictions:
            return sections
        
        for pred in predictions:
            confidence = pred.get('confidence', 0)
            if isinstance(confidence, str):
                confidence = float(confidence.replace('%', '')) / 100
            elif confidence > 1:
                confidence = confidence / 100
                
            edge = pred.get('value_edge', 0) or 0
            
            # Determine predicted outcome probabilities
            home_prob = pred.get('home_win_prob', 0) or 0
            draw_prob = pred.get('draw_prob', 0) or 0  
            away_prob = pred.get('away_win_prob', 0) or 0
            
            # Normalize if needed
            if home_prob > 1:
                home_prob /= 100
            if draw_prob > 1:
                draw_prob /= 100
            if away_prob > 1:
                away_prob /= 100
            
            max_prob = max(home_prob, draw_prob, away_prob)
            
            # Sure Win: 91%+ confidence
            if confidence >= 0.91 or max_prob >= 0.91:
                sections['sure_win'].append(self._enrich_prediction(pred, 'sure_win'))
            
            # Strong Picks: 80-90%
            elif confidence >= 0.80 or max_prob >= 0.80:
                sections['strong_picks'].append(self._enrich_prediction(pred, 'strong_picks'))
            
            # Value Hunters: 5%+ edge
            if edge >= 0.05:
                sections['value_hunters'].append(self._enrich_prediction(pred, 'value_hunters'))
            
            # Upset Watch: Underdog with reasonable probability
            is_upset = self._is_upset_potential(pred)
            if is_upset:
                sections['upset_watch'].append(self._enrich_prediction(pred, 'upset_watch'))
            
            # Risky Rewards: 30-50% confidence but good odds
            if 0.30 <= confidence <= 0.50:
                sections['risky_rewards'].append(self._enrich_prediction(pred, 'risky_rewards'))
        
        # Daily Banker: Highest confidence pick
        if predictions:
            best = max(predictions, key=lambda x: self._get_confidence(x))
            sections['daily_banker'] = [self._enrich_prediction(best, 'daily_banker')]
        
        # Sort each section by confidence descending
        for section_name in sections:
            if section_name != 'daily_banker':
                sections[section_name].sort(
                    key=lambda x: self._get_confidence(x), 
                    reverse=True
                )
        
        return sections
    
    def _get_confidence(self, pred: Dict) -> float:
        """Extract normalized confidence from prediction"""
        confidence = pred.get('confidence', 0)
        if isinstance(confidence, str):
            confidence = float(confidence.replace('%', '')) / 100
        elif confidence > 1:
            confidence = confidence / 100
        return confidence
    
    def _is_upset_potential(self, pred: Dict) -> bool:
        """
        Check if this is a potential upset.
        
        Criteria:
        - Lower-ranked team has 25%+ win probability
        - OR home team expected to lose but has 30%+ probability
        """
        home_prob = pred.get('home_win_prob', 0) or 0
        away_prob = pred.get('away_win_prob', 0) or 0
        
        if home_prob > 1:
            home_prob /= 100
        if away_prob > 1:
            away_prob /= 100
            
        # Check ELO difference if available
        home_elo = pred.get('home_elo', 1500)
        away_elo = pred.get('away_elo', 1500)
        
        # Underdog is team with lower ELO
        if home_elo < away_elo and home_prob >= 0.25:
            return True
        elif away_elo < home_elo and away_prob >= 0.25:
            return True
        
        # Also check if predicted outcome differs from ELO expectation
        predicted = pred.get('predicted_outcome', '')
        if home_elo > away_elo + 50 and predicted == 'Away':
            return True
        elif away_elo > home_elo + 50 and predicted == 'Home':
            return True
            
        return False
    
    def _enrich_prediction(self, pred: Dict, section: str) -> Dict:
        """Add section metadata to prediction"""
        enriched = pred.copy()
        enriched['section'] = section
        enriched['section_config'] = asdict(self.SECTIONS[section])
        return enriched
    
    def get_sure_wins(self, predictions: List[Dict]) -> List[Dict]:
        """Get only Sure Win picks (91%+ confidence)"""
        return self.categorize(predictions)['sure_win']
    
    def get_strong_picks(self, predictions: List[Dict]) -> List[Dict]:
        """Get Strong Picks (80-90% confidence)"""
        return self.categorize(predictions)['strong_picks']
    
    def get_value_bets(self, predictions: List[Dict]) -> List[Dict]:
        """Get Value Hunter picks (5%+ edge)"""
        return self.categorize(predictions)['value_hunters']
    
    def get_daily_banker(self, predictions: List[Dict]) -> Optional[Dict]:
        """Get the single Daily Banker pick"""
        bankers = self.categorize(predictions)['daily_banker']
        return bankers[0] if bankers else None
    
    def get_section_stats(self, predictions: List[Dict]) -> Dict:
        """Get statistics about each section"""
        sections = self.categorize(predictions)
        
        stats = {}
        for section_name, preds in sections.items():
            if preds:
                confidences = [self._get_confidence(p) for p in preds]
                stats[section_name] = {
                    'count': len(preds),
                    'avg_confidence': round(sum(confidences) / len(confidences) * 100, 1),
                    'min_confidence': round(min(confidences) * 100, 1),
                    'max_confidence': round(max(confidences) * 100, 1),
                    'config': asdict(self.SECTIONS[section_name])
                }
            else:
                stats[section_name] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'min_confidence': 0,
                    'max_confidence': 0,
                    'config': asdict(self.SECTIONS[section_name])
                }
        
        return stats
    
    def get_all_sections_config(self) -> Dict:
        """Get configuration for all sections"""
        return {
            name: asdict(config) 
            for name, config in self.SECTIONS.items()
        }


# Global instance
confidence_manager = ConfidenceSectionsManager()


def get_confidence_sections(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """Get predictions organized by confidence sections"""
    return confidence_manager.categorize(predictions)


def get_sure_wins(predictions: List[Dict]) -> List[Dict]:
    """Get 91%+ confidence predictions"""
    return confidence_manager.get_sure_wins(predictions)


def get_daily_banker(predictions: List[Dict]) -> Optional[Dict]:
    """Get single safest pick"""
    return confidence_manager.get_daily_banker(predictions)
