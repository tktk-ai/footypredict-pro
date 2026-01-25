"""
Pattern Recognition & Anomaly Detection

Advanced ML-style pattern recognition:
- Historical pattern matching
- Anomaly detection for unusual odds/predictions
- Streak pattern analysis
- Score pattern prediction
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum


class PatternType(Enum):
    """Types of patterns detected"""
    HOME_DOMINANCE = "home_dominance"
    AWAY_STRENGTH = "away_strength"
    DRAW_SPECIALIST = "draw_specialist"
    HIGH_SCORING = "high_scoring"
    LOW_SCORING = "low_scoring"
    MOMENTUM = "momentum"
    SLUMP = "slump"
    COMEBACK = "comeback"
    COLLAPSE = "collapse"
    CONSISTENT = "consistent"
    VOLATILE = "volatile"


@dataclass
class Pattern:
    """Detected pattern"""
    type: PatternType
    team: str
    strength: float  # 0-1
    confidence: float
    evidence: List[str]
    prediction_impact: float  # -1 to 1
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'team': self.team,
            'strength': round(self.strength, 2),
            'confidence': round(self.confidence, 2),
            'evidence': self.evidence[:3],
            'impact': round(self.prediction_impact, 3)
        }


@dataclass
class Anomaly:
    """Detected anomaly"""
    type: str
    severity: str  # low, medium, high, critical
    description: str
    expected_value: float
    actual_value: float
    deviation: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type,
            'severity': self.severity,
            'description': self.description,
            'deviation': f"{self.deviation:.1f}%",
            'timestamp': self.timestamp
        }


class PatternRecognitionEngine:
    """Advanced pattern recognition for football predictions"""
    
    def __init__(self):
        self.team_history: Dict[str, List[Dict]] = defaultdict(list)
        self.pattern_cache: Dict[str, List[Pattern]] = {}
        self.historical_odds: Dict[str, List[float]] = defaultdict(list)
        
    def add_match_result(self, team: str, result: Dict):
        """Add match result to history"""
        self.team_history[team].append({
            **result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 50 matches
        if len(self.team_history[team]) > 50:
            self.team_history[team] = self.team_history[team][-50:]
            
        # Invalidate cache
        if team in self.pattern_cache:
            del self.pattern_cache[team]
    
    def detect_patterns(self, team: str) -> List[Pattern]:
        """Detect all patterns for a team"""
        # Check cache
        if team in self.pattern_cache:
            return self.pattern_cache[team]
        
        history = self.team_history.get(team, [])
        if len(history) < 5:
            return []
        
        patterns = []
        
        # Detect various patterns
        patterns.extend(self._detect_form_patterns(team, history))
        patterns.extend(self._detect_scoring_patterns(team, history))
        patterns.extend(self._detect_home_away_patterns(team, history))
        patterns.extend(self._detect_momentum_patterns(team, history))
        
        # Cache results
        self.pattern_cache[team] = patterns
        return patterns
    
    def _detect_form_patterns(self, team: str, history: List[Dict]) -> List[Pattern]:
        """Detect form-based patterns"""
        patterns = []
        recent = history[-5:]
        
        wins = sum(1 for m in recent if m.get('result') == 'W')
        losses = sum(1 for m in recent if m.get('result') == 'L')
        draws = sum(1 for m in recent if m.get('result') == 'D')
        
        # Winning streak
        if wins >= 4:
            patterns.append(Pattern(
                type=PatternType.MOMENTUM,
                team=team,
                strength=wins / 5,
                confidence=0.85,
                evidence=[f"{wins} wins in last 5 matches", "Strong form"],
                prediction_impact=0.15
            ))
        
        # Losing streak
        if losses >= 4:
            patterns.append(Pattern(
                type=PatternType.SLUMP,
                team=team,
                strength=losses / 5,
                confidence=0.85,
                evidence=[f"{losses} losses in last 5 matches", "Poor form"],
                prediction_impact=-0.15
            ))
        
        # Consistent/Volatile
        results = [m.get('result') for m in recent]
        unique_results = len(set(results))
        
        if unique_results == 1:
            patterns.append(Pattern(
                type=PatternType.CONSISTENT,
                team=team,
                strength=0.9,
                confidence=0.8,
                evidence=["Same result in all 5 matches"],
                prediction_impact=0.1
            ))
        elif unique_results == 3:
            patterns.append(Pattern(
                type=PatternType.VOLATILE,
                team=team,
                strength=0.7,
                confidence=0.7,
                evidence=["Mixed results - unpredictable"],
                prediction_impact=-0.05
            ))
        
        return patterns
    
    def _detect_scoring_patterns(self, team: str, history: List[Dict]) -> List[Pattern]:
        """Detect scoring patterns"""
        patterns = []
        recent = history[-10:]
        
        goals_scored = [m.get('goals_for', 0) for m in recent]
        goals_conceded = [m.get('goals_against', 0) for m in recent]
        
        avg_scored = sum(goals_scored) / len(goals_scored)
        avg_conceded = sum(goals_conceded) / len(goals_conceded)
        
        # High scoring
        if avg_scored >= 2.0:
            patterns.append(Pattern(
                type=PatternType.HIGH_SCORING,
                team=team,
                strength=min(1.0, avg_scored / 3),
                confidence=0.8,
                evidence=[f"Avg {avg_scored:.1f} goals per game"],
                prediction_impact=0.08
            ))
        
        # Low scoring
        if avg_scored < 1.0:
            patterns.append(Pattern(
                type=PatternType.LOW_SCORING,
                team=team,
                strength=1.0 - avg_scored,
                confidence=0.8,
                evidence=[f"Only {avg_scored:.1f} goals per game avg"],
                prediction_impact=-0.05
            ))
        
        # Defensive strength/weakness
        if avg_conceded < 0.8:
            patterns.append(Pattern(
                type=PatternType.CONSISTENT,
                team=team,
                strength=1 - avg_conceded,
                confidence=0.75,
                evidence=[f"Strong defense: {avg_conceded:.1f} goals conceded"],
                prediction_impact=0.1
            ))
        
        return patterns
    
    def _detect_home_away_patterns(self, team: str, history: List[Dict]) -> List[Pattern]:
        """Detect home/away specific patterns"""
        patterns = []
        
        home_matches = [m for m in history if m.get('venue') == 'home']
        away_matches = [m for m in history if m.get('venue') == 'away']
        
        if len(home_matches) >= 5:
            home_wins = sum(1 for m in home_matches[-5:] if m.get('result') == 'W')
            if home_wins >= 4:
                patterns.append(Pattern(
                    type=PatternType.HOME_DOMINANCE,
                    team=team,
                    strength=home_wins / 5,
                    confidence=0.85,
                    evidence=[f"{home_wins}/5 home wins recently"],
                    prediction_impact=0.12
                ))
        
        if len(away_matches) >= 5:
            away_wins = sum(1 for m in away_matches[-5:] if m.get('result') == 'W')
            if away_wins >= 3:
                patterns.append(Pattern(
                    type=PatternType.AWAY_STRENGTH,
                    team=team,
                    strength=away_wins / 5,
                    confidence=0.8,
                    evidence=[f"{away_wins}/5 away wins recently"],
                    prediction_impact=0.1
                ))
        
        return patterns
    
    def _detect_momentum_patterns(self, team: str, history: List[Dict]) -> List[Pattern]:
        """Detect momentum/trend patterns"""
        patterns = []
        
        if len(history) < 10:
            return patterns
        
        # Compare recent vs older form
        recent_5 = history[-5:]
        older_5 = history[-10:-5]
        
        recent_wins = sum(1 for m in recent_5 if m.get('result') == 'W')
        older_wins = sum(1 for m in older_5 if m.get('result') == 'W')
        
        # Improving team
        if recent_wins > older_wins + 2:
            patterns.append(Pattern(
                type=PatternType.COMEBACK,
                team=team,
                strength=(recent_wins - older_wins) / 5,
                confidence=0.75,
                evidence=["Form improving significantly"],
                prediction_impact=0.08
            ))
        
        # Declining team
        if recent_wins < older_wins - 2:
            patterns.append(Pattern(
                type=PatternType.COLLAPSE,
                team=team,
                strength=(older_wins - recent_wins) / 5,
                confidence=0.75,
                evidence=["Form declining significantly"],
                prediction_impact=-0.08
            ))
        
        return patterns


class AnomalyDetector:
    """Detect anomalies in predictions and odds"""
    
    def __init__(self):
        self.historical_data: Dict[str, List[float]] = defaultdict(list)
        self.thresholds = {
            'odds_deviation': 0.15,  # 15% deviation
            'confidence_deviation': 0.20,
            'score_deviation': 2.5
        }
    
    def add_data_point(self, key: str, value: float):
        """Add data point for baseline calculation"""
        self.historical_data[key].append(value)
        if len(self.historical_data[key]) > 100:
            self.historical_data[key] = self.historical_data[key][-100:]
    
    def check_odds_anomaly(self, match_id: str, odds: Dict) -> Optional[Anomaly]:
        """Check for anomalous odds"""
        for outcome, value in odds.items():
            key = f"odds_{outcome}"
            history = self.historical_data.get(key, [])
            
            if len(history) < 10:
                self.add_data_point(key, value)
                continue
            
            mean = sum(history) / len(history)
            std_dev = math.sqrt(sum((x - mean) ** 2 for x in history) / len(history))
            
            if std_dev > 0:
                z_score = (value - mean) / std_dev
                
                if abs(z_score) > 2.5:
                    return Anomaly(
                        type='odds_anomaly',
                        severity='high' if abs(z_score) > 3 else 'medium',
                        description=f"Unusual {outcome} odds: {value:.2f}",
                        expected_value=mean,
                        actual_value=value,
                        deviation=(value - mean) / mean * 100
                    )
            
            self.add_data_point(key, value)
        
        return None
    
    def check_prediction_anomaly(
        self, 
        home: str, 
        away: str, 
        confidence: float,
        predicted_outcome: str
    ) -> Optional[Anomaly]:
        """Check for anomalous prediction"""
        key = f"conf_{predicted_outcome}"
        history = self.historical_data.get(key, [])
        
        if len(history) < 20:
            self.add_data_point(key, confidence)
            return None
        
        mean = sum(history) / len(history)
        
        # Unusually high or low confidence
        if confidence > mean + 0.25:
            return Anomaly(
                type='high_confidence',
                severity='medium',
                description=f"Unusually high confidence for {predicted_outcome}",
                expected_value=mean,
                actual_value=confidence,
                deviation=(confidence - mean) * 100
            )
        elif confidence < mean - 0.25:
            return Anomaly(
                type='low_confidence',
                severity='low',
                description=f"Lower than usual confidence",
                expected_value=mean,
                actual_value=confidence,
                deviation=(mean - confidence) * 100
            )
        
        self.add_data_point(key, confidence)
        return None
    
    def check_line_movement(self, match_id: str, old_odds: Dict, new_odds: Dict) -> Optional[Anomaly]:
        """Check for sharp line movements"""
        for outcome in ['home', 'draw', 'away']:
            if outcome in old_odds and outcome in new_odds:
                old = old_odds[outcome]
                new = new_odds[outcome]
                change = (new - old) / old * 100
                
                if abs(change) > 10:
                    return Anomaly(
                        type='sharp_movement',
                        severity='high' if abs(change) > 20 else 'medium',
                        description=f"Sharp {outcome} odds movement: {change:+.1f}%",
                        expected_value=old,
                        actual_value=new,
                        deviation=change
                    )
        
        return None


class ScorePredictor:
    """Predict exact scores using pattern analysis"""
    
    def __init__(self, pattern_engine: PatternRecognitionEngine):
        self.pattern_engine = pattern_engine
        self.score_history: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    
    def predict_score(self, home: str, away: str) -> Dict:
        """Predict most likely scores"""
        home_patterns = self.pattern_engine.detect_patterns(home)
        away_patterns = self.pattern_engine.detect_patterns(away)
        
        # Base scores from patterns
        home_attack = 1.2 + sum(
            p.prediction_impact for p in home_patterns 
            if p.type in [PatternType.HIGH_SCORING, PatternType.MOMENTUM, PatternType.HOME_DOMINANCE]
        )
        
        away_attack = 0.9 + sum(
            p.prediction_impact for p in away_patterns
            if p.type in [PatternType.HIGH_SCORING, PatternType.AWAY_STRENGTH]
        )
        
        # Adjust for defensive patterns
        home_attack *= 1 - sum(
            abs(p.prediction_impact) * 0.3 for p in away_patterns
            if p.type == PatternType.CONSISTENT
        )
        
        away_attack *= 1 - sum(
            abs(p.prediction_impact) * 0.3 for p in home_patterns
            if p.type == PatternType.CONSISTENT
        )
        
        # Round to get expected goals
        expected_home = max(0, round(home_attack * 1.1))
        expected_away = max(0, round(away_attack * 0.9))
        
        # Generate probability distribution
        scores = []
        for h in range(6):
            for a in range(5):
                prob = self._poisson_prob(h, home_attack) * self._poisson_prob(a, away_attack)
                if prob > 0.01:
                    scores.append({
                        'score': f"{h}-{a}",
                        'probability': round(prob * 100, 1)
                    })
        
        scores.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'most_likely': f"{expected_home}-{expected_away}",
            'expected_goals': {
                'home': round(home_attack, 1),
                'away': round(away_attack, 1),
                'total': round(home_attack + away_attack, 1)
            },
            'top_5_scores': scores[:5],
            'btts_probability': round(self._btts_prob(home_attack, away_attack) * 100, 1),
            'over_25_probability': round(self._over_prob(home_attack + away_attack, 2.5) * 100, 1),
            'patterns_used': len(home_patterns) + len(away_patterns)
        }
    
    def _poisson_prob(self, k: int, lam: float) -> float:
        """Calculate Poisson probability"""
        try:
            return (lam ** k) * math.exp(-lam) / math.factorial(k)
        except:
            return 0.0
    
    def _btts_prob(self, home_xg: float, away_xg: float) -> float:
        """Both teams to score probability"""
        home_scores = 1 - math.exp(-home_xg)
        away_scores = 1 - math.exp(-away_xg)
        return home_scores * away_scores
    
    def _over_prob(self, total_xg: float, line: float) -> float:
        """Over/Under probability using Poisson"""
        under = sum(self._poisson_prob(k, total_xg) for k in range(int(line) + 1))
        over = 1 - under
        return over


# Global instances
pattern_engine = PatternRecognitionEngine()
anomaly_detector = AnomalyDetector()
score_predictor = ScorePredictor(pattern_engine)


def detect_patterns(team: str) -> List[Dict]:
    """Detect patterns for a team"""
    patterns = pattern_engine.detect_patterns(team)
    return [p.to_dict() for p in patterns]


def check_anomalies(match_id: str, odds: Dict) -> Optional[Dict]:
    """Check for anomalies"""
    anomaly = anomaly_detector.check_odds_anomaly(match_id, odds)
    return anomaly.to_dict() if anomaly else None


def predict_exact_score(home: str, away: str) -> Dict:
    """Predict exact score for a match"""
    return score_predictor.predict_score(home, away)
