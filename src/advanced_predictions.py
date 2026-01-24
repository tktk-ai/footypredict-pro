"""
Advanced Predictions Module

Enhanced predictive features:
1. Form Momentum Tracking - Weight recent matches more heavily
2. Head-to-Head Analysis - Use historical matchup data
3. League Position Factor - Top vs bottom team adjustments
4. Market Odds Integration - Compare predictions to bookmaker odds
5. Confidence Calibration - Auto-adjust based on accuracy history
6. Multi-factor Ensemble - Combine all factors intelligently
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdvancedPrediction:
    """Enhanced prediction with all factors"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str
    raw_confidence: float
    calibrated_confidence: float
    factors: Dict[str, float]
    recommendations: List[str]
    value_bets: List[Dict]


class FormMomentumTracker:
    """
    Track team form with exponential decay weighting.
    Recent matches matter more than older ones.
    """
    
    # Simulated recent form (W=3, D=1, L=0) - most recent first
    TEAM_FORM = {
        # Bayern: WWWDW (last 5, most recent first)
        'Bayern': [3, 3, 3, 1, 3, 3, 1, 3, 3, 3],
        'Dortmund': [3, 1, 3, 0, 3, 1, 3, 0, 3, 1],
        'Leverkusen': [3, 3, 1, 3, 3, 3, 3, 3, 3, 0],
        'Leipzig': [1, 3, 0, 3, 1, 0, 3, 3, 1, 3],
        'Manchester City': [3, 3, 3, 3, 1, 3, 3, 3, 0, 3],
        'Liverpool': [3, 3, 1, 3, 3, 3, 0, 3, 3, 1],
        'Arsenal': [3, 1, 3, 3, 0, 3, 3, 1, 3, 0],
        'Real Madrid': [3, 3, 3, 1, 3, 3, 3, 1, 3, 3],
        'Barcelona': [1, 3, 3, 0, 3, 1, 3, 3, 0, 3],
        'Inter': [3, 3, 3, 3, 1, 3, 3, 0, 3, 3],
        'Juventus': [3, 1, 0, 3, 3, 1, 3, 0, 3, 1],
        'PSG': [3, 3, 1, 3, 0, 3, 3, 3, 1, 3],
    }
    
    def get_form_momentum(self, team: str, decay_rate: float = 0.85) -> float:
        """
        Calculate form score with exponential decay.
        More recent matches have higher weight.
        
        Args:
            team: Team name
            decay_rate: How much each older match is discounted (0.85 = 15% less weight)
        
        Returns:
            Momentum score 0.0-1.0 (higher = better form)
        """
        form = self._get_team_form(team)
        
        weighted_sum = 0
        weight_total = 0
        
        for i, result in enumerate(form):
            weight = decay_rate ** i
            weighted_sum += result * weight
            weight_total += 3 * weight  # Max per match is 3
        
        if weight_total == 0:
            return 0.5
        
        return weighted_sum / weight_total
    
    def _get_team_form(self, team: str) -> List[int]:
        """Get form data for team with fuzzy matching"""
        if team in self.TEAM_FORM:
            return self.TEAM_FORM[team]
        
        team_lower = team.lower()
        for name, form in self.TEAM_FORM.items():
            if name.lower() in team_lower or team_lower in name.lower():
                return form
        
        return [1, 1, 1, 1, 1]  # Average form
    
    def get_hot_streak(self, team: str) -> int:
        """Count consecutive wins (positive) or losses (negative)"""
        form = self._get_team_form(team)
        
        if not form:
            return 0
        
        streak = 0
        first = form[0]
        
        if first == 3:  # Win streak
            for r in form:
                if r == 3:
                    streak += 1
                else:
                    break
        elif first == 0:  # Losing streak
            for r in form:
                if r == 0:
                    streak -= 1
                else:
                    break
        
        return streak


class HeadToHeadAnalyzer:
    """
    Analyze historical head-to-head records between teams.
    """
    
    # Simulated H2H records: {matchup: [home_wins, draws, away_wins, home_goals, away_goals]}
    H2H_RECORDS = {
        ('Bayern', 'Dortmund'): [12, 5, 3, 38, 18],
        ('Bayern', 'Leipzig'): [10, 2, 2, 32, 12],
        ('Dortmund', 'Bayern'): [5, 3, 12, 22, 35],
        ('Liverpool', 'Manchester City'): [8, 6, 8, 28, 30],
        ('Real Madrid', 'Barcelona'): [10, 8, 10, 35, 32],
        ('Inter', 'Juventus'): [8, 10, 8, 25, 28],
        ('Arsenal', 'Tottenham'): [10, 6, 6, 35, 22],
    }
    
    def get_h2h_factor(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Get H2H adjustment factors.
        
        Returns:
            Dict with home/draw/away adjustments (-0.1 to +0.1)
        """
        key = (home_team, away_team)
        
        # Try exact match
        if key in self.H2H_RECORDS:
            record = self.H2H_RECORDS[key]
        else:
            # Try fuzzy match
            record = self._fuzzy_match(home_team, away_team)
        
        if not record:
            return {'home': 0, 'draw': 0, 'away': 0}
        
        home_wins, draws, away_wins, _, _ = record
        total = home_wins + draws + away_wins
        
        if total < 3:  # Not enough data
            return {'home': 0, 'draw': 0, 'away': 0}
        
        # Base expectation is 33% each
        home_pct = home_wins / total
        draw_pct = draws / total
        away_pct = away_wins / total
        
        # Adjustment is difference from baseline, capped at ±0.1
        return {
            'home': max(-0.1, min(0.1, home_pct - 0.4)),
            'draw': max(-0.1, min(0.1, draw_pct - 0.27)),
            'away': max(-0.1, min(0.1, away_pct - 0.33)),
        }
    
    def _fuzzy_match(self, home: str, away: str) -> Optional[List]:
        """Try fuzzy matching for team names"""
        for (h, a), record in self.H2H_RECORDS.items():
            if h.lower() in home.lower() and a.lower() in away.lower():
                return record
        return None
    
    def get_h2h_goals_avg(self, home_team: str, away_team: str) -> float:
        """Get average goals in H2H matches"""
        key = (home_team, away_team)
        
        if key in self.H2H_RECORDS:
            record = self.H2H_RECORDS[key]
            total_matches = record[0] + record[1] + record[2]
            if total_matches > 0:
                return (record[3] + record[4]) / total_matches
        
        return 2.7  # League average


class LeaguePositionAnalyzer:
    """
    Factor in league standings for predictions.
    Top teams vs bottom teams, etc.
    """
    
    # Simulated standings: {team: position}
    LEAGUE_POSITIONS = {
        # Bundesliga
        'Bayern': 1, 'Leverkusen': 2, 'Stuttgart': 3, 'Leipzig': 4,
        'Dortmund': 5, 'Frankfurt': 6, 'Freiburg': 7, 'Wolfsburg': 8,
        'Gladbach': 9, 'Hoffenheim': 10, 'Bremen': 11, 'Union Berlin': 12,
        'Mainz': 13, 'Augsburg': 14, 'Heidenheim': 15, 'St. Pauli': 16,
        'Bochum': 17, 'Holstein Kiel': 18,
        
        # Premier League
        'Liverpool': 1, 'Arsenal': 2, 'Nottingham Forest': 3, 'Chelsea': 4,
        'Manchester City': 5, 'Newcastle': 6, 'Bournemouth': 7, 'Brighton': 8,
        'Aston Villa': 9, 'Fulham': 10, 'Tottenham': 11, 'Brentford': 12,
        'Manchester United': 13, 'West Ham': 14, 'Crystal Palace': 15,
        'Everton': 16, 'Wolves': 17, 'Leicester': 18, 'Ipswich': 19, 'Southampton': 20,
    }
    
    def get_position_factor(
        self,
        home_team: str,
        away_team: str,
        league_size: int = 18
    ) -> Dict[str, float]:
        """
        Calculate position-based adjustment.
        
        Position Gap affects predictions:
        - Big gap (top vs bottom) = more confident in favorite
        - Small gap = more balanced probabilities
        """
        home_pos = self._get_position(home_team)
        away_pos = self._get_position(away_team)
        
        # Normalize positions to 0-1 scale (0 = top, 1 = bottom)
        home_norm = (home_pos - 1) / (league_size - 1) if league_size > 1 else 0.5
        away_norm = (away_pos - 1) / (league_size - 1) if league_size > 1 else 0.5
        
        # Position gap: positive = home team higher ranked
        gap = away_norm - home_norm  # -1 to +1
        
        # Convert to probability adjustments (max ±0.15)
        home_adj = gap * 0.15
        away_adj = -gap * 0.15
        
        return {
            'home_position': home_pos,
            'away_position': away_pos,
            'position_gap': round(gap, 3),
            'home_adjustment': round(home_adj, 3),
            'away_adjustment': round(away_adj, 3),
        }
    
    def _get_position(self, team: str) -> int:
        """Get league position with fuzzy matching"""
        if team in self.LEAGUE_POSITIONS:
            return self.LEAGUE_POSITIONS[team]
        
        team_lower = team.lower()
        for name, pos in self.LEAGUE_POSITIONS.items():
            if name.lower() in team_lower or team_lower in name.lower():
                return pos
        
        return 10  # Mid-table default


class OddsIntegrator:
    """
    Compare model predictions with market odds to find value.
    """
    
    # Simulated market odds (decimal)
    MARKET_ODDS = {
        ('Bayern', 'Augsburg'): {'home': 1.22, 'draw': 7.5, 'away': 12.0},
        ('Dortmund', 'Bremen'): {'home': 1.50, 'draw': 4.5, 'away': 6.0},
        ('Liverpool', 'Arsenal'): {'home': 2.1, 'draw': 3.4, 'away': 3.5},
        ('Real Madrid', 'Barcelona'): {'home': 2.4, 'draw': 3.3, 'away': 2.9},
    }
    
    def get_implied_probabilities(
        self,
        home_team: str,
        away_team: str
    ) -> Optional[Dict[str, float]]:
        """Convert market odds to implied probabilities"""
        key = (home_team, away_team)
        
        odds = self.MARKET_ODDS.get(key)
        if not odds:
            # Generate realistic odds
            return self._generate_odds()
        
        # Convert to probabilities (remove margin)
        raw_home = 1 / odds['home']
        raw_draw = 1 / odds['draw']
        raw_away = 1 / odds['away']
        
        # Remove margin (normalize to 100%)
        total = raw_home + raw_draw + raw_away
        
        return {
            'home': raw_home / total,
            'draw': raw_draw / total,
            'away': raw_away / total,
            'margin': (total - 1) * 100,
        }
    
    def _generate_odds(self) -> Dict[str, float]:
        """Generate realistic implied probabilities"""
        return {
            'home': 0.45,
            'draw': 0.27,
            'away': 0.28,
            'margin': 5.0,
        }
    
    def find_value(
        self,
        our_probs: Dict[str, float],
        market_probs: Dict[str, float],
        min_edge: float = 0.05
    ) -> List[Dict]:
        """Find value bets where our probability > market probability"""
        value_bets = []
        
        for outcome in ['home', 'draw', 'away']:
            our_prob = our_probs.get(outcome, 0.33)
            market_prob = market_probs.get(outcome, 0.33)
            
            edge = our_prob - market_prob
            
            if edge >= min_edge:
                value_bets.append({
                    'outcome': outcome,
                    'our_probability': round(our_prob, 3),
                    'market_probability': round(market_prob, 3),
                    'edge': round(edge, 3),
                    'implied_odds': round(1 / market_prob, 2),
                })
        
        return value_bets


class ConfidenceCalibrator:
    """
    Calibrate predictions based on historical accuracy.
    If model is overconfident, dial back. If underconfident, boost.
    """
    
    def __init__(self):
        # Simulated calibration data: {confidence_bin: actual_accuracy}
        self.calibration = {
            0.5: 0.48,   # 50% confidence → 48% accurate
            0.55: 0.52,
            0.6: 0.57,
            0.65: 0.62,
            0.7: 0.66,
            0.75: 0.71,
            0.8: 0.75,
            0.85: 0.79,
            0.9: 0.82,
        }
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Adjust confidence based on historical accuracy.
        
        Returns calibrated confidence that better reflects true probability.
        """
        # Find closest calibration bin
        bins = sorted(self.calibration.keys())
        
        for i, threshold in enumerate(bins):
            if raw_confidence <= threshold:
                if i == 0:
                    return self.calibration[threshold]
                
                # Interpolate between bins
                prev_bin = bins[i - 1]
                ratio = (raw_confidence - prev_bin) / (threshold - prev_bin)
                
                return (
                    self.calibration[prev_bin] * (1 - ratio) +
                    self.calibration[threshold] * ratio
                )
        
        # Above highest bin
        return min(0.85, raw_confidence * 0.9)  # Cap overconfidence
    
    def get_calibration_factor(self, confidence: float) -> float:
        """Get the calibration adjustment factor"""
        calibrated = self.calibrate(confidence)
        return calibrated / confidence if confidence > 0 else 1.0


class AdvancedPredictor:
    """
    Master prediction class combining all advanced factors.
    """
    
    def __init__(self):
        self.form_tracker = FormMomentumTracker()
        self.h2h_analyzer = HeadToHeadAnalyzer()
        self.position_analyzer = LeaguePositionAnalyzer()
        self.odds_integrator = OddsIntegrator()
        self.calibrator = ConfidenceCalibrator()
        
        # Factor weights (can be tuned)
        self.weights = {
            'base_elo': 0.35,
            'form_momentum': 0.20,
            'h2h': 0.15,
            'position': 0.15,
            'home_advantage': 0.15,
        }
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        base_prediction: Optional[Dict] = None
    ) -> AdvancedPrediction:
        """
        Generate advanced prediction with all factors.
        """
        # Get all factors
        form_home = self.form_tracker.get_form_momentum(home_team)
        form_away = self.form_tracker.get_form_momentum(away_team)
        h2h = self.h2h_analyzer.get_h2h_factor(home_team, away_team)
        position = self.position_analyzer.get_position_factor(home_team, away_team)
        market = self.odds_integrator.get_implied_probabilities(home_team, away_team)
        
        # Base probabilities (from ELO or passed in)
        if base_prediction:
            home_base = base_prediction.get('home_win_prob', 0.4)
            draw_base = base_prediction.get('draw_prob', 0.28)
            away_base = base_prediction.get('away_win_prob', 0.32)
        else:
            # Generate from factors
            home_base = 0.38 + 0.05  # Home advantage
            draw_base = 0.27
            away_base = 0.35 - 0.05
        
        # Apply form momentum adjustment
        form_diff = form_home - form_away  # -1 to +1
        home_base += form_diff * 0.10
        away_base -= form_diff * 0.10
        
        # Apply H2H adjustment
        home_base += h2h['home']
        draw_base += h2h['draw']
        away_base += h2h['away']
        
        # Apply position adjustment
        home_base += position['home_adjustment']
        away_base += position['away_adjustment']
        
        # Normalize probabilities
        total = home_base + draw_base + away_base
        home_prob = max(0.05, home_base / total)
        draw_prob = max(0.05, draw_base / total)
        away_prob = max(0.05, away_base / total)
        
        # Re-normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        # Determine prediction and confidence
        if home_prob >= away_prob and home_prob >= draw_prob:
            predicted = 'Home Win'
            raw_conf = home_prob
        elif away_prob >= home_prob and away_prob >= draw_prob:
            predicted = 'Away Win'
            raw_conf = away_prob
        else:
            predicted = 'Draw'
            raw_conf = draw_prob
        
        # Calibrate confidence
        calibrated_conf = self.calibrator.calibrate(raw_conf)
        
        # Find value bets
        our_probs = {'home': home_prob, 'draw': draw_prob, 'away': away_prob}
        value_bets = self.odds_integrator.find_value(our_probs, market or {})
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            predicted, calibrated_conf, form_home, form_away, value_bets
        )
        
        return AdvancedPrediction(
            home_win_prob=round(home_prob, 3),
            draw_prob=round(draw_prob, 3),
            away_win_prob=round(away_prob, 3),
            predicted_outcome=predicted,
            raw_confidence=round(raw_conf, 3),
            calibrated_confidence=round(calibrated_conf, 3),
            factors={
                'form_home': round(form_home, 3),
                'form_away': round(form_away, 3),
                'form_diff': round(form_diff, 3),
                'h2h_home_adj': h2h['home'],
                'h2h_draw_adj': h2h['draw'],
                'h2h_away_adj': h2h['away'],
                'position_gap': position['position_gap'],
                'home_position': position['home_position'],
                'away_position': position['away_position'],
            },
            recommendations=recommendations,
            value_bets=value_bets
        )
    
    def _generate_recommendations(
        self,
        predicted: str,
        confidence: float,
        form_home: float,
        form_away: float,
        value_bets: List[Dict]
    ) -> List[str]:
        """Generate betting recommendations"""
        recs = []
        
        # Main prediction recommendation
        if confidence >= 0.7:
            recs.append(f"🔥 Strong pick: {predicted} ({confidence*100:.0f}% confidence)")
        elif confidence >= 0.55:
            recs.append(f"✅ Recommended: {predicted} ({confidence*100:.0f}% confidence)")
        else:
            recs.append(f"⚠️ Uncertain: {predicted} ({confidence*100:.0f}% confidence)")
        
        # Form-based recommendations
        if form_home > 0.7:
            recs.append("📈 Home team in excellent form")
        elif form_home < 0.4:
            recs.append("📉 Home team in poor form")
        
        if form_away > 0.7:
            recs.append("📈 Away team in excellent form")
        elif form_away < 0.4:
            recs.append("📉 Away team in poor form")
        
        # Value bet recommendations
        for vb in value_bets:
            if vb['edge'] >= 0.1:
                recs.append(f"💰 Strong value on {vb['outcome'].title()} (+{vb['edge']*100:.0f}% edge)")
            else:
                recs.append(f"💵 Value bet: {vb['outcome'].title()} (+{vb['edge']*100:.0f}% edge)")
        
        return recs


# Global instance
advanced_predictor = AdvancedPredictor()


def get_advanced_prediction(
    home_team: str,
    away_team: str,
    base_prediction: Optional[Dict] = None
) -> Dict:
    """Get advanced prediction with all factors"""
    pred = advanced_predictor.predict(home_team, away_team, base_prediction)
    
    return {
        'home_win_prob': pred.home_win_prob,
        'draw_prob': pred.draw_prob,
        'away_win_prob': pred.away_win_prob,
        'predicted_outcome': pred.predicted_outcome,
        'raw_confidence': pred.raw_confidence,
        'calibrated_confidence': pred.calibrated_confidence,
        'factors': pred.factors,
        'recommendations': pred.recommendations,
        'value_bets': pred.value_bets,
    }
