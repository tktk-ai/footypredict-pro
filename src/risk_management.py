"""
Risk Management Module

Advanced betting analytics including:
1. Historical accuracy tracking
2. High-confidence pattern detection  
3. Kelly Criterion bet sizing
4. Backtesting engine
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    match_id: str
    date: str
    home_team: str
    away_team: str
    league: str
    predicted_outcome: str
    predicted_prob: float
    actual_outcome: Optional[str]
    is_correct: Optional[bool]
    confidence: float
    bet_type: str  # 'match', 'over_2.5', 'btts'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BacktestResult:
    """Results from backtesting a strategy"""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_stake: float
    total_returns: float
    profit_loss: float
    roi: float
    max_drawdown: float
    win_streak: int
    lose_streak: int
    
    def to_dict(self) -> Dict:
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': round(self.accuracy, 4),
            'total_stake': round(self.total_stake, 2),
            'total_returns': round(self.total_returns, 2),
            'profit_loss': round(self.profit_loss, 2),
            'roi': round(self.roi, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'win_streak': self.win_streak,
            'lose_streak': self.lose_streak
        }


class AccuracyTracker:
    """
    Track prediction accuracy over time
    
    Stores predictions and results to calculate:
    - Overall accuracy
    - Accuracy by league
    - Accuracy by confidence level
    - Accuracy by bet type
    """
    
    def __init__(self, data_dir: str = "data/predictions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions: List[PredictionRecord] = []
        self._load_predictions()
    
    def _load_predictions(self):
        """Load historical predictions from file"""
        filepath = self.data_dir / "history.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.predictions = [
                        PredictionRecord(**p) for p in data
                    ]
            except:
                self.predictions = []
    
    def _save_predictions(self):
        """Save predictions to file"""
        filepath = self.data_dir / "history.json"
        with open(filepath, 'w') as f:
            json.dump([p.to_dict() for p in self.predictions], f, indent=2)
    
    def record_prediction(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        predicted_outcome: str,
        predicted_prob: float,
        confidence: float,
        bet_type: str = 'match'
    ):
        """Record a new prediction"""
        record = PredictionRecord(
            match_id=match_id,
            date=datetime.now().isoformat(),
            home_team=home_team,
            away_team=away_team,
            league=league,
            predicted_outcome=predicted_outcome,
            predicted_prob=predicted_prob,
            actual_outcome=None,
            is_correct=None,
            confidence=confidence,
            bet_type=bet_type
        )
        self.predictions.append(record)
        self._save_predictions()
        return record
    
    def record_result(self, match_id: str, actual_outcome: str):
        """Record the actual outcome of a match"""
        for pred in self.predictions:
            if pred.match_id == match_id and pred.actual_outcome is None:
                pred.actual_outcome = actual_outcome
                pred.is_correct = (pred.predicted_outcome == actual_outcome)
        self._save_predictions()
    
    def get_accuracy_stats(self) -> Dict:
        """Get overall accuracy statistics"""
        completed = [p for p in self.predictions if p.is_correct is not None]
        
        if not completed:
            return {
                'total': 0,
                'correct': 0,
                'accuracy': 0,
                'by_league': {},
                'by_confidence': {},
                'by_bet_type': {}
            }
        
        correct = sum(1 for p in completed if p.is_correct)
        
        # By league
        by_league = {}
        for p in completed:
            if p.league not in by_league:
                by_league[p.league] = {'total': 0, 'correct': 0}
            by_league[p.league]['total'] += 1
            if p.is_correct:
                by_league[p.league]['correct'] += 1
        
        for league in by_league:
            t, c = by_league[league]['total'], by_league[league]['correct']
            by_league[league]['accuracy'] = c / t if t > 0 else 0
        
        # By confidence level
        by_confidence = {
            'low (50-60%)': {'total': 0, 'correct': 0},
            'medium (60-75%)': {'total': 0, 'correct': 0},
            'high (75%+)': {'total': 0, 'correct': 0}
        }
        
        for p in completed:
            if p.confidence < 0.60:
                level = 'low (50-60%)'
            elif p.confidence < 0.75:
                level = 'medium (60-75%)'
            else:
                level = 'high (75%+)'
            
            by_confidence[level]['total'] += 1
            if p.is_correct:
                by_confidence[level]['correct'] += 1
        
        for level in by_confidence:
            t, c = by_confidence[level]['total'], by_confidence[level]['correct']
            by_confidence[level]['accuracy'] = c / t if t > 0 else 0
        
        # By bet type
        by_bet_type = {}
        for p in completed:
            if p.bet_type not in by_bet_type:
                by_bet_type[p.bet_type] = {'total': 0, 'correct': 0}
            by_bet_type[p.bet_type]['total'] += 1
            if p.is_correct:
                by_bet_type[p.bet_type]['correct'] += 1
        
        for bet_type in by_bet_type:
            t, c = by_bet_type[bet_type]['total'], by_bet_type[bet_type]['correct']
            by_bet_type[bet_type]['accuracy'] = c / t if t > 0 else 0
        
        return {
            'total': len(completed),
            'correct': correct,
            'accuracy': correct / len(completed),
            'by_league': by_league,
            'by_confidence': by_confidence,
            'by_bet_type': by_bet_type
        }


class PatternDetector:
    """
    Identify high-confidence patterns in predictions
    
    Finds situations where the model performs best:
    - Specific confidence thresholds
    - Certain league/matchup types
    - Goal market patterns
    """
    
    def __init__(self, accuracy_tracker: AccuracyTracker):
        self.tracker = accuracy_tracker
    
    def find_high_confidence_patterns(self, min_sample: int = 10) -> Dict:
        """Find patterns with high accuracy"""
        stats = self.tracker.get_accuracy_stats()
        patterns = []
        
        # Check league patterns
        for league, data in stats.get('by_league', {}).items():
            if data['total'] >= min_sample and data['accuracy'] >= 0.65:
                patterns.append({
                    'type': 'league',
                    'condition': league,
                    'accuracy': data['accuracy'],
                    'sample_size': data['total'],
                    'edge': data['accuracy'] - 0.5  # Edge over random
                })
        
        # Check confidence level patterns
        for level, data in stats.get('by_confidence', {}).items():
            if data['total'] >= min_sample and data['accuracy'] >= 0.60:
                patterns.append({
                    'type': 'confidence',
                    'condition': level,
                    'accuracy': data['accuracy'],
                    'sample_size': data['total'],
                    'edge': data['accuracy'] - 0.5
                })
        
        # Check bet type patterns
        for bet_type, data in stats.get('by_bet_type', {}).items():
            if data['total'] >= min_sample and data['accuracy'] >= 0.55:
                patterns.append({
                    'type': 'bet_type',
                    'condition': bet_type,
                    'accuracy': data['accuracy'],
                    'sample_size': data['total'],
                    'edge': data['accuracy'] - 0.5
                })
        
        # Sort by edge
        patterns.sort(key=lambda x: x['edge'], reverse=True)
        
        return {
            'patterns': patterns,
            'recommendations': self._generate_recommendations(patterns)
        }
    
    def _generate_recommendations(self, patterns: List[Dict]) -> List[str]:
        """Generate actionable recommendations from patterns"""
        recs = []
        
        if not patterns:
            recs.append("Not enough data yet - keep tracking predictions!")
            return recs
        
        for p in patterns[:3]:  # Top 3 patterns
            if p['type'] == 'league':
                recs.append(f"✅ Focus on {p['condition']}: {p['accuracy']:.1%} accuracy")
            elif p['type'] == 'confidence':
                recs.append(f"✅ Best at {p['condition']}: {p['accuracy']:.1%} accuracy")
            elif p['type'] == 'bet_type':
                recs.append(f"✅ {p['condition']} bets: {p['accuracy']:.1%} accuracy")
        
        return recs


class KellyCriterion:
    """
    Kelly Criterion for optimal bet sizing
    
    Calculates the optimal fraction of bankroll to bet
    to maximize long-term growth while managing risk.
    
    Formula: f* = (p * b - q) / b
    Where:
        p = probability of winning
        b = decimal odds - 1 (net odds)
        q = probability of losing (1 - p)
    """
    
    def __init__(self, bankroll: float = 100.0, max_fraction: float = 0.25):
        self.bankroll = bankroll
        self.max_fraction = max_fraction  # Never bet more than 25% of bankroll
    
    def calculate_kelly(self, win_prob: float, decimal_odds: float) -> Dict:
        """
        Calculate Kelly bet size
        
        Args:
            win_prob: Estimated probability of winning
            decimal_odds: Decimal odds offered by bookmaker
        
        Returns:
            Dictionary with bet sizing recommendations
        """
        if win_prob <= 0 or win_prob >= 1:
            return {'error': 'Invalid probability'}
        
        if decimal_odds <= 1:
            return {'error': 'Invalid odds'}
        
        # Calculate Kelly fraction
        b = decimal_odds - 1  # Net odds
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # Apply constraints
        if kelly_fraction <= 0:
            return {
                'recommendation': 'NO BET',
                'kelly_fraction': kelly_fraction,
                'reason': 'Negative expected value - odds too low',
                'bet_amount': 0,
                'expected_value': (p * decimal_odds) - 1
            }
        
        # Cap at max fraction
        actual_fraction = min(kelly_fraction, self.max_fraction)
        bet_amount = self.bankroll * actual_fraction
        
        # Calculate fractional Kelly options
        full_kelly = bet_amount
        half_kelly = bet_amount * 0.5
        quarter_kelly = bet_amount * 0.25
        
        # Expected value
        ev = (p * decimal_odds) - 1
        
        return {
            'recommendation': 'BET' if kelly_fraction > 0.02 else 'SMALL BET',
            'kelly_fraction': round(kelly_fraction, 4),
            'actual_fraction': round(actual_fraction, 4),
            'bet_amount': round(bet_amount, 2),
            'full_kelly': round(full_kelly, 2),
            'half_kelly': round(half_kelly, 2),
            'quarter_kelly': round(quarter_kelly, 2),
            'expected_value': round(ev, 4),
            'edge_percent': round(kelly_fraction * 100, 2),
            'bankroll': self.bankroll
        }
    
    def update_bankroll(self, amount: float):
        """Update bankroll (after win/loss)"""
        self.bankroll += amount
        return self.bankroll


class Backtester:
    """
    Backtest betting strategies on historical data
    
    Simulates applying a betting strategy to past matches
    to evaluate expected performance.
    """
    
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
    
    def backtest_strategy(
        self,
        predictions: List[Dict],
        stake_type: str = 'fixed',  # 'fixed', 'kelly', 'percentage'
        stake_amount: float = 10.0,
        min_confidence: float = 0.55,
        min_odds: float = 1.3
    ) -> BacktestResult:
        """
        Run backtest on historical predictions
        
        Args:
            predictions: List of prediction dicts with 'predicted_prob', 'actual_outcome', 'odds'
            stake_type: How to size bets
            stake_amount: Fixed stake or percentage
            min_confidence: Minimum confidence to place bet
            min_odds: Minimum odds to accept
        """
        bankroll = self.initial_bankroll
        total_stake = 0
        total_returns = 0
        correct = 0
        total = 0
        
        max_bankroll = bankroll
        min_bankroll = bankroll
        max_drawdown = 0
        
        current_streak = 0
        win_streak = 0
        lose_streak = 0
        last_was_win = None
        
        kelly = KellyCriterion(bankroll)
        
        for pred in predictions:
            prob = pred.get('predicted_prob', 0)
            actual = pred.get('actual_outcome')
            predicted = pred.get('predicted_outcome')
            odds = pred.get('odds', 1.9)  # Default odds
            
            if prob < min_confidence or odds < min_odds:
                continue
            
            if actual is None:
                continue
            
            # Calculate stake
            if stake_type == 'fixed':
                stake = min(stake_amount, bankroll)
            elif stake_type == 'percentage':
                stake = bankroll * (stake_amount / 100)
            elif stake_type == 'kelly':
                kelly.bankroll = bankroll
                k_result = kelly.calculate_kelly(prob, odds)
                stake = k_result.get('quarter_kelly', stake_amount)
            else:
                stake = stake_amount
            
            if stake <= 0 or bankroll <= 0:
                continue
            
            total += 1
            total_stake += stake
            bankroll -= stake
            
            # Check result
            is_win = (predicted == actual)
            
            if is_win:
                winnings = stake * odds
                bankroll += winnings
                total_returns += winnings
                correct += 1
                
                if last_was_win:
                    current_streak += 1
                else:
                    current_streak = 1
                win_streak = max(win_streak, current_streak)
                last_was_win = True
            else:
                if not last_was_win:
                    current_streak += 1
                else:
                    current_streak = 1
                lose_streak = max(lose_streak, current_streak)
                last_was_win = False
            
            # Track drawdown
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)
            current_drawdown = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)
        
        profit_loss = bankroll - self.initial_bankroll
        roi = profit_loss / total_stake if total_stake > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return BacktestResult(
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            total_stake=total_stake,
            total_returns=total_returns,
            profit_loss=profit_loss,
            roi=roi,
            max_drawdown=max_drawdown,
            win_streak=win_streak,
            lose_streak=lose_streak
        )
    
    def simulate_strategies(
        self,
        predictions: List[Dict],
        confidence_levels: List[float] = [0.55, 0.60, 0.65, 0.70]
    ) -> Dict:
        """Compare different strategies"""
        results = {}
        
        for min_conf in confidence_levels:
            result = self.backtest_strategy(
                predictions,
                stake_type='fixed',
                min_confidence=min_conf
            )
            results[f'confidence_{int(min_conf*100)}%'] = result.to_dict()
        
        # Find best strategy
        best_roi = -999
        best_strategy = None
        for name, result in results.items():
            if result['roi'] > best_roi and result['total_predictions'] >= 10:
                best_roi = result['roi']
                best_strategy = name
        
        return {
            'strategies': results,
            'best_strategy': best_strategy,
            'best_roi': best_roi
        }


# Global instances
accuracy_tracker = AccuracyTracker()
pattern_detector = PatternDetector(accuracy_tracker)
kelly_calculator = KellyCriterion()
backtester = Backtester()
