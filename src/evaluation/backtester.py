"""
Backtester Module
Historical backtesting of prediction and betting strategies.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtests prediction models and betting strategies.
    
    Features:
    - Walk-forward testing
    - Out-of-sample validation
    - Strategy simulation
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000,
        stake_strategy: str = 'fixed'
    ):
        self.initial_bankroll = initial_bankroll
        self.stake_strategy = stake_strategy
        self.results: List[Dict] = []
    
    def run(
        self,
        matches: pd.DataFrame,
        predictor: Callable,
        start_date: str = None,
        end_date: str = None,
        train_window: int = 100
    ) -> Dict:
        """
        Run backtest on historical matches.
        
        Args:
            matches: DataFrame with match data
            predictor: Function that makes predictions
            start_date: Start of test period
            end_date: End of test period
            train_window: Number of matches for training
        """
        if 'match_date' in matches.columns:
            matches = matches.sort_values('match_date')
        
        if start_date:
            matches = matches[matches['match_date'] >= start_date]
        if end_date:
            matches = matches[matches['match_date'] <= end_date]
        
        bankroll = self.initial_bankroll
        results = []
        correct = 0
        total = 0
        
        for i in range(train_window, len(matches)):
            # Training data
            train = matches.iloc[:i]
            test_match = matches.iloc[i]
            
            try:
                # Make prediction
                prediction = predictor(
                    test_match['home_team'],
                    test_match['away_team'],
                    train
                )
                
                # Determine actual result
                actual = self._get_result(
                    test_match.get('home_goals', 0),
                    test_match.get('away_goals', 0)
                )
                
                # Check if correct
                predicted = prediction.get('prediction', prediction.get('result', ''))
                is_correct = predicted == actual
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Betting simulation
                stake = self._calculate_stake(bankroll, prediction)
                odds = prediction.get('odds', self._get_implied_odds(prediction))
                
                if prediction.get('bet', True) and stake > 0:
                    if is_correct:
                        profit = stake * (odds - 1)
                    else:
                        profit = -stake
                    
                    bankroll += profit
                else:
                    profit = 0
                
                results.append({
                    'date': test_match.get('match_date', i),
                    'home_team': test_match['home_team'],
                    'away_team': test_match['away_team'],
                    'predicted': predicted,
                    'actual': actual,
                    'correct': is_correct,
                    'stake': stake,
                    'odds': odds,
                    'profit': profit,
                    'bankroll': bankroll
                })
                
            except Exception as e:
                logger.warning(f"Prediction failed for match {i}: {e}")
                continue
        
        self.results = results
        
        return self.get_summary()
    
    def _get_result(self, home_goals: int, away_goals: int) -> str:
        """Get result code from goals."""
        if home_goals > away_goals:
            return 'H'
        elif home_goals < away_goals:
            return 'A'
        return 'D'
    
    def _calculate_stake(self, bankroll: float, prediction: Dict) -> float:
        """Calculate stake based on strategy."""
        if self.stake_strategy == 'fixed':
            return min(bankroll * 0.01, 10)
        elif self.stake_strategy == 'kelly':
            return prediction.get('kelly_stake', 0) * bankroll
        elif self.stake_strategy == 'proportional':
            confidence = prediction.get('confidence', 0.5)
            return bankroll * confidence * 0.02
        return 10
    
    def _get_implied_odds(self, prediction: Dict) -> float:
        """Get implied odds from prediction."""
        probs = prediction.get('1x2', {})
        max_prob = max(probs.values()) if probs else 0.33
        return 1 / max_prob if max_prob > 0 else 3.0
    
    def get_summary(self) -> Dict:
        """Get backtest summary."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        correct = df['correct'].sum()
        total = len(df)
        
        # Profit metrics
        total_profit = df['profit'].sum()
        total_staked = df['stake'].sum()
        
        # Max drawdown
        cumulative = df['profit'].cumsum()
        peak = cumulative.cummax()
        drawdown = peak - cumulative
        max_drawdown = drawdown.max()
        
        return {
            'period': {
                'start': str(df['date'].iloc[0]) if 'date' in df else None,
                'end': str(df['date'].iloc[-1]) if 'date' in df else None,
                'matches': total
            },
            'accuracy': {
                'correct': int(correct),
                'total': total,
                'accuracy_pct': round(correct / total * 100, 2) if total > 0 else 0
            },
            'profit': {
                'total_profit': round(total_profit, 2),
                'total_staked': round(total_staked, 2),
                'yield_pct': round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0,
                'roi_pct': round(total_profit / self.initial_bankroll * 100, 2)
            },
            'risk': {
                'max_drawdown': round(max_drawdown, 2),
                'final_bankroll': round(df['bankroll'].iloc[-1], 2),
                'peak_bankroll': round(df['bankroll'].max(), 2)
            }
        }
    
    def get_monthly_breakdown(self) -> pd.DataFrame:
        """Get monthly performance breakdown."""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly = df.groupby('month').agg({
                'profit': 'sum',
                'correct': 'sum',
                'stake': 'count'
            }).rename(columns={'stake': 'bets'})
            monthly['accuracy'] = monthly['correct'] / monthly['bets'] * 100
            
            return monthly
        
        return pd.DataFrame()


_backtester: Optional[Backtester] = None

def get_backtester() -> Backtester:
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester
