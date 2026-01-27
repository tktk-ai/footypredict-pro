"""
Evaluation Metrics Module
Calculates prediction and betting performance metrics.

Part of the complete blueprint implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score
)

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Calculates comprehensive prediction metrics.
    
    Categories:
    - Classification metrics
    - Probabilistic metrics
    - Betting metrics
    """
    
    def __init__(self):
        pass
    
    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None
    ) -> Dict:
        """Calculate classification metrics."""
        labels = labels or ['H', 'D', 'A']
        
        results = {
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            'f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
        }
        
        # Per-class metrics
        for i, label in enumerate(labels):
            y_true_binary = (y_true == i).astype(int) if isinstance(y_true[0], (int, np.integer)) else (y_true == label).astype(int)
            y_pred_binary = (y_pred == i).astype(int) if isinstance(y_pred[0], (int, np.integer)) else (y_pred == label).astype(int)
            
            results[f'{label}_precision'] = round(precision_score(y_true_binary, y_pred_binary, zero_division=0), 4)
            results[f'{label}_recall'] = round(recall_score(y_true_binary, y_pred_binary, zero_division=0), 4)
        
        return results
    
    def probabilistic_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Calculate probabilistic calibration metrics."""
        # Brier score (lower is better)
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
        
        # One-hot encode true labels
        if len(y_true.shape) == 1:
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i, label in enumerate(y_true):
                if isinstance(label, (int, np.integer)) and label < n_classes:
                    y_true_onehot[i, label] = 1
        else:
            y_true_onehot = y_true
        
        brier = np.mean((y_proba - y_true_onehot) ** 2)
        
        # Log loss
        try:
            ll = log_loss(y_true, y_proba)
        except Exception:
            ll = None
        
        # AUC (one-vs-rest)
        try:
            auc = roc_auc_score(y_true_onehot, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
        
        return {
            'brier_score': round(brier, 4),
            'log_loss': round(ll, 4) if ll else None,
            'auc_roc': round(auc, 4) if auc else None
        }
    
    def betting_metrics(
        self,
        predictions: List[Dict],
        outcomes: List[Dict],
        initial_bankroll: float = 1000
    ) -> Dict:
        """Calculate betting performance metrics."""
        if not predictions or not outcomes:
            return {}
        
        # Match predictions to outcomes
        results = []
        for pred, outcome in zip(predictions, outcomes):
            if pred.get('bet_placed', True):
                stake = pred.get('stake', pred.get('unit_stake', 10))
                odds = pred.get('odds', 2.0)
                won = pred.get('predicted_outcome') == outcome.get('actual_outcome')
                
                profit = stake * (odds - 1) if won else -stake
                results.append({
                    'stake': stake,
                    'odds': odds,
                    'won': won,
                    'profit': profit
                })
        
        if not results:
            return {}
        
        total_staked = sum(r['stake'] for r in results)
        total_profit = sum(r['profit'] for r in results)
        wins = sum(1 for r in results if r['won'])
        
        # Yield = profit / stake
        yield_pct = total_profit / total_staked * 100 if total_staked > 0 else 0
        
        # Max drawdown
        cumulative = [0]
        for r in results:
            cumulative.append(cumulative[-1] + r['profit'])
        
        peak = cumulative[0]
        max_dd = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe-like ratio
        returns = [r['profit'] / r['stake'] for r in results]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(results)) if np.std(returns) > 0 else 0
        
        return {
            'total_bets': len(results),
            'wins': wins,
            'losses': len(results) - wins,
            'win_rate': round(wins / len(results) * 100, 2),
            'total_staked': round(total_staked, 2),
            'total_profit': round(total_profit, 2),
            'yield': round(yield_pct, 2),
            'roi': round((total_profit / initial_bankroll) * 100, 2),
            'max_drawdown': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 4),
            'average_odds': round(np.mean([r['odds'] for r in results]), 2)
        }
    
    def calibration_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """Analyze probability calibration."""
        # Convert to binary for each class
        if len(y_proba.shape) == 1:
            y_proba = y_proba.reshape(-1, 1)
        
        n_classes = y_proba.shape[1]
        calibration = []
        
        for c in range(n_classes):
            class_proba = y_proba[:, c]
            class_true = (y_true == c).astype(int)
            
            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_means = []
            bin_true_frequencies = []
            
            for i in range(n_bins):
                mask = (class_proba >= bins[i]) & (class_proba < bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(class_proba[mask].mean())
                    bin_true_frequencies.append(class_true[mask].mean())
            
            if bin_means:
                calibration_error = np.mean(np.abs(np.array(bin_means) - np.array(bin_true_frequencies)))
            else:
                calibration_error = 0
            
            calibration.append({
                'class': c,
                'calibration_error': round(calibration_error, 4),
                'bin_means': [round(x, 4) for x in bin_means],
                'bin_frequencies': [round(x, 4) for x in bin_true_frequencies]
            })
        
        return {
            'by_class': calibration,
            'average_calibration_error': round(np.mean([c['calibration_error'] for c in calibration]), 4)
        }


_metrics: Optional[EvaluationMetrics] = None

def get_metrics() -> EvaluationMetrics:
    global _metrics
    if _metrics is None:
        _metrics = EvaluationMetrics()
    return _metrics
