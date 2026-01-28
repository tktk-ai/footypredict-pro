"""
Evaluation Reporting
====================
Generate comprehensive evaluation reports for model performance and predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "reports"
    include_charts: bool = True
    format: str = "markdown"  # 'markdown', 'html', 'json'
    date_range_days: int = 30


class EvaluationReporter:
    """
    Generates comprehensive evaluation reports.
    
    Reports include:
    - Model accuracy metrics
    - ROI and profit analysis
    - Market-specific performance
    - Time-series analysis
    - Confidence calibration
    """
    
    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_accuracy_report(
        self,
        predictions: List[Dict],
        results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate accuracy metrics report.
        
        Args:
            predictions: List of prediction records
            results: List of actual results
            
        Returns:
            Report dictionary with metrics
        """
        if not predictions or not results:
            return {'error': 'No data provided'}
        
        # Match predictions to results
        matched = self._match_predictions_results(predictions, results)
        
        if not matched:
            return {'error': 'No matched predictions'}
        
        # Calculate metrics
        correct = sum(1 for m in matched if m['correct'])
        total = len(matched)
        accuracy = correct / total if total > 0 else 0
        
        # By confidence level
        confidence_buckets = self._bucket_by_confidence(matched)
        
        # By market
        market_accuracy = self._accuracy_by_market(matched)
        
        # By time period
        time_analysis = self._accuracy_over_time(matched)
        
        report = {
            'summary': {
                'total_predictions': total,
                'correct': correct,
                'accuracy': round(accuracy * 100, 2),
                'generated_at': datetime.now().isoformat()
            },
            'by_confidence': confidence_buckets,
            'by_market': market_accuracy,
            'time_analysis': time_analysis
        }
        
        return report
    
    def generate_roi_report(
        self,
        bets: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate ROI and profit analysis report.
        
        Args:
            bets: List of bet records with stakes, odds, and results
            
        Returns:
            ROI report
        """
        if not bets:
            return {'error': 'No bet data'}
        
        total_stake = sum(b.get('stake', 0) for b in bets)
        total_returns = sum(
            b.get('stake', 0) * b.get('odds', 1) 
            for b in bets if b.get('won', False)
        )
        
        profit = total_returns - total_stake
        roi = (profit / total_stake * 100) if total_stake > 0 else 0
        
        # Win rate
        wins = sum(1 for b in bets if b.get('won', False))
        win_rate = wins / len(bets) * 100 if bets else 0
        
        # By market
        market_roi = {}
        for b in bets:
            market = b.get('market', 'unknown')
            if market not in market_roi:
                market_roi[market] = {'stake': 0, 'returns': 0, 'bets': 0}
            
            market_roi[market]['stake'] += b.get('stake', 0)
            market_roi[market]['bets'] += 1
            if b.get('won', False):
                market_roi[market]['returns'] += b.get('stake', 0) * b.get('odds', 1)
        
        for market, data in market_roi.items():
            data['roi'] = round(
                (data['returns'] - data['stake']) / data['stake'] * 100
                if data['stake'] > 0 else 0, 2
            )
        
        return {
            'summary': {
                'total_bets': len(bets),
                'total_stake': round(total_stake, 2),
                'total_returns': round(total_returns, 2),
                'profit': round(profit, 2),
                'roi': round(roi, 2),
                'win_rate': round(win_rate, 2)
            },
            'by_market': market_roi
        }
    
    def generate_model_comparison_report(
        self,
        model_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Compare performance across multiple models.
        
        Args:
            model_results: Dict mapping model name to prediction results
            
        Returns:
            Comparison report
        """
        comparison = {}
        
        for model_name, results in model_results.items():
            if not results:
                continue
            
            correct = sum(1 for r in results if r.get('correct', False))
            total = len(results)
            
            # Brier score
            brier_scores = []
            for r in results:
                prob = r.get('confidence', 0.5)
                outcome = 1 if r.get('correct', False) else 0
                brier_scores.append((prob - outcome) ** 2)
            
            avg_brier = np.mean(brier_scores) if brier_scores else 0
            
            comparison[model_name] = {
                'total': total,
                'correct': correct,
                'accuracy': round(correct / total * 100, 2) if total > 0 else 0,
                'brier_score': round(avg_brier, 4),
                'avg_confidence': round(np.mean([r.get('confidence', 0.5) for r in results]), 3)
            }
        
        # Rank models
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        return {
            'models': comparison,
            'ranking': [{'model': m, 'accuracy': d['accuracy']} for m, d in ranked],
            'best_model': ranked[0][0] if ranked else None
        }
    
    def generate_calibration_report(
        self,
        predictions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate probability calibration report.
        """
        buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        calibration = {f"0.{int(b*10-1)}-{b}": {'count': 0, 'wins': 0} for b in buckets}
        
        for pred in predictions:
            conf = pred.get('confidence', 0.5)
            correct = pred.get('correct', False)
            
            for b in buckets:
                if conf <= b:
                    key = f"0.{int(b*10-1)}-{b}"
                    calibration[key]['count'] += 1
                    if correct:
                        calibration[key]['wins'] += 1
                    break
        
        # Calculate actual vs expected
        for key, data in calibration.items():
            if data['count'] > 0:
                data['actual_rate'] = round(data['wins'] / data['count'], 3)
                expected = float(key.split('-')[1])
                data['expected_rate'] = expected
                data['calibration_error'] = round(abs(data['actual_rate'] - expected), 3)
            else:
                data['actual_rate'] = 0
                data['expected_rate'] = float(key.split('-')[1])
                data['calibration_error'] = 0
        
        # Overall calibration error
        errors = [d['calibration_error'] for d in calibration.values() if d['count'] > 0]
        avg_error = np.mean(errors) if errors else 0
        
        return {
            'buckets': calibration,
            'average_calibration_error': round(avg_error, 4),
            'is_well_calibrated': avg_error < 0.05
        }
    
    def save_report(
        self,
        report: Dict,
        filename: str
    ) -> str:
        """Save report to file."""
        filepath = self.output_dir / filename
        
        if self.config.format == 'json':
            filepath = filepath.with_suffix('.json')
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif self.config.format == 'markdown':
            filepath = filepath.with_suffix('.md')
            md_content = self._report_to_markdown(report)
            with open(filepath, 'w') as f:
                f.write(md_content)
        
        elif self.config.format == 'html':
            filepath = filepath.with_suffix('.html')
            html_content = self._report_to_html(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def _match_predictions_results(
        self,
        predictions: List[Dict],
        results: List[Dict]
    ) -> List[Dict]:
        """Match predictions with actual results."""
        matched = []
        
        results_map = {
            (r.get('home_team'), r.get('away_team'), r.get('date')): r
            for r in results
        }
        
        for pred in predictions:
            key = (pred.get('home_team'), pred.get('away_team'), pred.get('date'))
            if key in results_map:
                result = results_map[key]
                matched.append({
                    **pred,
                    'actual_result': result.get('result'),
                    'correct': pred.get('prediction') == result.get('result')
                })
        
        return matched
    
    def _bucket_by_confidence(self, matched: List[Dict]) -> Dict:
        """Bucket accuracy by confidence level."""
        buckets = {
            'very_high (90%+)': [],
            'high (75-90%)': [],
            'medium (60-75%)': [],
            'low (<60%)': []
        }
        
        for m in matched:
            conf = m.get('confidence', 0.5)
            if conf >= 0.9:
                buckets['very_high (90%+)'].append(m)
            elif conf >= 0.75:
                buckets['high (75-90%)'].append(m)
            elif conf >= 0.6:
                buckets['medium (60-75%)'].append(m)
            else:
                buckets['low (<60%)'].append(m)
        
        result = {}
        for bucket, items in buckets.items():
            if items:
                correct = sum(1 for i in items if i['correct'])
                result[bucket] = {
                    'total': len(items),
                    'correct': correct,
                    'accuracy': round(correct / len(items) * 100, 2)
                }
        
        return result
    
    def _accuracy_by_market(self, matched: List[Dict]) -> Dict:
        """Calculate accuracy by market type."""
        markets = {}
        
        for m in matched:
            market = m.get('market', 'unknown')
            if market not in markets:
                markets[market] = {'total': 0, 'correct': 0}
            
            markets[market]['total'] += 1
            if m['correct']:
                markets[market]['correct'] += 1
        
        for market, data in markets.items():
            data['accuracy'] = round(
                data['correct'] / data['total'] * 100, 2
            ) if data['total'] > 0 else 0
        
        return markets
    
    def _accuracy_over_time(self, matched: List[Dict]) -> Dict:
        """Analyze accuracy over time."""
        # Group by week
        weekly = {}
        
        for m in matched:
            date_str = m.get('date', '')
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str)
                    week = date.strftime('%Y-W%W')
                    
                    if week not in weekly:
                        weekly[week] = {'total': 0, 'correct': 0}
                    
                    weekly[week]['total'] += 1
                    if m['correct']:
                        weekly[week]['correct'] += 1
                except:
                    pass
        
        for week, data in weekly.items():
            data['accuracy'] = round(
                data['correct'] / data['total'] * 100, 2
            ) if data['total'] > 0 else 0
        
        return weekly
    
    def _report_to_markdown(self, report: Dict) -> str:
        """Convert report to markdown format."""
        lines = ["# Evaluation Report\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        if 'summary' in report:
            lines.append("\n## Summary\n")
            for key, value in report['summary'].items():
                lines.append(f"- **{key}**: {value}")
        
        if 'by_market' in report:
            lines.append("\n## Performance by Market\n")
            lines.append("| Market | Total | Correct | Accuracy |")
            lines.append("|--------|-------|---------|----------|")
            for market, data in report['by_market'].items():
                lines.append(
                    f"| {market} | {data.get('total', 0)} | "
                    f"{data.get('correct', 0)} | {data.get('accuracy', 0)}% |"
                )
        
        if 'by_confidence' in report:
            lines.append("\n## Performance by Confidence\n")
            for bucket, data in report['by_confidence'].items():
                lines.append(f"- **{bucket}**: {data.get('accuracy', 0)}% ({data.get('total', 0)} predictions)")
        
        return '\n'.join(lines)
    
    def _report_to_html(self, report: Dict) -> str:
        """Convert report to HTML format."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .metric { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Evaluation Report</h1>
"""
        
        if 'summary' in report:
            html += "<h2>Summary</h2><ul>"
            for key, value in report['summary'].items():
                html += f"<li><strong>{key}</strong>: {value}</li>"
            html += "</ul>"
        
        html += "</body></html>"
        return html


# Global instance
_reporter: Optional[EvaluationReporter] = None


def get_reporter() -> EvaluationReporter:
    """Get or create reporter instance."""
    global _reporter
    if _reporter is None:
        _reporter = EvaluationReporter()
    return _reporter


def generate_report(
    predictions: List[Dict],
    results: List[Dict],
    save: bool = True
) -> Dict:
    """Quick function to generate and optionally save a report."""
    reporter = get_reporter()
    report = reporter.generate_accuracy_report(predictions, results)
    
    if save:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reporter.save_report(report, filename)
    
    return report
