"""
Analytics Dashboard API

Provides endpoints for:
- Today's prediction stats (from real database)
- Historical accuracy tracking
- Performance breakdown by league/market
- ROI calculations

Connects to the predictions SQLite database.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Database path - same as scheduler.py
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.db')


class AnalyticsEngine:
    """Generate analytics from real prediction database."""
    
    def __init__(self):
        self.db_path = DB_PATH
    
    def _get_conn(self):
        """Get database connection."""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found at {self.db_path}")
            return None
        return sqlite3.connect(self.db_path)
    
    def get_today_summary(self) -> Dict:
        """Get today's prediction summary from real database."""
        conn = self._get_conn()
        
        if not conn:
            return self._get_fallback_summary()
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            today = datetime.now().date().isoformat()
            
            # Get today's predictions
            cursor.execute('''
                SELECT p.*, f.league_id, f.match_date, f.status as match_status,
                       f.home_score, f.away_score
                FROM predictions p
                LEFT JOIN fixtures f ON p.fixture_id = f.id
                WHERE date(p.created_at) = date(?)
                ORDER BY p.confidence DESC
            ''', (today,))
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            if not predictions:
                # Try getting most recent predictions if none today
                cursor.execute('''
                    SELECT p.*, f.league_id, f.match_date, f.status as match_status,
                           f.home_score, f.away_score
                    FROM predictions p
                    LEFT JOIN fixtures f ON p.fixture_id = f.id
                    ORDER BY p.created_at DESC
                    LIMIT 50
                ''')
                predictions = [dict(row) for row in cursor.fetchall()]
            
            if not predictions:
                return self._get_fallback_summary()
            
            # Calculate stats
            total = len(predictions)
            confidences = [p.get('confidence', 0) or 0 for p in predictions]
            
            high_conf = len([c for c in confidences if c >= 0.80])
            med_conf = len([c for c in confidences if 0.60 <= c < 0.80])
            low_conf = len([c for c in confidences if c < 0.60])
            
            # Calculate won/lost from match results
            won = 0
            lost = 0
            pending = 0
            
            for pred in predictions:
                match_status = pred.get('match_status', 'scheduled')
                if match_status in ['finished', 'ft', 'FINISHED']:
                    home_score = pred.get('home_score')
                    away_score = pred.get('away_score')
                    predicted = pred.get('predicted_outcome', '')
                    
                    if home_score is not None and away_score is not None:
                        actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                        if predicted == actual:
                            won += 1
                        else:
                            lost += 1
                    else:
                        pending += 1
                else:
                    pending += 1
            
            accuracy = round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0
            avg_conf = round(sum(confidences) / total * 100, 1) if total > 0 else 0
            
            # Group by league
            by_league = self._group_by_league(predictions)
            
            # Top picks
            top_picks = []
            for pred in predictions[:5]:
                status = 'pending'
                if pred.get('match_status') in ['finished', 'ft', 'FINISHED']:
                    home_score = pred.get('home_score')
                    away_score = pred.get('away_score')
                    if home_score is not None and away_score is not None:
                        actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                        status = 'won' if pred.get('predicted_outcome') == actual else 'lost'
                
                top_picks.append({
                    'match': f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}",
                    'confidence': int((pred.get('confidence', 0) or 0) * 100),
                    'prediction': pred.get('predicted_outcome', 'N/A'),
                    'status': status
                })
            
            return {
                'date': today,
                'total_predictions': total,
                'avg_confidence': avg_conf,
                'confidence_distribution': {
                    'high': high_conf,
                    'medium': med_conf,
                    'low': low_conf
                },
                'status': {
                    'pending': pending,
                    'won': won,
                    'lost': lost,
                    'accuracy': accuracy
                },
                'by_league': by_league,
                'top_picks': top_picks,
                'data_source': 'database'
            }
            
        except Exception as e:
            logger.error(f"Error getting today's summary: {e}")
            return self._get_fallback_summary()
        finally:
            conn.close()
    
    def _get_fallback_summary(self) -> Dict:
        """Return fallback summary when no data available."""
        today = datetime.now().date().isoformat()
        return {
            'date': today,
            'total_predictions': 0,
            'avg_confidence': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'status': {'pending': 0, 'won': 0, 'lost': 0, 'accuracy': 0},
            'by_league': [],
            'top_picks': [],
            'data_source': 'fallback',
            'message': 'No predictions in database. Run prediction generator job.'
        }
    
    def get_accuracy_history(self, days: int = 30) -> Dict:
        """Get accuracy history from database."""
        conn = self._get_conn()
        
        if not conn:
            return self._get_fallback_history(days)
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            history = []
            for i in range(days, 0, -1):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                
                # Get predictions for this date
                cursor.execute('''
                    SELECT p.*, f.home_score, f.away_score, f.status as match_status
                    FROM predictions p
                    LEFT JOIN fixtures f ON p.fixture_id = f.id
                    WHERE date(p.created_at) = date(?)
                ''', (date,))
                
                preds = [dict(row) for row in cursor.fetchall()]
                
                if preds:
                    total = len(preds)
                    won = 0
                    for pred in preds:
                        if pred.get('match_status') in ['finished', 'ft', 'FINISHED']:
                            home_score = pred.get('home_score')
                            away_score = pred.get('away_score')
                            if home_score is not None and away_score is not None:
                                actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                                if pred.get('predicted_outcome') == actual:
                                    won += 1
                    
                    accuracy = round(won / total * 100, 1) if total > 0 else 0
                    history.append({
                        'date': date,
                        'accuracy': accuracy,
                        'total': total,
                        'won': won
                    })
                else:
                    history.append({
                        'date': date,
                        'accuracy': 0,
                        'total': 0,
                        'won': 0
                    })
            
            # Calculate overall
            total_preds = sum(h['total'] for h in history)
            total_won = sum(h['won'] for h in history)
            
            return {
                'period': f'{days} days',
                'history': history,
                'overall': {
                    'total_predictions': total_preds,
                    'correct': total_won,
                    'accuracy': round(total_won / total_preds * 100, 1) if total_preds else 0
                },
                'data_source': 'database'
            }
            
        except Exception as e:
            logger.error(f"Error getting accuracy history: {e}")
            return self._get_fallback_history(days)
        finally:
            conn.close()
    
    def _get_fallback_history(self, days: int) -> Dict:
        """Return fallback history data."""
        history = []
        for i in range(days, 0, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            history.append({'date': date, 'accuracy': 0, 'total': 0, 'won': 0})
        
        return {
            'period': f'{days} days',
            'history': history,
            'overall': {'total_predictions': 0, 'correct': 0, 'accuracy': 0},
            'data_source': 'fallback'
        }
    
    def get_performance_by_league(self) -> List[Dict]:
        """Get performance breakdown by league from database."""
        conn = self._get_conn()
        
        if not conn:
            return []
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT f.league_id, p.predicted_outcome, f.home_score, f.away_score, f.status
                FROM predictions p
                JOIN fixtures f ON p.fixture_id = f.id
                WHERE f.status IN ('finished', 'ft', 'FINISHED')
            ''')
            
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results:
                return []
            
            # Group by league
            by_league = defaultdict(lambda: {'total': 0, 'won': 0})
            
            for r in results:
                league = r.get('league_id', 'unknown')
                by_league[league]['total'] += 1
                
                home_score = r.get('home_score')
                away_score = r.get('away_score')
                if home_score is not None and away_score is not None:
                    actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                    if r.get('predicted_outcome') == actual:
                        by_league[league]['won'] += 1
            
            leagues = []
            for league, stats in by_league.items():
                accuracy = round(stats['won'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0
                # Simple ROI estimate (assuming average odds of 1.9)
                roi = round((stats['won'] * 1.9 - stats['total']) / stats['total'] * 100, 1) if stats['total'] > 0 else 0
                
                leagues.append({
                    'league': league.replace('_', ' ').title(),
                    'predictions': stats['total'],
                    'won': stats['won'],
                    'accuracy': accuracy,
                    'roi': roi
                })
            
            return sorted(leagues, key=lambda x: x['accuracy'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting league performance: {e}")
            return []
        finally:
            conn.close()
    
    def get_performance_by_market(self) -> List[Dict]:
        """Get performance breakdown by market type."""
        conn = self._get_conn()
        
        if not conn:
            return []
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, f.home_score, f.away_score, f.status
                FROM predictions p
                JOIN fixtures f ON p.fixture_id = f.id
                WHERE f.status IN ('finished', 'ft', 'FINISHED')
            ''')
            
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results:
                return []
            
            markets = {
                '1X2': {'total': 0, 'won': 0, 'avg_odds': 1.85},
                'Over 2.5': {'total': 0, 'won': 0, 'avg_odds': 1.80},
                'BTTS': {'total': 0, 'won': 0, 'avg_odds': 1.85}
            }
            
            for r in results:
                home_score = r.get('home_score')
                away_score = r.get('away_score')
                
                if home_score is None or away_score is None:
                    continue
                
                total_goals = home_score + away_score
                both_scored = home_score > 0 and away_score > 0
                
                # 1X2 accuracy
                markets['1X2']['total'] += 1
                actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                if r.get('predicted_outcome') == actual:
                    markets['1X2']['won'] += 1
                
                # Over 2.5 accuracy
                over_25_prob = r.get('over_2_5', 0) or 0
                if over_25_prob > 0.5:
                    markets['Over 2.5']['total'] += 1
                    if total_goals > 2.5:
                        markets['Over 2.5']['won'] += 1
                
                # BTTS accuracy
                btts_prob = r.get('btts_yes', 0) or 0
                if btts_prob > 0.5:
                    markets['BTTS']['total'] += 1
                    if both_scored:
                        markets['BTTS']['won'] += 1
            
            result = []
            for market, stats in markets.items():
                if stats['total'] > 0:
                    accuracy = round(stats['won'] / stats['total'] * 100, 1)
                    result.append({
                        'market': market,
                        'predictions': stats['total'],
                        'won': stats['won'],
                        'accuracy': accuracy,
                        'avgOdds': stats['avg_odds']
                    })
            
            return sorted(result, key=lambda x: x['accuracy'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting market performance: {e}")
            return []
        finally:
            conn.close()
    
    def calculate_roi(self, stake: float = 10.0, period_days: int = 30) -> Dict:
        """Calculate ROI from database predictions."""
        conn = self._get_conn()
        
        if not conn:
            return self._get_fallback_roi(stake, period_days)
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=period_days)).isoformat()
            
            cursor.execute('''
                SELECT p.predicted_outcome, p.confidence, f.home_score, f.away_score
                FROM predictions p
                JOIN fixtures f ON p.fixture_id = f.id
                WHERE f.status IN ('finished', 'ft', 'FINISHED')
                AND p.created_at >= ?
            ''', (start_date,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results:
                return self._get_fallback_roi(stake, period_days)
            
            total_bets = len(results)
            wins = 0
            avg_odds = 1.85  # Assumed average odds
            
            for r in results:
                home_score = r.get('home_score')
                away_score = r.get('away_score')
                if home_score is not None and away_score is not None:
                    actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                    if r.get('predicted_outcome') == actual:
                        wins += 1
            
            total_staked = total_bets * stake
            total_returns = wins * stake * avg_odds
            profit = total_returns - total_staked
            roi = (profit / total_staked) * 100 if total_staked > 0 else 0
            
            return {
                'period': f'{period_days} days',
                'stake_per_bet': stake,
                'total_bets': total_bets,
                'wins': wins,
                'losses': total_bets - wins,
                'win_rate': round(wins / total_bets * 100, 1) if total_bets > 0 else 0,
                'avg_odds': avg_odds,
                'total_staked': round(total_staked, 2),
                'total_returns': round(total_returns, 2),
                'profit': round(profit, 2),
                'roi': round(roi, 1),
                'data_source': 'database'
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return self._get_fallback_roi(stake, period_days)
        finally:
            conn.close()
    
    def _get_fallback_roi(self, stake: float, period_days: int) -> Dict:
        """Return fallback ROI data."""
        return {
            'period': f'{period_days} days',
            'stake_per_bet': stake,
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'avg_odds': 1.85,
            'total_staked': 0,
            'total_returns': 0,
            'profit': 0,
            'roi': 0,
            'data_source': 'fallback'
        }
    
    def _group_by_league(self, predictions: List[Dict]) -> List[Dict]:
        """Group predictions by league."""
        by_league = defaultdict(list)
        for pred in predictions:
            league = pred.get('league_id', 'unknown')
            by_league[league].append(pred)
        
        result = []
        for league, preds in by_league.items():
            confidences = [(p.get('confidence', 0) or 0) for p in preds]
            result.append({
                'league': league.replace('_', ' ').title() if league else 'Unknown',
                'count': len(preds),
                'avg_conf': round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0
            })
        
        return sorted(result, key=lambda x: x['count'], reverse=True)
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_conn()
        
        if not conn:
            return {'status': 'no_database'}
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM predictions')
            pred_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM fixtures')
            fixture_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM odds')
            odds_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT MAX(created_at) FROM predictions')
            last_pred = cursor.fetchone()[0]
            
            return {
                'status': 'connected',
                'predictions_count': pred_count,
                'fixtures_count': fixture_count,
                'odds_count': odds_count,
                'last_prediction': last_pred,
                'database_path': self.db_path
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
        finally:
            conn.close()
    
    def get_section_analytics(self, section: str = 'all') -> Dict:
        """
        Get analytics by section:
        - daily_tips: Regular match predictions
        - money_zone: Time-based predictions (morning/afternoon/evening)
        - accas: Accumulator bets
        - all: Combined stats for all sections
        """
        conn = self._get_conn()
        
        if not conn:
            return self._get_section_fallback(section)
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all predictions
            cursor.execute('''
                SELECT p.*, f.league_id, f.match_date, f.status as match_status,
                       f.home_score, f.away_score
                FROM predictions p
                LEFT JOIN fixtures f ON p.fixture_id = f.id
                WHERE f.status IN ('finished', 'ft', 'FINISHED')
                ORDER BY p.created_at DESC
            ''')
            
            all_predictions = [dict(row) for row in cursor.fetchall()]
            
            if not all_predictions:
                return self._get_section_fallback(section)
            
            # Categorize predictions into sections
            sections = {
                'daily_tips': [],
                'money_zone': [],
                'accas': []
            }
            
            for pred in all_predictions:
                conf = pred.get('confidence', 0) or 0
                match_time = pred.get('match_date', '') or ''
                
                # Determine section based on prediction characteristics
                # ACCAs typically have multiple selections in prediction data
                prediction_data = pred.get('prediction_data', '')
                if prediction_data and isinstance(prediction_data, str):
                    try:
                        pred_json = json.loads(prediction_data)
                        if pred_json.get('is_acca') or pred_json.get('accumulator'):
                            sections['accas'].append(pred)
                            continue
                    except:
                        pass
                
                # Money Zone: Filter by match time (morning/afternoon/evening)
                # or high-confidence quick picks
                if conf >= 0.85:  # Sure Wins / High confidence
                    sections['money_zone'].append(pred)
                else:
                    sections['daily_tips'].append(pred)
            
            def calculate_section_stats(predictions: List[Dict]) -> Dict:
                if not predictions:
                    return {
                        'total': 0,
                        'won': 0,
                        'lost': 0,
                        'pending': 0,
                        'accuracy': 0,
                        'avg_confidence': 0,
                        'roi': 0
                    }
                
                total = len(predictions)
                won = 0
                lost = 0
                confidences = []
                
                for pred in predictions:
                    confidences.append((pred.get('confidence', 0) or 0) * 100)
                    home_score = pred.get('home_score')
                    away_score = pred.get('away_score')
                    predicted = pred.get('predicted_outcome', '')
                    
                    if home_score is not None and away_score is not None:
                        actual = 'Home Win' if home_score > away_score else ('Away Win' if away_score > home_score else 'Draw')
                        if predicted == actual:
                            won += 1
                        else:
                            lost += 1
                
                accuracy = round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0
                avg_conf = round(sum(confidences) / total, 1) if total > 0 else 0
                roi = round((won * 1.85 - (won + lost)) / max(1, won + lost) * 100, 1)
                
                return {
                    'total': total,
                    'won': won,
                    'lost': lost,
                    'pending': total - won - lost,
                    'accuracy': accuracy,
                    'avg_confidence': avg_conf,
                    'roi': roi
                }
            
            result = {}
            
            if section in ['all', 'daily_tips']:
                result['daily_tips'] = calculate_section_stats(sections['daily_tips'])
            if section in ['all', 'money_zone']:
                result['money_zone'] = calculate_section_stats(sections['money_zone'])
            if section in ['all', 'accas']:
                result['accas'] = calculate_section_stats(sections['accas'])
            
            if section == 'all':
                # Add combined stats
                all_stats = calculate_section_stats(all_predictions)
                result['combined'] = all_stats
                result['data_source'] = 'database'
            else:
                result['data_source'] = 'database'
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting section analytics: {e}")
            return self._get_section_fallback(section)
        finally:
            conn.close()
    
    def _get_section_fallback(self, section: str) -> Dict:
        """Return fallback section data."""
        empty_stats = {
            'total': 0,
            'won': 0,
            'lost': 0,
            'pending': 0,
            'accuracy': 0,
            'avg_confidence': 0,
            'roi': 0
        }
        
        if section == 'all':
            return {
                'daily_tips': empty_stats.copy(),
                'money_zone': empty_stats.copy(),
                'accas': empty_stats.copy(),
                'combined': empty_stats.copy(),
                'data_source': 'fallback'
            }
        else:
            return {section: empty_stats.copy(), 'data_source': 'fallback'}
    
    def get_time_period_analytics(self) -> Dict:
        """Get analytics broken down by time period (for Money Zone)."""
        conn = self._get_conn()
        
        if not conn:
            return self._get_time_period_fallback()
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, f.match_date, f.home_score, f.away_score, f.status
                FROM predictions p
                LEFT JOIN fixtures f ON p.fixture_id = f.id
                WHERE f.status IN ('finished', 'ft', 'FINISHED')
            ''')
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            if not predictions:
                return self._get_time_period_fallback()
            
            # Group by time period
            periods = {
                'morning': {'predictions': [], 'hours': range(6, 12)},    # 6 AM - 12 PM
                'afternoon': {'predictions': [], 'hours': range(12, 18)}, # 12 PM - 6 PM
                'evening': {'predictions': [], 'hours': range(18, 24)},   # 6 PM - 12 AM
                'night': {'predictions': [], 'hours': range(0, 6)}        # 12 AM - 6 AM
            }
            
            for pred in predictions:
                match_date = pred.get('match_date', '')
                try:
                    if match_date:
                        dt = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                        hour = dt.hour
                        for period_name, period_data in periods.items():
                            if hour in period_data['hours']:
                                period_data['predictions'].append(pred)
                                break
                except:
                    continue
            
            def calc_period_stats(preds):
                if not preds:
                    return {'total': 0, 'won': 0, 'accuracy': 0, 'roi': 0}
                
                total = len(preds)
                won = 0
                for p in preds:
                    hs = p.get('home_score')
                    aws = p.get('away_score')
                    if hs is not None and aws is not None:
                        actual = 'Home Win' if hs > aws else ('Away Win' if aws > hs else 'Draw')
                        if p.get('predicted_outcome') == actual:
                            won += 1
                
                accuracy = round(won / total * 100, 1) if total > 0 else 0
                roi = round((won * 1.85 - total) / total * 100, 1) if total > 0 else 0
                return {'total': total, 'won': won, 'accuracy': accuracy, 'roi': roi}
            
            return {
                'morning': calc_period_stats(periods['morning']['predictions']),
                'afternoon': calc_period_stats(periods['afternoon']['predictions']),
                'evening': calc_period_stats(periods['evening']['predictions']),
                'night': calc_period_stats(periods['night']['predictions']),
                'data_source': 'database'
            }
            
        except Exception as e:
            logger.error(f"Error getting time period analytics: {e}")
            return self._get_time_period_fallback()
        finally:
            conn.close()
    
    def _get_time_period_fallback(self) -> Dict:
        """Return fallback time period data."""
        empty = {'total': 0, 'won': 0, 'accuracy': 0, 'roi': 0}
        return {
            'morning': empty.copy(),
            'afternoon': empty.copy(),
            'evening': empty.copy(),
            'night': empty.copy(),
            'data_source': 'fallback'
        }
    
    def get_acca_analytics(self) -> Dict:
        """Get accumulator-specific analytics."""
        conn = self._get_conn()
        
        if not conn:
            return self._get_acca_fallback()
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get predictions with high confidence (likely acca candidates)
            cursor.execute('''
                SELECT p.*, f.home_score, f.away_score, f.status
                FROM predictions p
                LEFT JOIN fixtures f ON p.fixture_id = f.id
                WHERE p.confidence >= 0.75
                AND f.status IN ('finished', 'ft', 'FINISHED')
                ORDER BY p.created_at DESC
            ''')
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            if not predictions:
                return self._get_acca_fallback()
            
            # Categorize by acca type
            acca_types = {
                'doubles': {'size': 2, 'total': 0, 'won': 0, 'potential_odds': 3.4},
                'trebles': {'size': 3, 'total': 0, 'won': 0, 'potential_odds': 6.3},
                'quadruples': {'size': 4, 'total': 0, 'won': 0, 'potential_odds': 11.6},
                'five_folds': {'size': 5, 'total': 0, 'won': 0, 'potential_odds': 21.5}
            }
            
            # Simulate acca combinations
            total_preds = len(predictions)
            won_preds = 0
            for p in predictions:
                hs = p.get('home_score')
                aws = p.get('away_score')
                if hs is not None and aws is not None:
                    actual = 'Home Win' if hs > aws else ('Away Win' if aws > hs else 'Draw')
                    if p.get('predicted_outcome') == actual:
                        won_preds += 1
            
            win_rate = won_preds / total_preds if total_preds > 0 else 0
            
            # Calculate expected success rate for each acca type
            for acca_type, data in acca_types.items():
                size = data['size']
                # Probability of all selections winning
                success_prob = win_rate ** size
                sample_accas = max(1, total_preds // size)
                
                data['total'] = sample_accas
                data['won'] = round(sample_accas * success_prob)
                data['success_rate'] = round(success_prob * 100, 1)
                data['expected_roi'] = round((success_prob * data['potential_odds'] - 1) * 100, 1)
            
            return {
                'acca_types': {
                    name: {
                        'total': data['total'],
                        'won': data['won'],
                        'success_rate': data['success_rate'],
                        'expected_roi': data['expected_roi'],
                        'potential_odds': data['potential_odds']
                    }
                    for name, data in acca_types.items()
                },
                'single_pick_accuracy': round(win_rate * 100, 1),
                'total_high_conf_picks': total_preds,
                'data_source': 'database'
            }
            
        except Exception as e:
            logger.error(f"Error getting acca analytics: {e}")
            return self._get_acca_fallback()
        finally:
            conn.close()
    
    def _get_acca_fallback(self) -> Dict:
        """Return fallback acca data."""
        empty_acca = {'total': 0, 'won': 0, 'success_rate': 0, 'expected_roi': 0, 'potential_odds': 0}
        return {
            'acca_types': {
                'doubles': empty_acca.copy(),
                'trebles': empty_acca.copy(),
                'quadruples': empty_acca.copy(),
                'five_folds': empty_acca.copy()
            },
            'single_pick_accuracy': 0,
            'total_high_conf_picks': 0,
            'data_source': 'fallback'
        }


# Global instance
analytics_engine = AnalyticsEngine()


# API Functions
def get_today_analytics() -> Dict:
    """API: Get today's analytics."""
    return analytics_engine.get_today_summary()


def get_accuracy_analytics(days: int = 30) -> Dict:
    """API: Get accuracy history."""
    return analytics_engine.get_accuracy_history(days)


def get_league_analytics() -> List[Dict]:
    """API: Get league performance."""
    return analytics_engine.get_performance_by_league()


def get_market_analytics() -> List[Dict]:
    """API: Get market performance."""
    return analytics_engine.get_performance_by_market()


def get_roi_analytics(stake: float = 10.0, days: int = 30) -> Dict:
    """API: Calculate ROI."""
    return analytics_engine.calculate_roi(stake, days)


def get_db_stats() -> Dict:
    """API: Get database stats."""
    return analytics_engine.get_database_stats()


def get_section_analytics(section: str = 'all') -> Dict:
    """API: Get section-based analytics (daily_tips, money_zone, accas, all)."""
    return analytics_engine.get_section_analytics(section)


def get_time_period_analytics() -> Dict:
    """API: Get time period analytics for Money Zone."""
    return analytics_engine.get_time_period_analytics()


def get_acca_analytics() -> Dict:
    """API: Get accumulator analytics."""
    return analytics_engine.get_acca_analytics()
