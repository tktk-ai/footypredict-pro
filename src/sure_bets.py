"""
Sure Bets API - Curated High-Probability Lists
===============================================

Generates curated betting lists designed for safe accumulator building:
1. Top 5 Most Likely Winners (highest confidence across all markets)
2. Top 15 Over 0.5 from O2.5+ games (near-certain goals)
3. Top 7 Over 1.5 predictions
4. Top 10 Double Chance 1X (Home or Draw - safest result market)
5. Top 10 HT Over 0.5 (First half goal - high success rate)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SureBetsGenerator:
    """
    Generate curated betting lists with high success probability.
    """
    
    def __init__(self):
        self.predictions_cache = {}
        self.last_update = None
    
    def get_todays_matches(self) -> List[Dict]:
        """Get today's matches from the live data fetcher or API."""
        try:
            # Try to get from live data
            from src.live_data import get_todays_fixtures
            matches = get_todays_fixtures()
            if matches:
                return matches
        except Exception as e:
            logger.warning(f"Could not get live fixtures: {e}")
        
        # Fallback to predictions database
        try:
            import sqlite3
            db_path = Path(__file__).parent.parent / "data" / "predictions.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT DISTINCT home_team, away_team, league, kickoff_time, prediction, confidence
                    FROM predictions 
                    WHERE date(kickoff_time) = date(?)
                    ORDER BY confidence DESC
                """, (today,))
                rows = cursor.fetchall()
                conn.close()
                
                return [
                    {
                        'home_team': r[0],
                        'away_team': r[1],
                        'league': r[2],
                        'kickoff': r[3],
                        'prediction': r[4],
                        'confidence': r[5]
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"Could not get from database: {e}")
        
        return []
    
    def generate_predictions_for_match(self, home: str, away: str, league: str) -> Dict:
        """Generate all predictions for a match using unified ensemble."""
        cache_key = f"{home}_{away}_{league}"
        
        if cache_key in self.predictions_cache:
            return self.predictions_cache[cache_key]
        
        try:
            # Try different import paths
            try:
                from src.models.unified_ensemble import get_unified_predictor
            except ImportError:
                from models.unified_ensemble import get_unified_predictor
            
            predictor = get_unified_predictor()
            result = predictor.predict(home, away, league)
            self.predictions_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Prediction error for {home} vs {away}: {e}")
            return {}
    
    def get_top_5_winners(self, matches: List[Dict]) -> List[Dict]:
        """
        Category 1: Top 5 Most Likely Winnable Predictions
        - Highest confidence picks across all markets
        """
        all_picks = []
        
        for match in matches:
            home = match.get('home_team', '')
            away = match.get('away_team', '')
            league = match.get('league', '')
            kickoff = match.get('kickoff', '')
            
            if not home or not away:
                continue
            
            pred = self.generate_predictions_for_match(home, away, league)
            
            for pick in pred.get('best_picks', [])[:2]:
                if pick.get('probability', 0) >= 75:
                    all_picks.append({
                        'match': f"{home} vs {away}",
                        'league': league,
                        'kickoff': kickoff,
                        'market': pick.get('market', ''),
                        'prediction': pick.get('prediction', ''),
                        'probability': pick.get('probability', 0),
                        'confidence': 'HIGH'
                    })
        
        # Sort by probability and return top 5
        all_picks.sort(key=lambda x: x['probability'], reverse=True)
        return all_picks[:5]
    
    def get_over_05_from_high_scoring(self, matches: List[Dict]) -> List[Dict]:
        """
        Category 2: Top 15 Over 0.5 from O2.5+ games
        - If a game is predicted O2.5+, then O0.5 is almost guaranteed
        """
        over_05_picks = []
        
        for match in matches:
            home = match.get('home_team', '')
            away = match.get('away_team', '')
            league = match.get('league', '')
            kickoff = match.get('kickoff', '')
            
            if not home or not away:
                continue
            
            pred = self.generate_predictions_for_match(home, away, league)
            combined = pred.get('combined', {})
            
            # Check if Over 2.5 is predicted
            over25 = combined.get('over_25', combined.get('over25', {}))
            if over25 and over25.get('prediction') == 'Yes':
                over25_prob = over25.get('probability', 0)
                
                if over25_prob >= 60:
                    # Over 0.5 probability is much higher than O2.5
                    over05_prob = min(over25_prob + 25, 98)
                    
                    over_05_picks.append({
                        'match': f"{home} vs {away}",
                        'league': league,
                        'kickoff': kickoff,
                        'market': 'Over 0.5 Goals',
                        'prediction': 'Yes',
                        'probability': over05_prob,
                        'base_prediction': f"O2.5 at {over25_prob:.0f}%",
                        'confidence': 'VERY HIGH'
                    })
        
        # Sort by probability
        over_05_picks.sort(key=lambda x: x['probability'], reverse=True)
        return over_05_picks[:15]
    
    def get_over_15_predictions(self, matches: List[Dict]) -> List[Dict]:
        """
        Category 3: Top 7 Over 1.5 Predictions
        """
        over_15_picks = []
        
        for match in matches:
            home = match.get('home_team', '')
            away = match.get('away_team', '')
            league = match.get('league', '')
            kickoff = match.get('kickoff', '')
            
            if not home or not away:
                continue
            
            pred = self.generate_predictions_for_match(home, away, league)
            combined = pred.get('combined', {})
            
            over15 = combined.get('over_15', combined.get('over15', {}))
            if over15 and over15.get('prediction') == 'Yes':
                prob = over15.get('probability', 0)
                
                if prob >= 70:
                    over_15_picks.append({
                        'match': f"{home} vs {away}",
                        'league': league,
                        'kickoff': kickoff,
                        'market': 'Over 1.5 Goals',
                        'prediction': 'Yes',
                        'probability': prob,
                        'confidence': 'HIGH' if prob >= 80 else 'MEDIUM'
                    })
        
        over_15_picks.sort(key=lambda x: x['probability'], reverse=True)
        return over_15_picks[:7]
    
    def get_double_chance_1x(self, matches: List[Dict]) -> List[Dict]:
        """
        Category 4: Top 10 Double Chance 1X (Home or Draw)
        - Safest result market, covers 2 outcomes
        """
        dc_picks = []
        
        for match in matches:
            home = match.get('home_team', '')
            away = match.get('away_team', '')
            league = match.get('league', '')
            kickoff = match.get('kickoff', '')
            
            if not home or not away:
                continue
            
            pred = self.generate_predictions_for_match(home, away, league)
            combined = pred.get('combined', {})
            
            dc_1x = combined.get('dc_1x', {})
            if dc_1x and dc_1x.get('prediction') == 'Yes':
                prob = dc_1x.get('probability', 0)
                
                if prob >= 70:
                    dc_picks.append({
                        'match': f"{home} vs {away}",
                        'league': league,
                        'kickoff': kickoff,
                        'market': 'Double Chance 1X',
                        'prediction': f"{home} or Draw",
                        'probability': prob,
                        'confidence': 'HIGH' if prob >= 80 else 'MEDIUM'
                    })
        
        dc_picks.sort(key=lambda x: x['probability'], reverse=True)
        return dc_picks[:10]
    
    def get_ht_over_05(self, matches: List[Dict]) -> List[Dict]:
        """
        Category 5: Top 10 HT Over 0.5 (First Half Goal)
        - High success rate in high-scoring games
        - Derived from O2.5 predictions when direct HT data unavailable
        """
        ht_picks = []
        
        for match in matches:
            home = match.get('home_team', '')
            away = match.get('away_team', '')
            league = match.get('league', '')
            kickoff = match.get('kickoff', '')
            
            if not home or not away:
                continue
            
            pred = self.generate_predictions_for_match(home, away, league)
            special = pred.get('special', {})
            halftime = special.get('halftime', {})
            
            # Try to get HT Over 0.5 from halftime predictions
            try:
                ht_over = halftime.get('over_under', {}).get('over_05', {})
                if isinstance(ht_over, dict) and ht_over:
                    prob = ht_over.get('probability', 0)
                    if isinstance(prob, (int, float)) and prob >= 60:
                        ht_picks.append({
                            'match': f"{home} vs {away}",
                            'league': league,
                            'kickoff': kickoff,
                            'market': 'HT Over 0.5 Goals',
                            'prediction': 'Yes',
                            'probability': float(prob),
                            'confidence': 'HIGH' if prob >= 70 else 'MEDIUM'
                        })
            except (AttributeError, TypeError):
                pass  # Skip if data format is unexpected
        
        # If not enough HT predictions, derive from O2.5 games
        if len(ht_picks) < 10:
            for match in matches:
                home = match.get('home_team', '')
                away = match.get('away_team', '')
                league = match.get('league', '')
                kickoff = match.get('kickoff', '')
                
                if f"{home} vs {away}" in [p['match'] for p in ht_picks]:
                    continue
                
                pred = self.generate_predictions_for_match(home, away, league)
                combined = pred.get('combined', {})
                
                over25 = combined.get('over_25', combined.get('over25', {}))
                if isinstance(over25, dict) and over25:
                    try:
                        o25_prob = over25.get('probability', 0)
                        if isinstance(o25_prob, (int, float)) and o25_prob >= 70:
                            # If O2.5 is high, HT O0.5 is likely
                            ht_prob = min(float(o25_prob) - 5, 85)
                            
                            ht_picks.append({
                                'match': f"{home} vs {away}",
                                'league': league,
                                'kickoff': kickoff,
                                'market': 'HT Over 0.5 Goals',
                                'prediction': 'Yes',
                                'probability': ht_prob,
                                'confidence': 'MEDIUM',
                                'derived': True
                            })
                    except (TypeError, ValueError):
                        pass
        
        ht_picks.sort(key=lambda x: x['probability'], reverse=True)
        return ht_picks[:10]
    
    def generate_all_lists(self) -> Dict:
        """Generate all 5 curated lists."""
        matches = self.get_todays_matches()
        
        # No demo fallback - return empty lists if no real matches
        return {
            'generated_at': datetime.now().isoformat(),
            'total_matches_analyzed': len(matches),
            'lists': {
                'top_5_winners': {
                    'title': '🏆 Top 5 Most Likely Winners',
                    'description': 'Highest confidence picks - Build your safe accumulator here',
                    'picks': self.get_top_5_winners(matches)
                },
                'over_05_from_high_scoring': {
                    'title': '⚽ Top 15 Over 0.5 Goals (From O2.5+ Games)',
                    'description': 'Near-certain goal picks from high-scoring game predictions',
                    'picks': self.get_over_05_from_high_scoring(matches)
                },
                'over_15': {
                    'title': '🎯 Top 7 Over 1.5 Goals',
                    'description': 'Best over 1.5 predictions for the day',
                    'picks': self.get_over_15_predictions(matches)
                },
                'double_chance_1x': {
                    'title': '🛡️ Top 10 Double Chance 1X (Home or Draw)',
                    'description': 'Safest result market - covers 2 outcomes for home advantage',
                    'picks': self.get_double_chance_1x(matches)
                },
                'ht_over_05': {
                    'title': '⏱️ Top 10 First Half Goal (HT O0.5)',
                    'description': 'First half goal predictions - perfect for early cashout',
                    'picks': self.get_ht_over_05(matches)
                }
            }
        }


# Singleton
_generator = None

def get_sure_bets_generator() -> SureBetsGenerator:
    global _generator
    if _generator is None:
        _generator = SureBetsGenerator()
    return _generator


def get_sure_bets() -> Dict:
    """Get all curated betting lists."""
    return get_sure_bets_generator().generate_all_lists()


if __name__ == "__main__":
    result = get_sure_bets()
    
    print("\n" + "=" * 70)
    print("🎰 SURE BETS - CURATED HIGH-PROBABILITY LISTS")
    print("=" * 70)
    
    for list_key, list_data in result['lists'].items():
        print(f"\n{list_data['title']}")
        print(f"   {list_data['description']}")
        print("-" * 50)
        
        for pick in list_data['picks'][:5]:
            print(f"   {pick['match']}")
            print(f"      {pick['market']}: {pick['prediction']} ({pick['probability']:.0f}%)")
