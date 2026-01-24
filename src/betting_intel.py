"""
Betting Intelligence Module

Advanced betting features:
- Multi-bookmaker odds comparison
- Arbitrage opportunity detection
- Value bet identification
- Odds movement tracking
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BookmakerOdds:
    """Odds from a single bookmaker"""
    bookmaker: str
    home_odds: float
    draw_odds: float
    away_odds: float
    margin: float
    last_updated: str


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    match: str
    profit_percent: float
    stakes: Dict[str, float]  # outcome -> stake amount
    bookmakers: Dict[str, str]  # outcome -> bookmaker name
    odds: Dict[str, float]  # outcome -> odds
    total_stake: float
    guaranteed_return: float


@dataclass
class ValueBet:
    """Identified value bet"""
    match: str
    selection: str
    our_probability: float
    bookmaker_probability: float
    edge: float
    odds: float
    bookmaker: str
    kelly_stake: float


class OddsComparer:
    """
    Compare odds across bookmakers and find best prices
    
    Note: In production, you'd use:
    - The Odds API (https://the-odds-api.com) - Free tier available
    - Betfair Exchange API
    - Oddschecker scraping
    
    This implementation uses simulated odds data.
    """
    
    # Simulated bookmaker odds (in production, fetch from API)
    SAMPLE_ODDS = {
        'Bayern vs Dortmund': {
            'bet365': {'home': 1.45, 'draw': 4.50, 'away': 6.50},
            'betfair': {'home': 1.48, 'draw': 4.40, 'away': 6.80},
            'unibet': {'home': 1.44, 'draw': 4.60, 'away': 6.40},
            'williamhill': {'home': 1.47, 'draw': 4.33, 'away': 7.00},
            'pinnacle': {'home': 1.49, 'draw': 4.55, 'away': 6.60},
        },
        'Liverpool vs Arsenal': {
            'bet365': {'home': 2.10, 'draw': 3.40, 'away': 3.50},
            'betfair': {'home': 2.14, 'draw': 3.45, 'away': 3.45},
            'unibet': {'home': 2.05, 'draw': 3.50, 'away': 3.55},
            'williamhill': {'home': 2.15, 'draw': 3.30, 'away': 3.60},
            'pinnacle': {'home': 2.12, 'draw': 3.42, 'away': 3.52},
        },
        'Real Madrid vs Barcelona': {
            'bet365': {'home': 2.40, 'draw': 3.30, 'away': 2.90},
            'betfair': {'home': 2.42, 'draw': 3.35, 'away': 2.88},
            'unibet': {'home': 2.38, 'draw': 3.25, 'away': 2.95},
            'williamhill': {'home': 2.45, 'draw': 3.20, 'away': 2.85},
            'pinnacle': {'home': 2.44, 'draw': 3.32, 'away': 2.92},
        },
    }
    
    def __init__(self):
        self.odds_api_key = os.getenv('ODDS_API_KEY')
    
    def calculate_margin(self, home: float, draw: float, away: float) -> float:
        """Calculate bookmaker margin from odds"""
        if home <= 0 or draw <= 0 or away <= 0:
            return 0
        margin = (1/home + 1/draw + 1/away - 1) * 100
        return round(margin, 2)
    
    def get_odds_for_match(self, home_team: str, away_team: str) -> List[BookmakerOdds]:
        """Get odds from all bookmakers for a match"""
        match_key = f"{home_team} vs {away_team}"
        
        # Check if we have sample data
        if match_key in self.SAMPLE_ODDS:
            odds_data = self.SAMPLE_ODDS[match_key]
        else:
            # Generate realistic simulated odds
            odds_data = self._generate_simulated_odds(home_team, away_team)
        
        results = []
        for bookmaker, odds in odds_data.items():
            margin = self.calculate_margin(odds['home'], odds['draw'], odds['away'])
            results.append(BookmakerOdds(
                bookmaker=bookmaker,
                home_odds=odds['home'],
                draw_odds=odds['draw'],
                away_odds=odds['away'],
                margin=margin,
                last_updated=datetime.now().isoformat()
            ))
        
        return results
    
    def _generate_simulated_odds(self, home_team: str, away_team: str) -> Dict:
        """Generate realistic simulated odds"""
        import random
        
        # Base probabilities (slight home advantage)
        home_base = 0.42 + random.uniform(-0.15, 0.15)
        draw_base = 0.28 + random.uniform(-0.05, 0.05)
        away_base = 1 - home_base - draw_base
        
        bookmakers = ['bet365', 'betfair', 'unibet', 'williamhill', 'pinnacle']
        odds_data = {}
        
        for bookie in bookmakers:
            # Add some variance per bookmaker
            margin = random.uniform(0.03, 0.08)  # 3-8% margin
            
            home_odds = round(1 / (home_base * (1 + margin/3)) + random.uniform(-0.05, 0.05), 2)
            draw_odds = round(1 / (draw_base * (1 + margin/3)) + random.uniform(-0.1, 0.1), 2)
            away_odds = round(1 / (away_base * (1 + margin/3)) + random.uniform(-0.1, 0.1), 2)
            
            odds_data[bookie] = {
                'home': max(1.01, home_odds),
                'draw': max(1.01, draw_odds),
                'away': max(1.01, away_odds)
            }
        
        return odds_data
    
    def get_best_odds(self, home_team: str, away_team: str) -> Dict:
        """Find the best odds for each outcome across all bookmakers"""
        all_odds = self.get_odds_for_match(home_team, away_team)
        
        best = {
            'home': {'odds': 0, 'bookmaker': ''},
            'draw': {'odds': 0, 'bookmaker': ''},
            'away': {'odds': 0, 'bookmaker': ''},
        }
        
        for bookie_odds in all_odds:
            if bookie_odds.home_odds > best['home']['odds']:
                best['home'] = {'odds': bookie_odds.home_odds, 'bookmaker': bookie_odds.bookmaker}
            if bookie_odds.draw_odds > best['draw']['odds']:
                best['draw'] = {'odds': bookie_odds.draw_odds, 'bookmaker': bookie_odds.bookmaker}
            if bookie_odds.away_odds > best['away']['odds']:
                best['away'] = {'odds': bookie_odds.away_odds, 'bookmaker': bookie_odds.bookmaker}
        
        # Calculate combined margin with best odds
        combined_margin = self.calculate_margin(
            best['home']['odds'],
            best['draw']['odds'],
            best['away']['odds']
        )
        
        return {
            'match': f"{home_team} vs {away_team}",
            'best_odds': best,
            'combined_margin': combined_margin,
            'all_bookmakers': [
                {
                    'bookmaker': o.bookmaker,
                    'home': o.home_odds,
                    'draw': o.draw_odds,
                    'away': o.away_odds,
                    'margin': o.margin
                }
                for o in all_odds
            ]
        }


class ArbitrageFinder:
    """
    Detect arbitrage opportunities across bookmakers
    
    Arbitrage occurs when the combined probability from
    best odds across bookmakers is < 100%
    """
    
    def __init__(self):
        self.odds_comparer = OddsComparer()
    
    def check_arbitrage(self, home_team: str, away_team: str) -> Optional[ArbitrageOpportunity]:
        """Check if arbitrage opportunity exists for a match"""
        best_odds = self.odds_comparer.get_best_odds(home_team, away_team)
        
        home_odds = best_odds['best_odds']['home']['odds']
        draw_odds = best_odds['best_odds']['draw']['odds']
        away_odds = best_odds['best_odds']['away']['odds']
        
        # Calculate arbitrage percentage
        arb_pct = (1/home_odds + 1/draw_odds + 1/away_odds)
        
        if arb_pct < 1:  # Arbitrage exists!
            profit_pct = ((1 / arb_pct) - 1) * 100
            total_stake = 100  # Example with €100 total
            
            # Calculate stakes for guaranteed profit
            home_stake = round(total_stake * (1/home_odds) / arb_pct, 2)
            draw_stake = round(total_stake * (1/draw_odds) / arb_pct, 2)
            away_stake = round(total_stake * (1/away_odds) / arb_pct, 2)
            
            guaranteed_return = round(home_stake * home_odds, 2)
            
            return ArbitrageOpportunity(
                match=f"{home_team} vs {away_team}",
                profit_percent=round(profit_pct, 2),
                stakes={
                    'home': home_stake,
                    'draw': draw_stake,
                    'away': away_stake
                },
                bookmakers={
                    'home': best_odds['best_odds']['home']['bookmaker'],
                    'draw': best_odds['best_odds']['draw']['bookmaker'],
                    'away': best_odds['best_odds']['away']['bookmaker']
                },
                odds={
                    'home': home_odds,
                    'draw': draw_odds,
                    'away': away_odds
                },
                total_stake=total_stake,
                guaranteed_return=guaranteed_return
            )
        
        return None
    
    def find_all_arbitrage(self, matches: List[Tuple[str, str]]) -> List[ArbitrageOpportunity]:
        """Find all arbitrage opportunities from a list of matches"""
        opportunities = []
        
        for home, away in matches:
            arb = self.check_arbitrage(home, away)
            if arb:
                opportunities.append(arb)
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x.profit_percent, reverse=True)
        return opportunities


class ValueBetFinder:
    """
    Find value bets where our probability > bookmaker probability
    """
    
    def __init__(self):
        self.odds_comparer = OddsComparer()
    
    def find_value_bets(
        self,
        home_team: str,
        away_team: str,
        our_probs: Dict[str, float],  # {'home': 0.6, 'draw': 0.25, 'away': 0.15}
        min_edge: float = 0.05  # Minimum 5% edge
    ) -> List[ValueBet]:
        """Find value bets for a match"""
        best_odds = self.odds_comparer.get_best_odds(home_team, away_team)
        value_bets = []
        
        for outcome in ['home', 'draw', 'away']:
            our_prob = our_probs.get(outcome, 0.33)
            odds = best_odds['best_odds'][outcome]['odds']
            bookmaker = best_odds['best_odds'][outcome]['bookmaker']
            
            # Bookmaker implied probability
            bookie_prob = 1 / odds
            
            # Edge = our probability - bookmaker probability
            edge = our_prob - bookie_prob
            
            if edge >= min_edge:
                # Calculate Kelly stake
                b = odds - 1
                kelly = (our_prob * b - (1 - our_prob)) / b
                kelly_pct = max(0, min(0.25, kelly)) * 100  # Cap at 25%
                
                value_bets.append(ValueBet(
                    match=f"{home_team} vs {away_team}",
                    selection=outcome.replace('_', ' ').title().replace('Home', 'Home Win').replace('Away', 'Away Win'),
                    our_probability=round(our_prob, 3),
                    bookmaker_probability=round(bookie_prob, 3),
                    edge=round(edge, 3),
                    odds=odds,
                    bookmaker=bookmaker,
                    kelly_stake=round(kelly_pct, 1)
                ))
        
        # Sort by edge
        value_bets.sort(key=lambda x: x.edge, reverse=True)
        return value_bets


# Global instances
odds_comparer = OddsComparer()
arbitrage_finder = ArbitrageFinder()
value_bet_finder = ValueBetFinder()


def compare_odds(home_team: str, away_team: str) -> Dict:
    """Get odds comparison for a match"""
    return odds_comparer.get_best_odds(home_team, away_team)


def find_arbitrage(home_team: str, away_team: str) -> Optional[Dict]:
    """Find arbitrage opportunity for a match"""
    arb = arbitrage_finder.check_arbitrage(home_team, away_team)
    if arb:
        return {
            'match': arb.match,
            'profit_percent': arb.profit_percent,
            'stakes': arb.stakes,
            'bookmakers': arb.bookmakers,
            'odds': arb.odds,
            'total_stake': arb.total_stake,
            'guaranteed_return': arb.guaranteed_return
        }
    return None


def find_value(home_team: str, away_team: str, our_probs: Dict[str, float]) -> List[Dict]:
    """Find value bets for a match"""
    bets = value_bet_finder.find_value_bets(home_team, away_team, our_probs)
    return [
        {
            'match': b.match,
            'selection': b.selection,
            'our_probability': b.our_probability,
            'bookmaker_probability': b.bookmaker_probability,
            'edge': b.edge,
            'odds': b.odds,
            'bookmaker': b.bookmaker,
            'kelly_stake': b.kelly_stake
        }
        for b in bets
    ]
