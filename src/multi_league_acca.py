"""
Multi-League Accumulator Builder

Generate cross-league accumulators combining picks from multiple competitions:
- World Tour ACCA: Best picks from 5+ different leagues
- Euro Elite ACCA: Top 5 European leagues only
- Global Banker ACCA: Highest confidence worldwide
- Americas Special: MLS + South American leagues
- Asian Handicap Special: For handicap markets
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math
import random


@dataclass
class MultiLeaguePick:
    """Single pick in a multi-league accumulator"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    league_name: str
    selection: str
    odds: float
    probability: float
    confidence: float
    reasoning: str
    kickoff: Optional[str] = None


@dataclass 
class MultiLeagueAccumulator:
    """Complete multi-league accumulator"""
    id: str
    strategy: str
    name: str
    description: str
    picks: List[MultiLeaguePick]
    leagues_count: int
    combined_odds: float
    combined_probability: float
    stake_suggestion: float
    potential_return: float
    risk_level: str
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'strategy': self.strategy,
            'name': self.name,
            'description': self.description,
            'picks': [asdict(p) for p in self.picks],
            'leagues_count': self.leagues_count,
            'combined_odds': round(self.combined_odds, 2),
            'combined_probability': round(self.combined_probability * 100, 1),
            'stake_suggestion': self.stake_suggestion,
            'potential_return': round(self.potential_return, 2),
            'risk_level': self.risk_level,
            'created_at': self.created_at
        }


class MultiLeagueAccaBuilder:
    """
    Build accumulators from fixtures across all leagues.
    
    Strategies:
    - World Tour: 5+ leagues, diversified picks
    - Euro Elite: Top 5 European leagues
    - Global Banker: Highest confidence worldwide
    - Americas Special: North/South America focus
    - Weekend Warrior: Saturday/Sunday picks only
    """
    
    STRATEGIES = {
        'world_tour': {
            'name': '🌍 World Tour ACCA',
            'description': 'Best picks from 5+ different leagues worldwide',
            'min_leagues': 5,
            'max_picks': 7,
            'min_confidence': 0.70,
            'icon': '🌍',
            'risk_level': 'medium'
        },
        'euro_elite': {
            'name': '⭐ Euro Elite ACCA',
            'description': 'Top 5 European league picks only',
            'leagues': ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1'],
            'max_picks': 5,
            'min_confidence': 0.75,
            'icon': '⭐',
            'risk_level': 'low'
        },
        'global_banker': {
            'name': '🏆 Global Banker',
            'description': 'Highest confidence picks worldwide',
            'min_confidence': 0.85,
            'max_picks': 4,
            'icon': '🏆',
            'risk_level': 'very_low'
        },
        'value_worldwide': {
            'name': '💎 Value Worldwide',
            'description': 'Best value bets from around the globe',
            'min_edge': 0.05,
            'max_picks': 5,
            'icon': '💎',
            'risk_level': 'medium'
        },
        'goals_galore': {
            'name': '⚽ Goals Galore',
            'description': 'Over 2.5 goals picks across leagues',
            'selection_type': 'over_2.5',
            'max_picks': 6,
            'min_confidence': 0.65,
            'icon': '⚽',
            'risk_level': 'medium'
        },
        'underdogs_united': {
            'name': '🐺 Underdogs United',
            'description': 'High-odds underdog picks for big returns',
            'underdog_focus': True,
            'max_picks': 5,
            'min_probability': 0.25,
            'icon': '🐺',
            'risk_level': 'high'
        }
    }
    
    LEAGUE_NAMES = {
        'premier_league': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League',
        'la_liga': '🇪🇸 La Liga',
        'bundesliga': '🇩🇪 Bundesliga',
        'serie_a': '🇮🇹 Serie A',
        'ligue_1': '🇫🇷 Ligue 1',
        'eredivisie': '🇳🇱 Eredivisie',
        'primeira_liga': '🇵🇹 Primeira Liga',
        'championship': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship',
        'mls': '🇺🇸 MLS',
        'brasileirao': '🇧🇷 Brasileirão',
        'scottish_premiership': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership',
        'belgian_pro_league': '🇧🇪 Belgian Pro League',
        'super_lig': '🇹🇷 Süper Lig',
        'russian_premier': '🇷🇺 Russian Premier',
        'austrian_bundesliga': '🇦🇹 Austrian Bundesliga'
    }
    
    def __init__(self):
        self._counter = 0
    
    def _generate_id(self) -> str:
        self._counter += 1
        return f"mla_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._counter:04d}"
    
    def generate_world_tour(self, all_predictions: Dict[str, List[Dict]], max_picks: int = 7) -> Optional[MultiLeagueAccumulator]:
        """
        Generate World Tour ACCA with picks from 5+ different leagues.
        
        Args:
            all_predictions: Dict mapping league_id -> list of predictions
            max_picks: Maximum number of picks
            
        Returns:
            MultiLeagueAccumulator or None if not enough leagues
        """
        config = self.STRATEGIES['world_tour']
        picks = []
        used_leagues = set()
        
        # Get best pick from each league (ensuring diversity)
        league_best_picks = []
        
        for league_id, predictions in all_predictions.items():
            if not predictions:
                continue
                
            # Find best pick from this league
            valid_preds = [
                p for p in predictions 
                if self._get_confidence(p) >= config['min_confidence']
            ]
            
            if valid_preds:
                best = max(valid_preds, key=lambda x: self._get_confidence(x))
                league_best_picks.append((league_id, best))
        
        # Sort by confidence and take top picks ensuring diversity
        league_best_picks.sort(key=lambda x: self._get_confidence(x[1]), reverse=True)
        
        for league_id, pred in league_best_picks[:max_picks]:
            if len(picks) >= max_picks:
                break
                
            pick = self._create_pick(pred, league_id)
            picks.append(pick)
            used_leagues.add(league_id)
        
        # Check minimum league requirement
        if len(used_leagues) < config['min_leagues']:
            return None
        
        return self._build_accumulator('world_tour', picks)
    
    def generate_euro_elite(self, all_predictions: Dict[str, List[Dict]], max_picks: int = 5) -> Optional[MultiLeagueAccumulator]:
        """
        Generate Euro Elite ACCA from top 5 European leagues.
        """
        config = self.STRATEGIES['euro_elite']
        elite_leagues = config['leagues']
        picks = []
        
        # Get predictions only from elite leagues
        elite_predictions = []
        for league_id, predictions in all_predictions.items():
            if league_id in elite_leagues:
                for pred in predictions:
                    if self._get_confidence(pred) >= config['min_confidence']:
                        elite_predictions.append((league_id, pred))
        
        if not elite_predictions:
            return None
        
        # Sort by confidence
        elite_predictions.sort(key=lambda x: self._get_confidence(x[1]), reverse=True)
        
        # Take one from each league first, then fill
        used_leagues = set()
        
        # First pass: one from each league
        for league_id, pred in elite_predictions:
            if league_id not in used_leagues and len(picks) < max_picks:
                picks.append(self._create_pick(pred, league_id))
                used_leagues.add(league_id)
        
        # Second pass: fill remaining slots
        for league_id, pred in elite_predictions:
            if len(picks) >= max_picks:
                break
            pick = self._create_pick(pred, league_id)
            if pick not in picks:
                picks.append(pick)
        
        if not picks:
            return None
            
        return self._build_accumulator('euro_elite', picks[:max_picks])
    
    def generate_global_banker(self, all_predictions: Dict[str, List[Dict]], max_picks: int = 4) -> Optional[MultiLeagueAccumulator]:
        """
        Generate Global Banker ACCA with highest confidence picks worldwide.
        """
        config = self.STRATEGIES['global_banker']
        all_high_confidence = []
        
        for league_id, predictions in all_predictions.items():
            for pred in predictions:
                conf = self._get_confidence(pred)
                if conf >= config['min_confidence']:
                    all_high_confidence.append((league_id, pred, conf))
        
        if not all_high_confidence:
            return None
        
        # Sort by confidence descending
        all_high_confidence.sort(key=lambda x: x[2], reverse=True)
        
        picks = []
        for league_id, pred, _ in all_high_confidence[:max_picks]:
            picks.append(self._create_pick(pred, league_id))
        
        return self._build_accumulator('global_banker', picks)
    
    def generate_value_worldwide(self, all_predictions: Dict[str, List[Dict]], max_picks: int = 5) -> Optional[MultiLeagueAccumulator]:
        """
        Generate Value Worldwide ACCA with best value bets.
        """
        config = self.STRATEGIES['value_worldwide']
        all_value_picks = []
        
        for league_id, predictions in all_predictions.items():
            for pred in predictions:
                edge = pred.get('value_edge', 0) or 0
                if edge >= config['min_edge']:
                    all_value_picks.append((league_id, pred, edge))
        
        if not all_value_picks:
            return None
        
        # Sort by edge descending
        all_value_picks.sort(key=lambda x: x[2], reverse=True)
        
        picks = []
        for league_id, pred, _ in all_value_picks[:max_picks]:
            picks.append(self._create_pick(pred, league_id))
        
        return self._build_accumulator('value_worldwide', picks)
    
    def generate_goals_galore(self, all_predictions: Dict[str, List[Dict]], max_picks: int = 6) -> Optional[MultiLeagueAccumulator]:
        """
        Generate Goals Galore ACCA for Over 2.5 goals.
        """
        config = self.STRATEGIES['goals_galore']
        high_scoring_picks = []
        
        for league_id, predictions in all_predictions.items():
            for pred in predictions:
                # Check for over 2.5 probability or high-scoring teams
                over_prob = pred.get('over_2_5_prob', 0) or 0
                if over_prob == 0:
                    # Estimate from team goal averages if available
                    home_goals = pred.get('home_goals_avg', 1.3)
                    away_goals = pred.get('away_goals_avg', 1.1)
                    expected_total = home_goals + away_goals
                    if expected_total >= 2.5:
                        over_prob = 0.65
                
                if over_prob >= config['min_confidence']:
                    high_scoring_picks.append((league_id, pred, over_prob))
        
        if not high_scoring_picks:
            return None
        
        high_scoring_picks.sort(key=lambda x: x[2], reverse=True)
        
        picks = []
        for league_id, pred, prob in high_scoring_picks[:max_picks]:
            pick = self._create_pick(pred, league_id)
            pick.selection = 'Over 2.5 Goals'
            pick.probability = prob
            picks.append(pick)
        
        return self._build_accumulator('goals_galore', picks)
    
    def generate_underdogs_united(self, all_predictions: Dict[str, List[Dict]], max_picks: int = 5) -> Optional[MultiLeagueAccumulator]:
        """
        Generate Underdogs United ACCA for high-odds picks.
        """
        config = self.STRATEGIES['underdogs_united']
        underdog_picks = []
        
        for league_id, predictions in all_predictions.items():
            for pred in predictions:
                # Find underdog with reasonable probability
                home_prob = pred.get('home_win_prob', 0) or 0
                away_prob = pred.get('away_win_prob', 0) or 0
                home_elo = pred.get('home_elo', 1500)
                away_elo = pred.get('away_elo', 1500)
                
                if home_prob > 1:
                    home_prob /= 100
                if away_prob > 1:
                    away_prob /= 100
                
                # Away team is underdog (lower ELO)
                if away_elo < home_elo - 30 and away_prob >= config['min_probability']:
                    underdog_picks.append((league_id, pred, away_prob, 'Away'))
                # Home team is underdog
                elif home_elo < away_elo - 30 and home_prob >= config['min_probability']:
                    underdog_picks.append((league_id, pred, home_prob, 'Home'))
        
        if not underdog_picks:
            return None
        
        # Sort by probability descending (best underdog chances)
        underdog_picks.sort(key=lambda x: x[2], reverse=True)
        
        picks = []
        for league_id, pred, prob, selection in underdog_picks[:max_picks]:
            pick = self._create_pick(pred, league_id)
            pick.selection = selection
            pick.probability = prob
            pick.odds = 1 / prob if prob > 0 else 3.0  # Estimated odds
            picks.append(pick)
        
        return self._build_accumulator('underdogs_united', picks)
    
    def generate_all(self, all_predictions: Dict[str, List[Dict]]) -> Dict[str, Optional[MultiLeagueAccumulator]]:
        """
        Generate all multi-league accumulator types.
        
        Args:
            all_predictions: Dict mapping league_id -> list of predictions
            
        Returns:
            Dict mapping strategy name -> accumulator (or None)
        """
        return {
            'world_tour': self.generate_world_tour(all_predictions),
            'euro_elite': self.generate_euro_elite(all_predictions),
            'global_banker': self.generate_global_banker(all_predictions),
            'value_worldwide': self.generate_value_worldwide(all_predictions),
            'goals_galore': self.generate_goals_galore(all_predictions),
            'underdogs_united': self.generate_underdogs_united(all_predictions)
        }
    
    def _get_confidence(self, pred: Dict) -> float:
        """Extract normalized confidence from prediction"""
        confidence = pred.get('confidence', 0)
        if isinstance(confidence, str):
            confidence = float(confidence.replace('%', '')) / 100
        elif confidence > 1:
            confidence = confidence / 100
        return confidence
    
    def _create_pick(self, pred: Dict, league_id: str) -> MultiLeaguePick:
        """Create a pick from a prediction dict"""
        confidence = self._get_confidence(pred)
        outcome = pred.get('predicted_outcome', 'Home')
        
        # Get probability for predicted outcome
        if outcome == 'Home':
            prob = pred.get('home_win_prob', 0.5)
        elif outcome == 'Away':
            prob = pred.get('away_win_prob', 0.3)
        else:
            prob = pred.get('draw_prob', 0.25)
        
        if prob > 1:
            prob /= 100
        
        # Estimate odds from probability
        odds = 1 / prob if prob > 0 else 2.0
        
        return MultiLeaguePick(
            match_id=pred.get('match_id', f"{pred.get('home_team', '')}_{pred.get('away_team', '')}"),
            home_team=pred.get('home_team', 'Home'),
            away_team=pred.get('away_team', 'Away'),
            league=league_id,
            league_name=self.LEAGUE_NAMES.get(league_id, league_id.replace('_', ' ').title()),
            selection=outcome,
            odds=round(odds, 2),
            probability=prob,
            confidence=confidence,
            reasoning=pred.get('analysis_notes', ['High confidence pick'])[0] if pred.get('analysis_notes') else 'Selected based on model analysis',
            kickoff=pred.get('kickoff', pred.get('match_date'))
        )
    
    def _build_accumulator(self, strategy: str, picks: List[MultiLeaguePick]) -> MultiLeagueAccumulator:
        """Build accumulator from picks"""
        config = self.STRATEGIES[strategy]
        
        # Calculate combined odds and probability
        combined_odds = 1.0
        combined_prob = 1.0
        leagues = set()
        
        for pick in picks:
            combined_odds *= pick.odds
            combined_prob *= pick.probability
            leagues.add(pick.league)
        
        # Calculate stake suggestion based on risk
        risk_level = config.get('risk_level', 'medium')
        stake_map = {
            'very_low': 10.0,
            'low': 5.0,
            'medium': 3.0,
            'high': 2.0
        }
        stake = stake_map.get(risk_level, 3.0)
        
        return MultiLeagueAccumulator(
            id=self._generate_id(),
            strategy=strategy,
            name=config['name'],
            description=config['description'],
            picks=picks,
            leagues_count=len(leagues),
            combined_odds=combined_odds,
            combined_probability=combined_prob,
            stake_suggestion=stake,
            potential_return=stake * combined_odds,
            risk_level=risk_level,
            created_at=datetime.now().isoformat()
        )
    
    def get_strategy_info(self, strategy: str) -> Optional[Dict]:
        """Get information about a strategy"""
        return self.STRATEGIES.get(strategy)
    
    def get_all_strategies(self) -> Dict:
        """Get all strategy definitions"""
        return self.STRATEGIES


# Global instance
multi_league_builder = MultiLeagueAccaBuilder()


def generate_multi_league_acca(all_predictions: Dict[str, List[Dict]], strategy: str = 'world_tour') -> Optional[Dict]:
    """Generate a specific multi-league accumulator"""
    generators = {
        'world_tour': multi_league_builder.generate_world_tour,
        'euro_elite': multi_league_builder.generate_euro_elite,
        'global_banker': multi_league_builder.generate_global_banker,
        'value_worldwide': multi_league_builder.generate_value_worldwide,
        'goals_galore': multi_league_builder.generate_goals_galore,
        'underdogs_united': multi_league_builder.generate_underdogs_united
    }
    
    generator = generators.get(strategy)
    if not generator:
        return None
    
    acca = generator(all_predictions)
    return acca.to_dict() if acca else None


def generate_all_multi_league_accas(all_predictions: Dict[str, List[Dict]]) -> Dict[str, Optional[Dict]]:
    """Generate all multi-league accumulator types"""
    accas = multi_league_builder.generate_all(all_predictions)
    return {
        name: acca.to_dict() if acca else None
        for name, acca in accas.items()
    }
