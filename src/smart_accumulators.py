"""
Smart Auto-Accumulator Generator

Automatically generates multiple types of accumulators using cascade logic:
- If xG >= 3.5, then Over 0.5, 1.5, 2.5 are all "sure bets"
- Strong favorites imply Double Chance as safer option
- BTTS patterns from attacking strengths

Accumulator Types:
- Sure Wins (91%+ confidence)
- Over/Under Goals (0.5 to 3.5)
- BTTS (Both Teams To Score)
- Match Result (1X2)
- Double Chance
- HT/FT (Halftime/Fulltime)
- First Half Goals
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import math
import uuid


class BetType(Enum):
    """All supported bet types"""
    # Over/Under Goals
    OVER_0_5 = "over_0.5"
    OVER_1_5 = "over_1.5"
    OVER_2_5 = "over_2.5"
    OVER_3_5 = "over_3.5"
    OVER_4_5 = "over_4.5"
    UNDER_0_5 = "under_0.5"
    UNDER_1_5 = "under_1.5"
    UNDER_2_5 = "under_2.5"
    UNDER_3_5 = "under_3.5"
    
    # BTTS
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"
    
    # Match Result
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    
    # Double Chance
    DOUBLE_CHANCE_1X = "1x"  # Home or Draw
    DOUBLE_CHANCE_X2 = "x2"  # Draw or Away
    DOUBLE_CHANCE_12 = "12"  # Home or Away (no draw)
    
    # HT/FT
    HT_FT_HH = "ht_ft_1/1"   # Home leads at HT, Home wins
    HT_FT_HD = "ht_ft_1/x"   # Home leads at HT, Draw
    HT_FT_HA = "ht_ft_1/2"   # Home leads at HT, Away wins
    HT_FT_DH = "ht_ft_x/1"   # Draw at HT, Home wins
    HT_FT_DD = "ht_ft_x/x"   # Draw at HT, Draw
    HT_FT_DA = "ht_ft_x/2"   # Draw at HT, Away wins
    HT_FT_AH = "ht_ft_2/1"   # Away leads at HT, Home wins
    HT_FT_AD = "ht_ft_2/x"   # Away leads at HT, Draw
    HT_FT_AA = "ht_ft_2/2"   # Away leads at HT, Away wins
    
    # First Half
    FH_OVER_0_5 = "fh_over_0.5"
    FH_OVER_1_5 = "fh_over_1.5"
    FH_BTTS = "fh_btts"
    
    # Clean Sheet
    HOME_CLEAN_SHEET = "home_clean_sheet"
    AWAY_CLEAN_SHEET = "away_clean_sheet"
    
    # ===== COMBO BETS (Inspired by over25tips.com) =====
    
    # Win + Over 2.5 Goals (Very popular combo)
    HOME_WIN_OVER_2_5 = "home_win_over_2.5"
    AWAY_WIN_OVER_2_5 = "away_win_over_2.5"
    
    # BTTS + Win (High value combo)
    BTTS_HOME_WIN = "btts_home_win"
    BTTS_AWAY_WIN = "btts_away_win"
    
    # BTTS + Over 2.5 (Goals galore combo)
    BTTS_OVER_2_5 = "btts_over_2.5"
    
    # Score in Both Halves
    HOME_SCORE_BOTH_HALVES = "home_score_both_halves"
    AWAY_SCORE_BOTH_HALVES = "away_score_both_halves"
    
    # Win to Nil (Clean sheet + win)
    HOME_WIN_TO_NIL = "home_win_to_nil"
    AWAY_WIN_TO_NIL = "away_win_to_nil"
    
    # First Half Result
    FH_HOME_WIN = "fh_home_win"
    FH_DRAW = "fh_draw"
    FH_AWAY_WIN = "fh_away_win"


@dataclass
class SmartPick:
    """Individual pick in an accumulator"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    bet_type: str
    bet_label: str
    probability: float
    odds: float
    confidence: float
    reasoning: str
    kickoff: Optional[str] = None
    is_cascade: bool = False  # True if derived from cascade logic
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SmartAccumulator:
    """Complete smart accumulator"""
    id: str
    name: str
    category: str  # sure_wins, goals, btts, result, htft
    description: str
    emoji: str
    picks: List[SmartPick]
    combined_odds: float
    combined_probability: float
    confidence_rating: str  # "very_high", "high", "medium"
    risk_level: str  # "low", "medium", "high"
    suggested_stake: float
    potential_return: float
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'emoji': self.emoji,
            'picks': [p.to_dict() for p in self.picks],
            'num_picks': len(self.picks),
            'combined_odds': round(self.combined_odds, 2),
            'combined_probability': round(self.combined_probability, 4),
            'win_chance_pct': round(self.combined_probability * 100, 1),
            'confidence_rating': self.confidence_rating,
            'risk_level': self.risk_level,
            'suggested_stake': self.suggested_stake,
            'potential_return': round(self.potential_return, 2),
            'created_at': self.created_at
        }


class SmartAccumulatorGenerator:
    """
    Generates smart accumulators using cascade logic and probability analysis.
    
    Cascade Logic Examples:
    - xG >= 3.5 → Over 3.5 (70%), Over 2.5 (88%), Over 1.5 (96%), Over 0.5 (99%)
    - Home Win 85% → Double Chance 1X (95%+)
    - BTTS Yes 80% → Over 1.5 likely (90%+)
    """
    
    # Accumulator categories
    CATEGORIES = {
        'sure_wins': {
            'name': 'Sure Wins',
            'emoji': '🔥',
            'description': 'Ultra-high confidence picks (91%+)',
            'min_confidence': 0.91
        },
        'over_0_5': {
            'name': 'Over 0.5 Goals Banker',
            'emoji': '⚽',
            'description': 'At least 1 goal in the match',
            'min_confidence': 0.92
        },
        'over_1_5': {
            'name': 'Over 1.5 Goals',
            'emoji': '🎯',
            'description': 'At least 2 goals in the match',
            'min_confidence': 0.80
        },
        'over_2_5': {
            'name': 'Over 2.5 Goals',
            'emoji': '🔥',
            'description': 'High-scoring matches (3+ goals)',
            'min_confidence': 0.65
        },
        'over_3_5': {
            'name': 'Over 3.5 Goals',
            'emoji': '💥',
            'description': 'Goal-fest matches (4+ goals)',
            'min_confidence': 0.55
        },
        'btts': {
            'name': 'Both Teams to Score',
            'emoji': '⚔️',
            'description': 'Both teams will find the net',
            'min_confidence': 0.60
        },
        'result': {
            'name': 'Match Result (1X2)',
            'emoji': '🏆',
            'description': 'Strong match outcome predictions',
            'min_confidence': 0.65
        },
        'double_chance': {
            'name': 'Double Chance',
            'emoji': '🛡️',
            'description': 'Safer picks with 2 outcomes covered',
            'min_confidence': 0.80
        },
        'htft': {
            'name': 'HT/FT Predictions',
            'emoji': '⏱️',
            'description': 'Halftime and Fulltime combined',
            'min_confidence': 0.50
        },
        'first_half': {
            'name': 'First Half Goals',
            'emoji': '🥇',
            'description': 'Goals in the first 45 minutes',
            'min_confidence': 0.70
        },
        # ===== COMBO CATEGORIES (Inspired by over25tips.com) =====
        'win_over_2_5': {
            'name': 'Win + Over 2.5',
            'emoji': '🎰',
            'description': 'Team wins AND 3+ goals in match',
            'min_confidence': 0.50
        },
        'btts_win': {
            'name': 'BTTS + Win',
            'emoji': '💎',
            'description': 'Both teams score AND a winner',
            'min_confidence': 0.45
        },
        'btts_over_2_5': {
            'name': 'BTTS + Over 2.5',
            'emoji': '🌟',
            'description': 'Both teams score with 3+ goals',
            'min_confidence': 0.50
        },
        'win_to_nil': {
            'name': 'Win to Nil',
            'emoji': '🧤',
            'description': 'Team wins with clean sheet',
            'min_confidence': 0.40
        },
        'score_both_halves': {
            'name': 'Score Both Halves',
            'emoji': '⏰',
            'description': 'Team scores in 1st AND 2nd half',
            'min_confidence': 0.45
        }
    }
    
    def __init__(self):
        self.bookmaker_margin = 0.05  # 5% margin
    
    def _generate_id(self) -> str:
        """Generate unique accumulator ID"""
        return f"acca_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    
    def prob_to_odds(self, probability: float) -> float:
        """Convert probability to decimal odds with bookmaker margin"""
        if probability <= 0:
            return 50.0
        raw_odds = 1 / probability
        return round(raw_odds * (1 - self.bookmaker_margin), 2)
    
    def apply_cascade_logic(self, prediction: Dict) -> List[Dict]:
        """
        Apply cascade rules to derive implied bets from predictions.
        
        The key insight: High xG predictions IMPLY lower goal lines.
        If we predict xG = 3.5, then:
        - Over 3.5: ~65% (the base prediction)
        - Over 2.5: ~85% (implied - easier target)
        - Over 1.5: ~95% (implied - even easier)
        - Over 0.5: ~99% (implied - almost certain)
        """
        cascades = []
        
        # Extract prediction data
        goals = prediction.get('goals', {})
        final_pred = prediction.get('final_prediction', prediction.get('prediction', {}))
        
        # Get xG values
        xg_data = goals.get('expected_goals', {})
        home_xg = xg_data.get('home', 0) if isinstance(xg_data, dict) else 0
        away_xg = xg_data.get('away', 0) if isinstance(xg_data, dict) else 0
        total_xg = home_xg + away_xg
        
        # Get over/under probabilities
        over_under = goals.get('over_under', {})
        over_0_5 = over_under.get('over_0.5', 0)
        over_1_5 = over_under.get('over_1.5', 0)
        over_2_5 = over_under.get('over_2.5', 0)
        over_3_5 = over_under.get('over_3.5', 0)
        
        # Get BTTS
        btts_data = goals.get('btts', {})
        btts_yes = btts_data.get('yes', 0)
        
        # Get match result probabilities
        home_prob = final_pred.get('home_win_prob', 0)
        draw_prob = final_pred.get('draw_prob', 0)
        away_prob = final_pred.get('away_win_prob', 0)
        
        # ==== CASCADE RULES ====
        
        # Rule 1: High xG cascades to lower goal lines
        if total_xg >= 3.5:
            cascades.append({
                'bet_type': BetType.OVER_3_5.value,
                'probability': max(over_3_5, 0.65),
                'reasoning': f'High xG ({total_xg:.1f}) predicts 4+ goals',
                'is_cascade': False
            })
            cascades.append({
                'bet_type': BetType.OVER_2_5.value,
                'probability': max(over_2_5, 0.85),
                'reasoning': f'xG {total_xg:.1f} implies Over 2.5 (cascade)',
                'is_cascade': True
            })
            cascades.append({
                'bet_type': BetType.OVER_1_5.value,
                'probability': max(over_1_5, 0.95),
                'reasoning': f'xG {total_xg:.1f} implies Over 1.5 (cascade)',
                'is_cascade': True
            })
            cascades.append({
                'bet_type': BetType.OVER_0_5.value,
                'probability': max(over_0_5, 0.99),
                'reasoning': f'xG {total_xg:.1f} - goals virtually guaranteed',
                'is_cascade': True
            })
        elif total_xg >= 2.8:
            cascades.append({
                'bet_type': BetType.OVER_2_5.value,
                'probability': max(over_2_5, 0.70),
                'reasoning': f'Strong xG ({total_xg:.1f}) favors 3+ goals',
                'is_cascade': False
            })
            cascades.append({
                'bet_type': BetType.OVER_1_5.value,
                'probability': max(over_1_5, 0.90),
                'reasoning': f'xG {total_xg:.1f} implies Over 1.5 (cascade)',
                'is_cascade': True
            })
            cascades.append({
                'bet_type': BetType.OVER_0_5.value,
                'probability': max(over_0_5, 0.98),
                'reasoning': f'xG {total_xg:.1f} - goal expected',
                'is_cascade': True
            })
        elif total_xg >= 2.0:
            cascades.append({
                'bet_type': BetType.OVER_1_5.value,
                'probability': max(over_1_5, 0.80),
                'reasoning': f'Moderate xG ({total_xg:.1f}) suggests 2+ goals',
                'is_cascade': False
            })
            cascades.append({
                'bet_type': BetType.OVER_0_5.value,
                'probability': max(over_0_5, 0.95),
                'reasoning': f'xG {total_xg:.1f} implies at least 1 goal',
                'is_cascade': True
            })
        
        # Rule 2: Strong favorite cascades to Double Chance
        if home_prob >= 0.75:
            cascades.append({
                'bet_type': BetType.HOME_WIN.value,
                'probability': home_prob,
                'reasoning': f'Strong home favorite ({home_prob*100:.0f}%)',
                'is_cascade': False
            })
            cascades.append({
                'bet_type': BetType.DOUBLE_CHANCE_1X.value,
                'probability': min(0.98, home_prob + draw_prob),
                'reasoning': f'Home/Draw covers {(home_prob+draw_prob)*100:.0f}%',
                'is_cascade': True
            })
        elif away_prob >= 0.75:
            cascades.append({
                'bet_type': BetType.AWAY_WIN.value,
                'probability': away_prob,
                'reasoning': f'Strong away favorite ({away_prob*100:.0f}%)',
                'is_cascade': False
            })
            cascades.append({
                'bet_type': BetType.DOUBLE_CHANCE_X2.value,
                'probability': min(0.98, away_prob + draw_prob),
                'reasoning': f'Away/Draw covers {(away_prob+draw_prob)*100:.0f}%',
                'is_cascade': True
            })
        
        # Rule 3: BTTS cascades to Over 1.5
        if btts_yes >= 0.70:
            cascades.append({
                'bet_type': BetType.BTTS_YES.value,
                'probability': btts_yes,
                'reasoning': f'Both teams attacking ({btts_yes*100:.0f}%)',
                'is_cascade': False
            })
            if over_1_5 < btts_yes:  # BTTS implies Over 1.5
                cascades.append({
                    'bet_type': BetType.OVER_1_5.value,
                    'probability': max(over_1_5, btts_yes * 0.95),
                    'reasoning': f'BTTS Yes implies Over 1.5 (cascade)',
                    'is_cascade': True
                })
        
        # Rule 4: Both teams with good xG = BTTS
        if home_xg >= 1.2 and away_xg >= 1.0:
            if btts_yes >= 0.55:
                cascades.append({
                    'bet_type': BetType.BTTS_YES.value,
                    'probability': btts_yes,
                    'reasoning': f'Both teams have scoring xG ({home_xg:.1f} vs {away_xg:.1f})',
                    'is_cascade': False
                })
        
        # Rule 5: HT/FT for strong favorites
        if home_prob >= 0.80:
            ht_ft_prob = home_prob * 0.65  # Estimated HT/FT probability
            cascades.append({
                'bet_type': BetType.HT_FT_HH.value,
                'probability': ht_ft_prob,
                'reasoning': f'Strong home team likely leads at HT',
                'is_cascade': False
            })
        elif away_prob >= 0.80:
            ht_ft_prob = away_prob * 0.55  # Away HT leads less common
            cascades.append({
                'bet_type': BetType.HT_FT_AA.value,
                'probability': ht_ft_prob,
                'reasoning': f'Strong away team could lead early',
                'is_cascade': False
            })
        
        # ===== COMBO BETS (Over25Tips.com inspired) =====
        
        # Rule 6: Win + Over 2.5 Goals (Popular combo)
        # Strong favorite + high xG = likely to win with 3+ goals
        if home_prob >= 0.70 and over_2_5 >= 0.60:
            combo_prob = home_prob * over_2_5 * 1.1  # Slight positive correlation
            combo_prob = min(0.65, combo_prob)  # Cap it
            cascades.append({
                'bet_type': BetType.HOME_WIN_OVER_2_5.value,
                'probability': combo_prob,
                'reasoning': f'Home win ({home_prob*100:.0f}%) + Over 2.5 ({over_2_5*100:.0f}%)',
                'is_cascade': False
            })
        elif away_prob >= 0.70 and over_2_5 >= 0.60:
            combo_prob = away_prob * over_2_5 * 1.1
            combo_prob = min(0.60, combo_prob)
            cascades.append({
                'bet_type': BetType.AWAY_WIN_OVER_2_5.value,
                'probability': combo_prob,
                'reasoning': f'Away win ({away_prob*100:.0f}%) + Over 2.5 ({over_2_5*100:.0f}%)',
                'is_cascade': False
            })
        
        # Rule 7: BTTS + Win (High value combo from over25tips)
        # Team wins in a game where both teams score
        if home_prob >= 0.65 and btts_yes >= 0.60:
            combo_prob = home_prob * btts_yes * 0.85  # Negative correlation factor
            cascades.append({
                'bet_type': BetType.BTTS_HOME_WIN.value,
                'probability': combo_prob,
                'reasoning': f'Home wins with BTTS ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        elif away_prob >= 0.65 and btts_yes >= 0.60:
            combo_prob = away_prob * btts_yes * 0.80
            cascades.append({
                'bet_type': BetType.BTTS_AWAY_WIN.value,
                'probability': combo_prob,
                'reasoning': f'Away wins with BTTS ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # Rule 8: BTTS + Over 2.5 (Goals galore from over25tips)
        # Both teams score AND 3+ goals total
        if btts_yes >= 0.65 and over_2_5 >= 0.60:
            # BTTS + Over 2.5 has high positive correlation
            combo_prob = min(btts_yes, over_2_5) * 0.95
            cascades.append({
                'bet_type': BetType.BTTS_OVER_2_5.value,
                'probability': combo_prob,
                'reasoning': f'BTTS + Over 2.5 - attacking match ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # Rule 9: Win to Nil (Clean sheet + win)
        # Strong defense + attack mismatch
        clean_sheet_home = goals.get('clean_sheet', {}).get('home', 0)
        clean_sheet_away = goals.get('clean_sheet', {}).get('away', 0)
        
        if home_prob >= 0.75 and clean_sheet_home >= 0.40:
            combo_prob = home_prob * clean_sheet_home
            cascades.append({
                'bet_type': BetType.HOME_WIN_TO_NIL.value,
                'probability': combo_prob,
                'reasoning': f'Home wins to nil - strong defense ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        elif away_prob >= 0.75 and clean_sheet_away >= 0.35:
            combo_prob = away_prob * clean_sheet_away
            cascades.append({
                'bet_type': BetType.AWAY_WIN_TO_NIL.value,
                'probability': combo_prob,
                'reasoning': f'Away wins to nil ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # Rule 10: Score in Both Halves (from over25tips stats page)
        # High-scoring team likely to score in both halves
        if home_xg >= 1.8 and home_prob >= 0.60:
            both_halves_prob = min(0.55, home_xg / 4)  # Rough estimation
            cascades.append({
                'bet_type': BetType.HOME_SCORE_BOTH_HALVES.value,
                'probability': both_halves_prob,
                'reasoning': f'Home xG {home_xg:.1f} - likely goals in both halves',
                'is_cascade': False
            })
        
        return cascades
    
    def _extract_match_info(self, prediction: Dict) -> Dict:
        """Extract match information from prediction"""
        match = prediction.get('match', {})
        
        home_team = match.get('home_team', {})
        away_team = match.get('away_team', {})
        
        if isinstance(home_team, dict):
            home_name = home_team.get('name', 'Home')
        else:
            home_name = str(home_team) if home_team else 'Home'
            
        if isinstance(away_team, dict):
            away_name = away_team.get('name', 'Away')
        else:
            away_name = str(away_team) if away_team else 'Away'
        
        return {
            'match_id': match.get('id', str(uuid.uuid4())[:8]),
            'home_team': home_name,
            'away_team': away_name,
            'league': prediction.get('league', 'Unknown'),
            'kickoff': match.get('time', match.get('date', None))
        }
    
    def _get_bet_label(self, bet_type: str) -> str:
        """Get human-readable label for bet type"""
        labels = {
            'over_0.5': 'Over 0.5 Goals',
            'over_1.5': 'Over 1.5 Goals',
            'over_2.5': 'Over 2.5 Goals',
            'over_3.5': 'Over 3.5 Goals',
            'over_4.5': 'Over 4.5 Goals',
            'under_2.5': 'Under 2.5 Goals',
            'under_3.5': 'Under 3.5 Goals',
            'btts_yes': 'Both Teams to Score',
            'btts_no': 'No BTTS',
            'home_win': 'Home Win',
            'draw': 'Draw',
            'away_win': 'Away Win',
            '1x': 'Home or Draw',
            'x2': 'Draw or Away',
            '12': 'Home or Away',
            'ht_ft_1/1': 'HT/FT: Home-Home',
            'ht_ft_2/2': 'HT/FT: Away-Away',
            'ht_ft_x/1': 'HT/FT: Draw-Home',
            'ht_ft_x/2': 'HT/FT: Draw-Away',
            'fh_over_0.5': 'First Half Over 0.5',
            'fh_over_1.5': 'First Half Over 1.5',
            'home_clean_sheet': 'Home Clean Sheet',
            'away_clean_sheet': 'Away Clean Sheet',
            # Combo bets (over25tips inspired)
            'home_win_over_2.5': 'Home Win + Over 2.5',
            'away_win_over_2.5': 'Away Win + Over 2.5',
            'btts_home_win': 'BTTS + Home Win',
            'btts_away_win': 'BTTS + Away Win',
            'btts_over_2.5': 'BTTS + Over 2.5',
            'home_win_to_nil': 'Home Win to Nil',
            'away_win_to_nil': 'Away Win to Nil',
            'home_score_both_halves': 'Home Scores Both Halves',
            'away_score_both_halves': 'Away Scores Both Halves',
            'fh_home_win': 'First Half Home Win',
            'fh_draw': 'First Half Draw',
            'fh_away_win': 'First Half Away Win'
        }
        return labels.get(bet_type, bet_type.replace('_', ' ').title())
    
    def generate_sure_wins(
        self, 
        predictions: List[Dict], 
        max_picks: int = 5
    ) -> Optional[SmartAccumulator]:
        """
        Generate Sure Wins accumulator with 91%+ confidence picks.
        Uses cascade logic to find the safest bets.
        """
        picks = []
        category = self.CATEGORIES['sure_wins']
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            # Find highest probability cascade bet
            for cascade in cascades:
                if cascade['probability'] >= 0.91:
                    pick = SmartPick(
                        match_id=match_info['match_id'],
                        home_team=match_info['home_team'],
                        away_team=match_info['away_team'],
                        league=match_info['league'],
                        bet_type=cascade['bet_type'],
                        bet_label=self._get_bet_label(cascade['bet_type']),
                        probability=cascade['probability'],
                        odds=self.prob_to_odds(cascade['probability']),
                        confidence=cascade['probability'],
                        reasoning=cascade['reasoning'],
                        kickoff=match_info['kickoff'],
                        is_cascade=cascade['is_cascade']
                    )
                    picks.append(pick)
                    break  # One pick per match
        
        # Sort by probability and take top picks
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator('sure_wins', picks)
        return None
    
    def generate_goals_acca(
        self,
        predictions: List[Dict],
        goal_line: str = "2.5",
        max_picks: int = 5
    ) -> Optional[SmartAccumulator]:
        """Generate Over X.5 goals accumulator"""
        picks = []
        bet_type_map = {
            "0.5": BetType.OVER_0_5.value,
            "1.5": BetType.OVER_1_5.value,
            "2.5": BetType.OVER_2_5.value,
            "3.5": BetType.OVER_3_5.value,
        }
        target_bet = bet_type_map.get(goal_line, BetType.OVER_2_5.value)
        category_key = f"over_{goal_line.replace('.', '_')}"
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            for cascade in cascades:
                if cascade['bet_type'] == target_bet:
                    min_conf = self.CATEGORIES.get(category_key, {}).get('min_confidence', 0.60)
                    if cascade['probability'] >= min_conf:
                        pick = SmartPick(
                            match_id=match_info['match_id'],
                            home_team=match_info['home_team'],
                            away_team=match_info['away_team'],
                            league=match_info['league'],
                            bet_type=cascade['bet_type'],
                            bet_label=self._get_bet_label(cascade['bet_type']),
                            probability=cascade['probability'],
                            odds=self.prob_to_odds(cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator(category_key, picks)
        return None
    
    def generate_btts_acca(
        self,
        predictions: List[Dict],
        max_picks: int = 5
    ) -> Optional[SmartAccumulator]:
        """Generate Both Teams To Score accumulator"""
        picks = []
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            for cascade in cascades:
                if cascade['bet_type'] == BetType.BTTS_YES.value:
                    if cascade['probability'] >= 0.60:
                        pick = SmartPick(
                            match_id=match_info['match_id'],
                            home_team=match_info['home_team'],
                            away_team=match_info['away_team'],
                            league=match_info['league'],
                            bet_type=cascade['bet_type'],
                            bet_label=self._get_bet_label(cascade['bet_type']),
                            probability=cascade['probability'],
                            odds=self.prob_to_odds(cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator('btts', picks)
        return None
    
    def generate_result_acca(
        self,
        predictions: List[Dict],
        max_picks: int = 4
    ) -> Optional[SmartAccumulator]:
        """Generate Match Result (1X2) accumulator"""
        picks = []
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            for cascade in cascades:
                if cascade['bet_type'] in [BetType.HOME_WIN.value, BetType.AWAY_WIN.value]:
                    if cascade['probability'] >= 0.65:
                        pick = SmartPick(
                            match_id=match_info['match_id'],
                            home_team=match_info['home_team'],
                            away_team=match_info['away_team'],
                            league=match_info['league'],
                            bet_type=cascade['bet_type'],
                            bet_label=self._get_bet_label(cascade['bet_type']),
                            probability=cascade['probability'],
                            odds=self.prob_to_odds(cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator('result', picks)
        return None
    
    def generate_double_chance_acca(
        self,
        predictions: List[Dict],
        max_picks: int = 5
    ) -> Optional[SmartAccumulator]:
        """Generate Double Chance accumulator (safer bets)"""
        picks = []
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            for cascade in cascades:
                if cascade['bet_type'] in [BetType.DOUBLE_CHANCE_1X.value, BetType.DOUBLE_CHANCE_X2.value]:
                    if cascade['probability'] >= 0.80:
                        pick = SmartPick(
                            match_id=match_info['match_id'],
                            home_team=match_info['home_team'],
                            away_team=match_info['away_team'],
                            league=match_info['league'],
                            bet_type=cascade['bet_type'],
                            bet_label=self._get_bet_label(cascade['bet_type']),
                            probability=cascade['probability'],
                            odds=self.prob_to_odds(cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator('double_chance', picks)
        return None
    
    def generate_htft_acca(
        self,
        predictions: List[Dict],
        max_picks: int = 4
    ) -> Optional[SmartAccumulator]:
        """Generate HT/FT accumulator (high risk, high reward)"""
        picks = []
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            for cascade in cascades:
                if cascade['bet_type'] in [BetType.HT_FT_HH.value, BetType.HT_FT_AA.value]:
                    if cascade['probability'] >= 0.45:
                        pick = SmartPick(
                            match_id=match_info['match_id'],
                            home_team=match_info['home_team'],
                            away_team=match_info['away_team'],
                            league=match_info['league'],
                            bet_type=cascade['bet_type'],
                            bet_label=self._get_bet_label(cascade['bet_type']),
                            probability=cascade['probability'],
                            odds=self.prob_to_odds(cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator('htft', picks)
        return None
    
    def _build_accumulator(
        self,
        category: str,
        picks: List[SmartPick]
    ) -> SmartAccumulator:
        """Build complete accumulator from picks"""
        cat_info = self.CATEGORIES.get(category, self.CATEGORIES['sure_wins'])
        
        # Calculate combined probability and odds
        combined_prob = 1.0
        combined_odds = 1.0
        for pick in picks:
            combined_prob *= pick.probability
            combined_odds *= pick.odds
        
        # Determine confidence rating
        if combined_prob >= 0.50:
            confidence_rating = "very_high"
        elif combined_prob >= 0.30:
            confidence_rating = "high"
        elif combined_prob >= 0.15:
            confidence_rating = "medium"
        else:
            confidence_rating = "low"
        
        # Determine risk level (inverse of confidence)
        avg_pick_prob = sum(p.probability for p in picks) / len(picks)
        if avg_pick_prob >= 0.85:
            risk_level = "low"
        elif avg_pick_prob >= 0.65:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Suggested stake based on Kelly criterion
        edge = combined_prob * combined_odds - 1
        kelly_fraction = max(0, edge / (combined_odds - 1)) if combined_odds > 1 else 0
        suggested_stake = round(min(kelly_fraction * 100, 50), 2)  # Cap at €50
        
        return SmartAccumulator(
            id=self._generate_id(),
            name=cat_info['name'],
            category=category,
            description=cat_info['description'],
            emoji=cat_info['emoji'],
            picks=picks,
            combined_odds=combined_odds,
            combined_probability=combined_prob,
            confidence_rating=confidence_rating,
            risk_level=risk_level,
            suggested_stake=max(5, suggested_stake),  # Minimum €5
            potential_return=round(suggested_stake * combined_odds, 2),
            created_at=datetime.now().isoformat()
        )
    
    def generate_all(self, predictions: List[Dict]) -> Dict[str, SmartAccumulator]:
        """Generate all accumulator types from predictions"""
        accumulators = {}
        
        # Sure Wins (91%+)
        sure_wins = self.generate_sure_wins(predictions)
        if sure_wins:
            accumulators['sure_wins'] = sure_wins
        
        # Goals
        for goal_line in ["0.5", "1.5", "2.5", "3.5"]:
            acca = self.generate_goals_acca(predictions, goal_line)
            if acca:
                accumulators[f'over_{goal_line.replace(".", "_")}'] = acca
        
        # BTTS
        btts = self.generate_btts_acca(predictions)
        if btts:
            accumulators['btts'] = btts
        
        # Match Result
        result = self.generate_result_acca(predictions)
        if result:
            accumulators['result'] = result
        
        # Double Chance
        dc = self.generate_double_chance_acca(predictions)
        if dc:
            accumulators['double_chance'] = dc
        
        # HT/FT
        htft = self.generate_htft_acca(predictions)
        if htft:
            accumulators['htft'] = htft
        
        # ===== COMBO ACCAS (Over25Tips.com inspired) =====
        
        # Win + Over 2.5
        win_over = self.generate_combo_acca(predictions, 'win_over_2_5')
        if win_over:
            accumulators['win_over_2_5'] = win_over
        
        # BTTS + Win
        btts_win = self.generate_combo_acca(predictions, 'btts_win')
        if btts_win:
            accumulators['btts_win'] = btts_win
        
        # BTTS + Over 2.5
        btts_over = self.generate_combo_acca(predictions, 'btts_over_2_5')
        if btts_over:
            accumulators['btts_over_2_5'] = btts_over
        
        # Win to Nil
        win_nil = self.generate_combo_acca(predictions, 'win_to_nil')
        if win_nil:
            accumulators['win_to_nil'] = win_nil
        
        return accumulators
    
    def generate_combo_acca(
        self,
        predictions: List[Dict],
        combo_type: str,
        max_picks: int = 4
    ) -> Optional[SmartAccumulator]:
        """
        Generate combo bet accumulators (Over25Tips.com inspired).
        
        Combo types:
        - win_over_2_5: Home/Away Win + Over 2.5 Goals
        - btts_win: BTTS + Home/Away Win
        - btts_over_2_5: BTTS + Over 2.5 Goals
        - win_to_nil: Win with Clean Sheet
        """
        picks = []
        
        # Map combo type to bet types
        combo_bet_types = {
            'win_over_2_5': [BetType.HOME_WIN_OVER_2_5.value, BetType.AWAY_WIN_OVER_2_5.value],
            'btts_win': [BetType.BTTS_HOME_WIN.value, BetType.BTTS_AWAY_WIN.value],
            'btts_over_2_5': [BetType.BTTS_OVER_2_5.value],
            'win_to_nil': [BetType.HOME_WIN_TO_NIL.value, BetType.AWAY_WIN_TO_NIL.value],
            'score_both_halves': [BetType.HOME_SCORE_BOTH_HALVES.value, BetType.AWAY_SCORE_BOTH_HALVES.value]
        }
        
        target_bets = combo_bet_types.get(combo_type, [])
        min_conf = self.CATEGORIES.get(combo_type, {}).get('min_confidence', 0.40)
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            for cascade in cascades:
                if cascade['bet_type'] in target_bets:
                    if cascade['probability'] >= min_conf:
                        pick = SmartPick(
                            match_id=match_info['match_id'],
                            home_team=match_info['home_team'],
                            away_team=match_info['away_team'],
                            league=match_info['league'],
                            bet_type=cascade['bet_type'],
                            bet_label=self._get_bet_label(cascade['bet_type']),
                            probability=cascade['probability'],
                            odds=self.prob_to_odds(cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 2:
            return self._build_accumulator(combo_type, picks)
        return None


# Global instance
smart_acca_generator = SmartAccumulatorGenerator()


def generate_smart_accumulators(predictions: List[Dict]) -> Dict[str, Dict]:
    """Convenience function to generate all smart accumulators"""
    accas = smart_acca_generator.generate_all(predictions)
    return {k: v.to_dict() for k, v in accas.items()}


def get_sure_wins(predictions: List[Dict]) -> Optional[Dict]:
    """Get sure wins accumulator only"""
    acca = smart_acca_generator.generate_sure_wins(predictions)
    return acca.to_dict() if acca else None


def get_combo_accas(predictions: List[Dict]) -> Dict[str, Dict]:
    """Get all combo-style accumulators (Over25Tips inspired)"""
    results = {}
    for combo_type in ['win_over_2_5', 'btts_win', 'btts_over_2_5', 'win_to_nil']:
        acca = smart_acca_generator.generate_combo_acca(predictions, combo_type)
        if acca:
            results[combo_type] = acca.to_dict()
    return results

