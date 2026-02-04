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
    date: Optional[str] = None  # Match date (YYYY-MM-DD)
    time: Optional[str] = None  # Match time (HH:MM)
    venue: Optional[str] = None  # Stadium/venue
    
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
            'min_confidence': 0.70
        },
        'over_1_5': {
            'name': 'Over 1.5 Goals',
            'emoji': '🎯',
            'description': 'At least 2 goals in the match',
            'min_confidence': 0.60
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
            'min_confidence': 0.50
        },
        'result': {
            'name': 'Match Result (1X2)',
            'emoji': '🏆',
            'description': 'Strong match outcome predictions',
            'min_confidence': 0.55
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
        },
        # ===== JACKPOT CATEGORIES =====
        'jackpot': {
            'name': 'Jackpot 30',
            'emoji': '🎰',
            'description': 'Top 30 most winnable matches (any bet type)',
            'min_confidence': 0.60,
            'max_picks': 30,
            'bet_type': 'any'
        },
        'super_jackpot': {
            'name': 'Super Jackpot 40',
            'emoji': '💰',
            'description': 'Top 40 most winnable matches (any bet type)',
            'min_confidence': 0.55,
            'max_picks': 40,
            'bet_type': 'any'
        },
        'jackpot_over15': {
            'name': 'Jackpot Over 1.5',
            'emoji': '⚽',
            'description': 'Top 30 Over 1.5 Goals predictions',
            'min_confidence': 0.70,
            'max_picks': 30,
            'bet_type': 'over_1.5'
        },
        'super_jackpot_over15': {
            'name': 'Super Jackpot Over 1.5',
            'emoji': '🔥',
            'description': 'Top 40 Over 1.5 Goals predictions',
            'min_confidence': 0.65,
            'max_picks': 40,
            'bet_type': 'over_1.5'
        }
    }
    
    def __init__(self):
        self.bookmaker_margin = 0.05  # 5% margin
    
    def _generate_id(self) -> str:
        """Generate unique accumulator ID"""
        return f"acca_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    
    def get_real_odds(self, bet_type: str, real_odds: Dict, probability: float) -> float:
        """Get real odds for a bet type, falling back to calculated odds if not available.
        
        Args:
            bet_type: The type of bet (e.g., 'over_1.5', 'home_win', 'btts_yes')
            real_odds: Dictionary of real odds from SportyBet/SofaScore
            probability: Our calculated probability (used for fallback)
            
        Returns:
            Real odds if available, otherwise calculated odds from probability
        """
        if not real_odds:
            return self.prob_to_odds(probability)
        
        # Map bet types to odds keys
        bet_type_to_odds_key = {
            'home_win': 'home_win',
            'away_win': 'away_win',
            'draw': 'draw',
            'over_0.5': 'over_0.5',
            'over_1.5': 'over_1.5',
            'over_2.5': 'over_2.5',
            'over_3.5': 'over_3.5',
            'over_4.5': 'over_4.5',
            'under_2.5': 'under_2.5',
            'btts_yes': 'btts_yes',
            'btts_no': 'btts_no',
            'dc_1x': 'dc_1x',
            'dc_x2': 'dc_x2',
            'dc_12': 'dc_12',
        }
        
        odds_key = bet_type_to_odds_key.get(bet_type)
        if odds_key and real_odds.get(odds_key):
            real_odd = real_odds[odds_key]
            if isinstance(real_odd, (int, float)) and real_odd >= 1.01:
                return round(float(real_odd), 2)
        
        # Fallback to calculated odds
        return self.prob_to_odds(probability)
    
    def prob_to_odds(self, probability: float) -> float:
        """Convert probability to decimal odds with bookmaker margin.
        
        Decimal odds = 1 / (probability * (1 + margin))
        This ensures odds are always >= 1.0 and includes the house edge.
        
        NOTE: Use get_real_odds() instead when real odds are available.
        """
        if probability <= 0:
            return 50.0
        if probability >= 1:
            return 1.01  # Minimum valid odds
        
        # Apply bookmaker margin to probability (reduces effective probability)
        adjusted_prob = probability * (1 + self.bookmaker_margin)
        adjusted_prob = min(adjusted_prob, 0.99)  # Cap to avoid odds < 1.01
        
        raw_odds = 1 / adjusted_prob
        return max(1.01, round(raw_odds, 2))  # Ensure minimum 1.01 odds
    
    def is_easy_match(self, prediction: Dict) -> Tuple[bool, float]:
        """
        Identify matches with high predictability for accuracy.
        
        Easy matches have:
        - Strong favorite (>65% win probability)
        - High xG differential (>1.0)
        - Clear outcome indicators
        
        Returns: (is_easy, score)
        """
        final_pred = prediction.get('final_prediction', prediction.get('prediction', {}))
        goals = prediction.get('goals', {})
        
        home_prob = final_pred.get('home_win_prob', 0)
        away_prob = final_pred.get('away_win_prob', 0)
        draw_prob = final_pred.get('draw_prob', 0.33)
        max_prob = max(home_prob, away_prob)
        
        xg_data = goals.get('expected_goals', {})
        home_xg = xg_data.get('home', 0) if isinstance(xg_data, dict) else 0
        away_xg = xg_data.get('away', 0) if isinstance(xg_data, dict) else 0
        total_xg = home_xg + away_xg
        xg_diff = abs(home_xg - away_xg)
        
        # Calculate predictability score (0-10)
        score = 0
        
        # Factor 1: Strong favorite
        if max_prob >= 0.75: score += 4
        elif max_prob >= 0.65: score += 3
        elif max_prob >= 0.55: score += 2
        
        # Factor 2: Low draw probability (clearer outcome)
        if draw_prob <= 0.20: score += 2
        elif draw_prob <= 0.25: score += 1
        
        # Factor 3: xG differential
        if xg_diff >= 1.5: score += 3
        elif xg_diff >= 1.0: score += 2
        elif xg_diff >= 0.5: score += 1
        
        # Factor 4: High total xG (goals likely)
        if total_xg >= 3.0: score += 1
        
        return (score >= 5, score)

    
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
        
        # ===== FALLBACK RULES - Always generate entries from raw probabilities =====
        # These ensure we get regular ACCAs even when strict xG thresholds aren't met
        
        # Fallback Over 0.5 - almost always generates (most matches have goals)
        if over_0_5 >= 0.70:
            cascades.append({
                'bet_type': BetType.OVER_0_5.value,
                'probability': over_0_5,
                'reasoning': f'Over 0.5 goals ({over_0_5*100:.0f}%)',
                'is_cascade': False
            })
        
        # Fallback Over 1.5 - generates for attacking matches
        if over_1_5 >= 0.60:
            cascades.append({
                'bet_type': BetType.OVER_1_5.value,
                'probability': over_1_5,
                'reasoning': f'Over 1.5 goals ({over_1_5*100:.0f}%)',
                'is_cascade': False
            })
        
        # Fallback Over 2.5 - generates for high-scoring matches
        if over_2_5 >= 0.50:
            cascades.append({
                'bet_type': BetType.OVER_2_5.value,
                'probability': over_2_5,
                'reasoning': f'Over 2.5 goals ({over_2_5*100:.0f}%)',
                'is_cascade': False
            })
        
        # Fallback BTTS - generates for matches where both can score
        if btts_yes >= 0.50:
            cascades.append({
                'bet_type': BetType.BTTS_YES.value,
                'probability': btts_yes,
                'reasoning': f'BTTS Yes ({btts_yes*100:.0f}%)',
                'is_cascade': False
            })
        
        # Fallback Double Chance - safer bets
        dc_1x = home_prob + draw_prob
        dc_x2 = draw_prob + away_prob
        dc_12 = home_prob + away_prob
        
        if dc_1x >= 0.70:
            cascades.append({
                'bet_type': BetType.DOUBLE_CHANCE_1X.value,
                'probability': dc_1x,
                'reasoning': f'Home or Draw ({dc_1x*100:.0f}%)',
                'is_cascade': False
            })
        if dc_x2 >= 0.70:
            cascades.append({
                'bet_type': BetType.DOUBLE_CHANCE_X2.value,
                'probability': dc_x2,
                'reasoning': f'Draw or Away ({dc_x2*100:.0f}%)',
                'is_cascade': False
            })
        if dc_12 >= 0.75:
            cascades.append({
                'bet_type': BetType.DOUBLE_CHANCE_12.value,
                'probability': dc_12,
                'reasoning': f'No Draw ({dc_12*100:.0f}%)',
                'is_cascade': False
            })
        
        # Fallback Match Result - moderate preferences (lowered from 0.55)
        if home_prob >= 0.45:
            cascades.append({
                'bet_type': BetType.HOME_WIN.value,
                'probability': home_prob,
                'reasoning': f'Home Win ({home_prob*100:.0f}%)',
                'is_cascade': False
            })
        if away_prob >= 0.45:
            cascades.append({
                'bet_type': BetType.AWAY_WIN.value,
                'probability': away_prob,
                'reasoning': f'Away Win ({away_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # ===== FALLBACK COMBO BETS =====
        
        # Win + Over 2.5 (popular combo)
        if home_prob >= 0.50 and over_2_5 >= 0.45:
            combo_prob = home_prob * over_2_5 * 1.1
            cascades.append({
                'bet_type': BetType.HOME_WIN_OVER_2_5.value,
                'probability': min(0.65, combo_prob),
                'reasoning': f'Home + Over 2.5 ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        if away_prob >= 0.50 and over_2_5 >= 0.45:
            combo_prob = away_prob * over_2_5 * 1.1
            cascades.append({
                'bet_type': BetType.AWAY_WIN_OVER_2_5.value,
                'probability': min(0.65, combo_prob),
                'reasoning': f'Away + Over 2.5 ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # BTTS + Win
        if btts_yes >= 0.45 and home_prob >= 0.50:
            combo_prob = btts_yes * home_prob * 1.05
            cascades.append({
                'bet_type': BetType.BTTS_HOME_WIN.value,
                'probability': min(0.55, combo_prob),
                'reasoning': f'BTTS + Home Win ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        if btts_yes >= 0.45 and away_prob >= 0.50:
            combo_prob = btts_yes * away_prob * 1.05
            cascades.append({
                'bet_type': BetType.BTTS_AWAY_WIN.value,
                'probability': min(0.55, combo_prob),
                'reasoning': f'BTTS + Away Win ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # BTTS + Over 2.5
        if btts_yes >= 0.45 and over_2_5 >= 0.45:
            combo_prob = btts_yes * over_2_5 * 1.15  # Strong positive correlation
            cascades.append({
                'bet_type': BetType.BTTS_OVER_2_5.value,
                'probability': min(0.65, combo_prob),
                'reasoning': f'BTTS + Over 2.5 ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # Win to Nil
        clean_sheet_home = goals.get('clean_sheet', {}).get('home', 0.35)
        clean_sheet_away = goals.get('clean_sheet', {}).get('away', 0.25)
        
        if home_prob >= 0.55 and clean_sheet_home >= 0.30:
            combo_prob = home_prob * clean_sheet_home
            cascades.append({
                'bet_type': BetType.HOME_WIN_TO_NIL.value,
                'probability': combo_prob,
                'reasoning': f'Home to Nil ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        if away_prob >= 0.55 and clean_sheet_away >= 0.25:
            combo_prob = away_prob * clean_sheet_away
            cascades.append({
                'bet_type': BetType.AWAY_WIN_TO_NIL.value,
                'probability': combo_prob,
                'reasoning': f'Away to Nil ({combo_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        # HT/FT - strong home or away favorites
        if home_prob >= 0.60:
            ht_ft_prob = home_prob * 0.65  # ~65% of full-match prob
            cascades.append({
                'bet_type': BetType.HT_FT_HH.value,
                'probability': ht_ft_prob,
                'reasoning': f'HT/FT Home-Home ({ht_ft_prob*100:.0f}%)',
                'is_cascade': False
            })
        if away_prob >= 0.60:
            ht_ft_prob = away_prob * 0.55  # Away HT lead is rarer
            cascades.append({
                'bet_type': BetType.HT_FT_AA.value,
                'probability': ht_ft_prob,
                'reasoning': f'HT/FT Away-Away ({ht_ft_prob*100:.0f}%)',
                'is_cascade': False
            })
        
        return cascades
    
    def _extract_match_info(self, prediction: Dict) -> Dict:
        """Extract match information from prediction including real odds"""
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
        
        # Extract date and time
        match_date = match.get('date', '')
        match_time = match.get('time', '')
        
        # If only date exists, try to extract time from it
        if match_date and not match_time:
            if 'T' in str(match_date):
                parts = str(match_date).split('T')
                match_date = parts[0]
                match_time = parts[1][:5] if len(parts) > 1 else ''
        
        # Extract real odds from prediction data
        real_odds = prediction.get('real_odds', {})
        
        return {
            'match_id': str(match.get('id', str(uuid.uuid4())[:8])),
            'home_team': home_name,
            'away_team': away_name,
            'league': prediction.get('league', 'Unknown'),
            'kickoff': match.get('time', match.get('date', None)),
            'date': match_date or 'TBD',
            'time': match_time or 'TBD',
            'venue': match.get('venue', match.get('stadium', prediction.get('league', 'TBD'))),
            'real_odds': real_odds  # Real odds from SportyBet/SofaScore
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
                        odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                        confidence=cascade['probability'],
                        reasoning=cascade['reasoning'],
                        kickoff=match_info['kickoff'],
                        is_cascade=cascade['is_cascade'],
                        date=match_info['date'],
                        time=match_info['time'],
                        venue=match_info['venue']
                    )
                    picks.append(pick)
                    break  # One pick per match
        
        # Sort by probability and take top picks
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
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
                            odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade'],
                            date=match_info['date'],
                            time=match_info['time'],
                            venue=match_info['venue']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
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
                            odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade'],
                            date=match_info['date'],
                            time=match_info['time'],
                            venue=match_info['venue']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
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
                            odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade'],
                            date=match_info['date'],
                            time=match_info['time'],
                            venue=match_info['venue']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
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
                            odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade'],
                            date=match_info['date'],
                            time=match_info['time'],
                            venue=match_info['venue']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
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
                            odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade'],
                            date=match_info['date'],
                            time=match_info['time'],
                            venue=match_info['venue']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
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
        
        # ===== JACKPOT ACCAS =====
        
        # Jackpot (top 30 - any bet type)
        jackpot = self.generate_jackpot(predictions, 'jackpot')
        if jackpot:
            accumulators['jackpot'] = jackpot
        
        # Super Jackpot (top 40 - any bet type)
        super_jackpot = self.generate_jackpot(predictions, 'super_jackpot')
        if super_jackpot:
            accumulators['super_jackpot'] = super_jackpot
        
        # Jackpot Over 1.5 (top 30 - Over 1.5 goals only)
        jackpot_o15 = self.generate_jackpot(predictions, 'jackpot_over15')
        if jackpot_o15:
            accumulators['jackpot_over15'] = jackpot_o15
        
        # Super Jackpot Over 1.5 (top 40 - Over 1.5 goals only)
        super_jackpot_o15 = self.generate_jackpot(predictions, 'super_jackpot_over15')
        if super_jackpot_o15:
            accumulators['super_jackpot_over15'] = super_jackpot_o15
        
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
                            odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                            confidence=cascade['probability'],
                            reasoning=cascade['reasoning'],
                            kickoff=match_info['kickoff'],
                            is_cascade=cascade['is_cascade'],
                            date=match_info['date'],
                            time=match_info['time'],
                            venue=match_info['venue']
                        )
                        picks.append(pick)
                        break
        
        picks.sort(key=lambda x: x.probability, reverse=True)
        picks = picks[:max_picks]
        
        if len(picks) >= 1:
            return self._build_accumulator(combo_type, picks)
        return None
    
    def generate_jackpot(
        self,
        predictions: List[Dict],
        jackpot_type: str = 'jackpot',  # 'jackpot', 'super_jackpot', 'jackpot_over15', 'super_jackpot_over15'
    ) -> Optional[SmartAccumulator]:
        """
        Generate Jackpot accumulator with top winnable matches.
        
        Jackpot types:
        - jackpot: Top 30 matches (any bet type)
        - super_jackpot: Top 40 matches (any bet type)
        - jackpot_over15: Top 30 Over 1.5 goals predictions
        - super_jackpot_over15: Top 40 Over 1.5 goals predictions
        """
        picks = []
        cat_info = self.CATEGORIES.get(jackpot_type, self.CATEGORIES['jackpot'])
        max_picks = cat_info.get('max_picks', 30)
        min_conf = cat_info.get('min_confidence', 0.55)
        target_bet_type = cat_info.get('bet_type', 'any')  # 'any' or specific bet type
        
        # Collect best bet for each match
        match_best_bets = []
        
        for pred in predictions:
            cascades = self.apply_cascade_logic(pred)
            match_info = self._extract_match_info(pred)
            
            if not cascades:
                continue
            
            # Filter cascades based on jackpot type
            if target_bet_type != 'any':
                # Filter to specific bet type (e.g., over_1.5)
                filtered_cascades = [c for c in cascades if c['bet_type'] == target_bet_type]
                if not filtered_cascades:
                    continue
                best_cascade = max(filtered_cascades, key=lambda x: x['probability'])
            else:
                # Any bet type - find the best bet for this match
                best_cascade = max(cascades, key=lambda x: x['probability'])
            
            if best_cascade['probability'] >= min_conf:
                match_best_bets.append({
                    'match_info': match_info,
                    'cascade': best_cascade
                })
        
        # Sort all matches by probability (highest first)
        match_best_bets.sort(key=lambda x: x['cascade']['probability'], reverse=True)
        
        # Take top matches
        for item in match_best_bets[:max_picks]:
            match_info = item['match_info']
            cascade = item['cascade']
            
            pick = SmartPick(
                match_id=match_info['match_id'],
                home_team=match_info['home_team'],
                away_team=match_info['away_team'],
                league=match_info['league'],
                bet_type=cascade['bet_type'],
                bet_label=self._get_bet_label(cascade['bet_type']),
                probability=cascade['probability'],
                odds=self.get_real_odds(cascade['bet_type'], match_info.get('real_odds', {}), cascade['probability']),
                confidence=cascade['probability'],
                reasoning=cascade['reasoning'],
                kickoff=match_info['kickoff'],
                is_cascade=cascade['is_cascade'],
                date=match_info['date'],
                time=match_info['time'],
                venue=match_info['venue']
            )
            picks.append(pick)
        
        if len(picks) >= 5:  # Need at least 5 picks for a jackpot
            return self._build_accumulator(jackpot_type, picks)
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


def generate_daily_predictions(predictions: List[Dict], target_picks: int = 50) -> Dict:
    """
    Generate daily predictions focusing on easy-to-predict matches.
    
    Returns categorized picks targeting the specified number of accurate predictions.
    
    Args:
        predictions: List of match predictions
        target_picks: Target number of total picks (default 50)
    
    Returns:
        Dict with picks organized by category
    """
    from datetime import datetime
    
    generator = smart_acca_generator
    
    # Filter to easy matches first
    easy_predictions = []
    all_match_scores = []
    
    for pred in predictions:
        is_easy, score = generator.is_easy_match(pred)
        all_match_scores.append((pred, score, is_easy))
    
    # Sort by predictability score (highest first)
    all_match_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top matches (prioritize easy ones)
    easy_predictions = [p[0] for p in all_match_scores if p[2]]
    other_predictions = [p[0] for p in all_match_scores if not p[2]]
    
    # Combine: easy first, then others
    sorted_predictions = easy_predictions + other_predictions
    
    # Category targets (total ~50)
    category_targets = {
        'sure_wins': 8,       # 91%+ confidence
        'over_0_5': 10,       # 92%+ (almost certain)
        'over_1_5': 8,        # 85%+ 
        'over_2_5': 6,        # 70%+
        'double_chance': 8,   # 85%+
        'btts': 6,            # 65%+
        'result': 4,          # 70%+
    }
    
    results = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_matches_analyzed': len(predictions),
        'easy_matches': len(easy_predictions),
        'categories': {},
        'all_picks': [],
        'total_picks': 0
    }
    
    # Generate picks for each category
    for category, max_picks in category_targets.items():
        picks_list = []
        
        if category == 'sure_wins':
            acca = generator.generate_sure_wins(sorted_predictions, max_picks)
        elif category.startswith('over_'):
            goal_line = category.replace('over_', '').replace('_', '.')
            acca = generator.generate_goals_acca(sorted_predictions, goal_line, max_picks)
        elif category == 'btts':
            acca = generator.generate_btts_acca(sorted_predictions, max_picks)
        elif category == 'result':
            acca = generator.generate_result_acca(sorted_predictions, max_picks)
        elif category == 'double_chance':
            acca = generator.generate_double_chance_acca(sorted_predictions, max_picks)
        else:
            acca = None
        
        if acca:
            picks_list = acca.to_dict()['picks']
            results['categories'][category] = {
                'name': generator.CATEGORIES.get(category, {}).get('name', category),
                'emoji': generator.CATEGORIES.get(category, {}).get('emoji', '🎯'),
                'picks': picks_list,
                'count': len(picks_list),
                'avg_confidence': sum(p['confidence'] for p in picks_list) / len(picks_list) if picks_list else 0
            }
            results['all_picks'].extend(picks_list)
    
    results['total_picks'] = len(results['all_picks'])
    
    # Add summary statistics
    if results['all_picks']:
        results['summary'] = {
            'avg_confidence': sum(p['confidence'] for p in results['all_picks']) / len(results['all_picks']),
            'highest_confidence': max(p['confidence'] for p in results['all_picks']),
            'by_league': {}
        }
        
        # Count by league
        for pick in results['all_picks']:
            league = pick.get('league', 'Unknown')
            results['summary']['by_league'][league] = results['summary']['by_league'].get(league, 0) + 1
    
    return results


def get_jackpots(predictions: List[Dict], days: int = 1) -> Dict[str, Dict]:
    """
    Get Jackpot and Super Jackpot accumulators.
    
    Args:
        predictions: List of predictions
        days: Timeframe (1=today, 3=next 3 days, 7=next 7 days)
    
    Returns:
        Dict with 'jackpot' and 'super_jackpot' keys
    """
    results = {}
    
    # Jackpot (30 picks)
    jackpot = smart_acca_generator.generate_jackpot(predictions, 'jackpot')
    if jackpot:
        results['jackpot'] = jackpot.to_dict()
    
    # Super Jackpot (40 picks)
    super_jackpot = smart_acca_generator.generate_jackpot(predictions, 'super_jackpot')
    if super_jackpot:
        results['super_jackpot'] = super_jackpot.to_dict()
    
    return results
