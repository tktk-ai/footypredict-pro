"""
AI-Powered Sentiment Analysis

Analyzes news, social media, and market sentiment to enhance predictions:
- News sentiment from football sources
- Social media buzz detection
- Market movement analysis
- Public betting patterns
"""

import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import random  # For demo purposes


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    compound: float  # Overall score -1 to 1
    confidence: float
    
    @property
    def label(self) -> str:
        if self.compound > 0.3:
            return "Bullish"
        elif self.compound < -0.3:
            return "Bearish"
        return "Neutral"


@dataclass
class NewsItem:
    """News article structure"""
    title: str
    source: str
    url: str
    published: str
    sentiment: SentimentScore
    teams_mentioned: List[str]
    keywords: List[str]


class SentimentAnalyzer:
    """Advanced sentiment analysis for football predictions"""
    
    # Positive indicators
    POSITIVE_WORDS = {
        'win', 'victory', 'dominant', 'excellent', 'brilliant', 'superb',
        'confident', 'strong', 'powerful', 'unstoppable', 'momentum',
        'form', 'streak', 'unbeaten', 'clean sheet', 'boost', 'return',
        'fit', 'healthy', 'motivated', 'determined', 'focused',
        'upgrade', 'signing', 'reinforcement', 'star', 'key player'
    }
    
    # Negative indicators
    NEGATIVE_WORDS = {
        'loss', 'defeat', 'poor', 'struggling', 'weak', 'crisis',
        'injury', 'injured', 'suspended', 'banned', 'red card',
        'doubt', 'concern', 'worry', 'uncertain', 'inconsistent',
        'fatigue', 'tired', 'exhausted', 'pressure', 'slump',
        'missing', 'absent', 'out', 'sidelined', 'departure'
    }
    
    # Intensity modifiers
    INTENSIFIERS = {
        'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'highly': 1.4,
        'absolutely': 2.0, 'completely': 1.7, 'totally': 1.6
    }
    
    DIMINISHERS = {
        'slightly': 0.5, 'somewhat': 0.6, 'fairly': 0.7, 'relatively': 0.8
    }
    
    def __init__(self):
        self.cache: Dict[str, SentimentScore] = {}
        
    def analyze_text(self, text: str) -> SentimentScore:
        """Analyze sentiment of text"""
        # Check cache
        cache_key = hashlib.md5(text.lower().encode()).hexdigest()[:16]
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = 0
        negative_count = 0
        intensity = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifiers/diminishers
            if word in self.INTENSIFIERS:
                intensity = self.INTENSIFIERS[word]
            elif word in self.DIMINISHERS:
                intensity = self.DIMINISHERS[word]
            elif word in self.POSITIVE_WORDS:
                positive_count += intensity
                intensity = 1.0
            elif word in self.NEGATIVE_WORDS:
                negative_count += intensity
                intensity = 1.0
            else:
                # Check for multi-word phrases
                if i > 0:
                    phrase = f"{words[i-1]} {word}"
                    if phrase in ['clean sheet', 'red card', 'key player']:
                        if phrase in ['clean sheet', 'key player']:
                            positive_count += intensity
                        else:
                            negative_count += intensity
                intensity = 1.0
        
        total = max(positive_count + negative_count, 1)
        
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        neutral_ratio = 1 - (positive_ratio + negative_ratio)
        
        # Compound score: -1 to 1
        compound = (positive_count - negative_count) / total
        compound = max(-1, min(1, compound))
        
        # Confidence based on word count
        confidence = min(0.95, 0.5 + (len(words) / 200))
        
        score = SentimentScore(
            positive=round(positive_ratio, 3),
            negative=round(negative_ratio, 3),
            neutral=round(max(0, neutral_ratio), 3),
            compound=round(compound, 3),
            confidence=round(confidence, 3)
        )
        
        self.cache[cache_key] = score
        return score
    
    def analyze_team_sentiment(self, team: str, news_items: List[str]) -> Dict:
        """Analyze overall sentiment for a team"""
        if not news_items:
            return {
                'team': team,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'articles_analyzed': 0
            }
        
        scores = [self.analyze_text(item) for item in news_items]
        
        avg_compound = sum(s.compound for s in scores) / len(scores)
        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        
        return {
            'team': team,
            'sentiment': 'positive' if avg_compound > 0.1 else 'negative' if avg_compound < -0.1 else 'neutral',
            'score': round(avg_compound, 3),
            'confidence': round(avg_confidence, 3),
            'articles_analyzed': len(news_items),
            'breakdown': {
                'positive': sum(1 for s in scores if s.compound > 0.1),
                'negative': sum(1 for s in scores if s.compound < -0.1),
                'neutral': sum(1 for s in scores if -0.1 <= s.compound <= 0.1)
            }
        }


class MarketSentimentTracker:
    """Track betting market sentiment and movements"""
    
    def __init__(self):
        self.odds_history: Dict[str, List[Dict]] = defaultdict(list)
        self.public_bets: Dict[str, Dict] = {}
        
    def record_odds(self, match_id: str, bookmaker: str, odds: Dict):
        """Record odds snapshot"""
        self.odds_history[match_id].append({
            'bookmaker': bookmaker,
            'odds': odds,
            'timestamp': datetime.now().isoformat()
        })
        
    def analyze_odds_movement(self, match_id: str) -> Dict:
        """Analyze odds movement pattern"""
        history = self.odds_history.get(match_id, [])
        
        if len(history) < 2:
            return {
                'match_id': match_id,
                'movement': 'insufficient_data',
                'trend': 'unknown',
                'confidence': 0.0
            }
        
        first = history[0]['odds']
        last = history[-1]['odds']
        
        # Calculate movement for each outcome
        movements = {}
        for outcome in ['home', 'draw', 'away']:
            if outcome in first and outcome in last:
                change = ((last[outcome] - first[outcome]) / first[outcome]) * 100
                movements[outcome] = round(change, 2)
        
        # Determine trend
        max_decrease = min(movements.values()) if movements else 0
        if max_decrease < -5:
            trend = 'strong_steam'  # Significant money moving
        elif max_decrease < -2:
            trend = 'light_steam'
        else:
            trend = 'stable'
        
        # Find which outcome is shortening most
        shortening_outcome = min(movements, key=movements.get) if movements else None
        
        return {
            'match_id': match_id,
            'movements': movements,
            'trend': trend,
            'shortening': shortening_outcome,
            'snapshots': len(history),
            'confidence': min(0.9, 0.3 + len(history) * 0.1)
        }
    
    def get_public_betting_sentiment(self, match_id: str) -> Dict:
        """Get public betting patterns (simulated)"""
        # In production, this would integrate with betting exchanges
        
        # Simulate public betting data
        home_pct = random.randint(25, 55)
        away_pct = random.randint(20, 45)
        draw_pct = 100 - home_pct - away_pct
        
        return {
            'match_id': match_id,
            'public_bets': {
                'home': home_pct,
                'draw': draw_pct,
                'away': away_pct
            },
            'sharp_money': {
                'home': random.randint(30, 50),
                'draw': random.randint(20, 35),
                'away': random.randint(25, 45)
            },
            'consensus': 'home' if home_pct > 45 else 'away' if away_pct > 40 else 'split'
        }


class SmartBettingAdvisor:
    """AI-powered betting advice based on multiple signals"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_tracker = MarketSentimentTracker()
        
    def generate_advice(
        self,
        match: Dict,
        prediction: Dict,
        news: List[str] = None
    ) -> Dict:
        """Generate smart betting advice"""
        home = match.get('home', '')
        away = match.get('away', '')
        match_id = f"{home}_{away}"
        
        # Analyze sentiment
        home_sentiment = self.sentiment_analyzer.analyze_team_sentiment(
            home, news[:3] if news else []
        )
        away_sentiment = self.sentiment_analyzer.analyze_team_sentiment(
            away, news[3:6] if news and len(news) > 3 else []
        )
        
        # Get market sentiment
        market = self.market_tracker.analyze_odds_movement(match_id)
        public = self.market_tracker.get_public_betting_sentiment(match_id)
        
        # Calculate adjusted confidence
        base_confidence = prediction.get('confidence', 0.5)
        
        # Sentiment adjustment
        sentiment_boost = 0
        predicted_outcome = prediction.get('predicted_outcome', '')
        
        if predicted_outcome == 'Home Win' and home_sentiment['score'] > 0.2:
            sentiment_boost = 0.03
        elif predicted_outcome == 'Away Win' and away_sentiment['score'] > 0.2:
            sentiment_boost = 0.03
        elif home_sentiment['score'] < -0.2 and predicted_outcome == 'Home Win':
            sentiment_boost = -0.05
        elif away_sentiment['score'] < -0.2 and predicted_outcome == 'Away Win':
            sentiment_boost = -0.05
        
        # Market adjustment
        market_boost = 0
        if market.get('shortening') == 'home' and predicted_outcome == 'Home Win':
            market_boost = 0.02
        elif market.get('shortening') == 'away' and predicted_outcome == 'Away Win':
            market_boost = 0.02
        
        # Contrarian signal (betting against public)
        contrarian_signal = None
        public_bets = public.get('public_bets', {})
        if public_bets.get('home', 0) > 60 and predicted_outcome != 'Home Win':
            contrarian_signal = {'type': 'fade_public', 'strength': 'strong'}
        elif public_bets.get('away', 0) > 50 and predicted_outcome != 'Away Win':
            contrarian_signal = {'type': 'fade_public', 'strength': 'moderate'}
        
        # Final adjusted confidence
        adjusted_confidence = min(0.99, base_confidence + sentiment_boost + market_boost)
        
        # Generate recommendation
        if adjusted_confidence >= 0.85:
            recommendation = 'STRONG_BET'
            stake_pct = 3.0
        elif adjusted_confidence >= 0.70:
            recommendation = 'MODERATE_BET'
            stake_pct = 2.0
        elif adjusted_confidence >= 0.60:
            recommendation = 'SMALL_BET'
            stake_pct = 1.0
        else:
            recommendation = 'SKIP'
            stake_pct = 0
        
        return {
            'match': f"{home} vs {away}",
            'prediction': predicted_outcome,
            'base_confidence': round(base_confidence * 100, 1),
            'adjusted_confidence': round(adjusted_confidence * 100, 1),
            'recommendation': recommendation,
            'stake_percentage': stake_pct,
            'signals': {
                'sentiment': {
                    'home': home_sentiment,
                    'away': away_sentiment,
                    'impact': round(sentiment_boost * 100, 1)
                },
                'market': {
                    'movement': market,
                    'impact': round(market_boost * 100, 1)
                },
                'contrarian': contrarian_signal
            },
            'reasoning': self._generate_reasoning(
                predicted_outcome, adjusted_confidence, 
                sentiment_boost, market_boost, contrarian_signal
            )
        }
    
    def _generate_reasoning(
        self, outcome: str, confidence: float,
        sentiment_boost: float, market_boost: float,
        contrarian: Optional[Dict]
    ) -> str:
        """Generate human-readable reasoning"""
        reasons = []
        
        if confidence >= 0.85:
            reasons.append(f"High confidence ({confidence*100:.0f}%) in {outcome}")
        elif confidence >= 0.70:
            reasons.append(f"Moderate confidence ({confidence*100:.0f}%) in {outcome}")
        
        if sentiment_boost > 0:
            reasons.append("Positive news sentiment supports this pick")
        elif sentiment_boost < 0:
            reasons.append("⚠️ Negative sentiment - proceed with caution")
        
        if market_boost > 0:
            reasons.append("Smart money moving in this direction")
        
        if contrarian:
            reasons.append(f"Contrarian value: public heavily on opposite side")
        
        return " | ".join(reasons) if reasons else "Standard confidence pick"


# Global instances
sentiment_analyzer = SentimentAnalyzer()
market_tracker = MarketSentimentTracker()
betting_advisor = SmartBettingAdvisor()


def analyze_match_sentiment(home: str, away: str, news: List[str] = None) -> Dict:
    """Quick sentiment analysis for a match"""
    home_sent = sentiment_analyzer.analyze_team_sentiment(home, news[:5] if news else [])
    away_sent = sentiment_analyzer.analyze_team_sentiment(away, news[5:] if news else [])
    
    return {
        'home': home_sent,
        'away': away_sent,
        'advantage': 'home' if home_sent['score'] > away_sent['score'] + 0.1 
                     else 'away' if away_sent['score'] > home_sent['score'] + 0.1 
                     else 'neutral'
    }


def get_smart_advice(match: Dict, prediction: Dict, news: List[str] = None) -> Dict:
    """Get AI-powered betting advice"""
    return betting_advisor.generate_advice(match, prediction, news)
