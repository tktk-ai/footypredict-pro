"""
AI Betting Assistant - Natural Language Interface

Provides conversational AI interface for:
- Match queries and predictions
- Betting advice and strategy
- Performance insights
- Personalized recommendations
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AssistantResponse:
    """AI Assistant response"""
    message: str
    data: Optional[Dict] = None
    suggestions: List[str] = None
    action: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'message': self.message,
            'data': self.data,
            'suggestions': self.suggestions or [],
            'action': self.action
        }


class IntentClassifier:
    """Classify user intent from natural language"""
    
    INTENTS = {
        'get_prediction': [
            r'predict(?:ion)?.*(?:for|of|on)?\s+(\w+)\s+(?:vs|versus|v)\s+(\w+)',
            r'who.*win.*(\w+)\s+(?:vs|versus|v)\s+(\w+)',
            r'(\w+)\s+(?:vs|versus|v)\s+(\w+)',
            r'what.*odds.*(\w+)\s+(?:vs|versus|v)\s+(\w+)'
        ],
        'get_tips': [
            r'(?:give|get|show).*tip(?:s)?',
            r'best.*bet(?:s)?.*today',
            r'sure.*win(?:s)?',
            r'what.*bet.*today',
            r'recommend(?:ation)?s?'
        ],
        'get_accumulators': [
            r'accum(?:ulat)?(?:ors?)?',
            r'acca(?:s)?',
            r'parlay(?:s)?',
            r'multi.*bet(?:s)?'
        ],
        'get_stats': [
            r'(?:my)?.*stat(?:istic)?s?',
            r'(?:my)?.*performance',
            r'accuracy',
            r'win.*rate',
            r'roi'
        ],
        'get_bankroll': [
            r'bankroll',
            r'balance',
            r'stake.*(?:advice|recommend)',
            r'how.*much.*bet'
        ],
        'get_form': [
            r'form.*(?:of|for)?\s+(\w+)',
            r'(\w+).*(?:playing|doing)',
            r'how.*is.*(\w+)'
        ],
        'greeting': [
            r'^hi\b',
            r'^hello\b',
            r'^hey\b',
            r'good\s+(?:morning|afternoon|evening)'
        ],
        'help': [
            r'help',
            r'what.*can.*you.*do',
            r'commands?',
            r'features?'
        ]
    }
    
    def classify(self, text: str) -> Tuple[str, Dict]:
        """Classify intent and extract entities"""
        text_lower = text.lower().strip()
        
        for intent, patterns in self.INTENTS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities = {}
                    if match.groups():
                        if intent == 'get_prediction':
                            entities['home'] = match.group(1).title()
                            entities['away'] = match.group(2).title()
                        elif intent == 'get_form':
                            entities['team'] = match.group(1).title()
                    return intent, entities
        
        return 'unknown', {}


class AIBettingAssistant:
    """Conversational AI assistant for betting"""
    
    def __init__(self):
        self.classifier = IntentClassifier()
        self.context: Dict = {}
        self.conversation_history: List[Dict] = []
        
    def process_message(self, message: str, user_id: str = "default") -> AssistantResponse:
        """Process user message and generate response"""
        # Classify intent
        intent, entities = self.classifier.classify(message)
        
        # Store in history
        self.conversation_history.append({
            'user': message,
            'intent': intent,
            'entities': entities,
            'timestamp': datetime.now().isoformat()
        })
        
        # Route to appropriate handler
        handlers = {
            'greeting': self._handle_greeting,
            'help': self._handle_help,
            'get_prediction': self._handle_prediction,
            'get_tips': self._handle_tips,
            'get_accumulators': self._handle_accumulators,
            'get_stats': self._handle_stats,
            'get_bankroll': self._handle_bankroll,
            'get_form': self._handle_form,
            'unknown': self._handle_unknown
        }
        
        handler = handlers.get(intent, self._handle_unknown)
        return handler(entities, user_id)
    
    def _handle_greeting(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle greeting"""
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 18:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        return AssistantResponse(
            message=f"{greeting}! 👋 I'm your AI betting assistant. How can I help you today?",
            suggestions=[
                "Show me today's tips",
                "Get accumulators",
                "Check my stats"
            ]
        )
    
    def _handle_help(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle help request"""
        return AssistantResponse(
            message="""🤖 **I can help you with:**

**Predictions**
• "Predict Bayern vs Dortmund"
• "Who will win Liverpool vs Arsenal?"

**Tips & Recommendations**
• "Show me today's best bets"
• "Get sure wins"
• "What should I bet on?"

**Accumulators**
• "Show me accumulators"
• "Build me a parlay"

**Statistics**
• "Show my stats"
• "What's my win rate?"

**Bankroll**
• "How much should I stake?"
• "Check my bankroll"

Just type naturally and I'll understand! 🎯""",
            suggestions=[
                "Show today's tips",
                "Get predictions for tonight",
                "Build an accumulator"
            ]
        )
    
    def _handle_prediction(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle prediction request"""
        home = entities.get('home', 'Unknown')
        away = entities.get('away', 'Unknown')
        
        try:
            from src.enhanced_predictor_v2 import enhanced_predict_with_goals
            from src.ai_sentiment import get_smart_advice
            from src.pattern_recognition import predict_exact_score
            
            # Get prediction
            prediction = enhanced_predict_with_goals(home, away, 'bundesliga')
            
            # Get score prediction
            score_pred = predict_exact_score(home, away)
            
            # Get smart advice
            advice = get_smart_advice(
                {'home': home, 'away': away},
                prediction
            )
            
            outcome = prediction.get('predicted_outcome', 'Unknown')
            confidence = prediction.get('confidence', 0.5) * 100
            
            message = f"""⚽ **{home} vs {away}**

🎯 **Prediction:** {outcome}
📊 **Confidence:** {confidence:.0f}%
📈 **Expected Score:** {score_pred.get('most_likely', 'N/A')}

💡 **Recommendation:** {advice.get('recommendation', 'N/A')}
📝 **Reasoning:** {advice.get('reasoning', 'N/A')}

**Score Probabilities:**
• Over 2.5: {score_pred.get('over_25_probability', 0)}%
• BTTS: {score_pred.get('btts_probability', 0)}%"""
            
            return AssistantResponse(
                message=message,
                data={
                    'prediction': prediction,
                    'score': score_pred,
                    'advice': advice
                },
                suggestions=[
                    f"How much should I stake on {outcome}?",
                    "Show me similar matches",
                    "Add to accumulator"
                ],
                action='show_prediction'
            )
            
        except Exception as e:
            return AssistantResponse(
                message=f"⚽ **{home} vs {away}**\n\n🔮 Analyzing match data... Please check our predictions page for detailed analysis.",
                suggestions=[
                    "Show today's fixtures",
                    "Get sure wins"
                ]
            )
    
    def _handle_tips(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle tips request"""
        try:
            from src.confidence_sections import get_sure_wins, get_confidence_sections
            
            sure_wins = get_sure_wins(min_confidence=0.85)[:3]
            
            if sure_wins:
                tips = []
                for i, tip in enumerate(sure_wins, 1):
                    tips.append(f"{i}. **{tip.get('match', 'N/A')}**\n   • {tip.get('outcome', 'N/A')} @ {tip.get('odds', 'N/A')}\n   • Confidence: {tip.get('confidence', 0)*100:.0f}%")
                
                message = f"""🎯 **Today's Top Tips**

{chr(10).join(tips)}

💡 _These are our highest confidence picks for today._"""
                
                return AssistantResponse(
                    message=message,
                    data={'tips': sure_wins},
                    suggestions=[
                        "Build an accumulator with these",
                        "Show more tips",
                        "How much should I stake?"
                    ],
                    action='show_tips'
                )
            else:
                return AssistantResponse(
                    message="📊 I'm currently analyzing today's fixtures. Check back soon for tips!",
                    suggestions=["Show me fixtures", "Get accumulators"]
                )
                
        except Exception as e:
            return AssistantResponse(
                message="🎯 Head to our Tips page for today's best picks!",
                suggestions=["Check predictions", "Get accumulators"]
            )
    
    def _handle_accumulators(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle accumulator request"""
        try:
            from src.multi_league_acca import generate_all_multi_league_accas
            
            accas = generate_all_multi_league_accas()[:2]
            
            if accas:
                acca_text = []
                for acca in accas:
                    legs = acca.get('legs', [])
                    acca_text.append(f"""
**{acca.get('name', 'Accumulator')}** ({len(legs)} legs)
Combined Odds: {acca.get('total_odds', 0):.2f}
Confidence: {acca.get('confidence', 0)*100:.0f}%""")
                
                message = f"""🎰 **Today's Accumulators**
{chr(10).join(acca_text)}

💰 _Visit the Accumulators page for full details and betslip._"""
                
                return AssistantResponse(
                    message=message,
                    data={'accumulators': accas},
                    suggestions=[
                        "Show me the safe acca",
                        "Build custom accumulator",
                        "What's the best value acca?"
                    ],
                    action='show_accumulators'
                )
        except:
            pass
        
        return AssistantResponse(
            message="🎰 Check out our Accumulators page for today's best multi-bets!",
            suggestions=["Show tips", "Get predictions"]
        )
    
    def _handle_stats(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle stats request"""
        try:
            from src.accuracy_monitor import get_accuracy_stats
            from src.advanced_analytics import get_analytics_summary
            
            stats = get_accuracy_stats()
            summary = get_analytics_summary()
            
            accuracy = stats.get('accuracy', 0) * 100 if isinstance(stats.get('accuracy'), float) else stats.get('accuracy', 0)
            
            message = f"""📊 **Your Stats**

🎯 Overall Accuracy: **{accuracy:.1f}%**
📈 Total Predictions: **{stats.get('total', 0)}**
✅ Correct: **{stats.get('correct', 0)}**
💰 ROI (30d): **{summary.get('overall', {}).get('roi_30d', 0):.1f}%**

🔥 Current Streak: {summary.get('streak', {}).get('current_streak', 0)} {summary.get('streak', {}).get('streak_type', '')}"""
            
            return AssistantResponse(
                message=message,
                data={'stats': stats, 'summary': summary},
                suggestions=[
                    "Show league breakdown",
                    "What's my best performing league?",
                    "Show weekly trend"
                ]
            )
            
        except:
            return AssistantResponse(
                message="📊 Visit the Dashboard for your complete performance stats!",
                suggestions=["Get predictions", "Show tips"]
            )
    
    def _handle_bankroll(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle bankroll questions"""
        try:
            from src.smart_bankroll import get_bankroll_status, calculate_optimal_stake
            
            status = get_bankroll_status()
            bankroll = status.get('bankroll', {})
            
            message = f"""💰 **Bankroll Status**

📊 Current: **€{bankroll.get('current', 0):.2f}**
📈 ROI: **{bankroll.get('roi', 0):.1f}%**
🎯 Win Rate: **{bankroll.get('win_rate', 0):.1f}%**
📉 Drawdown: **{bankroll.get('drawdown', 0):.1f}%**

💡 **Stake Advice:**
Based on your risk level ({status.get('risk_level', 'moderate')}):
• Max stake: €{bankroll.get('current', 100) * 0.05:.2f} (5%)
• Recommended: €{bankroll.get('current', 100) * 0.02:.2f} (2%) per bet"""
            
            return AssistantResponse(
                message=message,
                data=status,
                suggestions=[
                    "Calculate stake for a bet",
                    "Change risk level",
                    "Show drawdown chart"
                ]
            )
        except:
            return AssistantResponse(
                message="💰 Use our bankroll manager for smart stake sizing!",
                suggestions=["Get predictions", "Show tips"]
            )
    
    def _handle_form(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle team form request"""
        team = entities.get('team', 'Unknown')
        
        try:
            from src.advanced_features import get_team_form
            from src.pattern_recognition import detect_patterns
            
            form = get_team_form(team)
            patterns = detect_patterns(team)
            
            message = f"""📊 **{team} Form Analysis**

**Recent Results:** {' '.join(form.get('last_5', ['N/A']))}
**Form Rating:** {form.get('form_rating', 50)}/100
**Goals Scored (avg):** {form.get('avg_scored', 0):.1f}
**Goals Conceded (avg):** {form.get('avg_conceded', 0):.1f}"""
            
            if patterns:
                pattern_text = "\n".join([f"• {p.get('type', 'N/A')}: {p.get('strength', 0)*100:.0f}%" for p in patterns[:3]])
                message += f"\n\n**Detected Patterns:**\n{pattern_text}"
            
            return AssistantResponse(
                message=message,
                data={'form': form, 'patterns': patterns},
                suggestions=[
                    f"Predict {team}'s next match",
                    f"Show {team}'s history"
                ]
            )
        except:
            return AssistantResponse(
                message=f"📊 Check the Dashboard for {team}'s complete form analysis.",
                suggestions=["Show predictions", "Get tips"]
            )
    
    def _handle_unknown(self, entities: Dict, user_id: str) -> AssistantResponse:
        """Handle unknown intent"""
        return AssistantResponse(
            message="🤔 I'm not sure what you mean. Here are some things I can help with:",
            suggestions=[
                "Predict Bayern vs Dortmund",
                "Show today's tips",
                "Get accumulators",
                "Check my stats",
                "Type 'help' for more options"
            ]
        )
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]


# Global assistant instance
ai_assistant = AIBettingAssistant()


def chat(message: str, user_id: str = "default") -> Dict:
    """Process a chat message"""
    response = ai_assistant.process_message(message, user_id)
    return response.to_dict()


def get_chat_history(user_id: str = "default", limit: int = 10) -> List[Dict]:
    """Get chat history"""
    return ai_assistant.get_conversation_history(limit)
