"""
Telegram Bot for Football Predictions

Sends daily predictions to your Telegram.
Setup: Create a bot with @BotFather, get token, add to .env
"""

import os
import asyncio
import requests
from datetime import datetime
from typing import Optional


class TelegramBot:
    """
    Telegram bot for sending predictions
    
    Usage:
        1. Create bot with @BotFather on Telegram
        2. Get your chat ID (message @userinfobot)
        3. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env
    """
    
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
    
    def is_configured(self) -> bool:
        """Check if bot is configured"""
        return bool(self.token and self.chat_id)
    
    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Send a message to the configured chat"""
        if not self.is_configured():
            print("Telegram bot not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': parse_mode
                }
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def format_prediction_message(self, predictions: list) -> str:
        """Format predictions for Telegram"""
        now = datetime.now()
        message = f"⚽ <b>Daily Predictions</b>\n"
        message += f"📅 {now.strftime('%A, %B %d, %Y')}\n"
        message += "─" * 25 + "\n\n"
        
        for pred in predictions[:10]:  # Limit to 10
            match = pred.get('match', {})
            prediction = pred.get('prediction', {})
            goals = pred.get('goals', {})
            
            if not prediction:
                continue
            
            home = match.get('home_team', {}).get('short_name') or match.get('home_team', {}).get('name', '?')
            away = match.get('away_team', {}).get('short_name') or match.get('away_team', {}).get('name', '?')
            
            # Format kickoff time
            kickoff = match.get('kickoff', '')
            try:
                dt = datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M')
            except:
                time_str = '??:??'
            
            # Get probabilities
            home_prob = prediction.get('home_win_prob', 0) * 100
            draw_prob = prediction.get('draw_prob', 0) * 100
            away_prob = prediction.get('away_win_prob', 0) * 100
            outcome = prediction.get('predicted_outcome', 'Unknown')
            confidence = prediction.get('confidence', 0) * 100
            
            # Goals data
            xg = goals.get('expected_goals', {})
            over_25 = goals.get('over_under', {}).get('over_2.5', 0) * 100
            
            # Build match block
            message += f"🏟 <b>{home}</b> vs <b>{away}</b>\n"
            message += f"⏰ {time_str}\n"
            message += f"📊 H: {home_prob:.0f}% | D: {draw_prob:.0f}% | A: {away_prob:.0f}%\n"
            message += f"🎯 <b>{outcome}</b> ({confidence:.0f}% conf)\n"
            
            if xg:
                message += f"⚽ xG: {xg.get('home', 0):.1f} - {xg.get('away', 0):.1f}"
                message += f" | O2.5: {over_25:.0f}%\n"
            
            message += "\n"
        
        message += "─" * 25 + "\n"
        message += "⚠️ For entertainment only"
        
        return message
    
    def send_daily_predictions(self, predictions: list) -> bool:
        """Send daily predictions digest"""
        message = self.format_prediction_message(predictions)
        return self.send_message(message)
    
    def send_value_alert(self, match: dict, prediction: dict, edge: float) -> bool:
        """Send alert for high-value bet"""
        home = match.get('home_team', {}).get('name', 'Home')
        away = match.get('away_team', {}).get('name', 'Away')
        outcome = prediction.get('predicted_outcome', 'Unknown')
        
        message = f"🔔 <b>VALUE BET ALERT</b>\n\n"
        message += f"🏟 {home} vs {away}\n"
        message += f"🎯 Pick: <b>{outcome}</b>\n"
        message += f"💰 Edge: <b>+{edge:.1f}%</b>\n"
        message += f"\n⏰ Act fast!"
        
        return self.send_message(message)


# Global bot instance
telegram_bot = TelegramBot()


def send_daily_digest(predictions: list) -> bool:
    """Convenience function to send daily digest"""
    return telegram_bot.send_daily_predictions(predictions)


def send_value_bet_alert(match: dict, prediction: dict, edge: float) -> bool:
    """Convenience function to send value bet alert"""
    return telegram_bot.send_value_alert(match, prediction, edge)


def send_daily_banker_alert(banker: dict) -> bool:
    """Send daily banker alert"""
    home = banker.get('home_team', 'Home')
    away = banker.get('away_team', 'Away')
    outcome = banker.get('predicted_outcome', 'Unknown')
    confidence = banker.get('confidence', 0) * 100
    
    message = f"🎯 <b>DAILY BANKER</b>\n\n"
    message += f"🏟 <b>{home}</b> vs <b>{away}</b>\n"
    message += f"📊 Prediction: <b>{outcome}</b>\n"
    message += f"💪 Confidence: <b>{confidence:.0f}%</b>\n"
    message += f"\n⚽ Today's safest pick!"
    
    return telegram_bot.send_message(message)


def send_sure_win_alerts(picks: list) -> bool:
    """Send sure win alerts (91%+ confidence)"""
    if not picks:
        return False
    
    message = f"🔒 <b>SURE WINS (91%+)</b>\n"
    message += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n"
    message += "─" * 20 + "\n\n"
    
    for pick in picks[:5]:
        home = pick.get('home_team', 'Home')
        away = pick.get('away_team', 'Away')
        outcome = pick.get('predicted_outcome', '')
        conf = pick.get('confidence', 0) * 100
        message += f"• {home} vs {away}\n"
        message += f"  → <b>{outcome}</b> ({conf:.0f}%)\n\n"
    
    return telegram_bot.send_message(message)


def send_match_result(match_id: str, home: str, away: str, predicted: str, actual: str, correct: bool) -> bool:
    """Send match result notification"""
    emoji = "✅" if correct else "❌"
    message = f"{emoji} <b>Result</b>\n\n"
    message += f"🏟 {home} vs {away}\n"
    message += f"🎯 Predicted: {predicted}\n"
    message += f"📊 Actual: {actual}\n"
    message += f"\n{'🎉 Correct!' if correct else '😔 Incorrect'}"
    
    return telegram_bot.send_message(message)


def send_accuracy_update(stats: dict) -> bool:
    """Send weekly accuracy update"""
    accuracy = stats.get('accuracy', 0) * 100
    total = stats.get('total', 0)
    
    message = f"📊 <b>Weekly Accuracy Update</b>\n\n"
    message += f"🎯 Accuracy: <b>{accuracy:.1f}%</b>\n"
    message += f"📈 Total predictions: {total}\n"
    
    by_conf = stats.get('by_confidence', {})
    if by_conf.get('high_90', {}).get('total', 0):
        high_acc = by_conf['high_90']['accuracy'] * 100
        message += f"💪 90%+ picks: <b>{high_acc:.1f}%</b>\n"
    
    return telegram_bot.send_message(message)


# ============================================================
# V3.0 Functions: Monte Carlo, Player Props, Value Betting
# ============================================================

def send_monte_carlo_prediction(home_team: str, away_team: str, result: dict) -> bool:
    """
    Send Monte Carlo simulation results.
    
    Args:
        home_team: Home team name
        away_team: Away team name  
        result: Monte Carlo simulation result dict
    """
    message = f"🎲 <b>Monte Carlo Simulation</b>\n"
    message += f"🏟 {home_team} vs {away_team}\n"
    message += "─" * 25 + "\n\n"
    
    # 1X2 Probabilities
    probs = result.get('1x2', {})
    message += f"📊 <b>Match Odds (100k sims)</b>\n"
    message += f"• Home Win: <b>{probs.get('home_win', 0):.1%}</b>\n"
    message += f"• Draw: <b>{probs.get('draw', 0):.1%}</b>\n"
    message += f"• Away Win: <b>{probs.get('away_win', 0):.1%}</b>\n\n"
    
    # Expected Goals
    xg = result.get('expected_goals', {})
    message += f"⚽ <b>Expected Goals</b>\n"
    message += f"• {home_team}: {xg.get('home', 0):.2f}\n"
    message += f"• {away_team}: {xg.get('away', 0):.2f}\n\n"
    
    # Over/Under
    ou = result.get('over_under', {})
    message += f"📈 <b>Over/Under</b>\n"
    message += f"• O1.5: {ou.get('over_1.5', 0):.1%}\n"
    message += f"• O2.5: {ou.get('over_2.5', 0):.1%}\n"
    message += f"• O3.5: {ou.get('over_3.5', 0):.1%}\n\n"
    
    # BTTS
    btts = result.get('btts', {})
    message += f"🔥 <b>BTTS</b>\n"
    message += f"• Yes: {btts.get('yes', 0):.1%}\n"
    message += f"• No: {btts.get('no', 0):.1%}\n\n"
    
    # Top Correct Scores
    cs = result.get('correct_scores', {})
    if cs:
        message += f"🎯 <b>Top Correct Scores</b>\n"
        for score, prob in list(cs.items())[:5]:
            message += f"• {score}: {prob:.1%}\n"
    
    return telegram_bot.send_message(message)


def send_player_props_prediction(player_name: str, predictions: dict) -> bool:
    """
    Send player props predictions.
    
    Args:
        player_name: Player name
        predictions: Player props prediction dict
    """
    message = f"⚽ <b>Player Props: {player_name}</b>\n"
    message += "─" * 25 + "\n\n"
    
    # Anytime Scorer
    ats = predictions.get('anytime_scorer', {})
    message += f"🎯 <b>Anytime Scorer</b>\n"
    message += f"• Probability: <b>{ats.get('probability', 0):.1%}</b>\n"
    message += f"• Fair Odds: {ats.get('fair_odds', 0):.2f}\n"
    message += f"• Expected Goals: {ats.get('expected_goals', 0):.2f}\n\n"
    
    # 2+ Goals
    g2 = predictions.get('2_plus_goals', {})
    message += f"⚽⚽ <b>2+ Goals</b>\n"
    message += f"• Probability: <b>{g2.get('probability', 0):.1%}</b>\n"
    message += f"• Fair Odds: {g2.get('fair_odds', 0):.2f}\n\n"
    
    # Shots
    shots = predictions.get('shots', {})
    message += f"🥅 <b>Shots</b>\n"
    message += f"• Expected: {shots.get('expected_shots', 0):.1f}\n"
    message += f"• Over 2.5: {shots.get('over_line_prob', 0):.1%}\n\n"
    
    # Cards
    cards = predictions.get('cards', {})
    message += f"🟨 <b>Yellow Card</b>\n"
    message += f"• Probability: {cards.get('yellow_card_prob', 0):.1%}\n"
    message += f"• Fair Odds: {cards.get('fair_odds_yellow', 0):.2f}\n"
    
    return telegram_bot.send_message(message)


def send_value_bets_alert(value_bets: list) -> bool:
    """
    Send value betting opportunities alert.
    
    Args:
        value_bets: List of value bet dicts with market, probability, odds, edge
    """
    if not value_bets:
        return False
    
    message = f"💰 <b>VALUE BETS DETECTED</b>\n"
    message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    message += "─" * 25 + "\n\n"
    
    for i, bet in enumerate(value_bets[:10], 1):
        rating = "⭐⭐⭐" if bet.get('edge', 0) >= 0.10 else ("⭐⭐" if bet.get('edge', 0) >= 0.06 else "⭐")
        
        message += f"{i}. <b>{bet.get('market', 'Unknown')}</b> {rating}\n"
        message += f"   📊 Prob: {bet.get('probability', 0):.1%}\n"
        message += f"   💵 Odds: {bet.get('odds', 0):.2f}\n"
        message += f"   📈 Edge: <b>+{bet.get('edge', 0):.1%}</b>\n"
        message += f"   💰 Kelly: {bet.get('kelly_stake_pct', 0):.1f}%\n\n"
    
    message += "─" * 25 + "\n"
    message += "⚠️ Bet responsibly"
    
    return telegram_bot.send_message(message)


def send_rl_recommendation(match: str, recommendation: dict) -> bool:
    """
    Send RL betting strategy recommendation.
    
    Args:
        match: Match description (e.g., "Man City vs Arsenal")
        recommendation: RL agent recommendation dict
    """
    action = recommendation.get('recommendation', 'skip')
    stake = recommendation.get('stake_percentage', 0)
    edge = recommendation.get('edge', 0)
    ev = recommendation.get('expected_value', 0)
    
    if action == 'skip':
        emoji = "⏭️"
        action_text = "SKIP (No Value)"
    else:
        emoji = "✅"
        action_text = f"BET {stake:.1f}% of bankroll"
    
    message = f"🤖 <b>AI Betting Recommendation</b>\n"
    message += f"🏟 {match}\n"
    message += "─" * 25 + "\n\n"
    message += f"{emoji} <b>{action_text}</b>\n\n"
    message += f"📊 Edge: {edge:.1%}\n"
    message += f"💰 Expected Value: {ev:.2f}\n"
    message += f"🎯 Confidence: {recommendation.get('confidence', 0):.1%}\n\n"
    message += "─" * 25 + "\n"
    message += "⚠️ AI suggestion - bet responsibly"
    
    return telegram_bot.send_message(message)

