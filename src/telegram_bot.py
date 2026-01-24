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
