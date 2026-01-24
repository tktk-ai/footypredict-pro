"""
WhatsApp Integration Module

Send predictions via WhatsApp Business API.
Requires: WhatsApp Business API access or Twilio.

Setup options:
1. WhatsApp Business API (official)
2. Twilio WhatsApp API (easier)
3. WhatsApp Web automation (unofficial)
"""

import os
import requests
from datetime import datetime
from typing import Dict, List, Optional


class WhatsAppBot:
    """
    WhatsApp bot for sending predictions
    
    Uses Twilio WhatsApp API (recommended for simplicity)
    
    Setup:
    1. Create Twilio account: https://www.twilio.com
    2. Enable WhatsApp Sandbox
    3. Add credentials to .env:
       TWILIO_ACCOUNT_SID=your_sid
       TWILIO_AUTH_TOKEN=your_token
       TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
    """
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886')
        self.api_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
    
    def is_configured(self) -> bool:
        """Check if Twilio is configured"""
        return bool(self.account_sid and self.auth_token)
    
    def send_message(self, to_number: str, message: str) -> Dict:
        """
        Send WhatsApp message
        
        Args:
            to_number: Phone number with country code (e.g., +1234567890)
            message: Message text (max 1600 chars for WhatsApp)
        """
        if not self.is_configured():
            return {
                'success': False,
                'error': 'WhatsApp not configured. Add Twilio credentials to .env'
            }
        
        # Format number for WhatsApp
        if not to_number.startswith('whatsapp:'):
            to_number = f"whatsapp:{to_number}"
        
        try:
            response = requests.post(
                self.api_url,
                auth=(self.account_sid, self.auth_token),
                data={
                    'From': self.from_number,
                    'To': to_number,
                    'Body': message[:1600]  # WhatsApp limit
                }
            )
            
            if response.status_code in [200, 201]:
                return {'success': True, 'message_sid': response.json().get('sid')}
            else:
                return {'success': False, 'error': response.text}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def format_predictions_message(self, predictions: List[Dict]) -> str:
        """Format predictions for WhatsApp"""
        now = datetime.now()
        
        # Use simple formatting (WhatsApp supports limited markdown)
        message = f"⚽ *Football Predictions*\n"
        message += f"📅 {now.strftime('%A, %d %B %Y')}\n"
        message += "━" * 20 + "\n\n"
        
        for pred in predictions[:8]:  # Limit for message size
            match = pred.get('match', {})
            prediction = pred.get('prediction', {})
            
            if not prediction:
                continue
            
            home = match.get('home_team', {}).get('short_name') or match.get('home_team', {}).get('name', '?')
            away = match.get('away_team', {}).get('short_name') or match.get('away_team', {}).get('name', '?')
            
            home_prob = prediction.get('home_win_prob', 0) * 100
            draw_prob = prediction.get('draw_prob', 0) * 100
            away_prob = prediction.get('away_win_prob', 0) * 100
            outcome = prediction.get('predicted_outcome', '?')
            confidence = prediction.get('confidence', 0) * 100
            
            message += f"🏟 *{home}* vs *{away}*\n"
            message += f"H: {home_prob:.0f}% | D: {draw_prob:.0f}% | A: {away_prob:.0f}%\n"
            message += f"🎯 Pick: *{outcome}* ({confidence:.0f}%)\n\n"
        
        message += "━" * 20 + "\n"
        message += "⚠️ _For entertainment only_"
        
        return message
    
    def send_daily_predictions(self, to_number: str, predictions: List[Dict]) -> Dict:
        """Send daily predictions digest"""
        message = self.format_predictions_message(predictions)
        return self.send_message(to_number, message)
    
    def send_value_alert(
        self,
        to_number: str,
        match: str,
        selection: str,
        odds: float,
        edge: float
    ) -> Dict:
        """Send value bet alert"""
        message = f"🔔 *VALUE BET ALERT*\n\n"
        message += f"🏟 {match}\n"
        message += f"🎯 Pick: *{selection}*\n"
        message += f"💰 Odds: {odds}\n"
        message += f"📈 Edge: +{edge:.1f}%\n\n"
        message += "⏰ _Act fast!_"
        
        return self.send_message(to_number, message)
    
    def send_match_reminder(
        self,
        to_number: str,
        home_team: str,
        away_team: str,
        kickoff: str,
        prediction: str
    ) -> Dict:
        """Send match reminder"""
        message = f"⏰ *Match Starting Soon!*\n\n"
        message += f"🏟 *{home_team}* vs *{away_team}*\n"
        message += f"🕐 Kickoff: {kickoff}\n"
        message += f"🎯 Our pick: *{prediction}*\n\n"
        message += "_Good luck!_ 🍀"
        
        return self.send_message(to_number, message)


# Global instance
whatsapp_bot = WhatsAppBot()


def send_whatsapp(to_number: str, message: str) -> Dict:
    """Send WhatsApp message"""
    return whatsapp_bot.send_message(to_number, message)


def send_predictions_whatsapp(to_number: str, predictions: List[Dict]) -> Dict:
    """Send predictions via WhatsApp"""
    return whatsapp_bot.send_daily_predictions(to_number, predictions)
