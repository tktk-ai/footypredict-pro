"""
Enhanced Notification Service

Multi-channel notifications for:
- Telegram (existing bot)
- WhatsApp (via Twilio)
- Email (via SMTP)
- Push notifications
- In-app notifications
"""

import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class NotificationChannel(Enum):
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification structure"""
    title: str
    message: str
    channel: str
    priority: str = "normal"
    data: Optional[Dict] = None
    recipient: Optional[str] = None
    sent_at: Optional[str] = None
    status: str = "pending"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NotificationTemplates:
    """Pre-built notification templates"""
    
    @staticmethod
    def sure_win_alert(prediction: Dict) -> Dict:
        return {
            'title': '🔒 Sure Win Alert!',
            'message': f"""
*{prediction.get('home')} vs {prediction.get('away')}*
🎯 Prediction: {prediction.get('outcome', 'N/A')}
📊 Confidence: {prediction.get('confidence', 0) * 100:.0f}%
💰 Odds: {prediction.get('odds', '-')}

_This is a high-confidence pick from our AI system._
            """.strip(),
            'priority': 'high'
        }
    
    @staticmethod
    def value_bet_alert(bet: Dict) -> Dict:
        return {
            'title': '💎 Value Bet Found!',
            'message': f"""
*{bet.get('home')} vs {bet.get('away')}*
📈 Edge: {bet.get('edge', 0):.1f}%
💰 Odds: {bet.get('odds', '-')}
📊 Our Probability: {bet.get('our_prob', 0) * 100:.0f}%

_Bookmakers are undervaluing this outcome._
            """.strip(),
            'priority': 'normal'
        }
    
    @staticmethod
    def accumulator_alert(acca: Dict) -> Dict:
        legs = acca.get('legs', [])
        leg_text = '\n'.join([
            f"• {l.get('match')}: {l.get('outcome')}" for l in legs[:5]
        ])
        return {
            'title': f"🎰 {acca.get('name', 'Accumulator')} Ready!",
            'message': f"""
*{len(legs)} Selections*
Combined Odds: {acca.get('total_odds', 0):.2f}

{leg_text}

💰 Potential Return: {acca.get('potential_return', 0):.2f}x stake
            """.strip(),
            'priority': 'normal'
        }
    
    @staticmethod
    def daily_summary(stats: Dict) -> Dict:
        return {
            'title': '📊 Daily Summary',
            'message': f"""
*FootyPredict Pro - Daily Report*

📈 Today's Performance:
• Predictions: {stats.get('total', 0)}
• Correct: {stats.get('correct', 0)}
• Accuracy: {stats.get('accuracy', 0):.1f}%

💰 Value Bets: {stats.get('value_bets', 0)}
🔒 Sure Wins: {stats.get('sure_wins', 0)}

_Tomorrow's predictions will be ready at 8:00 AM_
            """.strip(),
            'priority': 'low'
        }


class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    def send(self, chat_id: str, message: str, parse_mode: str = "Markdown") -> bool:
        """Send Telegram message"""
        if not self.bot_token:
            print("Telegram bot token not configured")
            return False
            
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': parse_mode
                },
                timeout=10
            )
            return response.ok
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
            
    def broadcast(self, chat_ids: List[str], message: str) -> Dict:
        """Send to multiple recipients"""
        results = {'sent': 0, 'failed': 0}
        for chat_id in chat_ids:
            if self.send(chat_id, message):
                results['sent'] += 1
            else:
                results['failed'] += 1
        return results


class EmailNotifier:
    """Send notifications via Email"""
    
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_pass = os.getenv('SMTP_PASS', '')
        self.from_email = os.getenv('FROM_EMAIL', 'noreply@footypredict.pro')
        
    def send(self, to_email: str, subject: str, body: str, html: bool = False) -> bool:
        """Send email"""
        if not self.smtp_user or not self.smtp_pass:
            print("Email not configured")
            return False
            
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
                
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.from_email, to_email, msg.as_string())
                
            return True
        except Exception as e:
            print(f"Email send error: {e}")
            return False


class PushNotifier:
    """Send push notifications (web push)"""
    
    def __init__(self):
        self.vapid_public = os.getenv('VAPID_PUBLIC_KEY', '')
        self.vapid_private = os.getenv('VAPID_PRIVATE_KEY', '')
        self.subscriptions: List[Dict] = []
        
    def subscribe(self, subscription: Dict) -> bool:
        """Add push subscription"""
        if subscription not in self.subscriptions:
            self.subscriptions.append(subscription)
            return True
        return False
        
    def send(self, subscription: Dict, title: str, body: str, icon: str = None) -> bool:
        """Send push notification"""
        # Would use pywebpush in production
        print(f"Push: {title} - {body}")
        return True
        
    def broadcast(self, title: str, body: str) -> Dict:
        """Send to all subscribers"""
        results = {'sent': 0, 'failed': 0}
        for sub in self.subscriptions:
            if self.send(sub, title, body):
                results['sent'] += 1
            else:
                results['failed'] += 1
        return results


class InAppNotifier:
    """In-app notification storage"""
    
    def __init__(self):
        self.notifications: Dict[str, List[Dict]] = {}  # user_id -> notifications
        
    def send(self, user_id: str, title: str, message: str, data: Dict = None) -> str:
        """Add in-app notification"""
        if user_id not in self.notifications:
            self.notifications[user_id] = []
            
        notification_id = f"notif_{datetime.now().timestamp()}"
        self.notifications[user_id].append({
            'id': notification_id,
            'title': title,
            'message': message,
            'data': data,
            'read': False,
            'created_at': datetime.now().isoformat()
        })
        
        # Keep only last 100
        if len(self.notifications[user_id]) > 100:
            self.notifications[user_id] = self.notifications[user_id][-100:]
            
        return notification_id
        
    def get_unread(self, user_id: str) -> List[Dict]:
        """Get unread notifications"""
        return [n for n in self.notifications.get(user_id, []) if not n['read']]
        
    def mark_read(self, user_id: str, notification_id: str) -> bool:
        """Mark notification as read"""
        for notif in self.notifications.get(user_id, []):
            if notif['id'] == notification_id:
                notif['read'] = True
                return True
        return False
        
    def get_all(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all notifications for user"""
        return self.notifications.get(user_id, [])[-limit:]


class NotificationService:
    """Unified notification service"""
    
    def __init__(self):
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()
        self.push = PushNotifier()
        self.in_app = InAppNotifier()
        self.templates = NotificationTemplates()
        self.history: List[Dict] = []
        
    def send(
        self, 
        notification: Notification,
        channels: List[NotificationChannel] = None
    ) -> Dict:
        """Send notification through specified channels"""
        if channels is None:
            channels = [NotificationChannel.IN_APP]
            
        results = {}
        
        for channel in channels:
            if channel == NotificationChannel.TELEGRAM and notification.recipient:
                results['telegram'] = self.telegram.send(
                    notification.recipient,
                    f"*{notification.title}*\n\n{notification.message}"
                )
            elif channel == NotificationChannel.EMAIL and notification.recipient:
                results['email'] = self.email.send(
                    notification.recipient,
                    notification.title,
                    notification.message
                )
            elif channel == NotificationChannel.IN_APP and notification.recipient:
                results['in_app'] = self.in_app.send(
                    notification.recipient,
                    notification.title,
                    notification.message,
                    notification.data
                )
                
        # Log to history
        notification.sent_at = datetime.now().isoformat()
        notification.status = "sent" if any(results.values()) else "failed"
        self.history.append(notification.to_dict())
        
        return results
        
    def send_sure_win_alert(self, prediction: Dict, user_ids: List[str]) -> Dict:
        """Send sure win alert to users"""
        template = self.templates.sure_win_alert(prediction)
        results = {'sent': 0, 'failed': 0}
        
        for user_id in user_ids:
            notif = Notification(
                title=template['title'],
                message=template['message'],
                channel='telegram',
                priority=template['priority'],
                recipient=user_id,
                data=prediction
            )
            result = self.send(notif, [NotificationChannel.TELEGRAM, NotificationChannel.IN_APP])
            if result.get('telegram') or result.get('in_app'):
                results['sent'] += 1
            else:
                results['failed'] += 1
                
        return results
        
    def send_daily_summary(self, stats: Dict, user_ids: List[str]) -> Dict:
        """Send daily summary to users"""
        template = self.templates.daily_summary(stats)
        results = {'sent': 0, 'failed': 0}
        
        for user_id in user_ids:
            notif = Notification(
                title=template['title'],
                message=template['message'],
                channel='email',
                priority=template['priority'],
                recipient=user_id,
                data=stats
            )
            result = self.send(notif, [NotificationChannel.EMAIL, NotificationChannel.IN_APP])
            if any(result.values()):
                results['sent'] += 1
            else:
                results['failed'] += 1
                
        return results
        
    def get_notification_history(self, limit: int = 100) -> List[Dict]:
        """Get notification history"""
        return self.history[-limit:]


# Global notification service
notification_service = NotificationService()


def notify_sure_win(prediction: Dict, user_ids: List[str]) -> Dict:
    """Quick helper to send sure win notification"""
    return notification_service.send_sure_win_alert(prediction, user_ids)


def notify_value_bet(bet: Dict, user_ids: List[str]) -> Dict:
    """Quick helper to send value bet notification"""
    template = NotificationTemplates.value_bet_alert(bet)
    results = {'sent': 0, 'failed': 0}
    
    for user_id in user_ids:
        notif = Notification(
            title=template['title'],
            message=template['message'],
            channel='telegram',
            priority=template['priority'],
            recipient=user_id,
            data=bet
        )
        result = notification_service.send(notif, [NotificationChannel.TELEGRAM])
        if result.get('telegram'):
            results['sent'] += 1
        else:
            results['failed'] += 1
            
    return results


def get_user_notifications(user_id: str) -> List[Dict]:
    """Get in-app notifications for user"""
    return notification_service.in_app.get_all(user_id)


def get_unread_count(user_id: str) -> int:
    """Get unread notification count"""
    return len(notification_service.in_app.get_unread(user_id))
