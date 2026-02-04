"""
User Management Module

Features:
- User registration/login
- Session management
- User preferences
- Subscription tiers
- Notification settings
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class User:
    """User account"""
    id: str
    email: str
    username: str
    password_hash: str
    created_at: str
    last_login: Optional[str]
    subscription_tier: str  # 'free', 'pro', 'premium'
    preferences: Dict
    telegram_chat_id: Optional[str]
    whatsapp_number: Optional[str]
    notification_settings: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_public_dict(self) -> Dict:
        """Return public info (no password hash)"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'subscription_tier': self.subscription_tier,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'preferences': self.preferences,
            'telegram_connected': bool(self.telegram_chat_id),
            'whatsapp_connected': bool(self.whatsapp_number),
        }


@dataclass
class Session:
    """User session"""
    token: str
    user_id: str
    created_at: str
    expires_at: str
    
    def is_valid(self) -> bool:
        return datetime.now() < datetime.fromisoformat(self.expires_at)


class UserManager:
    """
    Manage user accounts and authentication
    """
    
    SUBSCRIPTION_TIERS = {
        'free': {
            'name': 'Free',
            'predictions_per_day': 10,
            'leagues': ['bundesliga', 'bundesliga2'],
            'features': ['basic_predictions', 'match_alerts']
        },
        'pro': {
            'name': 'Pro',
            'predictions_per_day': 50,
            'leagues': ['all'],
            'features': ['basic_predictions', 'match_alerts', 'value_bets', 'accumulator']
        },
        'premium': {
            'name': 'Premium',
            'predictions_per_day': -1,  # Unlimited
            'leagues': ['all'],
            'features': ['all']
        }
    }
    
    def __init__(self, data_dir: str = "data/users"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self._load_users()
    
    def _load_users(self):
        """Load users from file"""
        filepath = self.data_dir / "users.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for user_data in data:
                        user = User(**user_data)
                        self.users[user.id] = user
            except:
                pass
    
    def _save_users(self):
        """Save users to file"""
        filepath = self.data_dir / "users.json"
        with open(filepath, 'w') as f:
            json.dump([u.to_dict() for u in self.users.values()], f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "football_predictions_2024"
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return secrets.token_hex(8)
    
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def register(
        self,
        email: str,
        username: str,
        password: str,
        tier: str = 'free'
    ) -> Dict:
        """Register new user"""
        # Check if email exists
        for user in self.users.values():
            if user.email.lower() == email.lower():
                return {'success': False, 'error': 'Email already registered'}
            if user.username.lower() == username.lower():
                return {'success': False, 'error': 'Username already taken'}
        
        # Create user
        user = User(
            id=self._generate_user_id(),
            email=email.lower(),
            username=username,
            password_hash=self._hash_password(password),
            created_at=datetime.now().isoformat(),
            last_login=None,
            subscription_tier=tier,
            preferences={
                'default_league': 'bundesliga',
                'dark_mode': True,
                'show_odds': True
            },
            telegram_chat_id=None,
            whatsapp_number=None,
            notification_settings={
                'daily_predictions': True,
                'value_bet_alerts': True,
                'match_reminders': True
            }
        )
        
        self.users[user.id] = user
        self._save_users()
        
        # Create session
        session = self._create_session(user.id)
        
        return {
            'success': True,
            'user': user.to_public_dict(),
            'token': session.token
        }
    
    def login(self, email: str, password: str) -> Dict:
        """Authenticate user"""
        password_hash = self._hash_password(password)
        
        for user in self.users.values():
            if user.email.lower() == email.lower():
                if user.password_hash == password_hash:
                    # Update last login
                    user.last_login = datetime.now().isoformat()
                    self._save_users()
                    
                    # Create session
                    session = self._create_session(user.id)
                    
                    return {
                        'success': True,
                        'user': user.to_public_dict(),
                        'token': session.token
                    }
                else:
                    return {'success': False, 'error': 'Invalid password'}
        
        return {'success': False, 'error': 'User not found'}
    
    def _create_session(self, user_id: str) -> Session:
        """Create new session for user"""
        session = Session(
            token=self._generate_session_token(),
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=7)).isoformat()
        )
        self.sessions[session.token] = session
        return session
    
    def validate_session(self, token: str) -> Optional[User]:
        """Validate session token and return user"""
        if token not in self.sessions:
            return None
        
        session = self.sessions[token]
        if not session.is_valid():
            del self.sessions[token]
            return None
        
        return self.users.get(session.user_id)
    
    def logout(self, token: str) -> bool:
        """Invalidate session"""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False
    
    def update_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences"""
        if user_id not in self.users:
            return False
        
        self.users[user_id].preferences.update(preferences)
        self._save_users()
        return True
    
    def connect_telegram(self, user_id: str, chat_id: str) -> bool:
        """Connect Telegram to user account"""
        if user_id not in self.users:
            return False
        
        self.users[user_id].telegram_chat_id = chat_id
        self._save_users()
        return True
    
    def connect_whatsapp(self, user_id: str, phone_number: str) -> bool:
        """Connect WhatsApp to user account"""
        if user_id not in self.users:
            return False
        
        self.users[user_id].whatsapp_number = phone_number
        self._save_users()
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_all_users(self) -> List[User]:
        """Get all users (admin only)"""
        return list(self.users.values())


# Global instance
user_manager = UserManager()
