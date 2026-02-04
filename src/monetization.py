"""
Monetization Module

Handles:
- Subscription tiers and features
- Premium content gating
- Affiliate bookmaker links
- Revenue tracking
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import secrets


@dataclass
class SubscriptionTier:
    """Subscription tier definition"""
    id: str
    name: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    predictions_per_day: int
    leagues: List[str]
    badge_color: str
    emoji: str


class MonetizationManager:
    """
    Manages subscriptions, premium features, and affiliate links.
    """
    
    TIERS = {
        'free': SubscriptionTier(
            id='free',
            name='Free',
            price_monthly=0,
            price_yearly=0,
            features=[
                '10 predictions per day',
                'Bundesliga only',
                'Basic accumulator',
                'Email support'
            ],
            predictions_per_day=10,
            leagues=['bundesliga', 'bundesliga2'],
            badge_color='#666',
            emoji='🆓'
        ),
        'pro': SubscriptionTier(
            id='pro',
            name='Pro',
            price_monthly=9.99,
            price_yearly=99.99,
            features=[
                '50 predictions per day',
                'All leagues',
                'All accumulators',
                'Value bet alerts',
                'Live odds comparison',
                'Email support (24h response)'
            ],
            predictions_per_day=50,
            leagues=['all'],
            badge_color='#6366f1',
            emoji='⭐'
        ),
        'premium': SubscriptionTier(
            id='premium',
            name='Premium',
            price_monthly=24.99,
            price_yearly=249.99,
            features=[
                'Unlimited predictions',
                'All leagues + international',
                'All accumulators',
                'Real-time Telegram/WhatsApp alerts',
                'API access',
                'Private Discord community',
                'Priority support',
                'Early access to new features'
            ],
            predictions_per_day=-1,  # Unlimited
            leagues=['all', 'international'],
            badge_color='#f59e0b',
            emoji='👑'
        )
    }
    
    # Affiliate bookmaker links (replace with real links)
    AFFILIATES = {
        'bet365': {
            'name': 'bet365',
            'url': 'https://www.bet365.com/?affiliate=YOUR_ID',
            'bonus': 'Up to $100 in Bet Credits',
            'logo': '/static/img/bookmakers/bet365.png',
            'rating': 4.8
        },
        'betway': {
            'name': 'Betway',
            'url': 'https://www.betway.com/?affiliate=YOUR_ID',
            'bonus': '$30 Free Bet',
            'logo': '/static/img/bookmakers/betway.png',
            'rating': 4.6
        },
        'unibet': {
            'name': 'Unibet',
            'url': 'https://www.unibet.com/?affiliate=YOUR_ID',
            'bonus': '$40 Money Back',
            'logo': '/static/img/bookmakers/unibet.png',
            'rating': 4.5
        },
        '1xbet': {
            'name': '1xBet',
            'url': 'https://www.1xbet.com/?affiliate=YOUR_ID',
            'bonus': '100% up to $130',
            'logo': '/static/img/bookmakers/1xbet.png',
            'rating': 4.4
        },
        'betfair': {
            'name': 'Betfair',
            'url': 'https://www.betfair.com/?affiliate=YOUR_ID',
            'bonus': 'Up to $20 Free',
            'logo': '/static/img/bookmakers/betfair.png',
            'rating': 4.7
        }
    }
    
    def __init__(self):
        self.active_subscriptions: Dict[str, Dict] = {}
    
    def get_tier(self, tier_id: str) -> Optional[SubscriptionTier]:
        """Get subscription tier by ID"""
        return self.TIERS.get(tier_id)
    
    def get_all_tiers(self) -> Dict[str, Dict]:
        """Get all tiers as dictionaries"""
        return {
            k: asdict(v) for k, v in self.TIERS.items()
        }
    
    def check_feature_access(self, user_tier: str, feature: str) -> bool:
        """Check if user tier has access to feature"""
        tier = self.TIERS.get(user_tier)
        if not tier:
            return False
        
        if 'all' in tier.features:
            return True
        
        return feature in tier.features
    
    def check_league_access(self, user_tier: str, league: str) -> bool:
        """Check if user tier has access to league"""
        tier = self.TIERS.get(user_tier)
        if not tier:
            return False
        
        if 'all' in tier.leagues:
            return True
        
        return league.lower() in tier.leagues
    
    def check_prediction_limit(self, user_tier: str, predictions_today: int) -> bool:
        """Check if user is within prediction limit"""
        tier = self.TIERS.get(user_tier)
        if not tier:
            return False
        
        if tier.predictions_per_day == -1:  # Unlimited
            return True
        
        return predictions_today < tier.predictions_per_day
    
    def get_upgrade_prompt(self, user_tier: str, feature: str) -> Dict:
        """Generate upgrade prompt for locked feature"""
        if user_tier == 'free':
            target_tier = self.TIERS['pro']
            return {
                'locked': True,
                'feature': feature,
                'upgrade_to': 'pro',
                'price': target_tier.price_monthly,
                'message': f"Unlock {feature} with Pro for ${target_tier.price_monthly}/month"
            }
        elif user_tier == 'pro':
            target_tier = self.TIERS['premium']
            return {
                'locked': True,
                'feature': feature,
                'upgrade_to': 'premium',
                'price': target_tier.price_monthly,
                'message': f"Unlock {feature} with Premium for ${target_tier.price_monthly}/month"
            }
        
        return {'locked': False}
    
    def get_affiliates(self) -> List[Dict]:
        """Get all affiliate bookmakers"""
        return list(self.AFFILIATES.values())
    
    def get_affiliate_link(self, bookmaker: str) -> Optional[str]:
        """Get affiliate link for bookmaker"""
        affiliate = self.AFFILIATES.get(bookmaker.lower())
        return affiliate['url'] if affiliate else None
    
    def get_promo_banner(self) -> Dict:
        """Get current promotional banner"""
        return {
            'active': True,
            'title': '🎉 New Year Special!',
            'message': 'Get 50% off your first month of Pro or Premium',
            'code': 'NEWYEAR50',
            'expires': (datetime.now() + timedelta(days=7)).isoformat(),
            'cta_text': 'Claim Offer',
            'cta_url': '/pricing?promo=NEWYEAR50'
        }
    
    def generate_checkout_session(self, user_id: str, tier_id: str, period: str = 'monthly') -> Dict:
        """Generate checkout session (integrate with Stripe/PayPal)"""
        tier = self.TIERS.get(tier_id)
        if not tier:
            return {'error': 'Invalid tier'}
        
        price = tier.price_monthly if period == 'monthly' else tier.price_yearly
        
        # Generate session ID (would integrate with payment provider)
        session_id = secrets.token_urlsafe(32)
        
        return {
            'session_id': session_id,
            'tier': tier_id,
            'tier_name': tier.name,
            'price': price,
            'period': period,
            'currency': 'USD',
            'user_id': user_id,
            # Would include Stripe/PayPal session URL
            'checkout_url': f'/checkout/{session_id}'
        }


# Global instance
monetization = MonetizationManager()


def get_pricing() -> Dict:
    """Get pricing information for display"""
    return {
        'tiers': monetization.get_all_tiers(),
        'promo': monetization.get_promo_banner(),
        'affiliates': monetization.get_affiliates()
    }


def check_access(user_tier: str, feature: str) -> Dict:
    """Check feature access with upgrade prompt if needed"""
    has_access = monetization.check_feature_access(user_tier, feature)
    
    if has_access:
        return {'access': True}
    
    return {
        'access': False,
        **monetization.get_upgrade_prompt(user_tier, feature)
    }
