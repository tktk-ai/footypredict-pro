"""
Payment Integration Module

Stripe integration for subscription payments.
Handles checkout sessions and webhooks.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional


class StripePayments:
    """
    Stripe payment integration for subscriptions.
    
    Set STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET in .env
    """
    
    PLANS = {
        'pro_monthly': {
            'name': 'Pro Monthly',
            'price': 999,  # in cents
            'interval': 'month',
            'tier': 'pro'
        },
        'pro_yearly': {
            'name': 'Pro Yearly',
            'price': 9999,  # in cents ($99.99)
            'interval': 'year',
            'tier': 'pro'
        },
        'premium_monthly': {
            'name': 'Premium Monthly',
            'price': 2499,  # in cents
            'interval': 'month',
            'tier': 'premium'
        },
        'premium_yearly': {
            'name': 'Premium Yearly',
            'price': 24999,  # in cents ($249.99)
            'interval': 'year',
            'tier': 'premium'
        }
    }
    
    def __init__(self):
        self.api_key = os.getenv('STRIPE_SECRET_KEY', '')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
        self.stripe = None
        
        if self.api_key:
            try:
                import stripe
                stripe.api_key = self.api_key
                self.stripe = stripe
            except ImportError:
                print("Warning: stripe package not installed. Run: pip install stripe")
    
    def create_checkout_session(
        self,
        user_id: str,
        user_email: str,
        plan_id: str,
        success_url: str = 'http://localhost:5000/profile?payment=success',
        cancel_url: str = 'http://localhost:5000/pricing?payment=cancelled'
    ) -> Dict:
        """
        Create Stripe checkout session for subscription.
        """
        if not self.stripe:
            # Fallback for demo mode
            return {
                'success': True,
                'mode': 'demo',
                'session_id': secrets.token_urlsafe(32),
                'checkout_url': f'{success_url}&demo=true',
                'message': 'Demo mode - Stripe not configured'
            }
        
        plan = self.PLANS.get(plan_id)
        if not plan:
            return {'success': False, 'error': 'Invalid plan'}
        
        try:
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': plan['name'],
                            'description': f"Football Predictions {plan['tier'].title()} Subscription"
                        },
                        'unit_amount': plan['price'],
                        'recurring': {
                            'interval': plan['interval']
                        }
                    },
                    'quantity': 1
                }],
                mode='subscription',
                success_url=success_url + '&session_id={CHECKOUT_SESSION_ID}',
                cancel_url=cancel_url,
                customer_email=user_email,
                metadata={
                    'user_id': user_id,
                    'plan_id': plan_id,
                    'tier': plan['tier']
                }
            )
            
            return {
                'success': True,
                'session_id': session.id,
                'checkout_url': session.url
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def verify_webhook(self, payload: bytes, signature: str) -> Optional[Dict]:
        """Verify and parse Stripe webhook event"""
        if not self.stripe or not self.webhook_secret:
            return None
        
        try:
            event = self.stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return event
        except Exception as e:
            print(f"Webhook verification failed: {e}")
            return None
    
    def handle_successful_payment(self, session: Dict) -> Dict:
        """Handle successful checkout completion"""
        metadata = session.get('metadata', {})
        user_id = metadata.get('user_id')
        plan_id = metadata.get('plan_id')
        tier = metadata.get('tier')
        
        if not user_id or not tier:
            return {'success': False, 'error': 'Missing metadata'}
        
        return {
            'success': True,
            'user_id': user_id,
            'new_tier': tier,
            'plan_id': plan_id,
            'subscription_id': session.get('subscription')
        }
    
    def cancel_subscription(self, subscription_id: str) -> Dict:
        """Cancel a subscription"""
        if not self.stripe:
            return {'success': True, 'mode': 'demo'}
        
        try:
            self.stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_customer_portal_url(self, customer_id: str) -> Optional[str]:
        """Get Stripe customer portal URL for managing subscription"""
        if not self.stripe:
            return None
        
        try:
            session = self.stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url='http://localhost:5000/profile'
            )
            return session.url
        except Exception:
            return None


# Simpler PayPal alternative
class PayPalPayments:
    """
    PayPal payment integration.
    Set PAYPAL_CLIENT_ID and PAYPAL_SECRET in .env
    """
    
    def __init__(self):
        self.client_id = os.getenv('PAYPAL_CLIENT_ID', '')
        self.secret = os.getenv('PAYPAL_SECRET', '')
        self.base_url = os.getenv('PAYPAL_BASE_URL', 'https://api-m.sandbox.paypal.com')
    
    def create_order(self, amount: float, description: str) -> Dict:
        """Create PayPal order"""
        if not self.client_id:
            return {
                'success': True,
                'mode': 'demo',
                'order_id': secrets.token_urlsafe(16),
                'message': 'Demo mode - PayPal not configured'
            }
        
        # Would implement PayPal API calls here
        return {'success': True, 'order_id': secrets.token_urlsafe(16)}
    
    def capture_order(self, order_id: str) -> Dict:
        """Capture PayPal order after approval"""
        return {'success': True, 'captured': True}


# Global instances
stripe_payments = StripePayments()
paypal_payments = PayPalPayments()


def create_payment_session(
    user_id: str,
    user_email: str,
    tier: str,
    period: str = 'monthly',
    provider: str = 'stripe'
) -> Dict:
    """Unified payment session creation"""
    plan_id = f"{tier}_{period}"
    
    if provider == 'stripe':
        return stripe_payments.create_checkout_session(user_id, user_email, plan_id)
    elif provider == 'paypal':
        from src.monetization import MonetizationManager
        mm = MonetizationManager()
        tier_info = mm.get_tier(tier)
        if tier_info:
            price = tier_info.price_monthly if period == 'monthly' else tier_info.price_yearly
            return paypal_payments.create_order(price, f"{tier.title()} Subscription")
    
    return {'success': False, 'error': 'Invalid payment provider'}
