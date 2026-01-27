"""
Rate Limiting Middleware
Request rate limiting for API.

Part of the complete blueprint implementation.
"""

from fastapi import Request, HTTPException
from typing import Dict, Optional
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API requests.
    
    Features:
    - Per-IP limiting
    - Per-user limiting
    - Sliding window
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.request_counts: Dict[str, list] = defaultdict(list)
    
    def _clean_old_requests(self, key: str, window_seconds: int):
        """Remove requests outside the window."""
        cutoff = time.time() - window_seconds
        self.request_counts[key] = [
            t for t in self.request_counts[key]
            if t > cutoff
        ]
    
    def check_rate_limit(
        self,
        identifier: str
    ) -> Dict:
        """Check if request is within rate limits."""
        now = time.time()
        
        # Clean old requests
        self._clean_old_requests(identifier, 3600)  # 1 hour window
        
        requests = self.request_counts[identifier]
        
        # Per-minute check
        minute_ago = now - 60
        minute_count = sum(1 for t in requests if t > minute_ago)
        
        if minute_count >= self.requests_per_minute:
            return {
                'allowed': False,
                'reason': 'Rate limit exceeded (per minute)',
                'retry_after': 60,
                'current_rate': minute_count
            }
        
        # Per-hour check
        hour_ago = now - 3600
        hour_count = sum(1 for t in requests if t > hour_ago)
        
        if hour_count >= self.requests_per_hour:
            return {
                'allowed': False,
                'reason': 'Rate limit exceeded (per hour)',
                'retry_after': 3600,
                'current_rate': hour_count
            }
        
        # Record request
        self.request_counts[identifier].append(now)
        
        return {
            'allowed': True,
            'remaining_minute': self.requests_per_minute - minute_count - 1,
            'remaining_hour': self.requests_per_hour - hour_count - 1
        }
    
    def get_identifier(self, request: Request) -> str:
        """Get unique identifier for request."""
        # Use API key if present
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"key:{api_key[:8]}"
        
        # Fall back to IP
        client_ip = request.client.host if request.client else 'unknown'
        return f"ip:{client_ip}"


_limiter: Optional[RateLimiter] = None

def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


async def rate_limit_middleware(request: Request):
    """Middleware to apply rate limiting."""
    limiter = get_limiter()
    identifier = limiter.get_identifier(request)
    
    result = limiter.check_rate_limit(identifier)
    
    if not result['allowed']:
        raise HTTPException(
            status_code=429,
            detail=result['reason'],
            headers={'Retry-After': str(result['retry_after'])}
        )
    
    return result
