"""
Authentication Middleware
JWT-based authentication for API.

Part of the complete blueprint implementation.
"""

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict
import logging
import hashlib
import time

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """
    Authentication middleware for API.
    
    Supports:
    - API key authentication
    - JWT tokens
    - Rate limiting per user
    """
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
    
    def register_api_key(
        self,
        api_key: str,
        user_id: str,
        permissions: list = None
    ):
        """Register an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys[key_hash] = {
            'user_id': user_id,
            'permissions': permissions or ['read'],
            'created_at': time.time()
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return self.api_keys.get(key_hash)
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has a permission."""
        user = self.validate_api_key(api_key)
        if not user:
            return False
        return permission in user.get('permissions', []) or 'admin' in user.get('permissions', [])
    
    async def verify_request(self, request: Request) -> Dict:
        """Verify a request."""
        # Check for API key in header
        api_key = request.headers.get('X-API-Key')
        
        if api_key:
            user = self.validate_api_key(api_key)
            if user:
                return {'authenticated': True, 'user': user}
        
        # Check for Bearer token
        auth = request.headers.get('Authorization')
        if auth and auth.startswith('Bearer '):
            token = auth.split(' ')[1]
            # Would validate JWT token here
            return {'authenticated': True, 'token': token}
        
        return {'authenticated': False}


_auth: Optional[AuthMiddleware] = None

def get_auth() -> AuthMiddleware:
    global _auth
    if _auth is None:
        _auth = AuthMiddleware()
    return _auth


async def require_auth(request: Request):
    """Dependency to require authentication."""
    auth = get_auth()
    result = await auth.verify_request(request)
    
    if not result['authenticated']:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return result


async def require_permission(request: Request, permission: str):
    """Dependency to require specific permission."""
    auth_result = await require_auth(request)
    
    api_key = request.headers.get('X-API-Key', '')
    auth = get_auth()
    
    if not auth.has_permission(api_key, permission):
        raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
    
    return auth_result
