"""
Cloudflare API Client

Client for connecting to the FootyPredict Cloudflare Worker API.
Provides prediction caching and fallback to local models.
"""

import requests
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CloudflareAPIClient:
    """Client for the Cloudflare Worker prediction API."""
    
    def __init__(self, base_url: str = None):
        """
        Initialize the Cloudflare API client.
        
        Args:
            base_url: Worker URL. Defaults to env var or hardcoded URL.
        """
        self.base_url = base_url or os.getenv(
            'CLOUDFLARE_API_URL',
            'https://footypredict-api.tirene857.workers.dev'
        )
        self.timeout = 10  # seconds
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    def predict(
        self, 
        home_team: str, 
        away_team: str, 
        league: str = None
    ) -> Dict[str, Any]:
        """
        Get prediction from Cloudflare Worker API.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: Optional league name
            
        Returns:
            Prediction response dict
        """
        # Check cache
        cache_key = f"{home_team}_{away_team}_{league}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={
                    "home_team": home_team,
                    "away_team": away_team,
                    "league": league
                },
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                data['source'] = 'cloudflare'
                self._set_cached(cache_key, data)
                return data
            else:
                logger.warning(f"Cloudflare API error: {response.status_code}")
                return self._error_response("API error", response.status_code)
                
        except requests.Timeout:
            logger.warning("Cloudflare API timeout")
            return self._error_response("Timeout", 504)
        except requests.RequestException as e:
            logger.error(f"Cloudflare API request failed: {e}")
            return self._error_response(str(e), 500)
    
    def batch_predict(self, matches: list) -> Dict[str, Any]:
        """
        Get batch predictions from Cloudflare Worker API.
        
        Args:
            matches: List of match dicts with home_team, away_team, league
            
        Returns:
            Batch prediction response
        """
        try:
            response = requests.post(
                f"{self.base_url}/batch",
                json={"matches": matches},
                timeout=self.timeout * 2,  # Double timeout for batch
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                data['source'] = 'cloudflare'
                return data
            else:
                return self._error_response("Batch API error", response.status_code)
                
        except requests.RequestException as e:
            logger.error(f"Cloudflare batch API failed: {e}")
            return self._error_response(str(e), 500)
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the Cloudflare API is healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return {
                'healthy': response.status_code == 200,
                'status_code': response.status_code,
                'url': self.base_url
            }
        except requests.RequestException as e:
            return {
                'healthy': False,
                'error': str(e),
                'url': self.base_url
            }
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about deployed models."""
        try:
            response = requests.get(
                f"{self.base_url}/models/info",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except requests.RequestException:
            return {}
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached prediction if still valid."""
        if key in self._cache:
            cached_time, data = self._cache[key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return data
        return None
    
    def _set_cached(self, key: str, data: Dict):
        """Cache a prediction response."""
        self._cache[key] = (datetime.now(), data)
        
        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][0]
            )
            for old_key in sorted_keys[:20]:
                del self._cache[old_key]
    
    def _error_response(self, error: str, code: int) -> Dict[str, Any]:
        """Create error response dict."""
        return {
            'success': False,
            'error': error,
            'status_code': code,
            'source': 'cloudflare',
            'timestamp': datetime.now().isoformat()
        }


# Global client instance
_cloudflare_client = None


def get_cloudflare_client() -> CloudflareAPIClient:
    """Get or create the global Cloudflare API client."""
    global _cloudflare_client
    if _cloudflare_client is None:
        _cloudflare_client = CloudflareAPIClient()
    return _cloudflare_client


def cloudflare_predict(
    home_team: str,
    away_team: str,
    league: str = None
) -> Dict[str, Any]:
    """
    Convenience function to get prediction from Cloudflare.
    
    Args:
        home_team: Home team name
        away_team: Away team name  
        league: Optional league name
        
    Returns:
        Prediction dict
    """
    client = get_cloudflare_client()
    return client.predict(home_team, away_team, league)
