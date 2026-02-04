"""
Enhanced Caching System for Predictions

Provides intelligent caching with:
- Redis-compatible in-memory cache
- TTL-based expiration
- Automatic cache invalidation
- Performance metrics
"""

import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
from functools import wraps
import threading


class PredictionCache:
    """High-performance prediction cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from args"""
        key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry['expires'] > time.time():
                    self.hits += 1
                    return entry['value']
                else:
                    del self._cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expires': time.time() + (ttl or self.default_ttl),
                'created': time.time()
            }
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if v['expires'] <= now]
            for key in expired:
                del self._cache[key]
            return len(expired)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            return {
                'entries': len(self._cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'memory_usage': self._estimate_memory()
            }
    
    def _estimate_memory(self) -> str:
        """Estimate memory usage"""
        try:
            import sys
            size = sys.getsizeof(self._cache)
            for v in self._cache.values():
                size += sys.getsizeof(v)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        except:
            return "Unknown"


# Global cache instances
prediction_cache = PredictionCache(default_ttl=300)  # 5 min for predictions
fixtures_cache = PredictionCache(default_ttl=600)    # 10 min for fixtures
odds_cache = PredictionCache(default_ttl=60)         # 1 min for odds


def cache_prediction(ttl: int = 300):
    """Decorator to cache prediction results"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = prediction_cache._generate_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached = prediction_cache.get(key)
            if cached is not None:
                cached['from_cache'] = True
                return cached
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            if result:
                prediction_cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cache_fixtures(ttl: int = 600):
    """Decorator to cache fixture results"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = fixtures_cache._generate_key(func.__name__, *args, **kwargs)
            cached = fixtures_cache.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if result:
                fixtures_cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


def invalidate_prediction_cache(home: str = None, away: str = None, league: str = None):
    """Invalidate cache entries matching criteria"""
    # For simplicity, clear all if specific criteria
    if home or away or league:
        prediction_cache.clear()
    return True


class RealTimeUpdater:
    """Real-time update manager for live data"""
    
    def __init__(self):
        self.subscribers: Dict[str, list] = {}
        self.last_updates: Dict[str, float] = {}
        
    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to a channel"""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(callback)
        
    def publish(self, channel: str, data: Any):
        """Publish data to channel subscribers"""
        self.last_updates[channel] = time.time()
        if channel in self.subscribers:
            for callback in self.subscribers[channel]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Subscriber error: {e}")
                    
    def get_channels(self) -> list:
        """Get list of active channels"""
        return list(self.subscribers.keys())


# Global real-time updater
realtime = RealTimeUpdater()


def get_cache_stats() -> Dict:
    """Get combined cache statistics"""
    return {
        'predictions': prediction_cache.get_stats(),
        'fixtures': fixtures_cache.get_stats(),
        'odds': odds_cache.get_stats(),
        'total_entries': (
            len(prediction_cache._cache) + 
            len(fixtures_cache._cache) + 
            len(odds_cache._cache)
        )
    }
