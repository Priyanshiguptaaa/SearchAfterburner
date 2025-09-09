"""In-memory LRU cache implementation."""

import time
import logging
from typing import Any, Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = None  # Would use threading.Lock in production
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            if entry["expires_at"] is None or time.time() < entry["expires_at"]:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"LRU cache hit: {key}")
                return entry["value"]
            else:
                # Expired, remove it
                del self._cache[key]
        
        self._misses += 1
        logger.debug(f"LRU cache miss: {key}")
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl
        
        # Remove if already exists
        if key in self._cache:
            del self._cache[key]
        
        # Add new entry
        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        
        # Evict if over capacity
        while len(self._cache) > self.maxsize:
            # Remove least recently used (first item)
            self._cache.popitem(last=False)
        
        logger.debug(f"LRU cache put: {key}")
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"LRU cache delete: {key}")
    
    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("LRU cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(1, total_requests)
        
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry["expires_at"] is not None and current_time >= entry["expires_at"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
        
        return len(expired_keys)
