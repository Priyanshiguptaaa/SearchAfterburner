"""Cache manager for multi-tier caching system."""

import hashlib
import time
import logging
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

from .memory import LRUCache
from .disk import DiskCache
from .serializer import CacheSerializer

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache system."""
    memory_size: int = 1000  # Number of items in memory cache
    disk_size_mb: int = 100  # Disk cache size in MB
    disk_path: str = "cache"
    ttl_seconds: int = 3600  # Time to live in seconds
    enable_compression: bool = True
    enable_encryption: bool = False

class CacheManager:
    """Multi-tier cache manager with memory and disk tiers."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.serializer = CacheSerializer(
            enable_compression=self.config.enable_compression,
            enable_encryption=self.config.enable_encryption
        )
        
        # Initialize cache tiers
        self.memory_cache = LRUCache(maxsize=self.config.memory_size)
        self.disk_cache = DiskCache(
            cache_dir=Path(self.config.disk_path),
            max_size_mb=self.config.disk_size_mb,
            serializer=self.serializer
        )
        
        logger.info(f"Cache manager initialized: memory={self.config.memory_size}, disk={self.config.disk_size_mb}MB")
    
    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prefix: str, *args) -> Optional[Any]:
        """Get value from cache (tries memory first, then disk)."""
        key = self._make_key(prefix, *args)
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit (memory): {prefix}")
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit (disk): {prefix}")
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value
        
        logger.debug(f"Cache miss: {prefix}")
        return None
    
    def put(self, prefix: str, value: Any, *args, ttl: Optional[int] = None) -> None:
        """Put value in cache (both memory and disk)."""
        key = self._make_key(prefix, *args)
        ttl = ttl or self.config.ttl_seconds
        
        # Add to memory cache
        self.memory_cache.put(key, value, ttl=ttl)
        
        # Add to disk cache
        self.disk_cache.put(key, value, ttl=ttl)
        
        logger.debug(f"Cached: {prefix}")
    
    def invalidate(self, prefix: str, *args) -> None:
        """Invalidate cache entry."""
        key = self._make_key(prefix, *args)
        
        # Remove from both caches
        self.memory_cache.delete(key)
        self.disk_cache.delete(key)
        
        logger.debug(f"Invalidated: {prefix}")
    
    def invalidate_pattern(self, prefix: str) -> None:
        """Invalidate all entries with given prefix."""
        # Memory cache
        keys_to_remove = [k for k in self.memory_cache._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            self.memory_cache.delete(key)
        
        # Disk cache
        self.disk_cache.invalidate_pattern(prefix)
        
        logger.debug(f"Invalidated pattern: {prefix}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        logger.info("All caches cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = self.memory_cache.stats()
        disk_stats = self.disk_cache.stats()
        
        return {
            "memory": memory_stats,
            "disk": disk_stats,
            "total_hits": memory_stats["hits"] + disk_stats["hits"],
            "total_misses": memory_stats["misses"] + disk_stats["misses"],
            "hit_rate": (memory_stats["hits"] + disk_stats["hits"]) / 
                       max(1, memory_stats["hits"] + disk_stats["hits"] + memory_stats["misses"] + disk_stats["misses"])
        }
    
    def cache_search_results(self, query: str, providers: List[str], results: Dict[str, Any]) -> None:
        """Cache search results."""
        self.put("search", results, query, *providers)
    
    def get_search_results(self, query: str, providers: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached search results."""
        return self.get("search", query, *providers)
    
    def cache_embeddings(self, text: str, embeddings: Any) -> None:
        """Cache embeddings."""
        self.put("embeddings", embeddings, text)
    
    def get_embeddings(self, text: str) -> Optional[Any]:
        """Get cached embeddings."""
        return self.get("embeddings", text)
    
    def cache_rerank_results(self, query: str, provider: str, results: Any) -> None:
        """Cache reranking results."""
        self.put("rerank", results, query, provider)
    
    def get_rerank_results(self, query: str, provider: str) -> Optional[Any]:
        """Get cached reranking results."""
        return self.get("rerank", query, provider)
