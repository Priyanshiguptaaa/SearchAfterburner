"""Disk-based cache implementation."""

import os
import time
import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List
import shutil

logger = logging.getLogger(__name__)

class DiskCache:
    """Disk-based cache with size limits and TTL support."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 100, serializer=None):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.serializer = serializer
        self._hits = 0
        self._misses = 0
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Disk cache initialized: {cache_dir}, max_size={max_size_mb}MB")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "entries": {},  # key -> {size, created_at, expires_at, access_count}
            "total_size": 0
        }
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            self._misses += 1
            logger.debug(f"Disk cache miss: {key}")
            return None
        
        # Check metadata
        if key not in self.metadata["entries"]:
            # Clean up orphaned file
            file_path.unlink(missing_ok=True)
            self._misses += 1
            return None
        
        entry = self.metadata["entries"][key]
        
        # Check TTL
        if entry["expires_at"] is not None and time.time() >= entry["expires_at"]:
            self.delete(key)
            self._misses += 1
            return None
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Deserialize
            if self.serializer:
                value = self.serializer.deserialize(data)
            else:
                value = json.loads(data.decode())
            
            # Update access count and timestamp
            entry["access_count"] += 1
            entry["last_accessed"] = time.time()
            self._save_metadata()
            
            self._hits += 1
            logger.debug(f"Disk cache hit: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Failed to read cache file {key}: {e}")
            self.delete(key)
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in disk cache."""
        try:
            # Serialize value
            if self.serializer:
                data = self.serializer.serialize(value)
            else:
                data = json.dumps(value).encode()
            
            # Check size limit
            if len(data) > self.max_size_bytes:
                logger.warning(f"Cache entry too large: {len(data)} bytes")
                return
            
            # Ensure we have space
            self._ensure_space(len(data))
            
            # Write file
            file_path = self._get_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Update metadata
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            self.metadata["entries"][key] = {
                "size": len(data),
                "created_at": time.time(),
                "expires_at": expires_at,
                "last_accessed": time.time(),
                "access_count": 0
            }
            self.metadata["total_size"] += len(data)
            self._save_metadata()
            
            logger.debug(f"Disk cache put: {key}")
            
        except Exception as e:
            logger.error(f"Failed to write cache file {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete key from disk cache."""
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            self.metadata["total_size"] -= entry["size"]
            del self.metadata["entries"][key]
            
            # Remove file
            file_path = self._get_file_path(key)
            file_path.unlink(missing_ok=True)
            
            self._save_metadata()
            logger.debug(f"Disk cache delete: {key}")
    
    def invalidate_pattern(self, prefix: str) -> None:
        """Invalidate all entries with given prefix."""
        keys_to_remove = [k for k in self.metadata["entries"].keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            self.delete(key)
        
        logger.debug(f"Invalidated pattern: {prefix}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Remove all files
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset metadata
        self.metadata = {
            "entries": {},
            "total_size": 0
        }
        self._save_metadata()
        
        self._hits = 0
        self._misses = 0
        logger.debug("Disk cache cleared")
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure there's enough space for new entry."""
        while (self.metadata["total_size"] + required_bytes > self.max_size_bytes and 
               self.metadata["entries"]):
            
            # Find least recently used entry
            lru_key = min(
                self.metadata["entries"].keys(),
                key=lambda k: self.metadata["entries"][k]["last_accessed"]
            )
            
            self.delete(lru_key)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(1, total_requests)
        
        return {
            "size": len(self.metadata["entries"]),
            "total_size_bytes": self.metadata["total_size"],
            "total_size_mb": self.metadata["total_size"] / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.metadata["entries"].items():
            if entry["expires_at"] is not None and current_time >= entry["expires_at"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
        
        return len(expired_keys)
