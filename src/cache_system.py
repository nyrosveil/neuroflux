"""
🧠 NeuroFlux Advanced Caching System
Intelligent caching for expensive operations with TTL and memory management.

Built with love by Nyros Veil 🚀

Features:
- TTL-based cache invalidation
- Memory usage monitoring
- LRU eviction policy
- Thread-safe operations
- Cache statistics and monitoring
"""

import time
import threading
import hashlib
import asyncio
from typing import Dict, Any, Optional, Callable, Tuple
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import os

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0

    def update_hit_rate(self):
        """Calculate and update hit rate."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0

class NeuroFluxCache:
    """
    Advanced caching system with TTL, LRU eviction, and memory management.
    """

    def __init__(self, max_memory_mb: int = 100, default_ttl: int = 300):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()

        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory usage of an object."""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return 1024  # Default estimate

    def _evict_lru(self):
        """Evict least recently used entries when memory limit is reached."""
        if not self.cache:
            return

        # Sort by last accessed time (oldest first)
        entries = sorted(self.cache.items(), key=lambda x: x[1].last_accessed)

        evicted_count = 0
        for key, entry in entries:
            if self.stats.memory_usage_bytes <= self.max_memory_bytes * 0.8:  # Keep 80% usage
                break

            del self.cache[key]
            self.stats.memory_usage_bytes -= entry.size_bytes
            self.stats.evictions += 1
            evicted_count += 1

        if evicted_count > 0:
            print(f"🧹 Cache eviction: removed {evicted_count} entries")

    def _background_cleanup(self):
        """Background thread for periodic cleanup."""
        while True:
            time.sleep(self.cleanup_interval)
            with self.lock:
                self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
                self.stats.memory_usage_bytes -= entry.size_bytes

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            print(f"🧹 Cache cleanup: removed {len(expired_keys)} expired entries")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            entry = self.cache.get(key)
            if entry and not entry.is_expired():
                entry.touch()
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return entry.data
            elif entry and entry.is_expired():
                # Clean up expired entry
                self.stats.memory_usage_bytes -= entry.size_bytes
                del self.cache[key]

            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self.lock:
            if ttl is None:
                ttl = self.default_ttl

            size = self._estimate_size(value)
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size
            )

            # Check memory limit
            if self.stats.memory_usage_bytes + size > self.max_memory_bytes:
                self._evict_lru()

            self.cache[key] = entry
            self.stats.memory_usage_bytes += size
            self.stats.total_entries = len(self.cache)

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.memory_usage_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.total_entries = len(self.cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'entries': self.stats.total_entries,
                'memory_usage_mb': self.stats.memory_usage_bytes / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': self.stats.hit_rate,
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'memory_usage_percent': (self.stats.memory_usage_bytes / self.max_memory_bytes) * 100
            }

def cached(cache_instance: NeuroFluxCache, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = cache_instance._generate_key(func.__name__, *args, **kwargs)
            cached_result = cache_instance.get(key)

            if cached_result is not None:
                return cached_result

            result = await func(*args, **kwargs)
            cache_instance.set(key, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = cache_instance._generate_key(func.__name__, *args, **kwargs)
            cached_result = cache_instance.get(key)

            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache_instance.set(key, result, ttl)
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Global cache instance
neuroflux_cache = NeuroFluxCache(max_memory_mb=50, default_ttl=300)  # 50MB, 5min default TTL