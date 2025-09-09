"""Hedged HTTP session with timeout and retry logic."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import httpx
import random

logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    """Configuration for network hedging."""
    timeout_ms: int = 5000
    hedge_delay_ms: int = 100
    max_retries: int = 3
    backoff_factor: float = 1.5
    jitter_ms: int = 50
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_ms: int = 30000

class HedgedSession:
    """HTTP session with hedging and circuit breaker logic."""
    
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout_ms=self.config.circuit_breaker_timeout_ms
        )
        self.session = httpx.AsyncClient(timeout=self.config.timeout_ms / 1000)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.aclose()
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Hedged GET request."""
        return await self._hedged_request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Hedged POST request."""
        return await self._hedged_request("POST", url, **kwargs)
    
    async def _hedged_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute request with hedging logic."""
        if self.circuit_breaker.is_open():
            raise httpx.RequestError("Circuit breaker is open")
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Add jitter to prevent thundering herd
                if attempt > 0:
                    jitter = random.uniform(0, self.config.jitter_ms) / 1000
                    await asyncio.sleep(jitter)
                
                # Execute request
                response = await self.session.request(method, url, **kwargs)
                
                # Success - update circuit breaker
                self.circuit_breaker.record_success()
                return response
                
            except Exception as e:
                last_error = e
                self.circuit_breaker.record_failure()
                
                # Calculate backoff delay
                if attempt < self.config.max_retries:
                    delay = (self.config.backoff_factor ** attempt) * 0.1
                    await asyncio.sleep(delay)
                
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
        
        # All attempts failed
        raise last_error or httpx.RequestError("All request attempts failed")

class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, threshold: int, timeout_ms: int):
        self.threshold = threshold
        self.timeout_ms = timeout_ms
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "OPEN":
            if time.time() * 1000 - self.last_failure_time > self.timeout_ms:
                self.state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time() * 1000
        
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
