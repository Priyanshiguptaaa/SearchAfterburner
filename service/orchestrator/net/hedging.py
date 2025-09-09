"""Request hedging utilities for parallel API calls."""

import asyncio
import time
import logging
from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

@dataclass
class RequestHedge:
    """Configuration for hedging a request."""
    url: str
    method: str = "GET"
    params: Optional[Dict] = None
    headers: Optional[Dict] = None
    json: Optional[Dict] = None
    timeout_ms: int = 5000
    priority: int = 1  # Higher number = higher priority

async def hedge_requests(
    hedges: List[RequestHedge],
    session: httpx.AsyncClient,
    max_concurrent: int = 3,
    hedge_delay_ms: int = 100
) -> Dict[str, Any]:
    """
    Execute multiple requests with hedging.
    
    Args:
        hedges: List of RequestHedge configurations
        session: HTTP session to use
        max_concurrent: Maximum concurrent requests
        hedge_delay_ms: Delay between hedged requests
        
    Returns:
        Dict with results from the first successful request
    """
    # Sort by priority (highest first)
    hedges.sort(key=lambda x: x.priority, reverse=True)
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_hedge(hedge: RequestHedge) -> Optional[Dict[str, Any]]:
        """Execute a single hedged request."""
        async with semaphore:
            try:
                # Add delay for hedging effect
                if hedge_delay_ms > 0:
                    await asyncio.sleep(hedge_delay_ms / 1000)
                
                # Prepare request parameters
                kwargs = {}
                if hedge.params:
                    kwargs["params"] = hedge.params
                if hedge.headers:
                    kwargs["headers"] = hedge.headers
                if hedge.json:
                    kwargs["json"] = hedge.json
                
                # Execute request
                start_time = time.time()
                response = await session.request(
                    hedge.method,
                    hedge.url,
                    timeout=hedge.timeout_ms / 1000,
                    **kwargs
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Return successful result
                return {
                    "url": hedge.url,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "success": True
                }
                
            except Exception as e:
                logger.warning(f"Hedged request to {hedge.url} failed: {e}")
                return {
                    "url": hedge.url,
                    "error": str(e),
                    "success": False
                }
    
    # Execute all hedges concurrently
    tasks = [execute_hedge(hedge) for hedge in hedges]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Find first successful result
    for result in results:
        if isinstance(result, dict) and result.get("success"):
            return result
    
    # If no successful results, return error info
    return {
        "success": False,
        "error": "All hedged requests failed",
        "results": [r for r in results if isinstance(r, dict)]
    }

async def hedge_search_requests(
    query: str,
    providers: List[str],
    session: httpx.AsyncClient,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Hedge search requests across multiple providers.
    
    Args:
        query: Search query
        providers: List of provider names
        session: HTTP session
        config: Optional configuration
        
    Returns:
        Dict with search results from first successful provider
    """
    config = config or {}
    
    # Create hedges for different providers
    hedges = []
    
    for i, provider in enumerate(providers):
        if provider == "ddg":
            # DuckDuckGo search
            hedges.append(RequestHedge(
                url="https://api.duckduckgo.com/",
                method="GET",
                params={"q": query, "format": "json", "no_html": "1"},
                priority=len(providers) - i
            ))
        elif provider == "wikipedia":
            # Wikipedia search
            hedges.append(RequestHedge(
                url="https://en.wikipedia.org/w/api.php",
                method="GET",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": "20"
                },
                priority=len(providers) - i
            ))
        elif provider == "mock":
            # Mock provider for testing
            hedges.append(RequestHedge(
                url="http://localhost:8080/mock/search",
                method="POST",
                json={"query": query},
                priority=len(providers) - i
            ))
    
    # Execute hedged requests
    return await hedge_requests(hedges, session)
