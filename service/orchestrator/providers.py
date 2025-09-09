"""Search providers for the evaluation system."""

import httpx
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging
from .net import HedgedSession, NetworkConfig, hedge_search_requests

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str

class BaseProvider:
    """Base class for search providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Search for results. Must be implemented by subclasses."""
        raise NotImplementedError

class DuckDuckGoProvider(BaseProvider):
    """DuckDuckGo search provider using duckduckgo-search package."""
    
    def __init__(self):
        super().__init__("ddg")
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
        except ImportError:
            logger.warning("duckduckgo-search not available, using fallback")
            self.ddgs = None
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Search using DuckDuckGo."""
        start_time = time.time()
        
        try:
            if self.ddgs:
                results = self.ddgs.text(query, max_results=max_results)
                search_results = []
                for result in results:
                    search_results.append(SearchResult(
                        title=result.get('title', ''),
                        url=result.get('href', ''),
                        snippet=result.get('body', ''),
                        provider=self.name
                    ))
                return search_results
            else:
                # Fallback to simple web search simulation
                return self._fallback_search(query, max_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return self._fallback_search(query, max_results)
        finally:
            search_time = (time.time() - start_time) * 1000
            logger.info(f"DuckDuckGo search took {search_time:.2f}ms")
    
    def _fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search with mock results."""
        logger.warning("Using fallback search results")
        return [
            SearchResult(
                title=f"Mock Result {i+1} for '{query}'",
                url=f"https://example.com/result{i+1}",
                snippet=f"This is a mock search result {i+1} for the query '{query}'. It contains relevant information that would be found in a real search.",
                provider=self.name
            )
            for i in range(min(max_results, 10))
        ]

class ExaProvider(BaseProvider):
    """Exa API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("exa")
        self.api_key = api_key
        self.base_url = "https://api.exa.ai"
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Search using Exa API."""
        if not self.api_key:
            logger.warning("No Exa API key provided, using fallback")
            return self._fallback_search(query, max_results)
        
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "numResults": min(max_results, 50),
                "type": "search",
                "useAutoprompt": True
            }
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/search",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                search_results = []
                
                for result in data.get('results', []):
                    search_results.append(SearchResult(
                        title=result.get('title', ''),
                        url=result.get('url', ''),
                        snippet=result.get('text', ''),
                        provider=self.name
                    ))
                
                return search_results
                
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return self._fallback_search(query, max_results)
        finally:
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Exa search took {search_time:.2f}ms")
    
    def _fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search with mock results."""
        logger.warning("Using fallback search results for Exa")
        return [
            SearchResult(
                title=f"Exa Mock Result {i+1} for '{query}'",
                url=f"https://exa-mock.com/result{i+1}",
                snippet=f"This is a mock Exa search result {i+1} for the query '{query}'. It contains high-quality information that would be found through Exa's search.",
                provider=self.name
            )
            for i in range(min(max_results, 10))
        ]

class WikipediaProvider(BaseProvider):
    """Wikipedia API provider (no API key required)."""
    def __init__(self):
        super().__init__("wikipedia")
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        try:
            import httpx
            import time
            
            # Wikipedia search API
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": min(max_results, 50),
                "srprop": "snippet|timestamp"
            }
            
            response = httpx.get(self.base_url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "query" in data and "search" in data["query"]:
                for item in data["query"]["search"]:
                    # Get page content for snippet
                    page_id = item["pageid"]
                    content_params = {
                        "action": "query",
                        "format": "json",
                        "pageids": page_id,
                        "prop": "extracts",
                        "exintro": True,
                        "exlimit": 1
                    }
                    
                    content_response = httpx.get(self.base_url, params=content_params, timeout=5.0)
                    content_data = content_response.json()
                    
                    snippet = item.get("snippet", "")
                    if "extract" in content_data.get("query", {}).get("pages", {}).get(str(page_id), {}):
                        snippet = content_data["query"]["pages"][str(page_id)]["extract"][:500]
                    
                    results.append(SearchResult(
                        title=item["title"],
                        url=f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                        snippet=snippet,
                        provider=self.name
                    ))
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return [
                SearchResult(
                    title=f"Wikipedia Mock Result {i+1} for '{query}'",
                    url=f"https://wikipedia-mock.com/result{i+1}",
                    snippet=f"This is a mock Wikipedia result {i+1} for the query '{query}'. It contains high-quality information that would be found through Wikipedia search.",
                    provider=self.name
                )
                for i in range(min(max_results, 5))
            ]

class GoogleProvider(BaseProvider):
    """Google Custom Search API provider."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        super().__init__("google")
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        if not self.api_key or not self.search_engine_id:
            logger.warning("No Google API key or search engine ID provided, using fallback")
            return self._fallback_search(query, max_results)
        
        start_time = time.time()
        
        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(max_results, 10)  # Google API limit
            }
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                search_results = []
                
                for item in data.get('items', []):
                    search_results.append(SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        provider=self.name
                    ))
                
                return search_results
                
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return self._fallback_search(query, max_results)
        finally:
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Google search took {search_time:.2f}ms")
    
    def _fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search with mock results."""
        logger.warning("Using fallback search results for Google")
        return [
            SearchResult(
                title=f"Google Mock Result {i+1} for '{query}'",
                url=f"https://google-mock.com/result{i+1}",
                snippet=f"This is a mock Google search result {i+1} for the query '{query}'. It contains information that would be found through Google's search.",
                provider=self.name
            )
            for i in range(min(max_results, 10))
        ]

class BaselineProvider(BaseProvider):
    """Baseline provider for comparison."""
    
    def __init__(self):
        super().__init__("baseline")
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Baseline search with simple results."""
        start_time = time.time()
        
        # Simulate some processing time
        time.sleep(0.1)
        
        search_results = [
            SearchResult(
                title=f"Baseline Result {i+1} for '{query}'",
                url=f"https://baseline.com/result{i+1}",
                snippet=f"This is a baseline search result {i+1} for the query '{query}'. It represents a simple search approach.",
                provider=self.name
            )
            for i in range(min(max_results, 15))
        ]
        
        search_time = (time.time() - start_time) * 1000
        logger.info(f"Baseline search took {search_time:.2f}ms")
        
        return search_results

def get_provider(name: str, **kwargs) -> BaseProvider:
    """Factory function to get a provider by name."""
    if name == "ddg":
        return DuckDuckGoProvider()
    elif name == "wikipedia":
        return WikipediaProvider()
    elif name == "exa":
        return ExaProvider(api_key=kwargs.get('api_key'))
    elif name == "google":
        return GoogleProvider(
            api_key=kwargs.get('api_key'),
            search_engine_id=kwargs.get('search_engine_id')
        )
    elif name == "baseline":
        return BaselineProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")

def search_multiple_providers(
    providers: List[str], 
    query: str, 
    max_results: int = 50,
    use_hedging: bool = False,
    **kwargs
) -> Dict[str, List[SearchResult]]:
    """Search using multiple providers and return results grouped by provider."""
    if use_hedging:
        return asyncio.run(_hedged_search_multiple_providers(providers, query, max_results, **kwargs))
    else:
        return _sync_search_multiple_providers(providers, query, max_results, **kwargs)

def _sync_search_multiple_providers(
    providers: List[str], 
    query: str, 
    max_results: int = 50,
    **kwargs
) -> Dict[str, List[SearchResult]]:
    """Synchronous search using multiple providers."""
    results = {}
    
    for provider_name in providers:
        try:
            provider = get_provider(provider_name, **kwargs)
            results[provider_name] = provider.search(query, max_results)
        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            results[provider_name] = []
    
    return results

async def _hedged_search_multiple_providers(
    providers: List[str], 
    query: str, 
    max_results: int = 50,
    **kwargs
) -> Dict[str, List[SearchResult]]:
    """Asynchronous hedged search using multiple providers."""
    results = {}
    
    # Create hedged session
    config = NetworkConfig(
        timeout_ms=5000,
        hedge_delay_ms=100,
        max_retries=2
    )
    
    async with HedgedSession(config) as session:
        # Try hedged approach first
        try:
            hedged_result = await hedge_search_requests(query, providers, session)
            if hedged_result.get("success"):
                # Process hedged result
                provider_name = hedged_result.get("url", "unknown")
                if "duckduckgo" in provider_name:
                    provider_name = "ddg"
                elif "wikipedia" in provider_name:
                    provider_name = "wikipedia"
                
                # Convert to SearchResult format
                data = hedged_result.get("data", {})
                if isinstance(data, dict) and "results" in data:
                    results[provider_name] = [
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", ""),
                            provider=provider_name
                        )
                        for item in data["results"][:max_results]
                    ]
                else:
                    # Fallback to individual provider searches
                    results = await _async_search_providers(providers, query, max_results, **kwargs)
            else:
                # Fallback to individual provider searches
                results = await _async_search_providers(providers, query, max_results, **kwargs)
        except Exception as e:
            logger.warning(f"Hedged search failed: {e}, falling back to individual providers")
            results = await _async_search_providers(providers, query, max_results, **kwargs)
    
    return results

async def _async_search_providers(
    providers: List[str], 
    query: str, 
    max_results: int = 50,
    **kwargs
) -> Dict[str, List[SearchResult]]:
    """Search individual providers asynchronously."""
    results = {}
    
    async def search_provider(provider_name: str):
        try:
            provider = get_provider(provider_name, **kwargs)
            # Run sync search in thread pool
            loop = asyncio.get_event_loop()
            return provider_name, await loop.run_in_executor(None, provider.search, query, max_results)
        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            return provider_name, []
    
    # Run all providers concurrently
    tasks = [search_provider(name) for name in providers]
    provider_results = await asyncio.gather(*tasks)
    
    for provider_name, search_results in provider_results:
        results[provider_name] = search_results
    
    return results
