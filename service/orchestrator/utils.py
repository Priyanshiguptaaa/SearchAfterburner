"""Utility functions for the evaluation system."""

import time
import hashlib
import json
import logging
import random
import numpy as np
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from providers import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class TimingStats:
    """Timing statistics for performance tracking."""
    search_ms: float = 0.0
    embed_ms: float = 0.0
    rerank_ms: float = 0.0
    judge_ms: float = 0.0
    total_ms: float = 0.0

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = (time.time() - self.start_time) * 1000
        logger.info(f"{self.name} took {self.duration:.2f}ms")

def hash_text(text: str) -> str:
    """Generate a hash for text deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def deduplicate_results(results: List[SearchResult], max_results: int = 50) -> List[SearchResult]:
    """Remove duplicate results based on URL and title similarity."""
    if not results:
        return results
    
    seen_urls = set()
    seen_titles = set()
    deduplicated = []
    
    for result in results:
        if len(deduplicated) >= max_results:
            break
            
        # Check URL uniqueness
        if result.url in seen_urls:
            continue
        
        # Check title similarity (simple approach)
        title_lower = result.title.lower()
        if any(title_lower in seen_title or seen_title in title_lower 
               for seen_title in seen_titles):
            continue
        
        seen_urls.add(result.url)
        seen_titles.add(title_lower)
        deduplicated.append(result)
    
    logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
    return deduplicated

def format_results_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format results as a table for console output."""
    from tabulate import tabulate
    
    headers = ["Provider", "Rel@5", "Coverage", "Search_ms", "Embed_ms", "Rerank_ms", "Judge_ms", "Total_ms"]
    rows = []
    
    for provider, stats in results.items():
        if isinstance(stats, dict) and 'timing' in stats:
            timing = stats['timing']
            evaluation = stats.get('evaluation', {})
            
            rel_at_5 = evaluation.get('A', 0.0) if isinstance(evaluation, dict) else 0.0
            coverage = evaluation.get('B', 0.0) if isinstance(evaluation, dict) else 0.0
            
            row = [
                provider,
                f"{rel_at_5:.2f}",
                f"{coverage:.2f}",
                f"{timing.search_ms:.0f}",
                f"{timing.embed_ms:.0f}",
                f"{timing.rerank_ms:.1f}",
                f"{timing.judge_ms:.0f}",
                f"{timing.total_ms:.0f}"
            ]
            rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def save_trace(trace_data: Dict[str, Any], filename: str = "trace.json"):
    """Save trace data to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)
        logger.info(f"Trace saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save trace: {e}")

def load_trace(filename: str = "trace.json") -> Optional[Dict[str, Any]]:
    """Load trace data from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load trace: {e}")
        return None

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def clean_text(text: str) -> str:
    """Clean text for processing."""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    
    return text.strip()

def validate_query(query: str) -> bool:
    """Validate that a query is reasonable."""
    if not query or not query.strip():
        return False
    
    # Check minimum length
    if len(query.strip()) < 3:
        return False
    
    # Check for reasonable content (not just special characters)
    clean = clean_text(query)
    if len(clean) < 3:
        return False
    
    return True

def create_summary_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics from results."""
    summary = {
        "total_providers": len(results),
        "total_queries": 0,
        "avg_latency": 0.0,
        "best_provider": None,
        "worst_provider": None
    }
    
    if not results:
        return summary
    
    # Calculate averages
    total_latency = 0
    provider_scores = {}
    
    for provider, data in results.items():
        if isinstance(data, dict) and 'timing' in data:
            timing = data['timing']
            total_latency += timing.total_ms
            
            # Calculate overall score
            evaluation = data.get('evaluation', {})
            if isinstance(evaluation, dict):
                score = (evaluation.get('A', 0) + evaluation.get('B', 0) + evaluation.get('C', 0)) / 3
                provider_scores[provider] = score
    
    if provider_scores:
        summary["avg_latency"] = total_latency / len(results)
        summary["best_provider"] = max(provider_scores, key=provider_scores.get)
        summary["worst_provider"] = min(provider_scores, key=provider_scores.get)
    
    return summary

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def cache_key(query: str, provider: str) -> str:
    """Generate cache key for query-provider combination."""
    return hashlib.md5(f"{query}_{provider}".encode()).hexdigest()

def save_cache(obj: Any, path: str) -> None:
    """Save object to cache file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info(f"Cache saved to {path}")

def load_cache(path: str) -> Any:
    """Load object from cache file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_cached_results(query: str, providers: List[str]) -> Dict[str, Any]:
    """Load cached results for offline mode."""
    cache_dir = "data/cache"
    results = {"query": query, "providers": providers, "ablation_results": []}
    
    for provider in providers:
        cache_path = os.path.join(cache_dir, f"{cache_key(query, provider)}.json")
        cached_data = load_cache(cache_path)
        if cached_data:
            results["ablation_results"].append(cached_data)
    
    return results

def ndcg_at_k(relevances: List[float], k: int = 10) -> float:
    """Calculate NDCG@k for a list of relevance scores."""
    if not relevances or k == 0:
        return 0.0
    
    # Take first k items
    rels = relevances[:k]
    
    # Calculate DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
    
    # Calculate IDCG (ideal DCG with perfect ranking)
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0

def kendall_tau(list_a: List[Any], list_b: List[Any]) -> float:
    """Calculate Kendall's tau correlation between two ranked lists."""
    if len(list_a) != len(list_b):
        return 0.0
    
    n = len(list_a)
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if order is consistent
            a_order = (list_a[i] > list_a[j]) - (list_a[i] < list_a[j])
            b_order = (list_b[i] > list_b[j]) - (list_b[i] < list_b[j])
            
            if a_order * b_order > 0:
                concordant += 1
            elif a_order * b_order < 0:
                discordant += 1
    
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = (time.time() - self.start_time) * 1000  # Convert to ms
        logger.info(f"{self.name}: {self.duration:.2f}ms")
