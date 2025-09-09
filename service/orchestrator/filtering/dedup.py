"""De-duplication strategies for search results."""

import hashlib
import logging
from typing import List, Dict, Any, Set, Optional
from abc import ABC, abstractmethod
from urllib.parse import urlparse, urlunparse
import re

logger = logging.getLogger(__name__)

class Deduplicator(ABC):
    """Abstract base class for de-duplication strategies."""
    
    @abstractmethod
    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates from results."""
        pass

class URLDeduplicator(Deduplicator):
    """De-duplicate based on URL canonicalization."""
    
    def __init__(self):
        self.seen_urls: Set[str] = set()
    
    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize URL for comparison."""
        try:
            parsed = urlparse(url)
            
            # Remove common tracking parameters
            query_params = []
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        # Skip tracking parameters
                        if key.lower() not in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'ref', 'source']:
                            query_params.append(f"{key}={value}")
            
            # Rebuild URL
            canonical = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                '&'.join(query_params) if query_params else '',
                ''  # Remove fragment
            ))
            
            return canonical
            
        except Exception as e:
            logger.warning(f"Failed to canonicalize URL {url}: {e}")
            return url
    
    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate URLs."""
        unique_results = []
        seen_urls = set()
        
        for result in results:
            url = result.get('url', '')
            canonical_url = self._canonicalize_url(url)
            
            if canonical_url not in seen_urls:
                seen_urls.add(canonical_url)
                unique_results.append(result)
            else:
                logger.debug(f"Duplicate URL found: {url}")
        
        logger.info(f"Deduplicated {len(results)} -> {len(unique_results)} results by URL")
        return unique_results

class MinHashDeduplicator(Deduplicator):
    """De-duplicate based on MinHash similarity."""
    
    def __init__(self, num_hashes: int = 128, similarity_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.similarity_threshold = similarity_threshold
    
    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Extract k-shingles from text."""
        if len(text) < k:
            return {text}
        
        shingles = set()
        for i in range(len(text) - k + 1):
            shingles.add(text[i:i+k])
        return shingles
    
    def _minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature."""
        hashes = []
        for i in range(self.num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                # Simple hash function
                hash_val = hash(f"{shingle}_{i}") % (2**32)
                min_hash = min(min_hash, hash_val)
            hashes.append(int(min_hash))
        return hashes
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Compute Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove similar results based on content."""
        if len(results) <= 1:
            return results
        
        # Compute signatures for all results
        signatures = []
        for result in results:
            # Combine title and snippet for similarity
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            shingles = self._get_shingles(text.lower())
            signature = self._minhash(shingles)
            signatures.append(signature)
        
        # Find duplicates
        unique_indices = []
        for i, sig1 in enumerate(signatures):
            is_duplicate = False
            for j in unique_indices:
                sig2 = signatures[j]
                similarity = self._jaccard_similarity(sig1, sig2)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Similar content found: {results[i]['title'][:50]}... (similarity: {similarity:.2f})")
                    break
            
            if not is_duplicate:
                unique_indices.append(i)
        
        unique_results = [results[i] for i in unique_indices]
        logger.info(f"Deduplicated {len(results)} -> {len(unique_results)} results by content similarity")
        return unique_results

class TitleDeduplicator(Deduplicator):
    """De-duplicate based on title similarity."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', title.lower().strip())
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in normalized.split() if w not in stop_words]
        return ' '.join(words)
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Compute title similarity."""
        norm1 = self._normalize_title(title1)
        norm2 = self._normalize_title(title2)
        
        if norm1 == norm2:
            return 1.0
        
        # Simple word overlap similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove results with similar titles."""
        if len(results) <= 1:
            return results
        
        unique_results = []
        for result in results:
            title = result.get('title', '')
            is_duplicate = False
            
            for existing in unique_results:
                existing_title = existing.get('title', '')
                similarity = self._title_similarity(title, existing_title)
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Similar title found: {title[:50]}... (similarity: {similarity:.2f})")
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        logger.info(f"Deduplicated {len(results)} -> {len(unique_results)} results by title similarity")
        return unique_results

class CompositeDeduplicator(Deduplicator):
    """Combine multiple de-duplication strategies."""
    
    def __init__(self, deduplicators: List[Deduplicator]):
        self.deduplicators = deduplicators
    
    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all de-duplication strategies in sequence."""
        current_results = results
        
        for deduplicator in self.deduplicators:
            current_results = deduplicator.deduplicate(current_results)
        
        return current_results
