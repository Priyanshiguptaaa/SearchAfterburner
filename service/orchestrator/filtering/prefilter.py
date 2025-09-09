"""Pre-filtering strategies for search results."""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import unicodedata

logger = logging.getLogger(__name__)

class Prefilter(ABC):
    """Abstract base class for pre-filtering strategies."""
    
    @abstractmethod
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results based on criteria."""
        pass

class QualityFilter(Prefilter):
    """Filter based on content quality indicators."""
    
    def __init__(self, min_title_length: int = 10, min_snippet_length: int = 20, 
                 max_title_length: int = 200, max_snippet_length: int = 1000):
        self.min_title_length = min_title_length
        self.min_snippet_length = min_snippet_length
        self.max_title_length = max_title_length
        self.max_snippet_length = max_snippet_length
    
    def _is_quality_content(self, result: Dict[str, Any]) -> bool:
        """Check if content meets quality standards."""
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        
        # Check length requirements
        if len(title) < self.min_title_length or len(title) > self.max_title_length:
            return False
        
        if len(snippet) < self.min_snippet_length or len(snippet) > self.max_snippet_length:
            return False
        
        # Check for spam indicators
        spam_indicators = [
            r'\b(click here|read more|learn more|see more)\b',
            r'\b(free|cheap|discount|sale|offer)\b.*\b(now|today|limited)\b',
            r'\b(guaranteed|promise|miracle|cure)\b',
            r'[!]{3,}',  # Multiple exclamation marks
            r'[A-Z]{5,}',  # All caps words
        ]
        
        text = f"{title} {snippet}".lower()
        for pattern in spam_indicators:
            if re.search(pattern, text):
                return False
        
        # Check for meaningful content
        if len(set(title.split())) < 2:  # At least 2 unique words
            return False
        
        return True
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by quality."""
        filtered = [r for r in results if self._is_quality_content(r)]
        logger.info(f"Quality filter: {len(results)} -> {len(filtered)} results")
        return filtered

class RelevanceFilter(Prefilter):
    """Filter based on query relevance."""
    
    def __init__(self, query: str, min_relevance_score: float = 0.3):
        self.query = query.lower()
        self.query_terms = set(self.query.split())
        self.min_relevance_score = min_relevance_score
    
    def _calculate_relevance(self, result: Dict[str, Any]) -> float:
        """Calculate relevance score for a result."""
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        text = f"{title} {snippet}"
        
        # Count query term matches
        text_terms = set(text.split())
        matches = self.query_terms.intersection(text_terms)
        
        if not self.query_terms:
            return 0.0
        
        # Basic relevance score
        term_score = len(matches) / len(self.query_terms)
        
        # Boost for title matches
        title_matches = self.query_terms.intersection(set(title.split()))
        title_boost = len(title_matches) / len(self.query_terms) * 0.3
        
        # Boost for exact phrase matches
        phrase_boost = 0.0
        if self.query in text:
            phrase_boost = 0.2
        
        return min(1.0, term_score + title_boost + phrase_boost)
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by relevance."""
        filtered = []
        for result in results:
            relevance = self._calculate_relevance(result)
            if relevance >= self.min_relevance_score:
                result['relevance_score'] = relevance
                filtered.append(result)
        
        # Sort by relevance score
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Relevance filter: {len(results)} -> {len(filtered)} results")
        return filtered

class LanguageFilter(Prefilter):
    """Filter based on language detection."""
    
    def __init__(self, target_language: str = 'en', confidence_threshold: float = 0.8):
        self.target_language = target_language
        self.confidence_threshold = confidence_threshold
    
    def _detect_language(self, text: str) -> tuple[str, float]:
        """Simple language detection based on character patterns."""
        if not text:
            return 'unknown', 0.0
        
        # Count ASCII vs non-ASCII characters
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        if total_chars == 0:
            return 'unknown', 0.0
        
        ascii_ratio = ascii_chars / total_chars
        
        # Simple heuristics for English
        if ascii_ratio > 0.9:
            # Check for common English patterns
            english_patterns = [
                r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
                r'\b(is|are|was|were|be|been|being)\b',
                r'\b(a|an|this|that|these|those)\b'
            ]
            
            pattern_matches = sum(1 for pattern in english_patterns if re.search(pattern, text, re.IGNORECASE))
            confidence = min(1.0, pattern_matches / len(english_patterns) + ascii_ratio * 0.5)
            
            return 'en', confidence
        else:
            # Likely non-English
            return 'non-en', 1.0 - ascii_ratio
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by language."""
        filtered = []
        
        for result in results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            text = f"{title} {snippet}"
            
            language, confidence = self._detect_language(text)
            
            if language == self.target_language and confidence >= self.confidence_threshold:
                result['language'] = language
                result['language_confidence'] = confidence
                filtered.append(result)
        
        logger.info(f"Language filter: {len(results)} -> {len(filtered)} results")
        return filtered

class DomainFilter(Prefilter):
    """Filter based on domain whitelist/blacklist."""
    
    def __init__(self, allowed_domains: Optional[List[str]] = None, 
                 blocked_domains: Optional[List[str]] = None):
        self.allowed_domains = set(allowed_domains or [])
        self.blocked_domains = set(blocked_domains or [])
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ''
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by domain."""
        filtered = []
        
        for result in results:
            url = result.get('url', '')
            domain = self._extract_domain(url)
            
            # Check blacklist first
            if self.blocked_domains and any(blocked in domain for blocked in self.blocked_domains):
                continue
            
            # Check whitelist if provided
            if self.allowed_domains and not any(allowed in domain for allowed in self.allowed_domains):
                continue
            
            filtered.append(result)
        
        logger.info(f"Domain filter: {len(results)} -> {len(filtered)} results")
        return filtered

class CompositeFilter(Prefilter):
    """Combine multiple pre-filtering strategies."""
    
    def __init__(self, filters: List[Prefilter]):
        self.filters = filters
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all filters in sequence."""
        current_results = results
        
        for filter_obj in self.filters:
            current_results = filter_obj.filter(current_results)
        
        return current_results
