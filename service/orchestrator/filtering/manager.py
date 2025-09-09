"""Filter manager for coordinating de-duplication and pre-filtering."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .dedup import Deduplicator, URLDeduplicator, MinHashDeduplicator, TitleDeduplicator, CompositeDeduplicator
from .prefilter import Prefilter, QualityFilter, RelevanceFilter, LanguageFilter, DomainFilter, CompositeFilter

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for filtering system."""
    enable_dedup: bool = True
    enable_quality_filter: bool = True
    enable_relevance_filter: bool = True
    enable_language_filter: bool = True
    enable_domain_filter: bool = False
    
    # Deduplication settings
    url_dedup: bool = True
    content_dedup: bool = True
    title_dedup: bool = False
    content_similarity_threshold: float = 0.8
    title_similarity_threshold: float = 0.9
    
    # Quality filter settings
    min_title_length: int = 10
    min_snippet_length: int = 20
    max_title_length: int = 200
    max_snippet_length: int = 1000
    
    # Relevance filter settings
    min_relevance_score: float = 0.3
    
    # Language filter settings
    target_language: str = 'en'
    language_confidence_threshold: float = 0.8
    
    # Domain filter settings
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None

class FilterManager:
    """Manages de-duplication and pre-filtering of search results."""
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.deduplicator = self._create_deduplicator()
        self.prefilter = self._create_prefilter()
        
        logger.info(f"Filter manager initialized with config: {self.config}")
    
    def _create_deduplicator(self) -> Optional[Deduplicator]:
        """Create de-duplicator based on config."""
        if not self.config.enable_dedup:
            return None
        
        deduplicators = []
        
        if self.config.url_dedup:
            deduplicators.append(URLDeduplicator())
        
        if self.config.content_dedup:
            deduplicators.append(MinHashDeduplicator(
                similarity_threshold=self.config.content_similarity_threshold
            ))
        
        if self.config.title_dedup:
            deduplicators.append(TitleDeduplicator(
                similarity_threshold=self.config.title_similarity_threshold
            ))
        
        if not deduplicators:
            return None
        
        if len(deduplicators) == 1:
            return deduplicators[0]
        else:
            return CompositeDeduplicator(deduplicators)
    
    def _create_prefilter(self) -> Optional[Prefilter]:
        """Create pre-filter based on config."""
        filters = []
        
        if self.config.enable_quality_filter:
            filters.append(QualityFilter(
                min_title_length=self.config.min_title_length,
                min_snippet_length=self.config.min_snippet_length,
                max_title_length=self.config.max_title_length,
                max_snippet_length=self.config.max_snippet_length
            ))
        
        if self.config.enable_domain_filter:
            filters.append(DomainFilter(
                allowed_domains=self.config.allowed_domains,
                blocked_domains=self.config.blocked_domains
            ))
        
        if not filters:
            return None
        
        if len(filters) == 1:
            return filters[0]
        else:
            return CompositeFilter(filters)
    
    def filter_results(self, results: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """Apply all filters to search results."""
        if not results:
            return results
        
        original_count = len(results)
        current_results = results
        
        # Apply pre-filtering first
        if self.prefilter:
            current_results = self.prefilter.filter(current_results)
            logger.debug(f"Pre-filtering: {original_count} -> {len(current_results)} results")
        
        # Apply relevance filtering if enabled
        if self.config.enable_relevance_filter and query:
            relevance_filter = RelevanceFilter(query, self.config.min_relevance_score)
            current_results = relevance_filter.filter(current_results)
            logger.debug(f"Relevance filtering: {len(current_results)} results")
        
        # Apply language filtering if enabled
        if self.config.enable_language_filter:
            language_filter = LanguageFilter(
                self.config.target_language,
                self.config.language_confidence_threshold
            )
            current_results = language_filter.filter(current_results)
            logger.debug(f"Language filtering: {len(current_results)} results")
        
        # Apply de-duplication last
        if self.deduplicator:
            current_results = self.deduplicator.deduplicate(current_results)
            logger.debug(f"De-duplication: {len(current_results)} results")
        
        final_count = len(current_results)
        logger.info(f"Filtering complete: {original_count} -> {final_count} results "
                   f"({(original_count - final_count) / original_count * 100:.1f}% filtered)")
        
        return current_results
    
    def filter_provider_results(self, provider_results: Dict[str, List[Dict[str, Any]]], 
                               query: str = "") -> Dict[str, List[Dict[str, Any]]]:
        """Filter results for each provider separately."""
        filtered_results = {}
        
        for provider, results in provider_results.items():
            filtered_results[provider] = self.filter_results(results, query)
        
        return filtered_results
    
    def get_filter_stats(self, original_results: List[Dict[str, Any]], 
                        filtered_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about filtering performance."""
        original_count = len(original_results)
        filtered_count = len(filtered_results)
        filtered_out = original_count - filtered_count
        
        return {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "filtered_out": filtered_out,
            "filter_rate": filtered_out / original_count if original_count > 0 else 0,
            "retention_rate": filtered_count / original_count if original_count > 0 else 0
        }
