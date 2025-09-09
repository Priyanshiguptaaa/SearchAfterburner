"""Early de-duplication and prefiltering for search results."""

from .dedup import Deduplicator, MinHashDeduplicator, URLDeduplicator
from .prefilter import Prefilter, QualityFilter, RelevanceFilter, LanguageFilter
from .manager import FilterManager, FilterConfig

__all__ = [
    "Deduplicator", "MinHashDeduplicator", "URLDeduplicator",
    "Prefilter", "QualityFilter", "RelevanceFilter", "LanguageFilter",
    "FilterManager", "FilterConfig"
]
