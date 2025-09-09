"""Input and output validators for the evaluation system."""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates input parameters and data."""
    
    def __init__(self):
        self.valid_providers = {"ddg", "wikipedia", "exa", "google", "baseline", "mock"}
        self.max_query_length = 1000
        self.max_providers = 10
        self.max_results = 100
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate search query."""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > self.max_query_length:
            return False, f"Query too long: {len(query)} > {self.max_query_length}"
        
        # Check for potentially malicious content
        if self._contains_suspicious_content(query):
            return False, "Query contains suspicious content"
        
        return True, "Valid query"
    
    def validate_providers(self, providers: List[str]) -> Tuple[bool, str]:
        """Validate provider list."""
        if not providers:
            return False, "No providers specified"
        
        if len(providers) > self.max_providers:
            return False, f"Too many providers: {len(providers)} > {self.max_providers}"
        
        invalid_providers = [p for p in providers if p not in self.valid_providers]
        if invalid_providers:
            return False, f"Invalid providers: {invalid_providers}"
        
        return True, "Valid providers"
    
    def validate_max_results(self, max_results: int) -> Tuple[bool, str]:
        """Validate max results parameter."""
        if max_results <= 0:
            return False, "Max results must be positive"
        
        if max_results > self.max_results:
            return False, f"Max results too high: {max_results} > {self.max_results}"
        
        return True, "Valid max results"
    
    def _contains_suspicious_content(self, query: str) -> bool:
        """Check for suspicious content in query."""
        suspicious_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',              # JavaScript URLs
            r'data:text/html',          # Data URLs
            r'<iframe.*?>',             # Iframe tags
            r'<object.*?>',             # Object tags
            r'<embed.*?>',              # Embed tags
        ]
        
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

class OutputValidator:
    """Validates output data and results."""
    
    def __init__(self):
        self.min_relevance_score = 0.0
        self.max_relevance_score = 1.0
        self.min_coverage = 0
        self.max_processing_time_ms = 60000
    
    def validate_search_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, str]:
        """Validate search results structure."""
        if not results:
            return False, "No search results"
        
        for provider, provider_results in results.items():
            if not isinstance(provider_results, list):
                return False, f"Provider {provider} results must be a list"
            
            for i, result in enumerate(provider_results):
                if not isinstance(result, dict):
                    return False, f"Provider {provider} result {i} must be a dictionary"
                
                # Check required fields
                required_fields = ["title", "url", "snippet"]
                for field in required_fields:
                    if field not in result:
                        return False, f"Provider {provider} result {i} missing field: {field}"
                
                # Validate URL
                if not self._is_valid_url(result["url"]):
                    return False, f"Provider {provider} result {i} has invalid URL: {result['url']}"
        
        return True, "Valid search results"
    
    def validate_evaluation_results(self, results: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate evaluation results."""
        if not results:
            return False, "No evaluation results"
        
        for provider, evaluation in results.items():
            if not isinstance(evaluation, dict):
                return False, f"Provider {provider} evaluation must be a dictionary"
            
            # Check required metrics
            required_metrics = ["relevance_at_5", "coverage"]
            for metric in required_metrics:
                if metric not in evaluation:
                    return False, f"Provider {provider} missing metric: {metric}"
                
                value = evaluation[metric]
                if not isinstance(value, (int, float)):
                    return False, f"Provider {provider} {metric} must be numeric"
                
                # Validate ranges
                if metric == "relevance_at_5":
                    if not (self.min_relevance_score <= value <= self.max_relevance_score):
                        return False, f"Provider {provider} {metric} out of range: {value}"
                elif metric == "coverage":
                    if not (self.min_coverage <= value):
                        return False, f"Provider {provider} {metric} out of range: {value}"
        
        return True, "Valid evaluation results"
    
    def validate_performance_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate performance metrics."""
        if not metrics:
            return False, "No performance metrics"
        
        # Check processing time
        if "processing_time_ms" in metrics:
            time_ms = metrics["processing_time_ms"]
            if not isinstance(time_ms, (int, float)) or time_ms < 0:
                return False, f"Invalid processing time: {time_ms}"
            
            if time_ms > self.max_processing_time_ms:
                return False, f"Processing time too high: {time_ms} > {self.max_processing_time_ms}"
        
        return True, "Valid performance metrics"
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

class QualityValidator:
    """Validates quality metrics and thresholds."""
    
    def __init__(self):
        self.min_relevance_threshold = 0.1
        self.min_coverage_threshold = 1
        self.max_processing_time_ms = 30000
        self.min_confidence_threshold = 0.0
    
    def validate_quality_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate quality metrics against thresholds."""
        issues = []
        
        # Check relevance
        if "relevance_at_5" in metrics:
            relevance = metrics["relevance_at_5"]
            if relevance < self.min_relevance_threshold:
                issues.append(f"Low relevance: {relevance:.3f} < {self.min_relevance_threshold}")
        
        # Check coverage
        if "coverage" in metrics:
            coverage = metrics["coverage"]
            if coverage < self.min_coverage_threshold:
                issues.append(f"Low coverage: {coverage} < {self.min_coverage_threshold}")
        
        # Check processing time
        if "processing_time_ms" in metrics:
            time_ms = metrics["processing_time_ms"]
            if time_ms > self.max_processing_time_ms:
                issues.append(f"High processing time: {time_ms:.1f}ms > {self.max_processing_time_ms}ms")
        
        # Check confidence
        if "confidence" in metrics:
            confidence = metrics["confidence"]
            if confidence < self.min_confidence_threshold:
                issues.append(f"Low confidence: {confidence:.3f} < {self.min_confidence_threshold}")
        
        return len(issues) == 0, issues
    
    def validate_consistency(self, results: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate consistency across providers."""
        issues = []
        
        if len(results) < 2:
            return True, issues
        
        # Check for significant differences in metrics
        relevance_scores = []
        coverage_scores = []
        
        for provider, metrics in results.items():
            if isinstance(metrics, dict):
                if "relevance_at_5" in metrics:
                    relevance_scores.append(metrics["relevance_at_5"])
                if "coverage" in metrics:
                    coverage_scores.append(metrics["coverage"])
        
        # Check relevance consistency
        if len(relevance_scores) > 1:
            min_rel = min(relevance_scores)
            max_rel = max(relevance_scores)
            if max_rel - min_rel > 0.5:  # Large difference
                issues.append(f"High relevance variance: {min_rel:.3f} to {max_rel:.3f}")
        
        # Check coverage consistency
        if len(coverage_scores) > 1:
            min_cov = min(coverage_scores)
            max_cov = max(coverage_scores)
            if max_cov - min_cov > 10:  # Large difference
                issues.append(f"High coverage variance: {min_cov} to {max_cov}")
        
        return len(issues) == 0, issues
