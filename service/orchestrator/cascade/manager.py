"""Cascade manager for coordinating judge evaluations."""

import logging
from typing import Dict, List, Any, Optional
from .cascade import JudgeCascade, CascadeConfig
from .judges import HeuristicJudge, LLMJudge, ConfidenceJudge

logger = logging.getLogger(__name__)

class CascadeManager:
    """Manages the judge cascade system."""
    
    def __init__(self, config: CascadeConfig = None):
        self.config = config or CascadeConfig()
        self.cascade = JudgeCascade(self.config)
        self._setup_judges()
        
        logger.info("Cascade manager initialized")
    
    def _setup_judges(self) -> None:
        """Set up judges in the cascade."""
        # Add heuristic judge (fast, always available)
        heuristic_judge = HeuristicJudge()
        self.cascade.add_judge(heuristic_judge)
        
        # Add LLM judge (slower, higher quality)
        llm_judge = LLMJudge()
        self.cascade.add_judge(llm_judge)
        
        # Add confidence judge (combines methods)
        confidence_judge = ConfidenceJudge()
        self.cascade.add_judge(confidence_judge)
        
        logger.info("Judges set up in cascade")
    
    def evaluate_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate search results using the cascade."""
        if not results:
            return {
                "relevance_at_5": 0.0,
                "coverage": 0,
                "judge_type": "none",
                "confidence": 0.0,
                "processing_time_ms": 0.0
            }
        
        # Get evaluation from cascade
        judge_result = self.cascade.evaluate(query, results)
        
        # Calculate additional metrics
        coverage = self._calculate_coverage(query, results)
        
        # Return evaluation results
        return {
            "relevance_at_5": judge_result.relevance_score,
            "coverage": coverage,
            "judge_type": judge_result.judge_type,
            "confidence": judge_result.confidence,
            "processing_time_ms": judge_result.processing_time_ms,
            "metadata": judge_result.metadata
        }
    
    def _calculate_coverage(self, query: str, results: List[Dict[str, Any]]) -> int:
        """Calculate coverage metric."""
        if not results:
            return 0
        
        # Simple coverage: count unique domains/sources
        domains = set()
        for result in results:
            url = result.get("url", "")
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain:
                        domains.add(domain)
                except Exception:
                    pass
        
        return len(domains)
    
    def get_cascade_stats(self) -> Dict[str, Any]:
        """Get cascade performance statistics."""
        return self.cascade.get_cascade_stats()
    
    def adapt_thresholds(self) -> None:
        """Adapt cascade thresholds based on performance."""
        self.cascade.adapt_thresholds()
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking."""
        self.cascade.performance_history = []
        for judge in self.cascade.judges.values():
            judge.evaluation_count = 0
            judge.total_time = 0.0
            judge.accuracy_scores = []
        
        logger.info("Performance tracking reset")
    
    def configure_judge(self, judge_name: str, **kwargs) -> bool:
        """Configure a specific judge."""
        if judge_name not in self.cascade.judges:
            logger.warning(f"Judge {judge_name} not found")
            return False
        
        # Update judge configuration
        judge = self.cascade.judges[judge_name]
        for key, value in kwargs.items():
            if hasattr(judge, key):
                setattr(judge, key, value)
                logger.info(f"Updated {judge_name}.{key} = {value}")
        
        return True
    
    def enable_judge(self, judge_name: str) -> bool:
        """Enable a specific judge."""
        if judge_name in self.cascade.judges:
            logger.info(f"Judge {judge_name} enabled")
            return True
        else:
            logger.warning(f"Judge {judge_name} not found")
            return False
    
    def disable_judge(self, judge_name: str) -> bool:
        """Disable a specific judge."""
        if judge_name in self.cascade.judges:
            del self.cascade.judges[judge_name]
            logger.info(f"Judge {judge_name} disabled")
            return True
        else:
            logger.warning(f"Judge {judge_name} not found")
            return False
