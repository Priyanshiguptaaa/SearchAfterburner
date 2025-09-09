"""Judge cascade implementation for efficient evaluation."""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class CascadeConfig:
    """Configuration for judge cascade."""
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3
    
    # Time budgets (milliseconds)
    heuristic_time_budget: int = 100
    llm_time_budget: int = 2000
    
    # Quality thresholds
    min_quality_for_heuristic: float = 0.6
    min_quality_for_llm: float = 0.4
    
    # Cascade behavior
    enable_heuristic_first: bool = True
    enable_llm_fallback: bool = True
    enable_confidence_based_routing: bool = True
    
    # Performance tracking
    track_performance: bool = True
    adaptation_window: int = 50

class JudgeResult:
    """Result from a judge evaluation."""
    
    def __init__(self, relevance_score: float, confidence: float, 
                 judge_type: str, processing_time_ms: float, 
                 metadata: Dict[str, Any] = None):
        self.relevance_score = relevance_score
        self.confidence = confidence
        self.judge_type = judge_type
        self.processing_time_ms = processing_time_ms
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if result is high confidence."""
        return self.confidence >= threshold
    
    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if result is low confidence."""
        return self.confidence <= threshold

class BaseJudge(ABC):
    """Abstract base class for judges."""
    
    def __init__(self, name: str):
        self.name = name
        self.evaluation_count = 0
        self.total_time = 0.0
        self.accuracy_scores = []
    
    @abstractmethod
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> JudgeResult:
        """Evaluate search results."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get judge statistics."""
        avg_time = self.total_time / max(1, self.evaluation_count)
        avg_accuracy = sum(self.accuracy_scores) / max(1, len(self.accuracy_scores))
        
        return {
            "name": self.name,
            "evaluation_count": self.evaluation_count,
            "avg_time_ms": avg_time,
            "avg_accuracy": avg_accuracy,
            "total_time_ms": self.total_time
        }

class JudgeCascade:
    """Cascade system that routes evaluations through multiple judges."""
    
    def __init__(self, config: CascadeConfig = None):
        self.config = config or CascadeConfig()
        self.judges = {}
        self.performance_history = []
        
        logger.info("Judge cascade initialized")
    
    def add_judge(self, judge: BaseJudge) -> None:
        """Add a judge to the cascade."""
        self.judges[judge.name] = judge
        logger.info(f"Added judge: {judge.name}")
    
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> JudgeResult:
        """Evaluate results using cascade strategy."""
        if not self.judges:
            raise ValueError("No judges available in cascade")
        
        start_time = time.time()
        
        # Try heuristic judge first if enabled
        if self.config.enable_heuristic_first and "heuristic" in self.judges:
            result = self._try_judge("heuristic", query, results)
            if result and self._should_accept_result(result):
                return self._finalize_result(result, start_time)
        
        # Try LLM judge if enabled and heuristic didn't work
        if self.config.enable_llm_fallback and "llm" in self.judges:
            result = self._try_judge("llm", query, results)
            if result:
                return self._finalize_result(result, start_time)
        
        # Fallback to any available judge
        for judge_name, judge in self.judges.items():
            result = self._try_judge(judge_name, query, results)
            if result:
                return self._finalize_result(result, start_time)
        
        # If all judges fail, return a default result
        logger.warning("All judges failed, returning default result")
        return JudgeResult(
            relevance_score=0.5,
            confidence=0.0,
            judge_type="default",
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _try_judge(self, judge_name: str, query: str, results: List[Dict[str, Any]]) -> Optional[JudgeResult]:
        """Try a specific judge."""
        judge = self.judges.get(judge_name)
        if not judge:
            return None
        
        try:
            start_time = time.time()
            result = judge.evaluate(query, results)
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update judge stats
            judge.evaluation_count += 1
            judge.total_time += result.processing_time_ms
            
            logger.debug(f"Judge {judge_name} completed in {result.processing_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Judge {judge_name} failed: {e}")
            return None
    
    def _should_accept_result(self, result: JudgeResult) -> bool:
        """Determine if result should be accepted."""
        if not self.config.enable_confidence_based_routing:
            return True
        
        # Accept high confidence results
        if result.is_high_confidence(self.config.high_confidence_threshold):
            return True
        
        # Reject low confidence results
        if result.is_low_confidence(self.config.low_confidence_threshold):
            return False
        
        # For medium confidence, check quality
        return result.relevance_score >= self.config.min_quality_for_heuristic
    
    def _finalize_result(self, result: JudgeResult, start_time: float) -> JudgeResult:
        """Finalize result and update performance tracking."""
        total_time = (time.time() - start_time) * 1000
        result.processing_time_ms = total_time
        
        # Track performance if enabled
        if self.config.track_performance:
            self.performance_history.append({
                "judge_type": result.judge_type,
                "confidence": result.confidence,
                "relevance_score": result.relevance_score,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": time.time()
            })
            
            # Keep only recent history
            if len(self.performance_history) > self.config.adaptation_window:
                self.performance_history = self.performance_history[-self.config.adaptation_window:]
        
        return result
    
    def get_cascade_stats(self) -> Dict[str, Any]:
        """Get cascade performance statistics."""
        judge_stats = {name: judge.get_stats() for name, judge in self.judges.items()}
        
        # Calculate cascade efficiency
        total_evaluations = sum(judge["evaluation_count"] for judge in judge_stats.values())
        heuristic_evaluations = judge_stats.get("heuristic", {}).get("evaluation_count", 0)
        llm_evaluations = judge_stats.get("llm", {}).get("evaluation_count", 0)
        
        efficiency = {
            "total_evaluations": total_evaluations,
            "heuristic_usage": heuristic_evaluations / max(1, total_evaluations),
            "llm_usage": llm_evaluations / max(1, total_evaluations),
            "cascade_hit_rate": heuristic_evaluations / max(1, total_evaluations)
        }
        
        return {
            "judges": judge_stats,
            "efficiency": efficiency,
            "performance_history_size": len(self.performance_history)
        }
    
    def adapt_thresholds(self) -> None:
        """Adapt confidence thresholds based on performance."""
        if not self.config.track_performance or len(self.performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_history = self.performance_history[-20:]  # Last 20 evaluations
        
        # Calculate average confidence by judge type
        judge_confidences = {}
        for entry in recent_history:
            judge_type = entry["judge_type"]
            if judge_type not in judge_confidences:
                judge_confidences[judge_type] = []
            judge_confidences[judge_type].append(entry["confidence"])
        
        # Adapt thresholds based on performance
        for judge_type, confidences in judge_confidences.items():
            if len(confidences) >= 5:  # Need enough data
                avg_confidence = sum(confidences) / len(confidences)
                
                # Adjust thresholds based on performance
                if judge_type == "heuristic" and avg_confidence > 0.7:
                    # Heuristic is performing well, can be more selective
                    self.config.high_confidence_threshold = min(0.9, self.config.high_confidence_threshold + 0.05)
                elif judge_type == "heuristic" and avg_confidence < 0.5:
                    # Heuristic is struggling, be more lenient
                    self.config.high_confidence_threshold = max(0.6, self.config.high_confidence_threshold - 0.05)
        
        logger.debug(f"Adapted thresholds: high={self.config.high_confidence_threshold:.2f}, "
                    f"low={self.config.low_confidence_threshold:.2f}")
