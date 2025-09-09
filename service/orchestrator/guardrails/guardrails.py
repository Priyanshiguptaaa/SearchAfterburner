"""Evaluation guardrails for ensuring system reliability and quality."""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class GuardrailLevel(Enum):
    """Guardrail enforcement levels."""
    WARN = "warn"      # Log warning but continue
    BLOCK = "block"    # Block operation and return error
    ADAPT = "adapt"    # Adapt behavior to meet constraints

@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    # Input validation
    max_query_length: int = 1000
    max_providers: int = 10
    max_results_per_provider: int = 100
    
    # Quality thresholds
    min_relevance_threshold: float = 0.1
    max_processing_time_ms: int = 30000
    min_coverage_threshold: int = 1
    
    # Rate limiting
    max_requests_per_minute: int = 100
    max_concurrent_requests: int = 10
    
    # Error handling
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_ms: int = 60000
    
    # Logging
    enable_detailed_logging: bool = True
    log_performance_metrics: bool = True
    log_quality_metrics: bool = True

class GuardrailViolation:
    """Represents a guardrail violation."""
    
    def __init__(self, rule: str, level: GuardrailLevel, message: str, 
                 value: Any = None, threshold: Any = None):
        self.rule = rule
        self.level = level
        self.message = message
        self.value = value
        self.threshold = threshold
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.rule}: {self.message}"

class GuardrailManager:
    """Manages evaluation guardrails and quality controls."""
    
    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.violations: List[GuardrailViolation] = []
        self.request_counts: Dict[str, int] = {}
        self.circuit_breaker_state = "CLOSED"
        self.circuit_breaker_failures = 0
        self.circuit_breaker_reset_time = 0
        
        logger.info("Guardrail manager initialized")
    
    def validate_input(self, query: str, providers: List[str], 
                      max_results: int = 50) -> List[GuardrailViolation]:
        """Validate input parameters."""
        violations = []
        
        # Query length validation
        if len(query) > self.config.max_query_length:
            violations.append(GuardrailViolation(
                "query_length",
                GuardrailLevel.WARN,
                f"Query too long: {len(query)} > {self.config.max_query_length}",
                len(query),
                self.config.max_query_length
            ))
        
        # Provider count validation
        if len(providers) > self.config.max_providers:
            violations.append(GuardrailViolation(
                "provider_count",
                GuardrailLevel.BLOCK,
                f"Too many providers: {len(providers)} > {self.config.max_providers}",
                len(providers),
                self.config.max_providers
            ))
        
        # Results per provider validation
        if max_results > self.config.max_results_per_provider:
            violations.append(GuardrailViolation(
                "max_results",
                GuardrailLevel.WARN,
                f"Too many results requested: {max_results} > {self.config.max_results_per_provider}",
                max_results,
                self.config.max_results_per_provider
            ))
        
        # Query content validation
        if not query.strip():
            violations.append(GuardrailViolation(
                "empty_query",
                GuardrailLevel.BLOCK,
                "Query cannot be empty"
            ))
        
        # Provider validation
        valid_providers = {"ddg", "wikipedia", "exa", "google", "baseline", "mock"}
        invalid_providers = [p for p in providers if p not in valid_providers]
        if invalid_providers:
            violations.append(GuardrailViolation(
                "invalid_providers",
                GuardrailLevel.BLOCK,
                f"Invalid providers: {invalid_providers}",
                invalid_providers,
                list(valid_providers)
            ))
        
        return violations
    
    def validate_output(self, results: Dict[str, Any], 
                       processing_time_ms: float) -> List[GuardrailViolation]:
        """Validate output quality and performance."""
        violations = []
        
        # Processing time validation
        if processing_time_ms > self.config.max_processing_time_ms:
            violations.append(GuardrailViolation(
                "processing_time",
                GuardrailLevel.WARN,
                f"Processing time exceeded: {processing_time_ms:.1f}ms > {self.config.max_processing_time_ms}ms",
                processing_time_ms,
                self.config.max_processing_time_ms
            ))
        
        # Results quality validation
        for provider, provider_results in results.items():
            if isinstance(provider_results, dict) and "relevance_at_5" in provider_results:
                relevance = provider_results["relevance_at_5"]
                if relevance < self.config.min_relevance_threshold:
                    violations.append(GuardrailViolation(
                        "low_relevance",
                        GuardrailLevel.WARN,
                        f"Low relevance for {provider}: {relevance:.3f} < {self.config.min_relevance_threshold}",
                        relevance,
                        self.config.min_relevance_threshold
                    ))
            
            if isinstance(provider_results, dict) and "coverage" in provider_results:
                coverage = provider_results["coverage"]
                if coverage < self.config.min_coverage_threshold:
                    violations.append(GuardrailViolation(
                        "low_coverage",
                        GuardrailLevel.WARN,
                        f"Low coverage for {provider}: {coverage} < {self.config.min_coverage_threshold}",
                        coverage,
                        self.config.min_coverage_threshold
                    ))
        
        return violations
    
    def check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        minute_key = f"{client_id}:{int(current_time // 60)}"
        
        # Clean old entries
        old_keys = [k for k in self.request_counts.keys() 
                   if not k.startswith(f"{client_id}:") or 
                   int(k.split(":")[1]) < int(current_time // 60) - 1]
        for key in old_keys:
            del self.request_counts[key]
        
        # Check current minute
        current_count = self.request_counts.get(minute_key, 0)
        if current_count >= self.config.max_requests_per_minute:
            return False
        
        # Update count
        self.request_counts[minute_key] = current_count + 1
        return True
    
    def check_circuit_breaker(self) -> bool:
        """Check circuit breaker state."""
        current_time = time.time()
        
        if self.circuit_breaker_state == "OPEN":
            if current_time > self.circuit_breaker_reset_time:
                self.circuit_breaker_state = "HALF_OPEN"
                self.circuit_breaker_failures = 0
                logger.info("Circuit breaker reset to HALF_OPEN")
            else:
                return False
        
        return True
    
    def record_success(self) -> None:
        """Record successful operation."""
        if self.circuit_breaker_state == "HALF_OPEN":
            self.circuit_breaker_state = "CLOSED"
            self.circuit_breaker_failures = 0
            logger.info("Circuit breaker closed after success")
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.circuit_breaker_failures += 1
        
        if self.circuit_breaker_failures >= self.config.circuit_breaker_threshold:
            self.circuit_breaker_state = "OPEN"
            self.circuit_breaker_reset_time = time.time() + (self.config.circuit_breaker_timeout_ms / 1000)
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")
    
    def handle_violations(self, violations: List[GuardrailViolation]) -> bool:
        """Handle guardrail violations. Returns True if operation should continue."""
        if not violations:
            return True
        
        # Log violations
        for violation in violations:
            if violation.level == GuardrailLevel.WARN:
                logger.warning(str(violation))
            elif violation.level == GuardrailLevel.BLOCK:
                logger.error(str(violation))
            elif violation.level == GuardrailLevel.ADAPT:
                logger.info(str(violation))
        
        # Store violations
        self.violations.extend(violations)
        
        # Check if any violations should block the operation
        blocking_violations = [v for v in violations if v.level == GuardrailLevel.BLOCK]
        if blocking_violations:
            logger.error(f"Operation blocked due to {len(blocking_violations)} blocking violations")
            return False
        
        return True
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get violation statistics."""
        if not self.violations:
            return {"total_violations": 0}
        
        # Group by rule
        rule_counts = {}
        level_counts = {}
        
        for violation in self.violations:
            rule_counts[violation.rule] = rule_counts.get(violation.rule, 0) + 1
            level_counts[violation.level.value] = level_counts.get(violation.level.value, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "rule_counts": rule_counts,
            "level_counts": level_counts,
            "circuit_breaker_state": self.circuit_breaker_state,
            "circuit_breaker_failures": self.circuit_breaker_failures
        }
    
    def reset_violations(self) -> None:
        """Reset violation history."""
        self.violations.clear()
        logger.info("Violation history reset")
    
    def adapt_config(self, performance_metrics: Dict[str, Any]) -> None:
        """Adapt guardrail configuration based on performance."""
        # Adapt based on processing time
        if "avg_processing_time_ms" in performance_metrics:
            avg_time = performance_metrics["avg_processing_time_ms"]
            if avg_time > self.config.max_processing_time_ms * 0.8:
                # Increase threshold by 20% if consistently high
                self.config.max_processing_time_ms = int(self.config.max_processing_time_ms * 1.2)
                logger.info(f"Adapted max_processing_time_ms to {self.config.max_processing_time_ms}")
        
        # Adapt based on relevance scores
        if "avg_relevance" in performance_metrics:
            avg_relevance = performance_metrics["avg_relevance"]
            if avg_relevance < self.config.min_relevance_threshold * 1.2:
                # Lower threshold if consistently low
                self.config.min_relevance_threshold = max(0.05, self.config.min_relevance_threshold * 0.9)
                logger.info(f"Adapted min_relevance_threshold to {self.config.min_relevance_threshold}")
