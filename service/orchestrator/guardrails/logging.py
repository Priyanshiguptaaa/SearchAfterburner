"""Enhanced logging system for comprehensive monitoring."""

import logging
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

@dataclass
class LogConfig:
    """Configuration for enhanced logging."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_json_logging: bool = False
    enable_performance_logging: bool = True
    enable_quality_logging: bool = True
    enable_audit_logging: bool = True
    max_log_file_size_mb: int = 100
    backup_count: int = 5

class EnhancedLogger:
    """Enhanced logger with structured logging and performance tracking."""
    
    def __init__(self, name: str, config: LogConfig = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self.performance_metrics = {}
        self.quality_metrics = {}
        self.audit_trail = []
        
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logger with appropriate handlers."""
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        console_formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_log_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            
            if self.config.enable_json_logging:
                file_formatter = JsonFormatter()
            else:
                file_formatter = logging.Formatter(self.config.log_format)
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_performance(self, operation: str, duration_ms: float, 
                       metadata: Dict[str, Any] = None) -> None:
        """Log performance metrics."""
        if not self.config.enable_performance_logging:
            return
        
        metadata = metadata or {}
        log_data = {
            "type": "performance",
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            **metadata
        }
        
        if self.config.enable_json_logging:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(f"PERF: {operation} took {duration_ms:.2f}ms")
        
        # Store for analysis
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(duration_ms)
    
    def log_quality(self, metric: str, value: float, 
                   context: Dict[str, Any] = None) -> None:
        """Log quality metrics."""
        if not self.config.enable_quality_logging:
            return
        
        context = context or {}
        log_data = {
            "type": "quality",
            "metric": metric,
            "value": value,
            "timestamp": time.time(),
            **context
        }
        
        if self.config.enable_json_logging:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(f"QUALITY: {metric} = {value:.3f}")
        
        # Store for analysis
        if metric not in self.quality_metrics:
            self.quality_metrics[metric] = []
        self.quality_metrics[metric].append(value)
    
    def log_audit(self, action: str, user: str = "system", 
                 details: Dict[str, Any] = None) -> None:
        """Log audit trail."""
        if not self.config.enable_audit_logging:
            return
        
        details = details or {}
        log_data = {
            "type": "audit",
            "action": action,
            "user": user,
            "timestamp": time.time(),
            "details": details
        }
        
        if self.config.enable_json_logging:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(f"AUDIT: {user} performed {action}")
        
        # Store for audit trail
        self.audit_trail.append(log_data)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with full context."""
        context = context or {}
        log_data = {
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
            **context
        }
        
        if self.config.enable_json_logging:
            self.logger.error(json.dumps(log_data))
        else:
            self.logger.error(f"ERROR: {type(error).__name__}: {error}")
    
    def log_search_request(self, query: str, providers: List[str], 
                          max_results: int = 50) -> None:
        """Log search request details."""
        log_data = {
            "type": "search_request",
            "query": query[:100] + "..." if len(query) > 100 else query,
            "providers": providers,
            "max_results": max_results,
            "timestamp": time.time()
        }
        
        if self.config.enable_json_logging:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(f"SEARCH: query='{query[:50]}...', providers={providers}")
    
    def log_search_results(self, results: Dict[str, Any], 
                          processing_time_ms: float) -> None:
        """Log search results summary."""
        total_results = sum(len(provider_results) if isinstance(provider_results, list) else 0
                           for provider_results in results.values())
        
        log_data = {
            "type": "search_results",
            "total_results": total_results,
            "providers": list(results.keys()),
            "processing_time_ms": processing_time_ms,
            "timestamp": time.time()
        }
        
        if self.config.enable_json_logging:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(f"RESULTS: {total_results} results from {len(results)} providers in {processing_time_ms:.1f}ms")
    
    def log_rerank_performance(self, provider: str, docs_scored: int, 
                              p50_ms: float, p95_ms: float) -> None:
        """Log reranking performance."""
        log_data = {
            "type": "rerank_performance",
            "provider": provider,
            "docs_scored": docs_scored,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "timestamp": time.time()
        }
        
        if self.config.enable_json_logging:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(f"RERANK: {provider} scored {docs_scored} docs, p50={p50_ms:.2f}ms, p95={p95_ms:.2f}ms")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for operation, times in self.performance_metrics.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]
                }
        
        return summary
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality metrics summary."""
        summary = {}
        
        for metric, values in self.quality_metrics.items():
            if values:
                summary[metric] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def export_audit_trail(self, filepath: str) -> None:
        """Export audit trail to file."""
        with open(filepath, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        
        self.logger.info(f"Audit trail exported to {filepath}")

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)
