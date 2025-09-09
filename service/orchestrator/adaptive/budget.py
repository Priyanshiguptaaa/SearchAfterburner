"""Adaptive budget management for search optimization."""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BudgetTier(Enum):
    """Budget allocation tiers."""
    FAST = "fast"      # Low latency, basic quality
    BALANCED = "balanced"  # Medium latency, good quality
    THOROUGH = "thorough"  # High latency, best quality

@dataclass
class BudgetConfig:
    """Configuration for budget management."""
    # Time budgets (milliseconds)
    fast_time_budget: int = 2000
    balanced_time_budget: int = 5000
    thorough_time_budget: int = 10000
    
    # Quality thresholds
    min_quality_threshold: float = 0.7
    quality_improvement_threshold: float = 0.1
    
    # Adaptive parameters
    adaptation_window: int = 10  # Number of queries to consider
    quality_decay_factor: float = 0.9
    time_decay_factor: float = 0.95
    
    # Tier switching thresholds
    upgrade_quality_threshold: float = 0.8
    downgrade_time_threshold: float = 0.9  # 90% of budget used

class BudgetManager:
    """Manages adaptive budget allocation for search operations."""
    
    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig()
        self.current_tier = BudgetTier.BALANCED
        self.performance_history: List[Dict[str, Any]] = []
        self.tier_stats = {
            tier: {"count": 0, "avg_time": 0.0, "avg_quality": 0.0}
            for tier in BudgetTier
        }
        
        logger.info(f"Budget manager initialized with tier: {self.current_tier}")
    
    def get_time_budget(self, tier: Optional[BudgetTier] = None) -> int:
        """Get time budget for tier."""
        tier = tier or self.current_tier
        
        if tier == BudgetTier.FAST:
            return self.config.fast_time_budget
        elif tier == BudgetTier.BALANCED:
            return self.config.balanced_time_budget
        elif tier == BudgetTier.THOROUGH:
            return self.config.thorough_time_budget
        else:
            return self.config.balanced_time_budget
    
    def get_quality_target(self, tier: Optional[BudgetTier] = None) -> float:
        """Get quality target for tier."""
        tier = tier or self.current_tier
        
        if tier == BudgetTier.FAST:
            return self.config.min_quality_threshold
        elif tier == BudgetTier.BALANCED:
            return self.config.min_quality_threshold + 0.1
        elif tier == BudgetTier.THOROUGH:
            return self.config.min_quality_threshold + 0.2
        else:
            return self.config.min_quality_threshold
    
    def should_upgrade_tier(self, current_quality: float, current_time: int) -> bool:
        """Determine if tier should be upgraded."""
        if self.current_tier == BudgetTier.THOROUGH:
            return False
        
        # Upgrade if quality is high and we have time budget
        quality_target = self.get_quality_target()
        time_budget = self.get_time_budget()
        
        return (current_quality >= self.config.upgrade_quality_threshold and 
                current_time < time_budget * 0.5)
    
    def should_downgrade_tier(self, current_quality: float, current_time: int) -> bool:
        """Determine if tier should be downgraded."""
        if self.current_tier == BudgetTier.FAST:
            return False
        
        time_budget = self.get_time_budget()
        
        # Downgrade if using too much time budget
        return current_time > time_budget * self.config.downgrade_time_threshold
    
    def adapt_tier(self, quality: float, time_ms: int) -> BudgetTier:
        """Adapt tier based on performance."""
        old_tier = self.current_tier
        
        # Record performance
        self.performance_history.append({
            "tier": self.current_tier,
            "quality": quality,
            "time_ms": time_ms,
            "timestamp": time.time()
        })
        
        # Update tier statistics
        self._update_tier_stats(self.current_tier, quality, time_ms)
        
        # Apply decay to old performance data
        self._apply_decay()
        
        # Determine new tier
        if self.should_upgrade_tier(quality, time_ms):
            if self.current_tier == BudgetTier.FAST:
                self.current_tier = BudgetTier.BALANCED
            elif self.current_tier == BudgetTier.BALANCED:
                self.current_tier = BudgetTier.THOROUGH
        elif self.should_downgrade_tier(quality, time_ms):
            if self.current_tier == BudgetTier.THOROUGH:
                self.current_tier = BudgetTier.BALANCED
            elif self.current_tier == BudgetTier.BALANCED:
                self.current_tier = BudgetTier.FAST
        
        if self.current_tier != old_tier:
            logger.info(f"Tier adapted: {old_tier} -> {self.current_tier}")
        
        return self.current_tier
    
    def _update_tier_stats(self, tier: BudgetTier, quality: float, time_ms: int) -> None:
        """Update statistics for a tier."""
        stats = self.tier_stats[tier]
        stats["count"] += 1
        
        # Update running averages
        if stats["count"] == 1:
            stats["avg_quality"] = quality
            stats["avg_time"] = time_ms
        else:
            alpha = 0.1  # Learning rate
            stats["avg_quality"] = (1 - alpha) * stats["avg_quality"] + alpha * quality
            stats["avg_time"] = (1 - alpha) * stats["avg_time"] + alpha * time_ms
    
    def _apply_decay(self) -> None:
        """Apply decay to performance history."""
        current_time = time.time()
        cutoff_time = current_time - (self.config.adaptation_window * 60)  # 10 queries worth
        
        # Remove old entries
        self.performance_history = [
            entry for entry in self.performance_history
            if entry["timestamp"] > cutoff_time
        ]
    
    def get_adaptive_config(self, query: str, providers: List[str]) -> Dict[str, Any]:
        """Get adaptive configuration for search."""
        tier = self.current_tier
        
        config = {
            "tier": tier,
            "time_budget_ms": self.get_time_budget(tier),
            "quality_target": self.get_quality_target(tier),
            "max_results": self._get_max_results(tier),
            "enable_hedging": tier != BudgetTier.FAST,
            "enable_caching": True,
            "enable_filtering": tier != BudgetTier.FAST,
            "rerank_topk": self._get_rerank_topk(tier),
            "pruning_config": self._get_pruning_config(tier)
        }
        
        logger.debug(f"Adaptive config for {tier}: {config}")
        return config
    
    def _get_max_results(self, tier: BudgetTier) -> int:
        """Get maximum results for tier."""
        if tier == BudgetTier.FAST:
            return 20
        elif tier == BudgetTier.BALANCED:
            return 50
        elif tier == BudgetTier.THOROUGH:
            return 100
        else:
            return 50
    
    def _get_rerank_topk(self, tier: BudgetTier) -> int:
        """Get rerank top-k for tier."""
        if tier == BudgetTier.FAST:
            return 10
        elif tier == BudgetTier.BALANCED:
            return 20
        elif tier == BudgetTier.THOROUGH:
            return 50
        else:
            return 20
    
    def _get_pruning_config(self, tier: BudgetTier) -> Dict[str, Any]:
        """Get pruning configuration for tier."""
        if tier == BudgetTier.FAST:
            return {"q_max": 8, "d_max": 32, "method": "idf_norm"}
        elif tier == BudgetTier.BALANCED:
            return {"q_max": 16, "d_max": 64, "method": "idf_norm"}
        elif tier == BudgetTier.THOROUGH:
            return {"q_max": 32, "d_max": 128, "method": "idf_norm"}
        else:
            return {"q_max": 16, "d_max": 64, "method": "idf_norm"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get budget management statistics."""
        return {
            "current_tier": self.current_tier.value,
            "tier_stats": {
                tier.value: stats for tier, stats in self.tier_stats.items()
            },
            "performance_history_size": len(self.performance_history),
            "recent_performance": self.performance_history[-5:] if self.performance_history else []
        }
