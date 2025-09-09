"""Adaptive orchestrator that combines budget management and tiering."""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .budget import BudgetManager, BudgetConfig, BudgetTier
from .tiering import TierManager, TierConfig, RetrievalTier

logger = logging.getLogger(__name__)

class AdaptiveOrchestrator:
    """Orchestrator that adapts search strategy based on performance and requirements."""
    
    def __init__(self, budget_config: BudgetConfig = None, tier_config: TierConfig = None):
        self.budget_manager = BudgetManager(budget_config)
        self.tier_manager = TierManager(tier_config)
        self.adaptation_enabled = True
        
        logger.info("Adaptive orchestrator initialized")
    
    def get_adaptive_config(self, query: str, providers: List[str], 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get adaptive configuration for search."""
        context = context or {}
        
        # Select tier based on query and context
        tier = self.tier_manager.select_tier(query, context)
        tier_config = self.tier_manager.get_tier_config(tier)
        
        # Get budget configuration
        budget_config = self.budget_manager.get_adaptive_config(query, providers)
        
        # Combine configurations
        adaptive_config = {
            "tier": tier.value,
            "budget_tier": budget_config["tier"],
            "time_budget_ms": budget_config["time_budget_ms"],
            "quality_target": budget_config["quality_target"],
            "max_results": min(tier_config["max_results"], budget_config["max_results"]),
            "enable_hedging": tier_config["enable_hedging"] and budget_config["enable_hedging"],
            "enable_caching": tier_config["enable_caching"] and budget_config["enable_caching"],
            "enable_filtering": tier_config["enable_filtering"] and budget_config["enable_filtering"],
            "enable_streaming": tier_config["enable_streaming"],
            "rerank_topk": min(tier_config["rerank_topk"], budget_config["rerank_topk"]),
            "pruning_config": budget_config["pruning_config"],
            "quality_threshold": tier_config["quality_threshold"]
        }
        
        logger.debug(f"Adaptive config: {adaptive_config}")
        return adaptive_config
    
    def adapt_after_search(self, query: str, quality: float, time_ms: int, 
                          config_used: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt strategy after search completion."""
        if not self.adaptation_enabled:
            return config_used
        
        # Update budget manager
        budget_tier = self.budget_manager.adapt_tier(quality, time_ms)
        
        # Update tier manager
        tier = RetrievalTier(config_used["tier"])
        self.tier_manager.update_tier_performance(tier, quality, time_ms)
        
        # Get new configuration
        new_config = self.get_adaptive_config(query, [], config_used)
        
        # Log adaptation
        if new_config["tier"] != config_used["tier"] or new_config["budget_tier"] != config_used["budget_tier"]:
            logger.info(f"Strategy adapted: tier {config_used['tier']}->{new_config['tier']}, "
                       f"budget {config_used['budget_tier']}->{new_config['budget_tier']}")
        
        return new_config
    
    def should_retry_with_higher_tier(self, query: str, quality: float, 
                                    time_ms: int, config_used: Dict[str, Any]) -> bool:
        """Determine if search should be retried with higher tier."""
        if not self.adaptation_enabled:
            return False
        
        # Don't retry if already at highest tier
        if config_used["tier"] == "premium":
            return False
        
        # Don't retry if quality is already good
        if quality >= config_used["quality_threshold"]:
            return False
        
        # Don't retry if time budget is nearly exhausted
        if time_ms >= config_used["time_budget_ms"] * 0.8:
            return False
        
        # Retry if quality is poor and we have time budget
        return quality < config_used["quality_threshold"] * 0.8
    
    def get_retry_config(self, original_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration for retry with higher tier."""
        current_tier = RetrievalTier(original_config["tier"])
        
        # Upgrade tier
        if current_tier == RetrievalTier.BASIC:
            new_tier = RetrievalTier.ENHANCED
        elif current_tier == RetrievalTier.ENHANCED:
            new_tier = RetrievalTier.PREMIUM
        else:
            new_tier = RetrievalTier.PREMIUM
        
        # Get new tier configuration
        tier_config = self.tier_manager.get_tier_config(new_tier)
        
        # Create retry configuration
        retry_config = original_config.copy()
        retry_config.update({
            "tier": new_tier.value,
            "max_results": tier_config["max_results"],
            "enable_hedging": tier_config["enable_hedging"],
            "enable_caching": tier_config["enable_caching"],
            "enable_filtering": tier_config["enable_filtering"],
            "enable_streaming": tier_config["enable_streaming"],
            "rerank_topk": tier_config["rerank_topk"],
            "pruning_config": tier_config["pruning_config"],
            "quality_threshold": tier_config["quality_threshold"],
            "is_retry": True
        })
        
        logger.info(f"Retry config: {original_config['tier']} -> {new_tier.value}")
        return retry_config
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all tiers and budgets."""
        budget_stats = self.budget_manager.get_stats()
        tier_stats = self.tier_manager.get_stats()
        
        return {
            "budget_manager": budget_stats,
            "tier_manager": tier_stats,
            "adaptation_enabled": self.adaptation_enabled
        }
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state."""
        self.budget_manager = BudgetManager()
        self.tier_manager = TierManager()
        logger.info("Adaptation state reset")
    
    def disable_adaptation(self) -> None:
        """Disable adaptive behavior."""
        self.adaptation_enabled = False
        logger.info("Adaptation disabled")
    
    def enable_adaptation(self) -> None:
        """Enable adaptive behavior."""
        self.adaptation_enabled = True
        logger.info("Adaptation enabled")
