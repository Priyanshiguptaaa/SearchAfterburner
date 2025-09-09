"""Retrieval tiering for adaptive search optimization."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalTier(Enum):
    """Retrieval quality tiers."""
    BASIC = "basic"        # Fast, simple retrieval
    ENHANCED = "enhanced"  # Balanced retrieval with improvements
    PREMIUM = "premium"    # High-quality retrieval with all features

@dataclass
class TierConfig:
    """Configuration for retrieval tiering."""
    # Tier-specific settings
    basic_max_results: int = 20
    enhanced_max_results: int = 50
    premium_max_results: int = 100
    
    # Quality thresholds for tier switching
    basic_quality_threshold: float = 0.6
    enhanced_quality_threshold: float = 0.8
    premium_quality_threshold: float = 0.9
    
    # Feature enablement by tier
    basic_features: List[str] = None
    enhanced_features: List[str] = None
    premium_features: List[str] = None
    
    def __post_init__(self):
        if self.basic_features is None:
            self.basic_features = ["search", "basic_rerank"]
        if self.enhanced_features is None:
            self.enhanced_features = ["search", "rerank", "filtering", "caching"]
        if self.premium_features is None:
            self.premium_features = ["search", "rerank", "filtering", "caching", "hedging", "streaming"]

class TierManager:
    """Manages retrieval tiering and feature selection."""
    
    def __init__(self, config: TierConfig = None):
        self.config = config or TierConfig()
        self.current_tier = RetrievalTier.ENHANCED
        self.tier_performance = {
            tier: {"queries": 0, "avg_quality": 0.0, "avg_time": 0.0}
            for tier in RetrievalTier
        }
        
        logger.info(f"Tier manager initialized with tier: {self.current_tier}")
    
    def select_tier(self, query: str, context: Dict[str, Any] = None) -> RetrievalTier:
        """Select appropriate tier based on query and context."""
        context = context or {}
        
        # Simple tier selection logic
        query_length = len(query.split())
        query_complexity = self._assess_query_complexity(query)
        
        # Start with basic tier
        selected_tier = RetrievalTier.BASIC
        
        # Upgrade based on complexity
        if query_complexity > 0.7 or query_length > 5:
            selected_tier = RetrievalTier.PREMIUM
        elif query_complexity > 0.4 or query_length > 3:
            selected_tier = RetrievalTier.ENHANCED
        
        # Consider context
        if context.get("user_preference") == "thorough":
            selected_tier = RetrievalTier.PREMIUM
        elif context.get("user_preference") == "fast":
            selected_tier = RetrievalTier.BASIC
        
        # Consider time constraints
        if context.get("time_budget_ms", 0) < 2000:
            selected_tier = RetrievalTier.BASIC
        elif context.get("time_budget_ms", 0) > 8000:
            selected_tier = RetrievalTier.PREMIUM
        
        self.current_tier = selected_tier
        logger.debug(f"Selected tier {selected_tier} for query: {query[:50]}...")
        
        return selected_tier
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 to 1.0)."""
        complexity_indicators = [
            # Question words
            r'\b(what|how|why|when|where|which|who)\b',
            # Complex conjunctions
            r'\b(and|or|but|however|although|despite|because)\b',
            # Technical terms
            r'\b(algorithm|implementation|analysis|optimization|performance)\b',
            # Multi-part queries
            r'\b(compare|difference|similar|versus|vs|between)\b',
            # Specificity indicators
            r'\b(specific|detailed|comprehensive|thorough)\b'
        ]
        
        import re
        query_lower = query.lower()
        
        matches = sum(1 for pattern in complexity_indicators if re.search(pattern, query_lower))
        complexity = min(1.0, matches / len(complexity_indicators))
        
        return complexity
    
    def get_tier_config(self, tier: RetrievalTier) -> Dict[str, Any]:
        """Get configuration for a specific tier."""
        if tier == RetrievalTier.BASIC:
            return {
                "max_results": self.config.basic_max_results,
                "features": self.config.basic_features,
                "quality_threshold": self.config.basic_quality_threshold,
                "enable_hedging": False,
                "enable_caching": False,
                "enable_filtering": False,
                "enable_streaming": False,
                "rerank_topk": 10,
                "pruning_config": {"q_max": 8, "d_max": 32, "method": "idf_norm"}
            }
        elif tier == RetrievalTier.ENHANCED:
            return {
                "max_results": self.config.enhanced_max_results,
                "features": self.config.enhanced_features,
                "quality_threshold": self.config.enhanced_quality_threshold,
                "enable_hedging": False,
                "enable_caching": True,
                "enable_filtering": True,
                "enable_streaming": False,
                "rerank_topk": 20,
                "pruning_config": {"q_max": 16, "d_max": 64, "method": "idf_norm"}
            }
        elif tier == RetrievalTier.PREMIUM:
            return {
                "max_results": self.config.premium_max_results,
                "features": self.config.premium_features,
                "quality_threshold": self.config.premium_quality_threshold,
                "enable_hedging": True,
                "enable_caching": True,
                "enable_filtering": True,
                "enable_streaming": True,
                "rerank_topk": 50,
                "pruning_config": {"q_max": 32, "d_max": 128, "method": "idf_norm"}
            }
        else:
            return self.get_tier_config(RetrievalTier.ENHANCED)
    
    def update_tier_performance(self, tier: RetrievalTier, quality: float, time_ms: float) -> None:
        """Update performance statistics for a tier."""
        stats = self.tier_performance[tier]
        stats["queries"] += 1
        
        # Update running averages
        if stats["queries"] == 1:
            stats["avg_quality"] = quality
            stats["avg_time"] = time_ms
        else:
            alpha = 0.1  # Learning rate
            stats["avg_quality"] = (1 - alpha) * stats["avg_quality"] + alpha * quality
            stats["avg_time"] = (1 - alpha) * stats["avg_time"] + alpha * time_ms
        
        logger.debug(f"Updated {tier} performance: quality={quality:.2f}, time={time_ms:.1f}ms")
    
    def get_recommended_tier(self, target_quality: float, max_time_ms: int) -> RetrievalTier:
        """Get recommended tier based on quality and time requirements."""
        # Find tiers that meet quality requirements
        suitable_tiers = []
        for tier in RetrievalTier:
            stats = self.tier_performance[tier]
            if stats["queries"] > 0 and stats["avg_quality"] >= target_quality:
                suitable_tiers.append((tier, stats["avg_time"]))
        
        if not suitable_tiers:
            # Fallback to highest tier
            return RetrievalTier.PREMIUM
        
        # Find tier with best time performance that meets requirements
        suitable_tiers.sort(key=lambda x: x[1])  # Sort by time
        
        for tier, avg_time in suitable_tiers:
            if avg_time <= max_time_ms:
                return tier
        
        # If no tier meets time requirements, return fastest suitable tier
        return suitable_tiers[0][0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tier management statistics."""
        return {
            "current_tier": self.current_tier.value,
            "tier_performance": {
                tier.value: stats for tier, stats in self.tier_performance.items()
            },
            "tier_configs": {
                tier.value: self.get_tier_config(tier) for tier in RetrievalTier
            }
        }
