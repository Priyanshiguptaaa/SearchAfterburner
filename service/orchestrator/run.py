#!/usr/bin/env python3
"""Main orchestrator for agentic search evaluation."""

import argparse
import json
import logging
import random
import time
from typing import Dict, List, Any, Optional
import httpx
import numpy as np

from providers import search_multiple_providers, SearchResult
from embed import get_embedder
from judge import get_judge
from prompts import get_synthesis_prompt
from utils import Timer, TimingStats, deduplicate_results, save_trace, set_seed, load_cached_results
from report import generate_markdown_report, save_markdown_report, print_console_summary, generate_json_report, save_json_report, print_ablation_table, generate_full_report
from metrics import MetricsCollector, StageTimer
from cache import CacheManager, CacheConfig
from filtering import FilterManager, FilterConfig
from adaptive import AdaptiveOrchestrator, BudgetConfig, TierConfig
from cascade import CascadeManager, CascadeConfig
from guardrails import GuardrailManager, GuardrailConfig, EnhancedLogger, LogConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchOrchestrator:
    """Main orchestrator for the search evaluation system."""
    
    def __init__(self, 
                 embed_model: str = "all-MiniLM-L6-v2",
                 use_local_embed: bool = True,
                 reranker_url: str = "http://localhost:8088",
                 judge_type: str = "heuristic",
                 metrics_collector: Optional[MetricsCollector] = None,
                 enable_caching: bool = True,
                 enable_filtering: bool = True,
                 enable_adaptive: bool = True,
                 enable_cascade: bool = True,
                 enable_guardrails: bool = True):
        
        self.embedder = get_embedder(use_local_embed, embed_model)
        self.judge = get_judge(judge_type)
        self.reranker_url = reranker_url
        self.metrics = metrics_collector or MetricsCollector()
        
        # Initialize caching
        if enable_caching:
            cache_config = CacheConfig(
                memory_size=1000,
                disk_size_mb=100,
                ttl_seconds=3600
            )
            self.cache = CacheManager(cache_config)
        else:
            self.cache = None
        
        # Initialize filtering
        if enable_filtering:
            filter_config = FilterConfig(
                enable_dedup=True,
                enable_quality_filter=True,
                enable_relevance_filter=True,
                enable_language_filter=True
            )
            self.filter_manager = FilterManager(filter_config)
        else:
            self.filter_manager = None
        
        # Initialize adaptive system
        if enable_adaptive:
            budget_config = BudgetConfig()
            tier_config = TierConfig()
            self.adaptive_orchestrator = AdaptiveOrchestrator(budget_config, tier_config)
        else:
            self.adaptive_orchestrator = None
        
        # Initialize cascade system
        if enable_cascade:
            cascade_config = CascadeConfig()
            self.cascade_manager = CascadeManager(cascade_config)
        else:
            self.cascade_manager = None
        
        # Initialize guardrails system
        if enable_guardrails:
            guardrail_config = GuardrailConfig()
            self.guardrail_manager = GuardrailManager(guardrail_config)
            
            log_config = LogConfig(
                enable_performance_logging=True,
                enable_quality_logging=True,
                enable_audit_logging=True
            )
            self.enhanced_logger = EnhancedLogger("orchestrator", log_config)
        else:
            self.guardrail_manager = None
            self.enhanced_logger = None
        
        logger.info(f"Initialized orchestrator with embed_model={embed_model}, judge_type={judge_type}, caching={enable_caching}, filtering={enable_filtering}, adaptive={enable_adaptive}, cascade={enable_cascade}, guardrails={enable_guardrails}")
    
    def plan_query(self, query: str) -> List[str]:
        """Plan sub-queries for comprehensive search."""
        # For now, use simple query variations
        # In a full implementation, this would use LLM planning
        sub_queries = [
            query,
            f"{query} best practices",
            f"{query} challenges problems"
        ]
        return sub_queries
    
    def search_providers(self, query: str, providers: List[str], max_results: int = 50) -> Dict[str, List[SearchResult]]:
        """Search using multiple providers with caching."""
        logger.info(f"Searching with providers: {providers}")
        
        # Check cache first
        if self.cache:
            cached_results = self.cache.get_search_results(query, providers)
            if cached_results:
                logger.info("Using cached search results")
                return cached_results
        
        with StageTimer(self.metrics, "search", {"providers": providers, "max_results": max_results}):
            results = search_multiple_providers(providers, query, max_results)
        
        # Apply filtering and de-duplication
        if self.filter_manager:
            # Convert SearchResult objects to dicts for filtering
            results_dict = {}
            for provider, provider_results in results.items():
                results_dict[provider] = [
                    {
                        'title': r.title,
                        'url': r.url,
                        'snippet': r.snippet,
                        'provider': r.provider
                    }
                    for r in provider_results
                ]
            
            # Apply filtering
            filtered_results_dict = self.filter_manager.filter_provider_results(results_dict, query)
            
            # Convert back to SearchResult objects
            from providers import SearchResult
            results = {}
            for provider, provider_results in filtered_results_dict.items():
                results[provider] = [
                    SearchResult(
                        title=r['title'],
                        url=r['url'],
                        snippet=r['snippet'],
                        provider=r['provider']
                    )
                    for r in provider_results
                ]
        else:
            # Fallback to basic deduplication
            for provider in results:
                results[provider] = deduplicate_results(results[provider], max_results)
        
        # Cache results
        if self.cache:
            self.cache.cache_search_results(query, providers, results)
        
        return results
    
    def embed_and_rerank(self, query: str, results: Dict[str, List[SearchResult]], 
                        topk: int = 20) -> tuple[Dict[str, List[SearchResult]], Dict[str, Dict[str, float]]]:
        """Embed texts and rerank using the Rust service."""
        logger.info("Starting embedding and reranking process")
        
        # Embed query tokens (with caching)
        with StageTimer(self.metrics, "embed_query", {"query_length": len(query)}):
            if self.cache:
                query_tokens = self.cache.get_embeddings(query)
                if query_tokens is None:
                    query_tokens = self.embedder.embed_query_tokens(query)
                    self.cache.cache_embeddings(query, query_tokens)
            else:
            query_tokens = self.embedder.embed_query_tokens(query)
        
        reranked_results = {}
        rerank_performance = {}
        
        for provider, provider_results in results.items():
            if not provider_results:
                reranked_results[provider] = []
                rerank_performance[provider] = {"p50_ms": 0.0, "p95_ms": 0.0}
                continue
            
            logger.info(f"Processing {len(provider_results)} results for {provider}")
            
            # Embed document tokens
            with StageTimer(self.metrics, "embed_docs", {"provider": provider, "num_docs": len(provider_results)}):
                doc_tokens_list = []
                for result in provider_results:
                    # Combine title and snippet for embedding
                    text = f"{result.title} {result.snippet}"
                    doc_tokens = self.embedder.embed_document_tokens(text)
                    doc_tokens_list.append(doc_tokens)
            
            # Prepare reranking request
            q_tokens = [token.tolist() for token in query_tokens]
            d_tokens = [[token.tolist() for token in doc_tokens] for doc_tokens in doc_tokens_list]
            
            rerank_request = {
                "q_tokens": q_tokens,
                "d_tokens": d_tokens,
                "topk": topk,
                "prune": {
                    "q_max": 16,
                    "d_max": 64,
                    "method": "idf_norm"
                }
            }
            
            # Call reranking service
            with StageTimer(self.metrics, "rerank", {"provider": provider, "num_docs": len(provider_results), "topk": topk}):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.post(
                            f"{self.reranker_url}/rerank",
                            json=rerank_request
                        )
                        response.raise_for_status()
                        
                        rerank_data = response.json()
                        order = rerank_data["order"]
                        scores = rerank_data["scores"]
                        perf = rerank_data["perf"]
                        
                        logger.info(f"Reranking for {provider}: p50={perf['per_doc_ms_p50']:.2f}ms, p95={perf['per_doc_ms_p95']:.2f}ms")
                        
                        # Store performance data (convert to microseconds for sub-ms values)
                        rerank_performance[provider] = {
                            "total_ms": perf.get('total_ms', 0.0),
                            "per_doc_p50_us": perf['per_doc_ms_p50'] * 1000,  # Convert to microseconds
                            "per_doc_p95_us": perf['per_doc_ms_p95'] * 1000,  # Convert to microseconds
                            "docs_scored": len(provider_results)
                        }
                        
                        # Reorder results based on reranking
                        reranked = [provider_results[i] for i in order]
                        reranked_results[provider] = reranked
                        
                except Exception as e:
                    logger.error(f"Reranking failed for {provider}: {e}")
                    # Fallback to original order
                    reranked_results[provider] = provider_results[:topk]
                    rerank_performance[provider] = {"p50_ms": 0.0, "p95_ms": 0.0}
        
        return reranked_results, rerank_performance
    
    def evaluate_results(self, query: str, results: Dict[str, List[SearchResult]]) -> Dict[str, Any]:
        """Evaluate search results using the cascade system."""
        logger.info("Starting evaluation process")
        
        evaluations = {}
        
        # Use cascade system if available
        if self.cascade_manager:
            for provider, provider_results in results.items():
                with StageTimer(self.metrics, "judge", {"provider": provider, "num_results": len(provider_results)}):
                    # Convert SearchResult objects to dicts for cascade
                    results_dict = [
                        {
                            "title": r.title,
                            "url": r.url,
                            "snippet": r.snippet,
                            "provider": r.provider
                        }
                        for r in provider_results
                    ]
                    
                    evaluation = self.cascade_manager.evaluate_results(query, results_dict)
                    evaluations[provider] = evaluation
        else:
            # Fallback to original judge system
        providers = list(results.keys())
        
        if len(providers) >= 2:
            # Compare first two providers
            provider1, provider2 = providers[0], providers[1]
            
                with StageTimer(self.metrics, "judge", {"provider1": provider1, "provider2": provider2, "num_results": len(results[provider1]) + len(results[provider2])}):
                evaluation = self.judge.evaluate(
                    query, provider1, results[provider1], 
                    provider2, results[provider2]
                )
                
                evaluations[provider1] = evaluation[provider1]
                evaluations[provider2] = evaluation[provider2]
        
        return evaluations
    
    def synthesize_results(self, query: str, results: Dict[str, List[SearchResult]]) -> str:
        """Synthesize results into a research brief."""
        logger.info("Synthesizing results")
        
        with StageTimer(self.metrics, "synthesize", {"num_providers": len(results), "total_results": sum(len(r) for r in results.values())}):
        # For now, return a simple synthesis
        # In a full implementation, this would use LLM synthesis
        synthesis = f"# Research Brief: {query}\n\n"
        
        for provider, provider_results in results.items():
            synthesis += f"## {provider.upper()} Results\n\n"
            for i, result in enumerate(provider_results[:3], 1):
                synthesis += f"{i}. **{result.title}**\n"
                synthesis += f"   - {result.snippet[:200]}...\n"
                synthesis += f"   - Source: [{provider}:{i}]\n\n"
        
        return synthesis
    
    def run_evaluation(self, query: str, providers: List[str], topk: int = 20, 
                      protocol: str = "both", attr: str = "on", agent_judge: str = "on", 
                      pairwise_trials: int = 5) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        logger.info(f"Starting evaluation for query: {query}")
        
        # Apply guardrails if enabled
        if self.guardrail_manager:
            # Validate input
            input_violations = self.guardrail_manager.validate_input(query, providers, topk)
            if not self.guardrail_manager.handle_violations(input_violations):
                return {"error": "Input validation failed", "violations": [str(v) for v in input_violations]}
            
            # Check rate limits
            if not self.guardrail_manager.check_rate_limit():
                return {"error": "Rate limit exceeded"}
            
            # Check circuit breaker
            if not self.guardrail_manager.check_circuit_breaker():
                return {"error": "Circuit breaker is open"}
            
            # Log search request
            if self.enhanced_logger:
                self.enhanced_logger.log_search_request(query, providers, topk)
                self.enhanced_logger.log_audit("evaluation_started", "system", {
                    "query": query[:100],
                    "providers": providers,
                    "topk": topk
                })
        
        # Set config flags for metrics
        self.metrics.set_config_flags({
            "query": query,
            "providers": providers,
            "topk": topk,
            "protocol": protocol,
            "attr": attr,
            "agent_judge": agent_judge,
            "pairwise_trials": pairwise_trials
        })
        
        with StageTimer(self.metrics, "total", {"query": query, "providers": providers}):
        # Step 1: Plan queries (simplified)
            with StageTimer(self.metrics, "plan", {"query_length": len(query)}):
        sub_queries = self.plan_query(query)
        logger.info(f"Generated {len(sub_queries)} sub-queries")
        
        # Step 2: Search providers
            with StageTimer(self.metrics, "search_all", {"providers": providers, "sub_queries": len(sub_queries)}):
        all_results = {}
        for sub_query in sub_queries:
            sub_results = self.search_providers(sub_query, providers, max_results=20)
            for provider, results in sub_results.items():
                if provider not in all_results:
                    all_results[provider] = []
                all_results[provider].extend(results)
        
        # Deduplicate across sub-queries
        for provider in all_results:
            all_results[provider] = deduplicate_results(all_results[provider], 50)
        
        # Step 3: Embed and rerank
        reranked_results, rerank_performance = self.embed_and_rerank(query, all_results, topk)
        
        # Step 4: Evaluate
        evaluations = self.evaluate_results(query, reranked_results)
        
        # Additional evaluations
        pairwise_results = {}
        attribution_results = {}
        agent_judge_results = {}
        
        for provider, provider_results in reranked_results.items():
            # Run pairwise evaluation
            if protocol in ["pairwise", "both"]:
                from judge import pairwise_evaluation_with_bias_controls
                pairwise_results[provider] = pairwise_evaluation_with_bias_controls(
                    query, provider_results, "heuristic", pairwise_trials
                )
            
            # Run attribution checking
            if attr == "on":
                from judge import check_attribution
                attribution_results[provider] = check_attribution(query, provider_results)
            
            # Run agent-as-judge evaluation
            if agent_judge == "on":
                from judge import agent_as_judge_evaluation
                agent_judge_results[provider] = agent_as_judge_evaluation(
                        query, provider_results, {"total_time_ms": self.metrics.get_total_duration()}
                )
        
        # Step 5: Synthesize
        synthesis = self.synthesize_results(query, reranked_results)
        
        # Compile results
        final_results = {}
        for provider in providers:
            if provider in reranked_results:
                # Get actual rerank performance data
                rerank_p95 = rerank_performance.get(provider, {}).get("p95_ms", 0.0)
                
                final_results[provider] = {
                    "top_results": reranked_results[provider][:topk],
                    "evaluation": evaluations.get(provider, {}),
                    "timing": TimingStats(
                        search_ms=0.0,  # Will be filled by metrics
                        embed_ms=0.0,   # Will be filled by metrics
                        rerank_ms=rerank_p95,  # Use actual p95 from Rust service
                        judge_ms=0.0,   # Will be filled by metrics
                        total_ms=self.metrics.get_total_duration()
                    ),
                    "rerank_performance": rerank_performance.get(provider, {}),
                    "pairwise_results": pairwise_results.get(provider, {}),
                    "attribution_results": attribution_results.get(provider, {}),
                    "agent_judge_results": agent_judge_results.get(provider, {})
                }
        
        # Set quality metrics for the metrics collector
        quality_metrics = {}
        for provider, data in final_results.items():
            if "evaluation" in data and isinstance(data["evaluation"], dict):
                quality_metrics[f"{provider}_rel_at_5"] = data["evaluation"].get("relevance_at_5", 0.0)
                quality_metrics[f"{provider}_coverage"] = data["evaluation"].get("coverage", 0)
            else:
                quality_metrics[f"{provider}_rel_at_5"] = 0.0
                quality_metrics[f"{provider}_coverage"] = 0
        
        self.metrics.set_quality_metrics(quality_metrics)
        
        # Save trace
        trace_file = self.metrics.save_trace()
        logger.info(f"Trace saved to {trace_file}")
        
        # Apply output validation and logging
        if self.guardrail_manager:
            total_time = self.metrics.get_total_duration()
            
            # Validate output
            output_violations = self.guardrail_manager.validate_output(final_results, total_time)
            if not self.guardrail_manager.handle_violations(output_violations):
                logger.warning("Output validation failed, but continuing")
            
            # Record success/failure
            if output_violations:
                self.guardrail_manager.record_failure()
            else:
                self.guardrail_manager.record_success()
            
            # Enhanced logging
            if self.enhanced_logger:
                self.enhanced_logger.log_performance("total_evaluation", total_time, {
                    "query": query,
                    "providers": providers,
                    "topk": topk
                })
                
                # Log quality metrics
                for provider, results in final_results.get("results", {}).items():
                    if isinstance(results, dict) and "relevance_at_5" in results:
                        self.enhanced_logger.log_quality("relevance_at_5", results["relevance_at_5"], {
                            "provider": provider,
                            "query": query
                        })
                    if isinstance(results, dict) and "coverage" in results:
                        self.enhanced_logger.log_quality("coverage", results["coverage"], {
                            "provider": provider,
                            "query": query
                        })
                
                self.enhanced_logger.log_audit("evaluation_completed", "system", {
                    "query": query[:100],
                    "providers": providers,
                    "total_time_ms": total_time,
                    "success": len(output_violations) == 0
                })
        
        # Print metrics summary
        self.metrics.print_summary()
        
        logger.info(f"Evaluation completed in {self.metrics.get_total_duration():.2f}ms")
        return final_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agentic Search Evaluation System")
    parser.add_argument("--q", "--query", dest="query", required=True, help="Search query")
    parser.add_argument("--providers", default="ddg,baseline", help="Comma-separated list of providers")
    parser.add_argument("--topk", type=int, default=20, help="Number of top results to keep")
    parser.add_argument("--judge", choices=["llm", "heuristic"], default="heuristic", help="Judge type")
    parser.add_argument("--embed", choices=["local", "openai"], default="local", help="Embedding method")
    parser.add_argument("--out", default="report.md", help="Output report file")
    parser.add_argument("--reranker-url", default="http://localhost:8088", help="Reranker service URL")
    parser.add_argument("--protocol", choices=["pointwise", "pairwise", "both"], default="both", help="Judge protocol")
    parser.add_argument("--distractor", choices=["on", "off"], default="off", help="Enable distractor injection")
    parser.add_argument("--late", choices=["on", "off"], default="on", help="Enable late-interaction reranking")
    parser.add_argument("--prune", choices=["none", "16/64", "8/32"], default="16/64", help="Token pruning setting")
    parser.add_argument("--attr", choices=["on", "off"], default="on", help="Enable attribution checking")
    parser.add_argument("--agent_judge", choices=["on", "off"], default="on", help="Enable agent-as-judge evaluation")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible results")
    parser.add_argument("--cache", choices=["on", "off"], default="off", help="Enable caching for deterministic demo")
    parser.add_argument("--offline", action="store_true", help="Use cached data only (offline mode)")
    parser.add_argument("--pairwise_trials", type=int, default=5, help="Number of pairwise comparison trials")
    parser.add_argument("--pruning_audit", action="store_true", help="Run pruning fidelity audit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse providers
    providers = [p.strip() for p in args.providers.split(",")]
    
    # Initialize orchestrator with metrics
    metrics_collector = MetricsCollector()
    orchestrator = SearchOrchestrator(
        use_local_embed=(args.embed == "local"),
        reranker_url=args.reranker_url,
        judge_type=args.judge,
        metrics_collector=metrics_collector
    )
    
    # Set random seed for reproducibility
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        set_seed(args.seed)
    
    # Run ablation study
    try:
        if args.offline:
            # Load cached results
            results = load_cached_results(args.query, providers)
        else:
            # Run ablation study with different configurations
            ablation_configs = [
                {"late": "off", "prune": "none", "name": "Single-vector baseline"},
                {"late": "on", "prune": "none", "name": "Late-interaction, no pruning"},
                {"late": "on", "prune": "16/64", "name": "Late-interaction, 16/64 pruning"},
                {"late": "on", "prune": "8/32", "name": "Late-interaction, 8/32 pruning"}
            ]
            
            ablation_results = []
            for config in ablation_configs:
                logger.info(f"Running ablation: {config['name']}")
                
                # Configure orchestrator for this ablation
                orchestrator.late_interaction = (config["late"] == "on")
                orchestrator.prune_config = config["prune"]
                
                # Run evaluation
                config_results = orchestrator.run_evaluation(
            args.query, providers, args.topk, 
            args.protocol, args.attr, args.agent_judge, args.pairwise_trials
        )
                
                # Add ablation metadata
                config_results["ablation_config"] = config
                ablation_results.append(config_results)
            
            # Generate combined results
            results = {
                "query": args.query,
                "providers": providers,
                "ablation_results": ablation_results,
                "timestamp": time.time()
            }
        
        # Generate and save reports
        report_content = generate_markdown_report(results)
        save_markdown_report(report_content, args.out)
        
        # Save JSON report
        json_report = generate_json_report(results)
        save_json_report(json_report, "results.json")
        
        # Save trace
        trace_data = {
            "query": args.query,
            "providers": providers,
            "topk": args.topk,
            "results": results,
            "timestamp": time.time()
        }
        save_trace(trace_data, "trace.json")
        
        # Print console summary
        print_console_summary(results)
        
        # Print ablation table
        print_ablation_table(results)
        
        print(f"\n✅ Evaluation complete! Report saved to {args.out}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
