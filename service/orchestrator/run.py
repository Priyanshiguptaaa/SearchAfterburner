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
                 judge_type: str = "heuristic"):
        
        self.embedder = get_embedder(use_local_embed, embed_model)
        self.judge = get_judge(judge_type)
        self.reranker_url = reranker_url
        
        logger.info(f"Initialized orchestrator with embed_model={embed_model}, judge_type={judge_type}")
    
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
        """Search using multiple providers."""
        logger.info(f"Searching with providers: {providers}")
        
        with Timer("search_all_providers"):
            results = search_multiple_providers(providers, query, max_results)
        
        # Deduplicate results
        for provider in results:
            results[provider] = deduplicate_results(results[provider], max_results)
        
        return results
    
    def embed_and_rerank(self, query: str, results: Dict[str, List[SearchResult]], 
                        topk: int = 20) -> tuple[Dict[str, List[SearchResult]], Dict[str, Dict[str, float]]]:
        """Embed texts and rerank using the Rust service."""
        logger.info("Starting embedding and reranking process")
        
        # Embed query tokens
        with Timer("embed_query"):
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
            with Timer(f"embed_docs_{provider}"):
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
            with Timer(f"rerank_{provider}"):
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
        """Evaluate search results using the judge."""
        logger.info("Starting evaluation process")
        
        evaluations = {}
        
        # Get provider names
        providers = list(results.keys())
        
        if len(providers) >= 2:
            # Compare first two providers
            provider1, provider2 = providers[0], providers[1]
            
            with Timer("evaluate_results"):
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
        start_time = time.time()
        
        # Step 1: Plan queries (simplified)
        sub_queries = self.plan_query(query)
        logger.info(f"Generated {len(sub_queries)} sub-queries")
        
        # Step 2: Search providers
        search_start = time.time()
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
        
        # Measure individual provider search times (simplified for now)
        search_time = (time.time() - search_start) * 1000
        provider_search_times = {}
        for provider in providers:
            provider_search_times[provider] = search_time / len(providers)  # Rough estimate
        
        # Step 3: Embed and rerank
        embed_start = time.time()
        reranked_results, rerank_performance = self.embed_and_rerank(query, all_results, topk)
        embed_time = (time.time() - embed_start) * 1000
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Step 4: Evaluate
        eval_start = time.time()
        evaluations = self.evaluate_results(query, reranked_results)
        eval_time = (time.time() - eval_start) * 1000
        
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
                    query, provider_results, {"total_time_ms": total_time}
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
                        search_ms=provider_search_times.get(provider, search_time),
                        embed_ms=embed_time,
                        rerank_ms=rerank_p95,  # Use actual p95 from Rust service
                        judge_ms=eval_time,
                        total_ms=total_time
                    ),
                    "rerank_performance": rerank_performance.get(provider, {}),
                    "pairwise_results": pairwise_results.get(provider, {}),
                    "attribution_results": attribution_results.get(provider, {}),
                    "agent_judge_results": agent_judge_results.get(provider, {})
                }
        
        logger.info(f"Evaluation completed in {total_time:.2f}ms")
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
    
    # Initialize orchestrator
    orchestrator = SearchOrchestrator(
        use_local_embed=(args.embed == "local"),
        reranker_url=args.reranker_url,
        judge_type=args.judge
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
