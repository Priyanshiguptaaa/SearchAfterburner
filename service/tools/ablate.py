#!/usr/bin/env python3
"""
Ablation runner for SearchEval Pro performance optimization.
Runs different configurations and compares performance metrics.
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "orchestrator"))

from metrics import MetricsCollector, load_trace, compare_traces, StageTimer
from run import SearchOrchestrator

class AblationRunner:
    """Runs ablation studies across different configurations."""
    
    def __init__(self, output_dir: str = "runs/ablation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        
    def get_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define ablation configurations."""
        return {
            "baseline": {
                "hedge": False,
                "cache": False,
                "stream": False,
                "adaptive": False,
                "judge_cascade": False,
                "rust_upgrades": False,
                "description": "Baseline configuration"
            },
            "+hedge": {
                "hedge": True,
                "cache": False,
                "stream": False,
                "adaptive": False,
                "judge_cascade": False,
                "rust_upgrades": False,
                "description": "Baseline + Network hedging"
            },
            "+hedge+cache": {
                "hedge": True,
                "cache": True,
                "stream": False,
                "adaptive": False,
                "judge_cascade": False,
                "rust_upgrades": False,
                "description": "Baseline + Hedging + Caching"
            },
            "+hedge+cache+stream": {
                "hedge": True,
                "cache": True,
                "stream": True,
                "adaptive": False,
                "judge_cascade": False,
                "rust_upgrades": False,
                "description": "Baseline + Hedging + Caching + Streaming"
            },
            "+all": {
                "hedge": True,
                "cache": True,
                "stream": True,
                "adaptive": True,
                "judge_cascade": True,
                "rust_upgrades": True,
                "description": "All optimizations enabled"
            }
        }
    
    def run_config(self, config_name: str, config: Dict[str, Any], 
                   query: str, providers: List[str], topk: int = 5, 
                   num_runs: int = 3) -> Dict[str, Any]:
        """Run a single configuration multiple times."""
        print(f"\n🔬 Running config: {config_name}")
        print(f"   Description: {config['description']}")
        
        run_results = []
        quality_metrics = []
        
        for run_idx in range(num_runs):
            print(f"   Run {run_idx + 1}/{num_runs}...")
            
            # Create metrics collector
            run_id = f"{config_name}_run_{run_idx}_{int(time.time())}"
            collector = MetricsCollector(run_id)
            collector.set_config_flags(config)
            
            try:
                # Initialize orchestrator with config
                orchestrator = SearchOrchestrator(
                    embed_model="all-MiniLM-L6-v2",
                    use_local_embed=True,
                    reranker_url="http://localhost:8088",
                    judge_type="heuristic"
                )
                
                # Run evaluation with timing
                start_time = time.time()
                
                # Run the complete evaluation pipeline
                results = orchestrator.run_evaluation(query, providers, topk)
                
                total_time = (time.time() - start_time) * 1000
                
                # Collect quality metrics from results
                quality = {
                    "rel_at_5": 0.0,
                    "coverage": 0,
                    "total_results": 0
                }
                
                for provider, data in results.items():
                    if "evaluation" in data and isinstance(data["evaluation"], dict):
                        quality["rel_at_5"] += data["evaluation"].get("relevance_at_5", 0.0)
                        quality["coverage"] += data["evaluation"].get("coverage", 0)
                    quality["total_results"] += len(data.get("top_results", []))
                
                if results:
                    quality["rel_at_5"] /= len(results)
                    quality["coverage"] /= len(results)
                
                collector.set_quality_metrics(quality)
                quality_metrics.append(quality)
                
                # Save trace
                trace_file = collector.save_trace()
                run_results.append({
                    "run_id": run_id,
                    "trace_file": str(trace_file),
                    "total_time_ms": total_time,
                    "quality": quality
                })
                
            except Exception as e:
                print(f"   ❌ Run {run_idx + 1} failed: {e}")
                continue
        
        if not run_results:
            print(f"   ❌ All runs failed for {config_name}")
            return None
        
        # Compute aggregated metrics
        total_times = [r["total_time_ms"] for r in run_results]
        avg_quality = {
            "rel_at_5": statistics.mean([q["rel_at_5"] for q in quality_metrics]),
            "coverage": statistics.mean([q["coverage"] for q in quality_metrics]),
            "total_results": statistics.mean([q["total_results"] for q in quality_metrics])
        }
        
        result = {
            "config_name": config_name,
            "config": config,
            "num_runs": len(run_results),
            "total_time_stats": {
                "p50_ms": statistics.quantiles(total_times, n=2)[0] if len(total_times) > 1 else total_times[0],
                "p90_ms": statistics.quantiles(total_times, n=10)[8] if len(total_times) > 1 else total_times[0],
                "p95_ms": statistics.quantiles(total_times, n=20)[18] if len(total_times) > 1 else total_times[0],
                "mean_ms": statistics.mean(total_times),
                "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0.0
            },
            "quality_metrics": avg_quality,
            "run_results": run_results
        }
        
        print(f"   ✅ Completed {len(run_results)} runs")
        print(f"   📊 P95: {result['total_time_stats']['p95_ms']:.1f}ms, "
              f"Rel@5: {avg_quality['rel_at_5']:.3f}, "
              f"Coverage: {avg_quality['coverage']:.1f}")
        
        return result
    
    def run_ablation_study(self, queries: List[str], providers: List[str], 
                          topk: int = 5, num_runs: int = 3) -> Dict[str, Any]:
        """Run complete ablation study across all configurations."""
        print(f"🚀 Starting ablation study")
        print(f"   Queries: {len(queries)}")
        print(f"   Providers: {providers}")
        print(f"   Top-K: {topk}")
        print(f"   Runs per config: {num_runs}")
        
        configs = self.get_configs()
        all_results = []
        
        for config_name, config in configs.items():
            for query in queries:
                print(f"\n📝 Query: {query[:50]}...")
                result = self.run_config(config_name, config, query, providers, topk, num_runs)
                if result:
                    all_results.append(result)
        
        # Generate comparison report
        comparison = self.generate_comparison_report(all_results)
        
        # Save results
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "queries": queries,
                "providers": providers,
                "topk": topk,
                "num_runs": num_runs,
                "results": all_results,
                "comparison": comparison
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
        return comparison
    
    def generate_comparison_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison report between configurations."""
        if not results:
            return {}
        
        # Group by config
        config_groups = {}
        for result in results:
            config_name = result["config_name"]
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(result)
        
        # Compute aggregated stats per config
        config_stats = {}
        for config_name, config_results in config_groups.items():
            all_times = []
            all_quality = {"rel_at_5": [], "coverage": [], "total_results": []}
            
            for result in config_results:
                all_times.append(result["total_time_stats"]["p95_ms"])
                all_quality["rel_at_5"].append(result["quality_metrics"]["rel_at_5"])
                all_quality["coverage"].append(result["quality_metrics"]["coverage"])
                all_quality["total_results"].append(result["quality_metrics"]["total_results"])
            
            config_stats[config_name] = {
                "p95_ms": statistics.mean(all_times),
                "p95_std": statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
                "rel_at_5": statistics.mean(all_quality["rel_at_5"]),
                "coverage": statistics.mean(all_quality["coverage"]),
                "total_results": statistics.mean(all_quality["total_results"]),
                "num_queries": len(config_results)
            }
        
        # Compute deltas from baseline
        baseline_stats = config_stats.get("baseline", {})
        deltas = {}
        
        for config_name, stats in config_stats.items():
            if config_name == "baseline":
                continue
            
            deltas[config_name] = {
                "p95_delta_ms": stats["p95_ms"] - baseline_stats.get("p95_ms", 0),
                "p95_delta_pct": ((stats["p95_ms"] - baseline_stats.get("p95_ms", 0)) / baseline_stats.get("p95_ms", 1)) * 100,
                "rel_at_5_delta": stats["rel_at_5"] - baseline_stats.get("rel_at_5", 0),
                "coverage_delta": stats["coverage"] - baseline_stats.get("coverage", 0)
            }
        
        return {
            "config_stats": config_stats,
            "deltas": deltas
        }
    
    def print_comparison_table(self, comparison: Dict[str, Any]):
        """Print a formatted comparison table."""
        if not comparison or "config_stats" not in comparison:
            print("No comparison data available")
            return
        
        config_stats = comparison["config_stats"]
        deltas = comparison.get("deltas", {})
        
        print(f"\n📊 Ablation Study Results")
        print("=" * 100)
        print(f"{'Config':<20} {'P95ms':<8} {'ΔP95%':<8} {'Rel@5':<8} {'ΔRel@5':<8} {'Coverage':<8} {'ΔCov':<8} {'Queries':<8}")
        print("-" * 100)
        
        for config_name in ["baseline", "+hedge", "+hedge+cache", "+hedge+cache+stream", "+all"]:
            if config_name not in config_stats:
                continue
            
            stats = config_stats[config_name]
            delta = deltas.get(config_name, {})
            
            p95_delta_pct = delta.get("p95_delta_pct", 0)
            rel_delta = delta.get("rel_at_5_delta", 0)
            cov_delta = delta.get("coverage_delta", 0)
            
            print(f"{config_name:<20} {stats['p95_ms']:<8.1f} {p95_delta_pct:<8.1f} "
                  f"{stats['rel_at_5']:<8.3f} {rel_delta:<8.3f} {stats['coverage']:<8.1f} "
                  f"{cov_delta:<8.1f} {stats['num_queries']:<8}")
        
        print("=" * 100)
        
        # Summary
        if deltas:
            best_config = min(deltas.keys(), key=lambda k: deltas[k]["p95_delta_ms"])
            best_delta = deltas[best_config]
            print(f"\n🏆 Best configuration: {best_config}")
            print(f"   P95 improvement: {abs(best_delta['p95_delta_pct']):.1f}%")
            print(f"   Quality change: Rel@5 {best_delta['rel_at_5_delta']:+.3f}, Coverage {best_delta['coverage_delta']:+.1f}")

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for SearchEval Pro")
    parser.add_argument("--queries", nargs="+", 
                       default=["Challenges in evaluating LLM-powered search quality",
                               "What are the latest advances in AI?",
                               "How does machine learning work?",
                               "Best practices for software engineering",
                               "Climate change solutions"],
                       help="Queries to test")
    parser.add_argument("--providers", nargs="+", default=["ddg", "wikipedia"],
                       help="Search providers to use")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument("--output", default="runs/ablation", help="Output directory")
    
    args = parser.parse_args()
    
    # Check if Rust service is running
    try:
        import httpx
        response = httpx.get("http://localhost:8088/bench", timeout=5)
        if response.status_code != 200:
            print("❌ Rust reranker service not responding properly")
            return 1
    except Exception as e:
        print(f"❌ Cannot connect to Rust reranker service: {e}")
        print("   Please start the service with: cd ranker-rs && cargo run --release")
        return 1
    
    # Run ablation study
    runner = AblationRunner(args.output)
    comparison = runner.run_ablation_study(args.queries, args.providers, args.topk, args.runs)
    
    # Print results
    runner.print_comparison_table(comparison)
    
    return 0

if __name__ == "__main__":
    exit(main())
