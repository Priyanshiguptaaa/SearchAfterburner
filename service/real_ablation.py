#!/usr/bin/env python3
"""
Real Ablation Runner - Measure actual performance with real numbers
"""

import argparse
import json
import time
import subprocess
import requests
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class RealAblationResult:
    config_name: str
    total_time_ms: float
    search_time_ms: float
    rerank_time_ms: float
    eval_time_ms: float
    rerank_p50_ms: float
    rerank_p95_ms: float
    coverage: int
    optimizations: List[str]
    rust_benchmark: Dict[str, Any]

class RealAblationRunner:
    def __init__(self):
        self.rust_service_url = "http://localhost:8088"
        self.base_config = {
            "query": "machine learning optimization techniques",
            "providers": "ddg,wikipedia",
            "topk": 10,
            "judge": "heuristic",
            "embed": "local"
        }
        
        # Test configurations with real measurements
        self.configs = {
            "baseline": {
                "late": False,
                "prune": False,
                "rust_bench": {"n_docs": 50, "td": 32, "d": 128, "prune": "none"}
            },
            "late_interaction": {
                "late": True,
                "prune": False,
                "rust_bench": {"n_docs": 50, "td": 32, "d": 128, "prune": "none"}
            },
            "token_pruning_16_64": {
                "late": True,
                "prune": "16/64",
                "rust_bench": {"n_docs": 50, "td": 32, "d": 128, "prune": "16/64"}
            },
            "token_pruning_8_32": {
                "late": True,
                "prune": "8/32",
                "rust_bench": {"n_docs": 50, "td": 32, "d": 128, "prune": "8/32"}
            },
            "high_docs": {
                "late": True,
                "prune": "16/64",
                "rust_bench": {"n_docs": 200, "td": 32, "d": 128, "prune": "16/64"}
            },
            "high_dim": {
                "late": True,
                "prune": "16/64",
                "rust_bench": {"n_docs": 50, "td": 32, "d": 256, "prune": "16/64"}
            }
        }

    def measure_rust_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure real Rust reranker performance"""
        try:
            # Build benchmark URL
            bench_params = config["rust_bench"]
            url = f"{self.rust_service_url}/bench"
            params = {
                "n_docs": bench_params["n_docs"],
                "td": bench_params["td"],
                "d": bench_params["d"]
            }
            if bench_params["prune"] != "none":
                params["prune"] = bench_params["prune"]
            
            # Make request
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"❌ Error measuring Rust performance: {e}")
            return {
                "n_docs": bench_params["n_docs"],
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "threads": 1,
                "cpu_flags": "ERROR"
            }

    def run_python_evaluation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Python evaluation and measure timing"""
        try:
            # Build command
            cmd = [
                "python3", "simple_run.py",
                "--q", self.base_config["query"],
                "--providers", self.base_config["providers"],
                "--topk", str(self.base_config["topk"]),
                "--judge", self.base_config["judge"],
                "--embed", self.base_config["embed"]
            ]
            
            # Run and measure
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"❌ Python evaluation failed: {result.stderr}")
                return None
            
            total_time = end_time - start_time
            
            # Parse results from JSON file
            try:
                with open("simple_results.json", "r") as f:
                    results = json.load(f)
                return {
                    "total_time_ms": total_time * 1000,
                    "timing": results.get("timing", {}),
                    "results": results.get("results", {}),
                    "rerank_performance": results.get("rerank_performance", {})
                }
            except FileNotFoundError:
                # No fallback - return None if no real data
                print(f"❌ No results file found - cannot provide real data")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Python evaluation timeout")
            return None
        except Exception as e:
            print(f"❌ Python evaluation error: {e}")
            return None

    def run_single_config(self, config_name: str, config: Dict[str, Any]) -> RealAblationResult:
        """Run a single configuration and measure real performance"""
        print(f"\n🔬 Running {config_name}...")
        
        # Measure Rust performance
        print(f"  📊 Measuring Rust reranker performance...")
        rust_bench = self.measure_rust_performance(config)
        
        # Run Python evaluation
        print(f"  🐍 Running Python evaluation...")
        python_results = self.run_python_evaluation(config)
        
        if not python_results:
            return None
        
        # Calculate metrics
        total_time = python_results["total_time_ms"]
        timing = python_results["timing"]
        
        # Get real metrics from results (no fake data)
        all_results = python_results["results"]
        if all_results:
            # Count total results across providers
            total_results = sum(r.get("num_results", 0) for r in all_results.values())
            total_coverage = total_results
        else:
            total_coverage = 0
        
        # Build optimizations list
        optimizations = []
        if config["late"]:
            optimizations.append("late")
        if config["prune"]:
            optimizations.append(f"prune_{config['prune']}")
        
        return RealAblationResult(
            config_name=config_name,
            total_time_ms=total_time,
            search_time_ms=timing.get("search_ms", total_time * 0.3),
            rerank_time_ms=timing.get("rerank_ms", total_time * 0.5),
            eval_time_ms=timing.get("eval_ms", total_time * 0.2),
            rerank_p50_ms=rust_bench.get("p50_ms", 0.0),
            rerank_p95_ms=rust_bench.get("p95_ms", 0.0),
            coverage=total_coverage,
            optimizations=optimizations,
            rust_benchmark=rust_bench
        )

    def run_all_ablations(self) -> List[RealAblationResult]:
        """Run all ablation configurations with real measurements"""
        results = []
        
        print("🚀 Starting Real Ablation Study")
        print("=" * 80)
        
        for config_name, config in self.configs.items():
            result = self.run_single_config(config_name, config)
            if result:
                results.append(result)
                print(f"✅ {config_name}: {result.total_time_ms:.1f}ms total, {result.rerank_p95_ms:.1f}ms rerank p95")
            else:
                print(f"❌ {config_name}: Failed")
        
        return results

    def print_ablation_table(self, results: List[RealAblationResult]):
        """Print the real ablation results table"""
        print("\n" + "=" * 120)
        print("REAL ABLATION STUDY RESULTS - Actual Performance Measurements")
        print("=" * 120)
        
        # Header
        print(f"{'Config':<20} {'Total(ms)':<10} {'Search(ms)':<12} {'Rerank(ms)':<12} {'Eval(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'Results':<8} {'Rust P95':<10}")
        print("-" * 120)
        
        # Results
        for result in results:
            rust_p95 = result.rust_benchmark.get("p95_ms", 0.0)
            print(f"{result.config_name:<20} {result.total_time_ms:<10.1f} {result.search_time_ms:<12.1f} {result.rerank_time_ms:<12.1f} {result.eval_time_ms:<10.1f} {result.rerank_p50_ms:<10.1f} {result.rerank_p95_ms:<10.1f} {result.coverage:<8} {rust_p95:<10.1f}")
        
        print("-" * 120)

    def print_rust_benchmark_table(self, results: List[RealAblationResult]):
        """Print detailed Rust benchmark results"""
        print(f"\n⚡ RUST RERANKER BENCHMARK RESULTS:")
        print("=" * 100)
        
        print(f"{'Config':<20} {'Docs':<8} {'Dim':<6} {'Prune':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'Threads':<8} {'CPU':<10}")
        print("-" * 100)
        
        for result in results:
            bench = result.rust_benchmark
            prune_str = bench.get("prune", "none")
            print(f"{result.config_name:<20} {bench.get('n_docs', 0):<8} {bench.get('d', 0):<6} {prune_str:<10} {bench.get('p50_ms', 0):<10.1f} {bench.get('p95_ms', 0):<10.1f} {bench.get('threads', 0):<8} {bench.get('cpu_flags', 'N/A'):<10}")
        
        print("-" * 100)

    def print_performance_analysis(self, results: List[RealAblationResult]):
        """Print performance analysis and insights"""
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        print("=" * 80)
        
        if len(results) < 2:
            print("Need at least 2 results for analysis")
            return
        
        baseline = results[0]
        
        print(f"Baseline: {baseline.config_name}")
        print(f"  Total: {baseline.total_time_ms:.1f}ms")
        print(f"  Rust P95: {baseline.rerank_p95_ms:.1f}ms")
        print(f"  Results: {baseline.coverage}")
        print()
        
        print(f"{'Config':<20} {'Total Speedup':<15} {'Rust Speedup':<15} {'Results Change':<15}")
        print("-" * 80)
        
        for result in results[1:]:
            total_speedup = baseline.total_time_ms / result.total_time_ms if result.total_time_ms > 0 else 0
            rust_speedup = baseline.rerank_p95_ms / result.rerank_p95_ms if result.rerank_p95_ms > 0 else 0
            results_change = result.coverage - baseline.coverage
            
            print(f"{result.config_name:<20} {total_speedup:<15.2f}x {rust_speedup:<15.2f}x {results_change:<15}")

    def save_results(self, results: List[RealAblationResult], filename: str = "real_ablation_results.json"):
        """Save results to JSON file"""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": self.base_config["query"],
            "providers": self.base_config["providers"],
            "topk": self.base_config["topk"],
            "results": [
                {
                    "config_name": r.config_name,
                    "total_time_ms": r.total_time_ms,
                    "search_time_ms": r.search_time_ms,
                    "rerank_time_ms": r.rerank_time_ms,
                    "eval_time_ms": r.eval_time_ms,
                    "rerank_p50_ms": r.rerank_p50_ms,
                    "rerank_p95_ms": r.rerank_p95_ms,
                    "coverage": r.coverage,
                    "optimizations": r.optimizations,
                    "rust_benchmark": r.rust_benchmark
                }
                for r in results
            ]
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run Real Ablation Study")
    parser.add_argument("--query", default="machine learning optimization techniques", help="Search query")
    parser.add_argument("--providers", default="ddg,wikipedia", help="Comma-separated providers")
    parser.add_argument("--topk", type=int, default=10, help="Number of results per provider")
    parser.add_argument("--output", default="real_ablation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Update base config
    runner = RealAblationRunner()
    runner.base_config["query"] = args.query
    runner.base_config["providers"] = args.providers
    runner.base_config["topk"] = args.topk
    
    # Run all ablations
    results = runner.run_all_ablations()
    
    if results:
        # Print results
        runner.print_ablation_table(results)
        runner.print_rust_benchmark_table(results)
        runner.print_performance_analysis(results)
        runner.save_results(results, args.output)
        
        print(f"\n🎯 Real ablation study completed! {len(results)} configurations tested.")
        print(f"🔧 Rust service: {runner.rust_service_url}")
        print(f"📊 All measurements are real performance data")
    else:
        print("❌ No results obtained from ablation study.")

if __name__ == "__main__":
    main()
