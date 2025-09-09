#!/usr/bin/env python3
"""
Performance metrics and timing instrumentation for SearchEval Pro.
Records p50/p90/p95 per stage and generates trace.json files.
"""

import json
import time
import statistics
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StageTiming:
    """Timing data for a single stage execution."""
    start_time: float
    end_time: float
    duration_ms: float
    stage_name: str
    metadata: Dict[str, Any] = None

@dataclass
class StageStats:
    """Statistical summary for a stage across multiple runs."""
    count: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float

class MetricsCollector:
    """Collects and analyzes performance metrics across stages."""
    
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.stage_timings: List[StageTiming] = []
        self.stage_stats: Dict[str, StageStats] = {}
        self.config_flags: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Create runs directory
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)
        
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized metrics collector for run {self.run_id}")
    
    def start_stage(self, stage_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start timing a stage and return a stage ID."""
        stage_id = f"{stage_name}_{len(self.stage_timings)}"
        timing = StageTiming(
            start_time=time.time(),
            end_time=0.0,
            duration_ms=0.0,
            stage_name=stage_name,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.stage_timings.append(timing)
        
        return stage_id
    
    def end_stage(self, stage_id: str, metadata: Dict[str, Any] = None):
        """End timing a stage."""
        end_time = time.time()
        
        with self._lock:
            for timing in self.stage_timings:
                if timing.stage_name in stage_id:  # Match by stage name
                    timing.end_time = end_time
                    timing.duration_ms = (end_time - timing.start_time) * 1000
                    if metadata:
                        timing.metadata.update(metadata)
                    break
    
    def record_stage(self, stage_name: str, duration_ms: float, metadata: Dict[str, Any] = None):
        """Record a completed stage timing."""
        timing = StageTiming(
            start_time=0.0,
            end_time=0.0,
            duration_ms=duration_ms,
            stage_name=stage_name,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.stage_timings.append(timing)
    
    def set_config_flags(self, flags: Dict[str, Any]):
        """Set configuration flags for this run."""
        self.config_flags = flags.copy()
    
    def set_quality_metrics(self, metrics: Dict[str, Any]):
        """Set quality metrics for this run."""
        self.quality_metrics = metrics.copy()
    
    def compute_stage_stats(self) -> Dict[str, StageStats]:
        """Compute statistical summaries for each stage."""
        stage_durations = defaultdict(list)
        
        with self._lock:
            for timing in self.stage_timings:
                stage_durations[timing.stage_name].append(timing.duration_ms)
        
        for stage_name, durations in stage_durations.items():
            if not durations:
                continue
                
            durations.sort()
            count = len(durations)
            
            stats = StageStats(
                count=count,
                p50_ms=statistics.quantiles(durations, n=2)[0] if count > 1 else durations[0],
                p90_ms=statistics.quantiles(durations, n=10)[8] if count > 1 else durations[0],
                p95_ms=statistics.quantiles(durations, n=20)[18] if count > 1 else durations[0],
                p99_ms=statistics.quantiles(durations, n=100)[98] if count > 1 else durations[0],
                min_ms=min(durations),
                max_ms=max(durations),
                mean_ms=statistics.mean(durations),
                std_ms=statistics.stdev(durations) if count > 1 else 0.0
            )
            
            self.stage_stats[stage_name] = stats
        
        return self.stage_stats
    
    def get_total_duration(self) -> float:
        """Get total duration of all stages."""
        if not self.stage_timings:
            return 0.0
        
        start_times = [t.start_time for t in self.stage_timings if t.start_time > 0]
        end_times = [t.end_time for t in self.stage_timings if t.end_time > 0]
        
        if not start_times or not end_times:
            return sum(t.duration_ms for t in self.stage_timings)
        
        return (max(end_times) - min(start_times)) * 1000
    
    def save_trace(self) -> Path:
        """Save trace data to runs/<run_id>/trace.json."""
        self.compute_stage_stats()
        
        trace_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config_flags": self.config_flags,
            "quality_metrics": self.quality_metrics,
            "total_duration_ms": self.get_total_duration(),
            "stage_stats": {name: asdict(stats) for name, stats in self.stage_stats.items()},
            "stage_timings": [asdict(timing) for timing in self.stage_timings]
        }
        
        trace_file = self.run_dir / "trace.json"
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        logger.info(f"Saved trace data to {trace_file}")
        return trace_file
    
    def print_summary(self):
        """Print a summary of performance metrics."""
        self.compute_stage_stats()
        
        print(f"\n📊 Performance Summary for Run {self.run_id}")
        print("=" * 80)
        
        # Stage performance table
        print(f"{'Stage':<15} {'Count':<6} {'P50ms':<8} {'P90ms':<8} {'P95ms':<8} {'P99ms':<8} {'Meanms':<8}")
        print("-" * 80)
        
        for stage_name, stats in self.stage_stats.items():
            print(f"{stage_name:<15} {stats.count:<6} {stats.p50_ms:<8.1f} {stats.p90_ms:<8.1f} "
                  f"{stats.p95_ms:<8.1f} {stats.p99_ms:<8.1f} {stats.mean_ms:<8.1f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<15} {'':<6} {'':<8} {'':<8} {self.get_total_duration():<8.1f} {'':<8} {'':<8}")
        print("=" * 80)
        
        # Quality metrics
        if self.quality_metrics:
            print(f"\n🎯 Quality Metrics:")
            for metric, value in self.quality_metrics.items():
                print(f"  {metric}: {value}")

class StageTimer:
    """Context manager for timing individual stages."""
    
    def __init__(self, collector: MetricsCollector, stage_name: str, metadata: Dict[str, Any] = None):
        self.collector = collector
        self.stage_name = stage_name
        self.metadata = metadata or {}
        self.stage_id = None
    
    def __enter__(self):
        self.stage_id = self.collector.start_stage(self.stage_name, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stage_id:
            self.collector.end_stage(self.stage_id, self.metadata)

def create_metrics_collector(run_id: Optional[str] = None) -> MetricsCollector:
    """Create a new metrics collector instance."""
    return MetricsCollector(run_id)

def load_trace(trace_file: Path) -> Dict[str, Any]:
    """Load trace data from a JSON file."""
    with open(trace_file, 'r') as f:
        return json.load(f)

def compare_traces(trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two trace files and return deltas."""
    stats1 = trace1.get('stage_stats', {})
    stats2 = trace2.get('stage_stats', {})
    
    deltas = {}
    for stage in set(stats1.keys()) | set(stats2.keys()):
        if stage in stats1 and stage in stats2:
            s1, s2 = stats1[stage], stats2[stage]
            deltas[stage] = {
                'p50_delta_ms': s2['p50_ms'] - s1['p50_ms'],
                'p90_delta_ms': s2['p90_ms'] - s1['p90_ms'],
                'p95_delta_ms': s2['p95_ms'] - s1['p95_ms'],
                'p99_delta_ms': s2['p99_ms'] - s1['p99_ms'],
                'total_delta_ms': s2['mean_ms'] - s1['mean_ms']
            }
    
    return deltas
