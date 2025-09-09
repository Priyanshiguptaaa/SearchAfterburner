"""Streaming pipeline for real-time search processing."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from .queues import SearchQueue, EmbeddingQueue, RerankQueue, JudgeQueue
from .workers import SearchWorker, EmbeddingWorker, RerankWorker, JudgeWorker

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for streaming pipeline."""
    max_concurrent_searches: int = 3
    max_concurrent_embeddings: int = 5
    max_concurrent_reranks: int = 2
    max_concurrent_judges: int = 2
    queue_size: int = 100
    batch_size: int = 10
    timeout_seconds: int = 30
    enable_backpressure: bool = True

class StreamingPipeline:
    """Asynchronous streaming pipeline for search processing."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize queues
        self.search_queue = SearchQueue(maxsize=self.config.queue_size)
        self.embedding_queue = EmbeddingQueue(maxsize=self.config.queue_size)
        self.rerank_queue = RerankQueue(maxsize=self.config.queue_size)
        self.judge_queue = JudgeQueue(maxsize=self.config.queue_size)
        
        # Initialize workers
        self.search_worker = SearchWorker(
            input_queue=self.search_queue,
            output_queue=self.embedding_queue,
            max_concurrent=self.config.max_concurrent_searches
        )
        
        self.embedding_worker = EmbeddingWorker(
            input_queue=self.embedding_queue,
            output_queue=self.rerank_queue,
            max_concurrent=self.config.max_concurrent_embeddings
        )
        
        self.rerank_worker = RerankWorker(
            input_queue=self.rerank_queue,
            output_queue=self.judge_queue,
            max_concurrent=self.config.max_concurrent_reranks
        )
        
        self.judge_worker = JudgeWorker(
            input_queue=self.judge_queue,
            max_concurrent=self.config.max_concurrent_judges
        )
        
        # Pipeline state
        self.is_running = False
        self.tasks = []
        self.results = {}
        self.callbacks = {}
        
        logger.info("Streaming pipeline initialized")
    
    async def start(self) -> None:
        """Start the streaming pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        
        # Start all workers
        self.tasks = [
            asyncio.create_task(self.search_worker.start()),
            asyncio.create_task(self.embedding_worker.start()),
            asyncio.create_task(self.rerank_worker.start()),
            asyncio.create_task(self.judge_worker.start()),
        ]
        
        logger.info("Streaming pipeline started")
    
    async def stop(self) -> None:
        """Stop the streaming pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Streaming pipeline stopped")
    
    async def submit_query(self, query: str, providers: List[str], 
                          query_id: Optional[str] = None) -> str:
        """Submit a query to the pipeline."""
        if not self.is_running:
            raise RuntimeError("Pipeline is not running")
        
        query_id = query_id or f"query_{datetime.now().timestamp()}"
        
        # Create search task
        search_task = {
            "query_id": query_id,
            "query": query,
            "providers": providers,
            "timestamp": datetime.now(),
            "status": "submitted"
        }
        
        # Submit to search queue
        await self.search_queue.put(search_task)
        
        logger.info(f"Submitted query {query_id}: {query}")
        return query_id
    
    async def get_result(self, query_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get result for a query."""
        timeout = timeout or self.config.timeout_seconds
        
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if query_id in self.results:
                return self.results[query_id]
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout waiting for result {query_id}")
        return None
    
    async def wait_for_completion(self, query_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for query completion and return result."""
        result = await self.get_result(query_id, timeout)
        if result is None:
            raise TimeoutError(f"Query {query_id} did not complete within timeout")
        return result
    
    def set_callback(self, event_type: str, callback: Callable) -> None:
        """Set callback for pipeline events."""
        self.callbacks[event_type] = callback
    
    async def _trigger_callback(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger callback for event."""
        if event_type in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(self.callbacks[event_type]):
                    await self.callbacks[event_type](data)
                else:
                    self.callbacks[event_type](data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
    
    async def process_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of queries."""
        query_ids = []
        
        # Submit all queries
        for query_data in queries:
            query_id = await self.submit_query(
                query_data["query"],
                query_data["providers"],
                query_data.get("query_id")
            )
            query_ids.append(query_id)
        
        # Wait for all to complete
        results = {}
        for query_id in query_ids:
            try:
                result = await self.wait_for_completion(query_id)
                results[query_id] = result
            except TimeoutError:
                logger.error(f"Query {query_id} timed out")
                results[query_id] = {"error": "timeout"}
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "is_running": self.is_running,
            "queue_sizes": {
                "search": self.search_queue.qsize(),
                "embedding": self.embedding_queue.qsize(),
                "rerank": self.rerank_queue.qsize(),
                "judge": self.judge_queue.qsize()
            },
            "worker_stats": {
                "search": self.search_worker.get_stats(),
                "embedding": self.embedding_worker.get_stats(),
                "rerank": self.rerank_worker.get_stats(),
                "judge": self.judge_worker.get_stats()
            },
            "completed_queries": len(self.results)
        }
