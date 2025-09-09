"""Worker processes for streaming pipeline."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from .queues import SearchQueue, EmbeddingQueue, RerankQueue, JudgeQueue, SearchTask, EmbeddingTask, RerankTask, JudgeTask

logger = logging.getLogger(__name__)

class BaseWorker:
    """Base class for pipeline workers."""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.is_running = False
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        self.errors = 0
    
    async def start(self) -> None:
        """Start the worker."""
        self.is_running = True
        logger.info(f"{self.__class__.__name__} started")
    
    async def stop(self) -> None:
        """Stop the worker."""
        self.is_running = False
        logger.info(f"{self.__class__.__name__} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        avg_time = self.total_processing_time / max(1, self.tasks_processed)
        return {
            "is_running": self.is_running,
            "tasks_processed": self.tasks_processed,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": avg_time,
            "errors": self.errors,
            "error_rate": self.errors / max(1, self.tasks_processed)
        }

class SearchWorker(BaseWorker):
    """Worker for search tasks."""
    
    def __init__(self, input_queue: SearchQueue, output_queue: EmbeddingQueue, max_concurrent: int = 3):
        super().__init__(max_concurrent)
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    async def start(self) -> None:
        """Start the search worker."""
        await super().start()
        
        while self.is_running:
            try:
                # Get task from input queue
                task = await self.input_queue.get_search_task()
                
                # Process task
                async with self.semaphore:
                    await self._process_search_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Search worker error: {e}")
                self.errors += 1
    
    async def _process_search_task(self, task: SearchTask) -> None:
        """Process a search task."""
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from ..providers import search_multiple_providers
            
            # Perform search
            search_results = search_multiple_providers(
                task.providers, 
                task.query, 
                max_results=50
            )
            
            # Create embedding task
            embedding_task = EmbeddingTask(
                query_id=task.query_id,
                query=task.query,
                search_results=search_results,
                timestamp=datetime.now(),
                status="ready"
            )
            
            # Send to embedding queue
            await self.output_queue.put_embedding_task(embedding_task)
            
            # Update stats
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed search task {task.query_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing search task {task.query_id}: {e}")
            self.errors += 1

class EmbeddingWorker(BaseWorker):
    """Worker for embedding tasks."""
    
    def __init__(self, input_queue: EmbeddingQueue, output_queue: RerankQueue, max_concurrent: int = 5):
        super().__init__(max_concurrent)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.embedder = None
    
    async def start(self) -> None:
        """Start the embedding worker."""
        await super().start()
        
        # Initialize embedder
        from ..embed import get_embedder
        self.embedder = get_embedder(use_local_embed=True)
        
        while self.is_running:
            try:
                # Get task from input queue
                task = await self.input_queue.get_embedding_task()
                
                # Process task
                async with self.semaphore:
                    await self._process_embedding_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
                self.errors += 1
    
    async def _process_embedding_task(self, task: EmbeddingTask) -> None:
        """Process an embedding task."""
        start_time = time.time()
        
        try:
            # Embed query
            query_tokens = self.embedder.embed_query_tokens(task.query)
            
            # Embed documents
            doc_tokens = {}
            for provider, results in task.search_results.items():
                doc_tokens[provider] = []
                for result in results:
                    text = f"{result.title} {result.snippet}"
                    doc_tokens[provider].append(self.embedder.embed_document_tokens(text))
            
            # Create rerank task
            rerank_task = RerankTask(
                query_id=task.query_id,
                query=task.query,
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
                timestamp=datetime.now(),
                status="ready"
            )
            
            # Send to rerank queue
            await self.output_queue.put_rerank_task(rerank_task)
            
            # Update stats
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed embedding task {task.query_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing embedding task {task.query_id}: {e}")
            self.errors += 1

class RerankWorker(BaseWorker):
    """Worker for reranking tasks."""
    
    def __init__(self, input_queue: RerankQueue, output_queue: JudgeQueue, max_concurrent: int = 2):
        super().__init__(max_concurrent)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.reranker_url = "http://localhost:8088"
    
    async def start(self) -> None:
        """Start the rerank worker."""
        await super().start()
        
        while self.is_running:
            try:
                # Get task from input queue
                task = await self.input_queue.get_rerank_task()
                
                # Process task
                async with self.semaphore:
                    await self._process_rerank_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rerank worker error: {e}")
                self.errors += 1
    
    async def _process_rerank_task(self, task: RerankTask) -> None:
        """Process a rerank task."""
        start_time = time.time()
        
        try:
            import httpx
            
            reranked_results = {}
            
            # Rerank each provider's results
            for provider, doc_tokens in task.doc_tokens.items():
                if not doc_tokens:
                    continue
                
                # Prepare reranking request
                payload = {
                    "q_tokens": task.query_tokens,
                    "d_tokens": doc_tokens,
                    "topk": 20,
                    "prune": {"q_max": 16, "d_max": 64, "method": "idf_norm"}
                }
                
                # Call reranking service
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.reranker_url}/rerank",
                        json=payload,
                        timeout=10.0
                    )
                    response.raise_for_status()
                    
                    rerank_data = response.json()
                    reranked_results[provider] = rerank_data.get("results", [])
            
            # Create judge task
            judge_task = JudgeTask(
                query_id=task.query_id,
                query=task.query,
                reranked_results=reranked_results,
                timestamp=datetime.now(),
                status="ready"
            )
            
            # Send to judge queue
            await self.output_queue.put_judge_task(judge_task)
            
            # Update stats
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed rerank task {task.query_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing rerank task {task.query_id}: {e}")
            self.errors += 1

class JudgeWorker(BaseWorker):
    """Worker for judging tasks."""
    
    def __init__(self, input_queue: JudgeQueue, max_concurrent: int = 2):
        super().__init__(max_concurrent)
        self.input_queue = input_queue
        self.judge = None
        self.results = {}
    
    async def start(self) -> None:
        """Start the judge worker."""
        await super().start()
        
        # Initialize judge
        from ..judge import get_judge
        self.judge = get_judge("heuristic")
        
        while self.is_running:
            try:
                # Get task from input queue
                task = await self.input_queue.get_judge_task()
                
                # Process task
                async with self.semaphore:
                    await self._process_judge_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Judge worker error: {e}")
                self.errors += 1
    
    async def _process_judge_task(self, task: JudgeTask) -> None:
        """Process a judge task."""
        start_time = time.time()
        
        try:
            # Evaluate results
            evaluations = {}
            for provider, results in task.reranked_results.items():
                if results:
                    evaluation = self.judge.evaluate(task.query, results)
                    evaluations[provider] = evaluation
                else:
                    evaluations[provider] = {"relevance_at_5": 0.0, "coverage": 0}
            
            # Create final result
            result = {
                "query_id": task.query_id,
                "query": task.query,
                "results": task.reranked_results,
                "evaluations": evaluations,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            # Store result
            self.results[task.query_id] = result
            
            # Update stats
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed judge task {task.query_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing judge task {task.query_id}: {e}")
            self.errors += 1
    
    def get_result(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a query."""
        return self.results.get(query_id)
