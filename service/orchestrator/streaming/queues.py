"""Specialized queues for streaming pipeline."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SearchTask:
    """Search task data structure."""
    query_id: str
    query: str
    providers: List[str]
    timestamp: datetime
    status: str = "pending"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EmbeddingTask:
    """Embedding task data structure."""
    query_id: str
    query: str
    search_results: Dict[str, List[Any]]
    timestamp: datetime
    status: str = "pending"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RerankTask:
    """Reranking task data structure."""
    query_id: str
    query: str
    query_tokens: Any
    doc_tokens: Dict[str, List[Any]]
    timestamp: datetime
    status: str = "pending"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class JudgeTask:
    """Judging task data structure."""
    query_id: str
    query: str
    reranked_results: Dict[str, List[Any]]
    timestamp: datetime
    status: str = "pending"
    metadata: Optional[Dict[str, Any]] = None

class BaseQueue:
    """Base class for specialized queues."""
    
    def __init__(self, maxsize: int = 100):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.total_put = 0
        self.total_get = 0
        self.total_dropped = 0
    
    async def put(self, item: Any) -> None:
        """Put item in queue."""
        try:
            await self.queue.put(item)
            self.total_put += 1
        except asyncio.QueueFull:
            self.total_dropped += 1
            logger.warning(f"Queue full, dropping item")
    
    async def get(self) -> Any:
        """Get item from queue."""
        item = await self.queue.get()
        self.total_get += 1
        return item
    
    def qsize(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "size": self.qsize(),
            "maxsize": self.maxsize,
            "total_put": self.total_put,
            "total_get": self.total_get,
            "total_dropped": self.total_dropped,
            "utilization": self.qsize() / self.maxsize if self.maxsize > 0 else 0
        }

class SearchQueue(BaseQueue):
    """Queue for search tasks."""
    
    async def put_search_task(self, task: SearchTask) -> None:
        """Put search task in queue."""
        await self.put(task)
        logger.debug(f"Added search task {task.query_id} to queue")
    
    async def get_search_task(self) -> SearchTask:
        """Get search task from queue."""
        return await self.get()

class EmbeddingQueue(BaseQueue):
    """Queue for embedding tasks."""
    
    async def put_embedding_task(self, task: EmbeddingTask) -> None:
        """Put embedding task in queue."""
        await self.put(task)
        logger.debug(f"Added embedding task {task.query_id} to queue")
    
    async def get_embedding_task(self) -> EmbeddingTask:
        """Get embedding task from queue."""
        return await self.get()

class RerankQueue(BaseQueue):
    """Queue for reranking tasks."""
    
    async def put_rerank_task(self, task: RerankTask) -> None:
        """Put rerank task in queue."""
        await self.put(task)
        logger.debug(f"Added rerank task {task.query_id} to queue")
    
    async def get_rerank_task(self) -> RerankTask:
        """Get rerank task from queue."""
        return await self.get()

class JudgeQueue(BaseQueue):
    """Queue for judging tasks."""
    
    async def put_judge_task(self, task: JudgeTask) -> None:
        """Put judge task in queue."""
        await self.put(task)
        logger.debug(f"Added judge task {task.query_id} to queue")
    
    async def get_judge_task(self) -> JudgeTask:
        """Get judge task from queue."""
        return await self.get()
