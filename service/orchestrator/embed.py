"""Embedding utilities for the evaluation system."""

import numpy as np
from typing import List, Tuple
import logging
import time
import re

logger = logging.getLogger(__name__)

class Embedder:
    """Handles text embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder with a local model."""
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        
        try:
            # Try to import and use sentence-transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model loaded in {load_time:.2f}ms, dimension: {self.dimension}")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embedder")
            self.model = None
            self.dimension = 384  # Standard dimension for all-MiniLM-L6-v2
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.warning("Falling back to mock embedder")
            self.model = None
            self.dimension = 384
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalized vectors."""
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        try:
            if self.model is not None:
                # Get embeddings from sentence-transformers
                embeddings = self.model.encode(texts, convert_to_numpy=True)
            else:
                # Fallback to mock embeddings
                embeddings = self._generate_mock_embeddings(len(texts))
            
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
            
            embed_time = (time.time() - start_time) * 1000
            logger.info(f"Embedded {len(texts)} texts in {embed_time:.2f}ms")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Fallback to mock embeddings
            return self._generate_mock_embeddings(len(texts))
    
    def _generate_mock_embeddings(self, num_texts: int) -> np.ndarray:
        """Generate mock embeddings for testing."""
        # Generate random embeddings
        embeddings = np.random.randn(num_texts, self.dimension).astype(np.float32)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        logger.info(f"Generated {num_texts} mock embeddings")
        return embeddings
    
    def chunk_to_tokens(self, text: str, max_tokens: int = 100) -> List[np.ndarray]:
        """Split text into sentence-level tokens and embed each."""
        if not text or not text.strip():
            return []
        
        # Simple sentence splitting
        sentences = self._split_into_sentences(text)
        
        # Limit number of sentences
        sentences = sentences[:max_tokens]
        
        if not sentences:
            return []
        
        # Embed each sentence
        embeddings = self.embed_texts(sentences)
        
        # Convert to list of numpy arrays
        token_embeddings = [embeddings[i] for i in range(len(sentences))]
        
        logger.info(f"Created {len(token_embeddings)} token embeddings from {len(sentences)} sentences")
        
        return token_embeddings
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def embed_query_tokens(self, query: str) -> List[np.ndarray]:
        """Embed query as tokens for late-interaction scoring."""
        return self.chunk_to_tokens(query, max_tokens=20)
    
    def embed_document_tokens(self, text: str) -> List[np.ndarray]:
        """Embed document as tokens for late-interaction scoring."""
        return self.chunk_to_tokens(text, max_tokens=100)

class MockEmbedder:
    """Mock embedder for testing without sentence-transformers."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.info(f"Using mock embedder with dimension {dimension}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate random normalized embeddings."""
        if not texts:
            return np.array([])
        
        # Generate random embeddings
        embeddings = np.random.randn(len(texts), self.dimension).astype(np.float32)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        logger.info(f"Generated {len(texts)} mock embeddings")
        return embeddings
    
    def chunk_to_tokens(self, text: str, max_tokens: int = 100) -> List[np.ndarray]:
        """Split text and generate mock token embeddings."""
        sentences = self._split_into_sentences(text)
        sentences = sentences[:max_tokens]
        
        if not sentences:
            return []
        
        embeddings = self.embed_texts(sentences)
        return [embeddings[i] for i in range(len(sentences))]
    
    def embed_query_tokens(self, query: str) -> List[np.ndarray]:
        """Embed query as tokens."""
        return self.chunk_to_tokens(query, max_tokens=20)
    
    def embed_document_tokens(self, text: str) -> List[np.ndarray]:
        """Embed document as tokens."""
        return self.chunk_to_tokens(text, max_tokens=100)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

def get_embedder(use_local: bool = True, model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Get an embedder instance."""
    if use_local:
        try:
            return Embedder(model_name)
        except Exception as e:
            logger.warning(f"Failed to load local model, using mock: {e}")
            return MockEmbedder()
    else:
        return MockEmbedder()
