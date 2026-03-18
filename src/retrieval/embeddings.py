"""Embedding model manager with lazy loading and caching."""

import time
from functools import lru_cache
from typing import Optional, List
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingsManager:
    """
    Singleton manager for HuggingFace embeddings model.
    Provides lazy loading and caching to avoid reloading model.
    """

    _instance: Optional['EmbeddingsManager'] = None
    _model: Optional[HuggingFaceEmbeddings] = None
    _model_name: Optional[str] = None

    def __new__(cls):
        """Singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
        """
        Get or create embeddings model.

        Args:
            model_name: Name of the sentence-transformers model

        Returns:
            HuggingFaceEmbeddings instance
        """
        # Return cached model if same model requested
        if self._model is not None and self._model_name == model_name:
            logger.debug(f"Using cached embedding model: {model_name}")
            return self._model

        # Load new model
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()

        try:
            self._model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self._model_name = model_name

            elapsed = time.time() - start_time
            logger.info(f"Embedding model loaded successfully in {elapsed:.2f}s")

            return self._model

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @lru_cache(maxsize=100)
    def embed_query_cached(self, query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
        """
        Embed a query with LRU caching.

        Args:
            query: Query string to embed
            model_name: Name of the model to use

        Returns:
            Embedding vector
        """
        model = self.get_embeddings(model_name)
        return model.embed_query(query)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string (non-cached version).

        Args:
            query: Query string to embed

        Returns:
            Embedding vector
        """
        if self._model is None:
            raise RuntimeError("Embeddings model not initialized. Call get_embeddings() first.")
        return self._model.embed_query(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if self._model is None:
            raise RuntimeError("Embeddings model not initialized. Call get_embeddings() first.")
        return self._model.embed_documents(texts)


# Global instance for easy import
embeddings_manager = EmbeddingsManager()
