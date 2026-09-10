"""Query and document embeddings, via ChromaDB's ONNX build of all-MiniLM-L6-v2.

Why ONNX rather than sentence-transformers:

The knowledge base was never embedded by sentence-transformers. The ingestion
scripts call `collection.add(documents=...)` without supplying vectors, on a
collection created without an embedding function, so ChromaDB embedded every
stored chunk with its own default — an ONNX export of all-MiniLM-L6-v2. The
collection's persisted configuration records exactly that
(`embedding_function: default`), and re-embedding stored chunks with ChromaDB's
DefaultEmbeddingFunction reproduces the stored vectors to 7e-9.

Only the *queries* went through sentence-transformers, which pulled in torch and
transformers — about 1.7 GB of the virtual environment, to embed one short string
per question. The two implementations agree to six decimal places on this corpus,
so it never produced wrong answers, but it meant index and query were embedded by
different code paths. Using the ONNX function for queries makes them identical to
the index by construction, and removes torch entirely. That is what makes the app
fit the memory of a free-tier host at all.

The model file (~80 MB) is downloaded by ChromaDB on first use into
~/.cache/chroma/onnx_models and reused afterwards.
"""

import time
from functools import lru_cache
from typing import List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# The only model the ONNX path provides, and the one the index was built with.
SUPPORTED_MODEL = "all-MiniLM-L6-v2"


def _normalise_model_name(model_name: str) -> str:
    """Accept both "all-MiniLM-L6-v2" and the sentence-transformers style
    "sentence-transformers/all-MiniLM-L6-v2" found in the existing configs."""
    return (model_name or SUPPORTED_MODEL).split("/")[-1]


class OnnxEmbeddings:
    """Drop-in for the LangChain embeddings interface the callers already use:
    `embed_query(text) -> list[float]` and `embed_documents(texts) -> list[list]`.
    """

    def __init__(self, model_name: str = SUPPORTED_MODEL):
        name = _normalise_model_name(model_name)
        if name != SUPPORTED_MODEL:
            # Silently substituting a different model would make every query
            # vector incomparable with the stored index.
            raise ValueError(
                f"Embedding model {model_name!r} is not available. The knowledge "
                f"base was built with {SUPPORTED_MODEL}, which is the only model "
                "the ONNX embedder provides; querying with any other model would "
                "produce vectors incomparable with the stored index."
            )
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.model_name = name
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [list(map(float, v)) for v in self._fn(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class EmbeddingsManager:
    """Process-wide singleton, so the ONNX session is created once."""

    _instance: Optional["EmbeddingsManager"] = None
    _model: Optional[OnnxEmbeddings] = None
    _model_name: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embeddings(self, model_name: str = SUPPORTED_MODEL) -> OnnxEmbeddings:
        """Get or create the embedder. Returns the cached instance when the same
        model is requested again."""
        name = _normalise_model_name(model_name)
        if self._model is not None and self._model_name == name:
            return self._model

        logger.info(f"Loading ONNX embedding model: {name}")
        start = time.time()
        self._model = OnnxEmbeddings(name)
        self._model_name = name
        logger.info(f"Embedding model ready in {time.time() - start:.2f}s")
        return self._model

    @lru_cache(maxsize=100)
    def embed_query_cached(self, query: str,
                           model_name: str = SUPPORTED_MODEL) -> tuple:
        """Embed a query with LRU caching. Returns a tuple so the cached value
        cannot be mutated by a caller."""
        return tuple(self.get_embeddings(model_name).embed_query(query))

    def embed_query(self, query: str) -> List[float]:
        if self._model is None:
            raise RuntimeError(
                "Embeddings model not initialised. Call get_embeddings() first.")
        return self._model.embed_query(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError(
                "Embeddings model not initialised. Call get_embeddings() first.")
        return self._model.embed_documents(texts)


# Global instance for easy import
embeddings_manager = EmbeddingsManager()
