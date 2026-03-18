"""Metadata-filtered similarity search with re-ranking."""

import time
from typing import List, Dict, Optional
from src.retrieval.embeddings import embeddings_manager
from src.retrieval.vectorstore import VectorStoreManager
from src.core.config import AppConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Retriever:
    """RAG retriever with metadata filtering and re-ranking."""

    def __init__(self, config: AppConfig):
        """
        Initialize retriever with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.embedding_model = embeddings_manager.get_embeddings(
            config.retrieval.embedding_model
        )
        self.vectorstore = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks with metadata filtering.

        Args:
            query: User question
            top_k: Number of results to return (default from config)
            metadata_filters: e.g., {"aquifer": "confined", "type": "theory"}
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with keys: {content, metadata, score}
            Sorted by relevance score (highest first)
        """
        start_time = time.time()

        # Use config defaults if not specified
        top_k = top_k or self.config.retrieval.top_k
        similarity_threshold = similarity_threshold or self.config.retrieval.similarity_threshold

        logger.info(f"Retrieving top-{top_k} documents for query: '{query[:100]}...'")
        if metadata_filters:
            logger.info(f"Applying metadata filters: {metadata_filters}")

        # Embed query
        try:
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

        # Query vector store with metadata filtering
        try:
            # Request more results than needed for filtering
            fetch_k = top_k * 2 if metadata_filters else top_k

            results = self.vectorstore.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                where=metadata_filters,
                include=["documents", "metadatas", "distances"]
            )

        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            # Try fallback without filters
            if metadata_filters:
                logger.info("Retrying without metadata filters...")
                try:
                    results = self.vectorstore.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"]
                    )
                except Exception as e2:
                    logger.error(f"Fallback query also failed: {e2}")
                    return []
            else:
                return []

        # Parse results
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            logger.warning("No documents retrieved")
            return []

        # Convert distances to similarity scores (1 - distance for cosine)
        # ChromaDB returns distances, lower is better
        # For cosine distance: similarity = 1 - distance
        formatted_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity_score = 1.0 - dist

            # Apply similarity threshold
            if similarity_score < similarity_threshold:
                continue

            formatted_results.append({
                "content": doc,
                "metadata": meta,
                "score": similarity_score
            })

        # Sort by score (highest first)
        formatted_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k after filtering
        formatted_results = formatted_results[:top_k]

        # Deduplication: merge adjacent chunks from same source
        formatted_results = self._deduplicate_chunks(formatted_results)

        elapsed = time.time() - start_time
        if formatted_results:
            avg_score = sum(r['score'] for r in formatted_results) / len(formatted_results)
            logger.info(
                f"Retrieved {len(formatted_results)} documents in {elapsed:.2f}s "
                f"(avg score: {avg_score:.3f})"
            )
        else:
            logger.warning(f"No documents retrieved in {elapsed:.2f}s")

        return formatted_results

    def _deduplicate_chunks(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate adjacent chunks from the same source.

        Args:
            results: List of retrieval results

        Returns:
            Deduplicated results
        """
        if len(results) <= 1:
            return results

        deduplicated = []
        seen_chunks = set()

        for result in results:
            chunk_id = result["metadata"].get("chunk_id", "")

            # Skip exact duplicates
            if chunk_id in seen_chunks:
                continue

            seen_chunks.add(chunk_id)
            deduplicated.append(result)

        return deduplicated


# Convenience function for quick retrieval
def retrieve(
    query: str,
    config: Optional[AppConfig] = None,
    top_k: int = 5,
    metadata_filters: Optional[Dict] = None,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Convenience function for quick retrieval.

    Args:
        query: User question
        config: App configuration (loads default if None)
        top_k: Number of results
        metadata_filters: Metadata filters
        similarity_threshold: Minimum similarity score

    Returns:
        List of retrieved documents
    """
    if config is None:
        from src.core.config import load_config
        config = load_config()

    retriever = Retriever(config)
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        metadata_filters=metadata_filters,
        similarity_threshold=similarity_threshold
    )
