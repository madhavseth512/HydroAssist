"""Metadata-filtered similarity search with re-ranking."""

from collections import Counter

import time
from typing import List, Dict, Optional
from src.retrieval.embeddings import embeddings_manager
from src.retrieval.vectorstore import VectorStoreManager
from src.core.config import AppConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Most chunks any one paper may contribute to an answer. Two leaves room for a
# document to make its point twice while still citing several sources: the
# consultant's value is that it argues from the literature, and an answer drawn
# entirely from one document does not.
MAX_PER_DOCUMENT = 2


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
        # Read-only: the store is committed to git and shipped to the host, and
        # merely opening a chromadb PersistentClient writes a lock row into it.
        # Retrieval reads a per-process snapshot instead; only the ingestion
        # scripts write to the real directory.
        self.vectorstore = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name,
            read_only=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None,
        max_per_document: int = MAX_PER_DOCUMENT
    ) -> List[Dict]:
        """
        Retrieve relevant chunks with metadata filtering.

        Args:
            query: User question
            top_k: Number of results to return (default from config)
            metadata_filters: e.g., {"aquifer": "confined", "type": "theory"}
            similarity_threshold: Minimum similarity score (0-1)
            max_per_document: Most chunks any one paper may contribute; 0 to
                allow a single document to fill the answer.

        Returns:
            List of dicts with keys: {content, metadata, score}
            Sorted by relevance score (highest first)
        """
        start_time = time.time()

        # Use config defaults if not specified
        # `is None` rather than `or`: an explicit threshold of 0.0 is a valid
        # request for "no filtering" and must not fall back to the config value.
        top_k = self.config.retrieval.top_k if top_k is None else top_k
        if similarity_threshold is None:
            similarity_threshold = self.config.retrieval.similarity_threshold

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
            # Fetch beyond top_k so the per-document cap and the threshold
            # have something to choose from; without this, capping just
            # returns fewer results.
            fetch_k = top_k * 2 if metadata_filters else top_k
            if max_per_document > 0:
                fetch_k = max(fetch_k, top_k * 4)

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

        # A filter that matches nothing is not an error, so the fallback above
        # never sees it — the query succeeds and returns an empty list, and the
        # consultant answers "I don't have enough information" about a corpus
        # that does. Asking about a method no paper is tagged with is exactly
        # that case. Retry without the filter instead.
        if not documents and metadata_filters:
            logger.info("Metadata filters matched nothing; retrying without them")
            try:
                results = self.vectorstore.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_k,
                    include=["documents", "metadatas", "distances"]
                )
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
            except Exception as e:
                logger.error(f"Unfiltered retry failed: {e}")
                return []

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

        # Then spread the answer across papers, before cutting to top_k.
        formatted_results = self._limit_per_document(formatted_results, top_k,
                                                     max_per_document)

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

    @staticmethod
    def _document_of(result: Dict) -> str:
        """The paper a chunk came from. doc_id is "<pdf stem>_page_<n>"."""
        doc_id = str(result.get("metadata", {}).get("doc_id", ""))
        return doc_id.rsplit("_page_", 1)[0] if "_page_" in doc_id else doc_id

    def _limit_per_document(self, results: List[Dict], top_k: int,
                            max_per_document: int) -> List[Dict]:
        """At most `max_per_document` chunks from any one paper.

        Without this, one long document can take every slot. The AQTESOLV
        manual is 503 of 4,017 chunks and lists assumptions in the phrasing
        people ask questions in, so "the assumptions of the Hantush-Jacob
        method" returned five chunks of manual and never Hantush — his paper
        ranked 15th, and Theis's own paper ranked 30th on the same question
        about Theis. The answer was drawn from one source and cited no
        literature.

        Chunks displaced this way are kept and appended, so a query that simply
        has no other source still returns as many results as before.
        """
        if max_per_document <= 0:
            return results

        kept, displaced, seen = [], [], Counter()
        for result in results:
            document = self._document_of(result)
            if seen[document] < max_per_document:
                seen[document] += 1
                kept.append(result)
            else:
                displaced.append(result)

        if len(kept) < top_k:
            # Backfilling appends chunks that outscore what is already kept, so
            # the list has to be put back in order: the consultant's prompt and
            # the sources panel both present these as a ranking.
            kept.extend(displaced[:top_k - len(kept)])
            kept.sort(key=lambda result: result["score"], reverse=True)
        return kept

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
