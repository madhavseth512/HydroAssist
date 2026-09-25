"""Searching HydroAssist's own documentation.

The consultant retrieves from a library of hydrogeology papers. Asked where the
pumping rate goes, it searched Theis (1935) for an answer that was never in it.
The guide searches this instead: `docs/help/`, indexed by
`scripts/ingest_help.py` into its own collection.

Separate from the papers on purpose. Mixed in, a help paragraph would compete
with Hantush on a question about leakage, and a question about the interface
would come back citing 1955.
"""

from typing import Dict, List, Optional

from src.core.config import AppConfig
from src.retrieval.embeddings import embeddings_manager
from src.retrieval.vectorstore import VectorStoreManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

COLLECTION = "hydroassist_help"

# Help is 32 chunks, so a low bar returns something for almost any phrasing.
# The guide is told to say when the passages do not answer the question, which
# is the check that matters rather than the threshold.
DEFAULT_TOP_K = 4


class HelpRetriever:
    """The help index, or nothing at all if it was never built."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.embedding_model = embeddings_manager.get_embeddings(
            config.retrieval.embedding_model)
        self.available = False
        self.store: Optional[VectorStoreManager] = None

        try:
            self.store = VectorStoreManager(
                persist_directory=config.vectorstore.persist_directory,
                collection_name=COLLECTION,
                read_only=True,
            )
            self.available = self.store.collection.count() > 0
        except Exception as error:
            # A missing help index must not stop the application from starting:
            # the calculators and the consultant do not depend on it.
            logger.warning(f"Help index unavailable ({error}); the guide will "
                           f"answer from the workspace alone.")

        if self.available:
            logger.info(f"Help index: {self.store.collection.count()} chunks")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Passages of help relevant to a question, best first."""
        if not self.available or not query:
            return []

        try:
            embedding = self.embedding_model.embed_query(query)
            results = self.store.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            logger.error(f"Help retrieval failed: {error}")
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"content": document, "metadata": metadata, "score": 1.0 - distance}
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
