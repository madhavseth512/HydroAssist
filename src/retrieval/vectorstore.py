"""ChromaDB vector store initialization and management."""

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from pathlib import Path
from typing import Optional, Dict, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStoreManager:
    """Manager for ChromaDB vector store operations."""

    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        collection_name: str = "hydrogeology_kb"
    ):
        """
        Initialize ChromaDB client and collection.

        Args:
            persist_directory: Path to persist vector database
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Collection] = None

        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "HydroAssist hydrogeology knowledge base",
                    "hnsw:space": "cosine"  # Cosine similarity
                }
            )

            logger.info(f"Collection '{self.collection_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique document IDs
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None,
        include: List[str] = None
    ) -> Dict:
        """
        Query the vector store.

        Args:
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Metadata filters (e.g., {"aquifer": "confined"})
            include: Fields to include in results

        Returns:
            Query results dictionary
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include
            )
            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def is_collection_empty(self) -> bool:
        """
        Check if collection is empty.

        Returns:
            True if collection has no documents
        """
        if not self.collection:
            return True

        try:
            count = self.collection.count()
            return count == 0
        except Exception as e:
            logger.error(f"Failed to check collection: {e}")
            return True

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {"total_chunks": 0, "sources": []}

        try:
            # Get all documents (just metadata for stats)
            results = self.collection.get(
                include=["metadatas"]
            )

            metadatas = results.get("metadatas", [])
            total = len(metadatas)

            # Count sources
            sources = {}
            aquifer_types = {}
            methods = {}

            for meta in metadatas:
                source = meta.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

                aquifer = meta.get("aquifer", "unknown")
                aquifer_types[aquifer] = aquifer_types.get(aquifer, 0) + 1

                method = meta.get("method", "unknown")
                methods[method] = methods.get(method, 0) + 1

            return {
                "total_chunks": total,
                "sources": sources,
                "aquifer_types": aquifer_types,
                "methods": methods
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_chunks": 0, "error": str(e)}

    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        try:
            logger.warning(f"Resetting collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self._initialize_client()
            logger.info("Collection reset complete")

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
