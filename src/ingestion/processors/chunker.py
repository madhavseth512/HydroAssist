"""Text chunking with metadata preservation using LangChain."""

from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import setup_logger
from src.utils.metadata import create_chunk_id

logger = setup_logger(__name__)


class MetadataPreservingChunker:
    """Wrapper around RecursiveCharacterTextSplitter with metadata preservation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators (default: paragraph-aware)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators prioritize paragraph boundaries
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence breaks
                ", ",    # Clause breaks
                " ",     # Word breaks
                ""       # Character breaks
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )

        logger.info(f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}")

    def chunk_text(
        self,
        text: str,
        metadata: Dict,
        doc_id: str
    ) -> List[Document]:
        """
        Chunk text while preserving metadata.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            doc_id: Document identifier for creating chunk IDs

        Returns:
            List of LangChain Document objects with metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for doc_id: {doc_id}")
            return []

        # Create initial document
        doc = Document(page_content=text, metadata=metadata.copy())

        # Split into chunks
        chunks = self.splitter.split_documents([doc])

        # Add unique chunk IDs to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = create_chunk_id(doc_id, i)
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['doc_id'] = doc_id

        logger.info(f"Created {len(chunks)} chunks from {doc_id}")

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, any]]
    ) -> List[Document]:
        """
        Chunk multiple documents.

        Args:
            documents: List of dicts with 'text', 'metadata', and 'doc_id' keys

        Returns:
            List of chunked Document objects
        """
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            doc_id = doc.get('doc_id', 'unknown')

            chunks = self.chunk_text(text, metadata, doc_id)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks created: {len(all_chunks)} from {len(documents)} documents")

        return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict] = None,
    doc_id: str = "doc"
) -> List[Document]:
    """
    Convenience function for quick text chunking.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        metadata: Metadata to attach
        doc_id: Document identifier

    Returns:
        List of Document chunks
    """
    chunker = MetadataPreservingChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    metadata = metadata or {}

    return chunker.chunk_text(text, metadata, doc_id)
