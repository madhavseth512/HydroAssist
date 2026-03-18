"""Pydantic models for validating ingested data."""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class ChunkData(BaseModel):
    """Validation model for a processed chunk."""

    content: str = Field(..., min_length=10, description="Chunk text content")
    metadata: dict = Field(..., description="Chunk metadata")
    doc_id: str = Field(..., min_length=1, description="Document identifier")

    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Ensure content is not too short or too long."""
        if len(v) < 10:
            raise ValueError("Content too short (< 10 chars)")
        if len(v) > 10000:
            raise ValueError("Content too long (> 10000 chars)")
        return v

    @field_validator('metadata')
    @classmethod
    def validate_metadata_fields(cls, v: dict) -> dict:
        """Ensure required metadata fields are present."""
        required_fields = ['source', 'type', 'aquifer', 'method', 'section', 'chunk_id']

        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required metadata field: {field}")

        return v


class DocumentData(BaseModel):
    """Validation model for a source document."""

    text: str = Field(..., min_length=100, description="Document text")
    metadata: dict = Field(..., description="Document metadata")
    doc_id: str = Field(..., min_length=1, description="Unique document ID")

    @field_validator('text')
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """Ensure document has substantial content."""
        if len(v) < 100:
            raise ValueError("Document text too short (< 100 chars)")
        return v


def validate_chunk(chunk_data: dict) -> bool:
    """
    Validate a single chunk.

    Args:
        chunk_data: Dictionary containing chunk data

    Returns:
        True if valid

    Raises:
        ValidationError if invalid
    """
    ChunkData(**chunk_data)
    return True


def validate_document(doc_data: dict) -> bool:
    """
    Validate a document.

    Args:
        doc_data: Dictionary containing document data

    Returns:
        True if valid

    Raises:
        ValidationError if invalid
    """
    DocumentData(**doc_data)
    return True


def validate_chunks_batch(chunks: List[dict]) -> tuple[int, List[str]]:
    """
    Validate a batch of chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Tuple of (valid_count, error_messages)
    """
    valid_count = 0
    errors = []

    for i, chunk in enumerate(chunks):
        try:
            validate_chunk(chunk)
            valid_count += 1
        except Exception as e:
            errors.append(f"Chunk {i}: {str(e)}")

    return valid_count, errors
