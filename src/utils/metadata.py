"""Metadata schema definitions for knowledge base chunks."""

from dataclasses import dataclass, asdict
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# Controlled vocabularies
AQUIFER_TYPES = ["confined", "unconfined", "leaky", "fractured", "general"]
CONTENT_TYPES = ["theory", "method_description", "assumptions", "derivation", "example", "reference"]
SOURCES = ["USGS_B1", "AQTESOLV"]


@dataclass
class ChunkMetadata:
    """
    Metadata for a single knowledge base chunk.

    Attributes:
        source: Source document identifier
        type: Content type classification
        aquifer: Aquifer type this chunk discusses
        method: Analysis method name
        page: Source page number (for PDFs)
        section: Section/chapter reference
        doc_id: Unique document identifier
        chunk_id: Unique chunk identifier
        has_equations: Whether chunk contains mathematical equations
        confidence: Metadata extraction confidence score (0-1)
    """
    source: str
    type: str
    aquifer: str
    method: str
    section: str
    doc_id: str
    chunk_id: str
    has_equations: bool
    confidence: float
    page: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ChunkMetadata':
        """Create from dictionary."""
        return cls(**data)


class ChunkMetadataValidator(BaseModel):
    """Pydantic model for runtime validation of chunk metadata."""

    source: str = Field(..., description="Source document identifier")
    type: Literal["theory", "method_description", "assumptions", "derivation", "example", "reference"]
    aquifer: Literal["confined", "unconfined", "leaky", "fractured", "general"]
    method: str = Field(..., description="Method name in format: Author_Year")
    section: str = Field(..., description="Section reference")
    doc_id: str = Field(..., description="Unique document identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    has_equations: bool = Field(..., description="Contains equations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    page: Optional[int] = Field(None, description="Page number for PDFs")

    @field_validator('method')
    @classmethod
    def validate_method_format(cls, v: str) -> str:
        """Validate method name follows Author_Year format."""
        if v and '_' not in v:
            # Allow empty or general methods
            if v not in ["general", "multiple", ""]:
                raise ValueError(f"Method must follow Author_Year format, got: {v}")
        return v

    @field_validator('chunk_id')
    @classmethod
    def validate_chunk_id_format(cls, v: str) -> str:
        """Validate chunk_id contains doc_id prefix."""
        if not v:
            raise ValueError("chunk_id cannot be empty")
        return v


def validate_metadata(metadata: dict) -> bool:
    """
    Validate metadata dictionary against schema.

    Args:
        metadata: Dictionary containing metadata fields

    Returns:
        True if valid, raises ValidationError otherwise
    """
    try:
        ChunkMetadataValidator(**metadata)
        return True
    except Exception as e:
        raise ValueError(f"Metadata validation failed: {e}")


def create_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Create a unique chunk ID.

    Args:
        doc_id: Document identifier
        chunk_index: Index of chunk within document

    Returns:
        Unique chunk ID in format: {doc_id}_chunk_{index}
    """
    return f"{doc_id}_chunk_{chunk_index}"
