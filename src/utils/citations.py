"""Citation formatting utilities for HydroAssist."""

from typing import Dict, List, Optional


def format_citation(source: str, page: Optional[int] = None) -> str:
    """
    Format a single citation.

    Args:
        source: Source identifier (e.g., "USGS_B1", "AQTESOLV")
        page: Optional page number

    Returns:
        Formatted citation string

    Examples:
        >>> format_citation("USGS_B1", 42)
        '[USGS TWI Book 3-B1, p.42]'
        >>> format_citation("AQTESOLV")
        '[AQTESOLV]'
    """
    # Format source name for display
    if source == "USGS_B1":
        display_name = "USGS TWI Book 3-B1"
    elif source.startswith("AQTESOLV"):
        display_name = "AQTESOLV"
    else:
        display_name = source

    # Add page number if available
    if page is not None:
        return f"[{display_name}, p.{page}]"
    else:
        return f"[{display_name}]"


def format_multiple_citations(citations: List[Dict[str, any]]) -> str:
    """
    Format multiple citations into a single string.

    Args:
        citations: List of dicts with 'source' and optional 'page' keys

    Returns:
        Formatted citation string with multiple sources

    Examples:
        >>> format_multiple_citations([
        ...     {"source": "USGS_B1", "page": 42},
        ...     {"source": "AQTESOLV"}
        ... ])
        '[USGS TWI Book 3-B1, p.42; AQTESOLV]'
    """
    formatted = []
    for cite in citations:
        source = cite.get('source', 'Unknown')
        page = cite.get('page')

        if source == "USGS_B1":
            display_name = "USGS TWI Book 3-B1"
        elif source.startswith("AQTESOLV"):
            display_name = "AQTESOLV"
        else:
            display_name = source

        if page is not None:
            formatted.append(f"{display_name}, p.{page}")
        else:
            formatted.append(display_name)

    return f"[{'; '.join(formatted)}]"


def extract_citations_from_metadata(metadata_list: List[Dict]) -> str:
    """
    Extract and format citations from a list of document metadata.

    Args:
        metadata_list: List of metadata dicts from retrieved documents

    Returns:
        Formatted citation string
    """
    citations = []
    seen = set()  # Avoid duplicate citations

    for metadata in metadata_list:
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page')

        # Create unique key for deduplication
        key = f"{source}_{page}" if page else source

        if key not in seen:
            citations.append({'source': source, 'page': page})
            seen.add(key)

    if len(citations) == 1:
        return format_citation(citations[0]['source'], citations[0].get('page'))
    else:
        return format_multiple_citations(citations)
