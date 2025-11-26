# services/rag/chunking.py
"""
Module for chunking text content into smaller segments for embedding.
Supports simple fixed-size chunking with overlap.
Since all content is now normalized to str, simplified to text chunking.
"""

import json
from typing import List, Dict, Any
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of approximately `chunk_size` characters with overlap.
    
    Args:
        text (str): Input text to chunk.
        chunk_size (int): Target size of each chunk (default: 500).
        overlap (int): Overlap between consecutive chunks (default: 100).
    
    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# Removed chunk_csv and chunk_json since content is now always str
def chunk_json(data: Dict[str, Any]) -> List[str]:
    json_str = json.dumps(data, indent=2)
    return chunk_text(json_str)

def chunk_content(content: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Generic chunker for string content (normalized from all file types).
    
    Args:
        content (str): Parsed content as string.
        chunk_size (int): Chunk size parameter.
        overlap (int): Overlap for text-based chunking.
    
    Returns:
        List[str]: List of chunks.
    """
    return chunk_text(content, chunk_size, overlap)