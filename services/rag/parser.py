# services/rag/parser.py
"""
Module for parsing different file types: PDF, CSV, JSON.
Extracts text content from files as strings for further processing in RAG.
"""

import os
import csv
import json
from typing import Dict, Any
from PyPDF2 import PdfReader  # Requires: pip install PyPDF2


def parse_pdf(file_path: str) -> str:
    """
    Parse a PDF file and extract all text content.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Concatenated text from all pages.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    text = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()


def parse_csv(file_path: str) -> str:
    """
    Parse a CSV file into a string representation (headers + rows).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: CSV content as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        data = list(reader)

    if not headers or not data:
        return ""

    header_str = ",".join(headers)
    rows_str = "\n".join(
        [",".join([str(value) for value in row.values()]) for row in data]
    )
    return header_str + "\n" + rows_str


def parse_json(file_path: str) -> str:
    """
    Parse a JSON file into a string representation.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        str: JSON content as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return json.dumps(data, indent=0)  # Compact string without extra spaces


def parse_file(file_path: str) -> str:
    """
    Generic parser that dispatches based on file extension and always returns a string.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Parsed content as a string.

    Raises:
        ValueError: If unsupported file type.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".json":
        return parse_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


from pathlib import Path
from typing import List


def list_files(directory: str, exts: List[str] = None) -> List[Path]:
    """
    List all files in a directory recursively. Optionally filter by extensions.
    """
    p = Path(directory)
    if not p.exists():
        return []
    if exts:
        exts = [e.lower() for e in exts]
        return [f for f in p.rglob("*") if f.is_file() and f.suffix.lower()[1:] in exts]
    else:
        return [f for f in p.rglob("*") if f.is_file()]

# python -m services.rag.rag_tool --ingest --interactive
