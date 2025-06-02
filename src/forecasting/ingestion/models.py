# src/forecasting/ingestion/models.py
import datetime
from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    source_name: str  # e.g., "SEC EDGAR", "Reuters News"
    identifier: (
        str  # e.g., URL, file path, API ID - should be unique per source content
    )
    content: str | bytes  # Raw content (text or binary for PDF)
    content_type: str  # "text/html", "application/pdf", "text/plain"
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # Publication date, title, etc.
    fetched_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class ParsedDocument(BaseModel):
    source_name: str
    identifier: str  # from RawDocument
    cleaned_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)  # from RawDocument
    original_content_hash: str  # SHA256 of raw_document.content


class TextChunk(BaseModel):  # For chunking before embedding
    text: str
    # These fields help link the chunk back to its origin within our system & for display
    source_document_db_id: int  # FK to our SourceDocument table in Postgres
    source_name_for_display: str  # e.g., specific 10-K report title, article title
    source_identifier_for_display: str | None = (
        None  # e.g. URL of the specific article/filing
    )
    chunk_specific_metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # e.g., original page number, paragraph
