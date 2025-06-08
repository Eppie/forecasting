import pytest
from datetime import datetime
from forecasting.ingestion.models import RawDocument, ParsedDocument, TextChunk
from forecasting.ingestion.chunker import NaiveTextChunker


def create_test_parsed_document(text: str, source_name: str = "test_source", doc_id: str = "test_id") -> ParsedDocument:
    """Helper function to create a test ParsedDocument."""
    return ParsedDocument(
        source_name=source_name,
        identifier=doc_id,
        cleaned_text=text,
        metadata={"title": "Test Document"},
        original_content_hash="test_hash_123"
    )


def test_chunker_initialization():
    """Test that the chunker initializes with correct parameters."""
    chunker = NaiveTextChunker(chunk_size_chars=1000, chunk_overlap_chars=100)
    assert chunker.chunk_size_chars == 1000
    assert chunker.chunk_overlap_chars == 100


def test_empty_document():
    """Test chunking an empty document returns no chunks."""
    chunker = NaiveTextChunker(chunk_size_chars=100, chunk_overlap_chars=10)
    doc = create_test_parsed_document("")
    chunks = chunker.chunk_parsed_document(doc, source_document_db_id=1)
    assert len(chunks) == 0


def test_single_chunk():
    """Test that a short document fits in a single chunk."""
    text = "This is a short document that should fit in one chunk."
    chunker = NaiveTextChunker(chunk_size_chars=100, chunk_overlap_chars=10)
    doc = create_test_parsed_document(text)
    chunks = chunker.chunk_parsed_document(doc, source_document_db_id=1)
    
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].source_document_db_id == 1
    assert chunks[0].source_name_for_display == "Test Document"
    assert chunks[0].chunk_specific_metadata["char_start_index"] == 0


def test_multiple_chunks_no_overlap():
    """Test chunking with multiple chunks and no overlap."""
    text = "a" * 50 + "b" * 50  # 100 characters total
    chunker = NaiveTextChunker(chunk_size_chars=50, chunk_overlap_chars=0)
    doc = create_test_parsed_document(text)
    chunks = chunker.chunk_parsed_document(doc, source_document_db_id=1)
    
    assert len(chunks) == 2
    assert chunks[0].text == "a" * 50
    assert chunks[1].text == "b" * 50
    assert chunks[0].chunk_specific_metadata["char_start_index"] == 0
    assert chunks[1].chunk_specific_metadata["char_start_index"] == 50


def test_chunk_metadata():
    """Test that chunk metadata is correctly set."""
    text = "a" * 150
    doc = ParsedDocument(
        source_name="test_source",
        identifier="test_id",
        cleaned_text=text,
        metadata={"title": "Test Title"},
        original_content_hash="test_hash"
    )
    
    chunker = NaiveTextChunker(chunk_size_chars=100, chunk_overlap_chars=20)
    chunks = chunker.chunk_parsed_document(doc, source_document_db_id=42)
    
    assert len(chunks) == 2
    assert chunks[0].source_document_db_id == 42
    assert chunks[0].source_name_for_display == "Test Title"
    assert chunks[0].source_identifier_for_display is None  # No HTTP in identifier
    assert chunks[0].chunk_specific_metadata["original_doc_hash"] == "test_hash"
    assert chunks[0].chunk_specific_metadata["char_start_index"] == 0
    assert chunks[1].chunk_specific_metadata["char_start_index"] == 80  # 100 - 20 overlap


def test_http_identifier():
    """Test that HTTP identifiers are properly set in chunks."""
    doc = ParsedDocument(
        source_name="test_source",
        identifier="http://example.com/doc123",
        cleaned_text="Test content",
        metadata={"title": "HTTP Test"},
        original_content_hash="hash123"
    )
    
    chunker = NaiveTextChunker(chunk_size_chars=100, chunk_overlap_chars=0)
    chunks = chunker.chunk_parsed_document(doc, source_document_db_id=1)
    
    assert chunks[0].source_identifier_for_display == "http://example.com/doc123"
