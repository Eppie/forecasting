import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from forecasting.config import settings

Base = declarative_base()


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Column[int] = Column(Integer, primary_key=True, index=True)
    timestamp: Column[datetime.datetime] = Column(
        DateTime(timezone=True), server_default=func.now()
    )
    question_text: Column[str] = Column(Text, nullable=False)
    predicted_probability: Column[float | None] = Column(Float)
    rationale_text: Column[str | None] = Column(Text)
    llm_model_used: Column[str | None] = Column(String(255))

    resolved: Column[bool] = Column(Boolean, default=False)
    outcome: Column[bool | None] = Column(
        Boolean
    )  # True if event happened, False if not
    brier_score: Column[float | None] = Column(Float)

    # Relationship to cited sources
    cited_chunks = relationship("CitedChunkLink", back_populates="forecast")


class SourceDocument(Base):  # Represents an ingested document
    __tablename__ = "source_documents"
    id: Column[int] = Column(Integer, primary_key=True, index=True)
    source_name: Column[str] = Column(
        String(255), index=True
    )  # e.g., "SEC EDGAR", "Reuters"
    document_url: Column[str | None] = Column(String(1024), unique=True)
    original_content_hash: Column[str] = Column(
        String(64), unique=True, index=True
    )  # SHA256 of raw content
    processed_at: Column[datetime.datetime] = Column(
        DateTime(timezone=True), server_default=func.now()
    )
    metadata_: Column[dict | None] = Column(JSON)  # e.g., publication date, title

    chunks = relationship("DocumentChunk", back_populates="source_document")


class DocumentChunk(
    Base
):  # Represents a chunk of a SourceDocument, stored in Vector DB
    __tablename__ = "document_chunks"
    id: Column[int] = Column(Integer, primary_key=True, index=True)  # Internal DB ID
    vector_id: Column[str | int] = Column(
        String(36), unique=True, index=True
    )  # ID in Qdrant (UUID) or other vector store
    source_document_id: Column[int] = Column(Integer, ForeignKey("source_documents.id"))
    text_content: Column[str] = Column(Text, nullable=False)
    chunk_metadata: Column[dict | None] = Column(
        JSON
    )  # e.g., page number, paragraph_id

    source_document = relationship("SourceDocument", back_populates="chunks")
    cited_in_forecasts = relationship("CitedChunkLink", back_populates="document_chunk")


class CitedChunkLink(
    Base
):  # Association table for many-to-many between Forecasts and DocumentChunks
    __tablename__ = "cited_chunk_links"
    forecast_id: Column[int] = Column(
        Integer, ForeignKey("forecasts.id"), primary_key=True
    )
    document_chunk_id: Column[int] = Column(
        Integer, ForeignKey("document_chunks.id"), primary_key=True
    )
    citation_label: Column[str] = Column(String(10))  # e.g., "[A]", "[B]"

    forecast = relationship("Forecast", back_populates="cited_chunks")
    document_chunk = relationship("DocumentChunk", back_populates="cited_in_forecasts")


engine = create_engine(str(settings.DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_db_and_tables():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
