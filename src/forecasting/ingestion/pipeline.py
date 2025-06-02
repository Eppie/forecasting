# src/forecasting/ingestion/pipeline.py
import uuid

import numpy as np
from sqlalchemy.orm import Session

from forecasting.db.models import (
    DocumentChunk as DBORMChunk,
)
from forecasting.db.models import (
    SourceDocument,
)  # DB ORM model
from forecasting.db.session import get_session_context
from forecasting.ingestion.chunker import NaiveTextChunker  # Or your chosen chunker
from forecasting.ingestion.embedder import SentenceEmbedderABC
from forecasting.ingestion.fetchers import BaseFetcher
from forecasting.ingestion.models import (
    TextChunk as IngestionTextChunk,
)  # Your Pydantic model for chunked text
from forecasting.ingestion.parser import DocumentParser
from forecasting.ingestion.vector_store import (
    ChunkData as VectorStoreChunkData,
)
from forecasting.ingestion.vector_store import (
    VectorStoreABC,
)  # Qdrant's Pydantic model


class IngestionPipeline:
    def __init__(
        self,
        fetchers: list[BaseFetcher],
        parser: DocumentParser,
        chunker: NaiveTextChunker,  # Adjust type if using a different chunker
        embedder: SentenceEmbedderABC,
        vector_store: VectorStoreABC,
    ):
        self.fetchers = fetchers
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

        # Ensure vector store collection exists with the correct dimension
        # This is also done in main.py, but good to have it here if pipeline is run independently
        try:
            dim = self.embedder.get_embedding_dimension()
            self.vector_store.ensure_collection_exists(vector_size=dim)
        except Exception as e:
            print(
                f"Error ensuring vector collection exists during IngestionPipeline init: {e}"
            )
            # Depending on severity, you might want to raise this or handle it.

    def run_single_fetcher(self, fetcher: BaseFetcher, db: Session):
        print(f"Running ingestion for fetcher: {fetcher.__class__.__name__}...")
        raw_docs = fetcher.fetch_latest()
        processed_count = 0
        for raw_doc in raw_docs:
            # 1. Parse raw document
            parsed_doc = self.parser.parse(raw_doc)
            if not parsed_doc:
                print(
                    f"Skipping document {raw_doc.identifier} due to parsing failure or no text."
                )
                continue

            # 2. Check if document already processed using hash
            existing_doc = (
                db.query(SourceDocument)
                .filter(
                    SourceDocument.original_content_hash
                    == parsed_doc.original_content_hash
                )
                .first()
            )

            if existing_doc:
                print(
                    f"Document {raw_doc.identifier} (hash: {parsed_doc.original_content_hash}) already processed. Skipping."
                )
                continue

            print(
                f"Processing new document: {raw_doc.identifier} from {raw_doc.source_name}"
            )

            # 3. Store SourceDocument in relational DB
            db_source_doc = SourceDocument(
                source_name=parsed_doc.source_name,
                document_url=(
                    parsed_doc.identifier if "http" in parsed_doc.identifier else None
                ),
                original_content_hash=parsed_doc.original_content_hash,
                metadata_=parsed_doc.metadata,  # Ensure your DB model uses metadata_ or adjust here
            )
            db.add(db_source_doc)
            db.flush()  # To get db_source_doc.id

            # 4. Chunk document (using IngestionTextChunk Pydantic model)
            text_chunks_from_chunker: list[IngestionTextChunk] = (
                self.chunker.chunk_parsed_document(parsed_doc, db_source_doc.id)
            )

            if not text_chunks_from_chunker:
                print(
                    f"No chunks generated for document {parsed_doc.identifier}. Committing source document entry only."
                )
                db.commit()  # Commit source doc even if no chunks, or rollback if that's desired
                continue

            # 5. Embed chunks
            chunk_texts_to_embed = [tc.text for tc in text_chunks_from_chunker]
            embeddings_np: list[np.ndarray] = self.embedder.embed_texts(
                chunk_texts_to_embed
            )

            vector_store_points_to_upsert: list[
                tuple[VectorStoreChunkData, np.ndarray]
            ] = []
            db_orm_chunks_to_add: list[DBORMChunk] = []

            for i, ingested_text_chunk in enumerate(text_chunks_from_chunker):
                # This ID is for Qdrant/VectorStore. It must be unique.
                # ChunkData Pydantic model can generate one by default.
                qdrant_point_id = str(uuid.uuid4())

                # Prepare data for Qdrant's Pydantic model (VectorStoreChunkData)
                # This model is defined in your vector_store.py
                data_for_qdrant_payload = VectorStoreChunkData(
                    id=qdrant_point_id,  # Critical: Qdrant uses this as its point ID
                    text=ingested_text_chunk.text,
                    document_db_id=ingested_text_chunk.source_document_db_id,
                    # This is SourceDocument.id from our PG DB
                    source_name=ingested_text_chunk.source_name_for_display,
                    source_url=ingested_text_chunk.source_identifier_for_display,
                    extra_metadata=ingested_text_chunk.chunk_specific_metadata,
                )
                vector_store_points_to_upsert.append(
                    (data_for_qdrant_payload, embeddings_np[i])
                )

                # Prepare data for our relational DB's DocumentChunk ORM model
                db_orm_chunk = DBORMChunk(
                    vector_id=qdrant_point_id,  # Store Qdrant's ID for potential linking/debugging
                    source_document_id=ingested_text_chunk.source_document_db_id,
                    text_content=ingested_text_chunk.text,
                    chunk_metadata=ingested_text_chunk.chunk_specific_metadata,
                )
                db_orm_chunks_to_add.append(db_orm_chunk)

            # 6. Upsert to vector store
            if vector_store_points_to_upsert:
                self.vector_store.upsert_chunks(vector_store_points_to_upsert)

            # 7. Add DocumentChunk ORM entries to relational DB
            if db_orm_chunks_to_add:
                db.add_all(db_orm_chunks_to_add)

            db.commit()  # Commit this document and its chunks
            processed_count += 1
            print(
                f"Successfully processed and stored document: {parsed_doc.identifier} with {len(text_chunks_from_chunker)} chunks."
            )
        print(
            f"Finished ingestion for fetcher: {fetcher.__class__.__name__}. Processed {processed_count} new documents."
        )

    def run_all(self):
        print("Starting full ingestion pipeline for all fetchers...")
        with (
            get_session_context() as db
        ):  # Single session for the whole run_all process
            for fetcher in self.fetchers:
                self.run_single_fetcher(fetcher, db)
        print("Full ingestion pipeline finished.")
