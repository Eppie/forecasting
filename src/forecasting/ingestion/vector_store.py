import uuid
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Pydantic model for a chunk to be stored/retrieved
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from forecasting.config import settings


class ChunkData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Vector DB ID
    text: str
    document_db_id: int  # ID from our relational DB (SourceDocument or DocumentChunk)
    source_name: str  # e.g., "SEC 10-K Report for XYZ"
    source_url: str | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStoreABC(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks_with_embeddings: list[tuple[ChunkData, np.ndarray]]):
        pass

    @abstractmethod
    def similarity_search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> list[ChunkData]:
        pass

    @abstractmethod
    def ensure_collection_exists(
        self, vector_size: int, distance_metric: str = "Cosine"
    ):
        pass


class QdrantVectorStore(VectorStoreABC):
    def __init__(
        self,
        url: str = settings.VECTOR_DB_URL,
        api_key: str | None = None,
        collection_name: str = settings.VECTOR_DB_COLLECTION_NAME,
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def ensure_collection_exists(
        self, vector_size: int, distance_metric: str = "Cosine"
    ):
        try:
            self.client.get_collection(collection_name=self.collection_name)
            # print(f"Collection '{self.collection_name}' already exists.")
        except Exception:  # Should be more specific, e.g. UnexpectedResponse
            # print(f"Collection '{self.collection_name}' not found, creating...")
            dist = models.Distance.COSINE
            if distance_metric.lower() == "dot":
                dist = models.Distance.DOT
            elif distance_metric.lower() == "euclid":
                dist = models.Distance.EUCLID

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=dist),
            )
            # print(f"Collection '{self.collection_name}' created.")

    def upsert_chunks(self, chunks_with_embeddings: list[tuple[ChunkData, np.ndarray]]):
        points = []
        for chunk, embedding in chunks_with_embeddings:
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector=embedding.tolist(),
                    payload=chunk.model_dump(
                        exclude={"id"}
                    ),  # Store other fields in payload
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

    def similarity_search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> list[ChunkData]:
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
        )
        return [ChunkData(id=hit.id, **hit.payload) for hit in search_result]
