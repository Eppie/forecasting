from forecasting.ingestion.embedder import SentenceEmbedderABC
from forecasting.ingestion.vector_store import ChunkData, VectorStoreABC


class Retriever:
    def __init__(self, store: VectorStoreABC, embedder: SentenceEmbedderABC):
        self.store = store
        self.embedder = embedder

    def retrieve(self, question: str, k: int = 5) -> list[ChunkData]:
        query_embedding = self.embedder.embed_text(question)
        # Ensure vector store collection exists before searching
        # This check might be better placed at app startup or during ingestion pipeline init
        # For MVP, doing it here is okay, but consider refactoring.
        # self.store.ensure_collection_exists(vector_size=self.embedder.get_embedding_dimension())

        retrieved_chunks = self.store.similarity_search(query_embedding, k=k)
        return retrieved_chunks
