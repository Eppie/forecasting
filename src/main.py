from pydantic import BaseModel

from forecasting.config import settings
from forecasting.db.session import get_session_context
from forecasting.ingestion.embedder import SBERTEmbedder, SentenceEmbedderABC
from forecasting.ingestion.vector_store import (
    ChunkData,
    QdrantVectorStore,
    VectorStoreABC,
)
from forecasting.log.forecast_logger import ForecastLogger, LoggedForecastData
from forecasting.reasoning.llm_engine import LlamaCppEngine, LlmEngineABC, LLMOutput
from forecasting.reasoning.prompt import build_rag_prompt
from forecasting.retrieval.retriever import Retriever


class ForecastResult(BaseModel):
    forecast_id: int | None = None  # Populated after log
    probability_str: str | None
    probability_float: float | None
    rationale: str
    # Citations: label -> (snippet_text, full_source_reference_str)
    citations: dict[str, tuple[str, str | None]] = {}


# Global instances (consider dependency injection for more complex app)
# These will be initialized once
embedder: SentenceEmbedderABC = SBERTEmbedder()
vector_store: VectorStoreABC = QdrantVectorStore()
# Ensure collection exists on startup (or first use)
vector_store.ensure_collection_exists(vector_size=embedder.get_embedding_dimension())

retriever: Retriever = Retriever(store=vector_store, embedder=embedder)
llm_engine: LlmEngineABC = LlamaCppEngine()
forecast_logger: ForecastLogger = ForecastLogger()


def run_forecast_pipeline(
    question: str, stream_to_console: bool = False
) -> ForecastResult:  # Add stream_to_console
    # 1. Retrieve relevant chunks
    retrieved_chunks: list[ChunkData] = retriever.retrieve(question, k=5)
    print("---- RETRIEVED CHUNKS ----")
    if not retrieved_chunks:
        print("No chunks retrieved.")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i + 1} (ID: {chunk.id}, Source: {chunk.source_name}):")
        print(chunk.text[:200] + "...")  # Print a snippet
    print("--------------------------")

    # 2. Build prompt
    prompt, citation_label_to_chunk_map = build_rag_prompt(question, retrieved_chunks)
    print("---- PROMPT ----")
    print(prompt)
    # 3. Generate forecast with LLM
    if stream_to_console:
        from rich.console import Console
        from rich.live import Live

        console = Console()
        full_rationale_text = ""
        with Live(console=console, refresh_per_second=10) as live:
            live.update("Assistant thinking...")
            for token in llm_engine.stream_generate(prompt):
                full_rationale_text += token
                live.update(full_rationale_text)
        llm_response = LLMOutput.from_raw_text(full_rationale_text)
    else:
        llm_response = llm_engine.generate(prompt)

    if llm_response.probability_float is None:
        # Basic fallback or re-prompt logic could go here
        # For now, we'll just indicate it wasn't found
        print("Warning: Probability not explicitly found in LLM output.")

    # 4. Prepare citations for output
    output_citations: dict[str, tuple[str, str | None]] = {}
    for label, chunk_data in citation_label_to_chunk_map.items():
        if label in llm_response.rationale:  # Only include if actually cited
            output_citations[label] = (
                chunk_data.text,
                chunk_data.source_url or chunk_data.source_name,
            )

    # 5. Log forecast
    log_data = LoggedForecastData(
        question_text=question,
        predicted_probability=llm_response.probability_float,
        rationale_text=llm_response.rationale,
        llm_model_used=settings.LLM_MODEL_PATH.split("/")[-1],  # Simplified model name
        # Map citation_label_to_chunk_map to what logger expects (e.g., list of DocumentChunk IDs)
        # This requires DocumentChunk.vector_id to be the ChunkData.id
        # and that we can map ChunkData.document_db_id to actual DocumentChunk.id
        # For MVP, we might simplify log of specific chunks if this mapping is complex initially.
        # Or, the logger takes the `citation_label_to_chunk_map` directly.
        cited_chunk_vector_ids_with_labels={
            label: chunk.id
            for label, chunk in citation_label_to_chunk_map.items()
            if label in llm_response.rationale
        },
    )

    forecast_id = None
    with (
        get_session_context() as db_session
    ):  # Assuming get_session_context from db.session
        forecast_id = forecast_logger.log_forecast(log_data, db_session)

    return ForecastResult(
        forecast_id=forecast_id,
        probability_str=llm_response.probability_str,
        probability_float=llm_response.probability_float,
        rationale=llm_response.rationale,
        citations=output_citations,
    )
