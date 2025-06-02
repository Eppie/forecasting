from forecasting.ingestion.vector_store import ChunkData


def build_rag_prompt(
    question: str, retrieved_chunks: list[ChunkData]
) -> tuple[str, dict[str, ChunkData]]:
    """
    Builds the prompt for the LLM, including retrieved context and citation mapping.
    Returns the prompt string and a dictionary mapping citation labels to ChunkData.
    """
    context_str = ""
    citation_map: dict[str, ChunkData] = {}

    if not retrieved_chunks:
        context_str = "No specific documents found. Please answer based on your general knowledge.\n"
    else:
        context_str += "Use the following retrieved context to answer the question. Cite sources using labels like [A], [B], etc., for each factual claim from these sources:\n\n"
        for i, chunk_data in enumerate(retrieved_chunks):
            label = f"[{chr(ord('A') + i)}]"
            context_str += f"{label}: {chunk_data.text}\n(Source: {chunk_data.source_name}, Document DB ID: {chunk_data.document_db_id})\n\n"
            citation_map[label] = chunk_data

    system_message = (
        "You are an expert forecasting assistant. Your goal is to provide a probabilistic forecast "
        "for the user's question. Please follow these steps:\n"
        "1. Decompose the problem if it's complex.\n"
        "2. Identify a suitable reference class of analogous historical cases and state the base rate (frequency) of the outcome in that class. "
        "If no direct reference class is available from the provided context or your general knowledge, state that explicitly.\n"
        "3. Consider any inside-view factors – specifics of the current situation from the context or general knowledge – that might make it deviate from the base rate.\n"
        "4. Adjust the probability based on these factors.\n"
        '5. Output a final numeric probability (e.g., "Final Probability: X%").\n'
        "6. Provide a step-by-step reasoning for your forecast. Explicitly cite facts using the provided source labels (e.g., [A], [B]) where appropriate.\n"
        "If the provided context is insufficient, clearly state that and rely on your general knowledge, mentioning its limitations if any.\n"
    )

    # For llama.cpp, a common chat format is:
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
    # {{ user_prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    # Using a simpler format for now, adaptable to specific model needs.
    # Some models prefer system prompt separate, others combined.
    # This example assumes a model that takes a single combined prompt.

    full_prompt = f"{system_message}\n\nContext:\n{context_str}\nUser Question: {question}\n\nAssistant's Step-by-Step Forecast:"
    return full_prompt, citation_map
