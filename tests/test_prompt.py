# tests/test_prompt.py
from forecasting.reasoning.prompt import build_rag_prompt
from src.forecasting.ingestion.vector_store import ChunkData


def test_build_rag_prompt_with_chunks():
    question = "What is the capital of France?"
    chunks = [
        ChunkData(
            id="v1", text="Paris is a nice city.", document_db_id=1, source_name="Doc1"
        ),
        ChunkData(
            id="v2", text="France is in Europe.", document_db_id=2, source_name="Doc2"
        ),
    ]
    prompt, citation_map = build_rag_prompt(question, chunks)
    assert question in prompt
    assert "[A]: Paris is a nice city." in prompt
    assert "(Source: Doc1, Document DB ID: 1)" in prompt
    assert "[B]: France is in Europe." in prompt
    assert citation_map["[A]"].text == "Paris is a nice city."
    assert "User Question: What is the capital of France?" in prompt
    assert "You are an expert forecasting assistant." in prompt


def test_build_rag_prompt_no_chunks():
    question = "What is the capital of Mars?"
    prompt, citation_map = build_rag_prompt(question, [])
    assert "No specific documents found." in prompt
    assert not citation_map
