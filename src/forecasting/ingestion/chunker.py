from forecasting.ingestion.models import ParsedDocument, TextChunk

# For more advanced chunking, consider LangChain's text_splitters
# from langchain.text_splitter import RecursiveCharacterTextSplitter


class NaiveTextChunker:
    def __init__(self, chunk_size_chars: int = 1000, chunk_overlap_chars: int = 150):
        # For this naive version, sizes are in characters.
        # A better approach uses token counts (e.g., with tiktoken or HuggingFace tokenizers)
        # settings.CHUNK_SIZE and settings.CHUNK_OVERLAP might refer to tokens. Adjust accordingly.
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

        # Example using LangChain's splitter (would require adding 'langchain' to dependencies)
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=settings.CHUNK_SIZE, # Assumes settings.CHUNK_SIZE is in tokens
        #     chunk_overlap=settings.CHUNK_OVERLAP, # Assumes settings.CHUNK_OVERLAP is in tokens
        #     length_function=len, # Replace with a tokenizer's length function for token-based
        #     # add_start_index = True, # Useful for some applications
        # )

    def chunk_parsed_document(
        self, parsed_doc: ParsedDocument, source_document_db_id: int
    ) -> list[TextChunk]:
        text = parsed_doc.cleaned_text
        chunks: list[TextChunk] = []

        # Naive character-based splitting:
        start_index = 0
        doc_len = len(text)
        while start_index < doc_len:
            end_index = min(start_index + self.chunk_size_chars, doc_len)
            chunk_text_content = text[start_index:end_index]

            chunks.append(
                TextChunk(
                    text=chunk_text_content,
                    source_document_db_id=source_document_db_id,
                    source_name_for_display=parsed_doc.metadata.get(
                        "title", parsed_doc.source_name
                    ),
                    source_identifier_for_display=(
                        parsed_doc.identifier
                        if "http" in parsed_doc.identifier
                        else None
                    ),
                    chunk_specific_metadata={
                        "original_doc_hash": parsed_doc.original_content_hash,
                        "char_start_index": start_index,
                    },  # Example of chunk-specific metadata
                )
            )

            if end_index == doc_len:
                break
            start_index += self.chunk_size_chars - self.chunk_overlap_chars
            if (
                start_index >= end_index
            ):  # Prevent infinite loops if overlap is too large
                start_index = end_index

        # # Example using LangChain splitter (if you chose to use it):
        # # split_texts = self.text_splitter.split_text(text)
        # # for i, chunk_text_content in enumerate(split_texts):
        # #     chunks.append(TextChunk(
        # #         text=chunk_text_content,
        # #         source_document_db_id=source_document_db_id,
        # #         source_name_for_display=parsed_doc.metadata.get("title", parsed_doc.source_name),
        # #         source_identifier_for_display=parsed_doc.identifier if "http" in parsed_doc.identifier else None,
        # #         chunk_specific_metadata={"original_doc_hash": parsed_doc.original_content_hash, "chunk_index": i}
        # #     ))

        return chunks
