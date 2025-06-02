# src/forecasting/ingestion/parser.py
import hashlib

import fitz  # PyMuPDF, add 'PyMuPDF' to dependencies
from bs4 import BeautifulSoup  # Add 'beautifulsoup4' and 'lxml' to dependencies

from forecasting.ingestion.models import ParsedDocument, RawDocument


class DocumentParser:
    def parse(self, raw_doc: RawDocument) -> ParsedDocument | None:
        """Parses a RawDocument into a ParsedDocument. Returns None if parsing fails."""
        text = ""
        try:
            if raw_doc.content_type.lower() == "text/html":
                text = self._html_to_text(raw_doc.content)
            elif raw_doc.content_type.lower() == "application/pdf":
                if not isinstance(raw_doc.content, bytes):
                    print(
                        f"Error: PDF content for {raw_doc.identifier} is not bytes. Skipping."
                    )
                    return None
                text = self._pdf_to_text(raw_doc.content)
            elif raw_doc.content_type.lower() == "text/plain":
                text = (
                    raw_doc.content
                    if isinstance(raw_doc.content, str)
                    else raw_doc.content.decode("utf-8", errors="ignore")
                )
            else:
                print(
                    f"Warning: Unsupported content type '{raw_doc.content_type}' for {raw_doc.identifier}. Skipping."
                )
                return None
        except Exception as e:
            print(
                f"Error parsing document {raw_doc.identifier} with content type {raw_doc.content_type}: {e}"
            )
            return None

        if not text.strip():  # If parsing resulted in no text
            print(f"Warning: No text extracted from {raw_doc.identifier}. Skipping.")
            return None

        content_for_hash = (
            raw_doc.content
            if isinstance(raw_doc.content, bytes)
            else str(raw_doc.content).encode("utf-8", errors="ignore")
        )
        content_hash = hashlib.sha256(content_for_hash).hexdigest()

        return ParsedDocument(
            source_name=raw_doc.source_name,
            identifier=raw_doc.identifier,
            cleaned_text=text.strip(),
            metadata=raw_doc.metadata,  # Pass along metadata
            original_content_hash=content_hash,
        )

    def _html_to_text(self, html_content: str | bytes) -> str:
        soup = BeautifulSoup(html_content, "lxml")
        for script_or_style in soup(
            ["script", "style", "header", "footer", "nav", "aside"]
        ):  # Remove common non-content tags
            script_or_style.decompose()
        # Get text, join paragraphs, and clean up whitespace
        # text = ' '.join(soup.stripped_strings) # This can sometimes be too aggressive
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all(["p", "div", "article"])
        ]  # More targeted
        text = "\n\n".join(
            filter(None, paragraphs)
        )  # Join paragraphs with double newlines
        if not text:  # Fallback if no p/div/article tags yielded content
            text = soup.get_text(separator=" ", strip=True)
        return text

    def _pdf_to_text(self, pdf_bytes: bytes) -> str:
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text(
                    "text", sort=True
                )  # "text" for plain text, sort for reading order
                if page_text:
                    text += page_text + "\n\n"  # Add double newline between pages
        return text.strip()
