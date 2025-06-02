from pydantic import BaseModel
from sqlalchemy.orm import Session

from forecasting.db.models import CitedChunkLink, DocumentChunk, Forecast


class LoggedForecastData(BaseModel):
    question_text: str
    predicted_probability: float | None
    rationale_text: str | None
    llm_model_used: str | None
    # Stores {citation_label: vector_id_of_chunk}
    cited_chunk_vector_ids_with_labels: dict[str, str] = {}


class ForecastLogger:
    def log_forecast(self, data: LoggedForecastData, db: Session) -> int:
        forecast_entry = Forecast(
            question_text=data.question_text,
            predicted_probability=data.predicted_probability,
            rationale_text=data.rationale_text,
            llm_model_used=data.llm_model_used,
        )
        db.add(forecast_entry)
        db.flush()  # To get forecast_entry.id

        # Link cited chunks
        for label, vector_id in data.cited_chunk_vector_ids_with_labels.items():
            # Find DocumentChunk by its vector_id
            # This assumes DocumentChunk.vector_id is populated and unique
            chunk_db_entry = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.vector_id == vector_id)
                .first()
            )
            if chunk_db_entry:
                link = CitedChunkLink(
                    forecast_id=forecast_entry.id,
                    document_chunk_id=chunk_db_entry.id,
                    citation_label=label,
                )
                db.add(link)
            else:
                print(
                    f"Warning: Could not find DocumentChunk with vector_id {vector_id} to link citation {label}"
                )

        db.commit()
        db.refresh(forecast_entry)
        return forecast_entry.id
