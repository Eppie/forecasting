from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

from forecasting.config import settings


class SentenceEmbedderABC(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        pass


class SBERTEmbedder(SentenceEmbedderABC):
    _model = None  # Lazy load

    def __init__(self, model_id: str = settings.EMBEDDING_MODEL_ID):
        self.model_id = model_id
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        if SBERTEmbedder._model is None:
            # print(f"Loading embedding model: {self.model_id}...")
            SBERTEmbedder._model = SentenceTransformer(self.model_id)
            # print("Embedding model loaded.")

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_model_loaded()
        return SBERTEmbedder._model.encode(texts, convert_to_numpy=True)

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_model_loaded()
        embeddings = SBERTEmbedder._model.encode([text], convert_to_numpy=True)
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        self._ensure_model_loaded()
        return SBERTEmbedder._model.get_sentence_embedding_dimension()
