from pydantic import HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: PostgresDsn = "postgresql://eppie@localhost:5432/forecastdb"
    LLM_MODEL_PATH: str = (
        "/Users/eppie/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q3_K_L.gguf"
    )
    EMBEDDING_MODEL_ID: str = "all-MiniLM-L6-v2"
    VECTOR_DB_URL: HttpUrl | str = "http://localhost:6333"  # Qdrant URL
    VECTOR_DB_COLLECTION_NAME: str = "forecast_docs"
    CHUNK_SIZE: int = 300  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
