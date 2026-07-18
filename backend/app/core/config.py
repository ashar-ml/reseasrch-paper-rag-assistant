import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

# Resolve workspace directory: config.py is at backend/app/core/config.py, so parent.parent.parent.parent is root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    # LLM settings
    LLM_PROVIDER: str = Field("groq", description="LLM provider: 'groq' or 'ollama'")
    GROQ_API_KEY: str = Field("gsk_your_api_key_here", description="Groq API Key")
    GROQ_MODEL: str = Field("llama3-8b-8192", description="Groq Model name")
    
    OLLAMA_BASE_URL: str = Field("http://localhost:11434", description="Ollama API base URL")
    OLLAMA_MODEL: str = Field("llama3", description="Ollama Model name")
    
    # Embedding settings
    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", description="HuggingFace Embedding model")
    
    # Storage settings
    VECTOR_DB_DIR: str = Field("data/vector_db", description="Vector DB storage directory")
    RAW_DATA_DIR: str = Field("data/raw", description="Raw PDF upload directory")
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # UI Connection settings
    BACKEND_API_URL: str = "http://localhost:8000"

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    @property
    def vector_db_path(self) -> Path:
        path = Path(self.VECTOR_DB_DIR)
        if not path.is_absolute():
            path = BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_data_path(self) -> Path:
        path = Path(self.RAW_DATA_DIR)
        if not path.is_absolute():
            path = BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()
