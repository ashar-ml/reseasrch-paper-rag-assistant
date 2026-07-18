import time
from typing import Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
from loguru import logger

class EmbeddingService:
    _instances: Dict[str, HuggingFaceEmbeddings] = {}

    @classmethod
    def get_embeddings(cls, model_name: str = None) -> HuggingFaceEmbeddings:
        """
        Returns a cached instance of HuggingFaceEmbeddings for the specified model.
        """
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL_NAME

        # Map short name to Hugging Face model path
        hf_model_path = model_name
        if model_name == "all-MiniLM-L6-v2":
            hf_model_path = "sentence-transformers/all-MiniLM-L6-v2"
        elif model_name == "BAAI/bge-large-en-v1.5":
            hf_model_path = "BAAI/bge-large-en-v1.5"
        elif model_name == "BAAI/bge-small-en-v1.5":
            hf_model_path = "BAAI/bge-small-en-v1.5"

        if model_name not in cls._instances:
            logger.info(f"Initializing embedding model: {model_name} ({hf_model_path})...")
            start_time = time.time()
            
            # Check if CUDA is available for acceleration
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Embedding model execution device configured: {device}")

            cls._instances[model_name] = HuggingFaceEmbeddings(
                model_name=hf_model_path,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}  # Standardizes cosine distance calculations
            )
            logger.info(f"Loaded {model_name} in {time.time() - start_time:.2f} seconds.")

        return cls._instances[model_name]

    @classmethod
    def get_dimension(cls, model_name: str) -> int:
        """
        Returns vector output size for the chosen model.
        """
        if "MiniLM" in model_name:
            return 384
        elif "bge-large" in model_name:
            return 1024
        elif "bge-small" in model_name:
            return 384
        return 768

