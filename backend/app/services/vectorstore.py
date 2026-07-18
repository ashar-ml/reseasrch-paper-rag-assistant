import re
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger

from app.core.config import settings
from app.services.embeddings import EmbeddingService

class VectorStoreService:
    @staticmethod
    def _get_clean_collection_name(model_name: str) -> str:
        """
        Converts model name into a valid Chroma collection name.
        Rules: 3-63 characters, alphanumeric start/end, contains only alphanumeric, underscores, hyphens.
        """
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        clean_name = clean_name.strip('_')
        if len(clean_name) < 3:
            clean_name = f"col_{clean_name}"
        return f"col_{clean_name}"[:63]

    @classmethod
    def get_vectorstore(cls, model_name: str = None) -> Chroma:
        """
        Initializes and returns a Chroma instance for the specified embedding model.
        """
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL_NAME

        embeddings = EmbeddingService.get_embeddings(model_name)
        collection_name = cls._get_clean_collection_name(model_name)
        
        # Keep collections separate in the directory path to avoid dimension mismatch errors
        persist_dir = str(settings.vector_db_path / model_name.replace('/', '_'))

        logger.info(f"Connecting to Chroma collection '{collection_name}' at '{persist_dir}'...")
        
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

    @classmethod
    def add_documents(cls, documents: List[Document], model_name: str = None):
        """
        Adds a list of parsed chunk Documents to the persistent Chroma index.
        """
        if not documents:
            logger.warning("Empty document list passed. Skipping vector indexing.")
            return
            
        vectorstore = cls.get_vectorstore(model_name)
        logger.info(f"Indexing {len(documents)} chunks in ChromaDB vector store...")
        vectorstore.add_documents(documents)
        logger.info("Vector database indexing complete.")

    @classmethod
    def similarity_search(cls, query: str, k: int = 5, model_name: str = None) -> List[Document]:
        """
        Performs vector similarity search on the index.
        """
        vectorstore = cls.get_vectorstore(model_name)
        logger.info(f"Performing vector similarity search for query: '{query}' with k={k}...")
        return vectorstore.similarity_search(query, k=k)
