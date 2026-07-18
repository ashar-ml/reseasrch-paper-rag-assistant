import time
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from loguru import logger

class RerankerService:
    _model = None

    @classmethod
    def get_model(cls) -> CrossEncoder:
        if cls._model is None:
            logger.info("Initializing Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
            start_time = time.time()
            
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Cross-Encoder execution device configured: {device}")
            
            cls._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            logger.info(f"Cross-Encoder model loaded in {time.time() - start_time:.2f} seconds.")
            
        return cls._model

    @classmethod
    def rerank(cls, query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
        """
        Scores query-document content pairs and sorts documents by relevance.
        Returns the top_n documents.
        """
        if not documents:
            logger.warning("Empty document list passed for reranking.")
            return []
            
        model = cls.get_model()
        logger.info(f"Reranking {len(documents)} document chunks with Cross-Encoder...")
        
        # Construct query-chunk text pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict relevance scores
        scores = model.predict(pairs)
        
        # Sort documents based on prediction scores
        ranked_docs = []
        for idx, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
            doc = documents[idx]
            doc.metadata["rerank_score"] = float(score)
            ranked_docs.append(doc)
            
        logger.info(f"Reranking complete. Top score: {ranked_docs[0].metadata['rerank_score']:.4f}")
        return ranked_docs[:top_n]
