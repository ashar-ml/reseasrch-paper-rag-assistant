from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from loguru import logger

from app.services.vectorstore import VectorStoreService

class HybridRetriever:
    @classmethod
    def retrieve(cls, query: str, k: int = 10, model_name: str = None) -> List[Document]:
        """
        Performs hybrid search blending BM25 (sparse matching) and Chroma Vector similarity (dense matching).
        """
        # 1. Vector Dense Retrieval
        try:
            vector_docs = VectorStoreService.similarity_search(query, k=k, model_name=model_name)
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            vector_docs = []

        # 2. BM25 Sparse Retrieval
        sparse_docs = []
        try:
            vectorstore = VectorStoreService.get_vectorstore(model_name)
            # Retrieve all document chunks in collection to build the BM25 index on the fly
            db_data = vectorstore.get(include=["documents", "metadatas"])
            
            documents_text = db_data.get("documents", [])
            metadatas = db_data.get("metadatas", [])
            
            if documents_text:
                # Tokenize corpus for BM25 matching
                tokenized_corpus = [doc.lower().split(" ") for doc in documents_text]
                bm25 = BM25Okapi(tokenized_corpus)
                
                tokenized_query = query.lower().split(" ")
                doc_scores = bm25.get_scores(tokenized_query)
                
                # Sort documents by score and take top K
                top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
                
                for idx in top_indices:
                    # Only add if there is a positive match score
                    if doc_scores[idx] > 0:
                        doc = Document(
                            page_content=documents_text[idx],
                            metadata=metadatas[idx]
                        )
                        sparse_docs.append(doc)
        except Exception as e:
            logger.error(f"BM25 sparse search failed: {e}")

        # 3. Interleaving Blending & Deduplication
        combined_docs = []
        seen_keys = set()
        
        max_len = max(len(vector_docs), len(sparse_docs))
        for i in range(max_len):
            # Interleave Vector docs
            if i < len(vector_docs):
                doc = vector_docs[i]
                key = (doc.page_content.strip(), doc.metadata.get("source"), doc.metadata.get("page"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined_docs.append(doc)
            
            # Interleave BM25 docs
            if i < len(sparse_docs):
                doc = sparse_docs[i]
                key = (doc.page_content.strip(), doc.metadata.get("source"), doc.metadata.get("page"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined_docs.append(doc)

        logger.info(f"Hybrid retrieval blending finished. Blended {len(combined_docs)} candidate chunks (Dense: {len(vector_docs)}, Sparse: {len(sparse_docs)}).")
        return combined_docs
