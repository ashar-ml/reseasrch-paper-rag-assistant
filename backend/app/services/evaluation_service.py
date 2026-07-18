import time
from typing import Any, Dict, List
import numpy as np
from loguru import logger

from app.services.embeddings import EmbeddingService

class EmbeddingEvaluator:
    @staticmethod
    def benchmark_models(test_queries: List[str] = None, test_documents: List[str] = None) -> Dict[str, Any]:
        """
        Benchmarks all-MiniLM-L6-v2 vs BAAI/bge-large-en-v1.5.
        Measures encoding latency, vector dimensions, and cosine similarity spread.
        """
        if not test_queries:
            test_queries = [
                "What is the core architecture of the transformer model?",
                "What are the optimization techniques for hybrid retrieval?",
                "How does agentic RAG prevent hallucinations?"
            ]
        if not test_documents:
            test_documents = [
                "The transformer model uses self-attention mechanisms to capture dependencies in sequence processing.",
                "Hybrid retrieval combines dense vector embeddings with sparse BM25 indexing algorithms to rank items.",
                "Agentic RAG queries agents dynamically, routing retrieved chunks to grading nodes to verify facts."
            ]

        models_to_test = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

        results = {}

        for model_name in models_to_test:
            try:
                logger.info(f"Evaluating embedding model benchmark: {model_name}...")
                embeddings = EmbeddingService.get_embeddings(model_name)
                
                # Measure Query Encoding Latency
                start_time = time.time()
                query_vectors = [embeddings.embed_query(q) for q in test_queries]
                query_latency = (time.time() - start_time) / len(test_queries)
                
                # Measure Document Encoding Latency
                start_time = time.time()
                doc_vectors = [embeddings.embed_documents(test_documents)]
                doc_latency = (time.time() - start_time) / len(test_documents)
                
                # Flatten the list of vectors since embed_documents returns List[List[float]]
                doc_vectors_flat = doc_vectors[0]
                
                # Compute Cosine Similarities (vectors are normalized to length 1 by default, dot product is cosine similarity)
                similarities = []
                for q_vec in query_vectors:
                    for d_vec in doc_vectors_flat:
                        sim = np.dot(q_vec, d_vec)
                        similarities.append(float(sim))
                
                results[model_name] = {
                    "model_name": model_name,
                    "dimension": EmbeddingService.get_dimension(model_name),
                    "query_encoding_latency_ms": round(query_latency * 1000, 2),
                    "doc_encoding_latency_ms": round(doc_latency * 1000, 2),
                    "average_similarity": round(float(np.mean(similarities)), 4),
                    "max_similarity": round(float(np.max(similarities)), 4),
                    "min_similarity": round(float(np.min(similarities)), 4),
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error benchmarking model {model_name}: {e}")
                results[model_name] = {
                    "model_name": model_name,
                    "status": "error",
                    "error": str(e)
                }

        return results

