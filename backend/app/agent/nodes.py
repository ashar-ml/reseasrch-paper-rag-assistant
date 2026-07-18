import re
from typing import Dict, Any, List
from langchain_core.documents import Document
from loguru import logger

from app.agent.state import AgentState
from app.services.parser import get_llm
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import RerankerService

# Centralized LLM fetcher reuse
llm = get_llm()

class AgentNodes:
    @staticmethod
    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """
        Retrieves candidate documents from hybrid search (Vector + BM25)
        and reranks them using the Cross-Encoder.
        """
        query = state["query"]
        logger.info(f"[LangGraph Node: Retrieve] Fetching documents for query: '{query}'...")
        
        # 1. Hybrid retrieval (fetch top 12 blended candidates)
        candidates = HybridRetriever.retrieve(query, k=12)
        
        # 2. Rerank to select top 4 most relevant chunks
        reranked_docs = RerankerService.rerank(query, candidates, top_n=4)
        
        return {"documents": reranked_docs}

    @staticmethod
    def grade_documents_node(state: AgentState) -> Dict[str, Any]:
        """
        Grades retrieved documents for relevance to the original query.
        Filters out irrelevant chunks.
        """
        original_query = state["original_query"]
        documents = state["documents"]
        logger.info(f"[LangGraph Node: Grade] Evaluating {len(documents)} docs for relevance to '{original_query}'...")
        
        relevant_docs = []
        
        for idx, doc in enumerate(documents):
            prompt = f"""You are an academic relevance grader. Grade whether the research paper chunk below is relevant to the user query.
Query: {original_query}
Paper Chunk Content: {doc.page_content}

Return exactly 'yes' if relevant, or 'no' if irrelevant. Do not write explanation, comments, or extra text. Output only a single word.
Relevance Grade (yes/no):"""
            
            try:
                response = llm.invoke(prompt)
                grade = response.content.strip().lower()
                
                # Check for standard response matches
                if "yes" in grade:
                    logger.info(f"  - Chunk [{idx+1}] graded: RELEVANT")
                    relevant_docs.append(doc)
                else:
                    logger.info(f"  - Chunk [{idx+1}] graded: IRRELEVANT")
            except Exception as e:
                logger.error(f"Failed to grade document {idx+1}: {e}")
                # Fallback: keep document if grading fails to prevent data loss
                relevant_docs.append(doc)
                
        return {"documents": relevant_docs}

    @staticmethod
    def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
        """
        Optimizes/rewrites the query using LLM if retrieval candidates are found irrelevant.
        """
        original_query = state["original_query"]
        rewrite_count = state.get("rewrite_count", 0)
        logger.info(f"[LangGraph Node: Rewrite] Rewriting search query for: '{original_query}' (Current count: {rewrite_count})")
        
        prompt = f"""You are an academic search query optimizer. Rewrite the research question below to make it better suited for semantic vector database search. Focus on extracting main terms, synonyms, and academic concepts.
Original Query: {original_query}

Optimized Query (return ONLY the search query, do not write explanations, tags, or quotes):"""
        
        try:
            response = llm.invoke(prompt)
            rewritten_query = response.content.strip().strip('"').strip("'")
            logger.info(f"  - Optimized Query: '{rewritten_query}'")
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            rewritten_query = original_query
            
        return {
            "query": rewritten_query,
            "rewrite_count": rewrite_count + 1
        }

    @staticmethod
    def generate_node(state: AgentState) -> Dict[str, Any]:
        """
        Generates the final response citing sources and mapping citations.
        """
        original_query = state["original_query"]
        documents = state["documents"]
        logger.info(f"[LangGraph Node: Generate] Generating final answer for query: '{original_query}'...")
        
        if not documents:
            return {
                "answer": "I could not find any relevant information in the uploaded research papers to answer your query.",
                "citations": []
            }
            
        # 1. Format document sources for prompt context
        context_blocks = []
        citations_list = []
        
        for idx, doc in enumerate(documents):
            cite_num = idx + 1
            title = doc.metadata.get("paper_title", "Unknown Title")
            authors = doc.metadata.get("authors", "Unknown")
            page = doc.metadata.get("page", 1)
            
            context_blocks.append(f"Source [{cite_num}] (Paper: {title} | Authors: {authors} | Page: {page}):\n{doc.page_content}")
            
            citations_list.append({
                "index": cite_num,
                "paper_title": title,
                "page": page,
                "text_snippet": doc.metadata.get("original_text", doc.page_content[:200])
            })
            
        context_str = "\n\n".join(context_blocks)
        
        prompt = f"""You are a production-grade Senior Academic Research Assistant. Answer the user query based ONLY on the research paper sources provided below.
For every claim or fact you state, you MUST append the corresponding source citation number like [1], [2], etc.
Answer objectively, analytically, and match academic standards. If the sources do not contain enough info, state this clearly.

Sources Context:
---
{context_str}
---

Query: {original_query}

Answer with Citations:"""
        
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            answer = f"Error generating answer: {str(e)}"
            
        return {
            "answer": answer,
            "citations": citations_list
        }
