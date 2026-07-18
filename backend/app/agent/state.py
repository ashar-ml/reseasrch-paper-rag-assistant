from typing import Any, Dict, List, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    State tracking schema representing the context parameters passed between LangGraph nodes.
    """
    query: str
    original_query: str
    documents: List[Document]
    answer: str
    citations: List[Dict[str, Any]]
    rewrite_count: int
