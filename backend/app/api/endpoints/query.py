from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from app.agent.graph import app_graph

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the main findings of the paper?")

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[dict] = []

@router.post("", response_model=QueryResponse)
async def query_assistant(request: QueryRequest):
    """
    Endpoint to query the Agentic RAG assistant.
    Invokes the Corrective RAG LangGraph workflow (Retrieve -> Grade -> Rewrite -> Generate).
    """
    logger.info(f"API query received: '{request.query}'")
    
    # 1. Initialize State Context
    initial_state = {
        "query": request.query,
        "original_query": request.query,
        "documents": [],
        "answer": "",
        "citations": [],
        "rewrite_count": 0
    }
    
    try:
        # 2. Invoke State Machine Graph
        final_state = app_graph.invoke(initial_state)
        
        return QueryResponse(
            query=request.query,
            answer=final_state.get("answer", "No answer could be generated."),
            citations=final_state.get("citations", [])
        )
        
    except Exception as e:
        logger.error(f"Error executing agentic RAG flow: {e}")
        raise HTTPException(status_code=500, detail=f"Agent workflow execution failed: {str(e)}")
